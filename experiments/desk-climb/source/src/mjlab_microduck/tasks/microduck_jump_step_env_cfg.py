"""Microduck single-step jump task ("jump onto a low platform").

Built on the velocity recipe (BAM actuators, full DR + obs-noise + NaN-guard
stack, 61-D observation contract) with flat terrain plus one mocap platform.

Task
----
* The robot spawns facing +x with a 60 mm deep x 230 mm wide platform ahead.
  Spawn mix (v11): ``ballistic_spawn_prob`` (0.5 -> 0 by curriculum) — the
  "just jumped" state: HOME pose, soles 1-2 cm off the ground 0.05-0.15 m
  from the platform, with a SOLVED launch velocity (arc apex 5-15 cm above
  the top, coming down on the platform centre).  v10 showed the crouch (v8)
  and the landing (v9, 57.9 % deterministic) exist but the coordinated
  takeoff is never discovered from the floor — so the nearly-done spawn
  moves one step back from the end state (landing) to mid-flight.
  ``airborne_spawn_prob`` (constant 0.2) keeps the drop-onto-platform
  landing skill alive (v10's retiring it diluted the skill);
  ``edge_spawn_prob`` (0.15) near-edge floor; the rest far floor.  The
  ballistic fraction retires as the BALLISTIC-episode landed rate climbs
  (thresholds 0.10/0.20/0.35 -> prob 0.3/0.15/0.0, 200-episode windows,
  bidirectional).  The platform top height IS the twist command: ``vx`` is
  sampled in 0.025-0.035 m and the reset event places the mocap platform so
  its top sits at that height.  Jump onto the platform and stand.
* Reward is episodic-style (AGENTS.md): potential-based trunk-height progress
  (paid on the DELTA of min(trunk_z, platform_top + 0.09), per-step capped —
  rising pays, holding pays zero, falling pays back, so it cannot be farmed),
  a potential-based forward progress toward the platform centre measured on
  the FEET (v7 — the trunk version was collected by leaning over the platform
  head-first; feet don't advance during a lean), a per-foot z-progress that
  pays only within 15 cm of the platform centre (teaches clearing the top
  with the FEET), a one-shot first-foot-on-top bonus that only pays mid-
  transition with the trailing foot airborne (v8 — the v7 static one-foot
  park farmed it), a head/trunk platform-contact cost (the anti-lean price),
  a one-shot success bonus (both feet above the platform top, feet supported
  by platform contact, trunk upright — full pay only for episodes that
  started on the FLOOR; platform-top spawns collect a small fraction so
  "stand still" cannot out-earn the approach, the v5 lesson), an |a_z|-class
  landing-impact cost, and a light action-rate tax that stays small during
  skill discovery.
* A fall terminates with no positive payoff; landing is a time_out success
  after the success condition holds for 0.4 s.

Command/platform synchronisation: ``command_manager.reset`` resamples commands
AFTER reset-mode events run (``ManagerBasedRlEnv._reset_idx``), so a reset
event can only read the PREVIOUS episode's command.  The reset event therefore
places the platform as a best effort, and a ``step``-mode event re-pins the
platform top to the live command (only when it changed) — the platform always
matches what the actor observes in the twist slot.  The resampling interval
(8-12 s) is longer than the 5 s episode, so the command is constant per
episode and the sync fires once per reset in practice.

Config landmine (cost a full 2000-iter run, 2026-09-23): the base velocity
template sets ``rel_forward_envs=0.2`` — forward envs get
``vx = |vx|.clamp(min=0.3)`` on every resample, silently overwriting the
0.03-0.06 platform height with a 30 cm slab floating in the sky on ~20% of
episodes.  ``rel_forward_envs`` (like rel_standing/heading/turn) MUST be
zeroed when repurposing the twist slot for a non-locomotion command.

Observation contract is untouched: twist slot = [platform height, 0, 0],
head/body slots keep the velocity defaults (tiny alive ranges).
"""

from __future__ import annotations

import math
import os
from copy import deepcopy

import mujoco
import torch

from mjlab.entity import Entity, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    NUM_STEPS_PER_ENV,
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)

# --- platform geometry ---------------------------------------------------------
PLATFORM_ENTITY = "jump_platform"
PLATFORM_CONTACT_SENSOR = "feet_platform_contact"
PLATFORM_DEPTH_M = 0.060  # x extent: the whole 54 mm sole fits on top
PLATFORM_WIDTH_M = 0.230  # y extent
PLATFORM_HALF_THICKNESS_M = 0.015  # 30 mm slab; the top follows the command
PLATFORM_HEIGHT_RANGE = (0.025, 0.035)  # m, sampled as the twist vx command
PLATFORM_DISTANCE_RANGE = (0.18, 0.25)  # m, robot root to the platform front face
EDGE_DISTANCE_RANGE = (0.04, 0.08)  # m, edge-spawn root to the front face
AIRBORNE_CLEARANCE_RANGE = (0.08, 0.15)  # m, sole clearance above the platform top
# v11 ballistic ("just jumped") spawn anchors.  Apex margin A above the TOP:
# the arc must clear the 25-35 mm top with landing margin.  Sole clearance c
# and front-face distance d at spawn.  Launch velocities are SOLVED, not
# sampled: vz = sqrt(2 g (h + A - c)) so the arc apex is exactly A above the
# top; flight time t = vz/g + sqrt(2 A/g) (up to apex, then down to top
# level); vx = (d + 0.02) / t so the trajectory comes down on the platform
# centre minus the 10 mm standing offset.  Independent vx sampling would
# systematically miss the platform.
BALLISTIC_APEX_RANGE = (0.05, 0.15)  # m, arc apex above the platform top
BALLISTIC_DISTANCE_RANGE = (0.05, 0.15)  # m, spawn root to the front face
BALLISTIC_SOLE_CLEARANCE_RANGE = (0.01, 0.02)  # m, soles just off the ground
GRAVITY = 9.81

# Spawn types (per-episode, recorded at reset for spawn-typed rewards).
SPAWN_FLOOR = 0  # far floor spawn (PLATFORM_DISTANCE_RANGE)
SPAWN_EDGE = 1  # near-edge floor spawn (EDGE_DISTANCE_RANGE)
SPAWN_PLATFORM = 2  # reverse-curriculum spawn on the platform top
SPAWN_AIRBORNE = 3  # reverse-curriculum spawn dropping onto the platform top
SPAWN_BALLISTIC = 4  # reverse-curriculum spawn mid-flight toward the platform

# Airborne curriculum (v10 thresholds, superseded as the DRIVER by ballistic
# in v11 — the function is kept for the stage-logic tests; airborne spawns
# are now a CONSTANT 0.2 share so the landing skill is not diluted).
AIRBORNE_CURRICULUM = (
    {"rate": 0.50, "prob": 0.0},
    {"rate": 0.30, "prob": 0.2},
    {"rate": 0.15, "prob": 0.4},
    {"rate": 0.0, "prob": 0.6},
)
# v11 ballistic curriculum: driver = BALLISTIC-episode landed rate.  First
# rung 0.10 — more conservative than airborne's 0.15 because ballistic adds
# flight management on top of the landing (the v10 airborne training curve
# 0.0001 -> 0.48 over 2000 iters at 60% share says a harder skill starts
# lower and climbs slower); retirement at 0.35 (below airborne's 0.50 — the
# ballistic state is a means to the floor jump, so floor pressure should
# dominate earlier).  Bidirectional, 200-episode windows.
BALLISTIC_CURRICULUM = (
    {"rate": 0.35, "prob": 0.0},
    {"rate": 0.20, "prob": 0.15},
    {"rate": 0.10, "prob": 0.3},
    {"rate": 0.0, "prob": 0.5},
)
AIRBORNE_CURRICULUM_MIN_EPISODES = 200

EPISODE_LENGTH_S = 5.0
PROGRESS_CAP_ABOVE_TOP_M = 0.09  # pay min(trunk_z, top + this) deltas
PROGRESS_MAX_DELTA_M = 0.005  # per-step pay cap (anti-jackpot rate limit)
FOOT_PROGRESS_GATE_M = 0.15  # foot z-progress pays only this near the platform centre
SUCCESS_FOOT_MARGIN_M = 0.005  # both feet above top - this
SUCCESS_UPRIGHT_MIN = 0.9  # -projected_gravity_b z
SUCCESS_HOLD_S = 0.4
# Nearly-done spawns collect a fraction of the 40-weight success (v6 lesson:
# goal-state spawns must not out-earn the real maneuver).  Platform-top and
# airborne-above: 0.075 = 3 points (free states).  Ballistic: 0.15 = 6 —
# farther from done than a free fall (the flight still has to be managed and
# the landing stuck), so it earns more, but well below the floor-born 40.
SUCCESS_PLATFORM_SPAWN_SCALE = 0.075
SUCCESS_BALLISTIC_SPAWN_SCALE = 0.15
FALLEN_GRAVITY_Z = -0.5  # fallen: projected_gravity_b z above this


def _platform_spec() -> mujoco.MjSpec:
    """Mocap body with a single box geom (see floor_desk.spec_for for the pattern)."""
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=PLATFORM_ENTITY, mocap=True)
    geom = body.add_geom(
        name=PLATFORM_ENTITY,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(0.0, 0.0, 0.0),
        size=(0.5 * PLATFORM_DEPTH_M, 0.5 * PLATFORM_WIDTH_M, PLATFORM_HALF_THICKNESS_M),
    )
    geom.friction = [1.0, 0.005, 0.0001]
    geom.solref = [0.01, 1.0]
    geom.solimp = [0.95, 0.99, 0.001, 0.5, 2.0]
    geom.priority = 1
    geom.condim = 3
    geom.rgba = [0.62, 0.48, 0.30, 1]
    return spec


def _platform_entity_cfg() -> EntityCfg:
    """One mocap platform per env, parked below the floor until reset."""
    return EntityCfg(spec_fn=_platform_spec, init_state=EntityCfg.InitialStateCfg(pos=(0, 0, -2)))


# --- per-env episode state -------------------------------------------------------


class _JumpStepState:
    def __init__(self, env: ManagerBasedRlEnv):
        n, dev = env.num_envs, env.device
        self.center_xy = torch.zeros(n, 2, device=dev)  # mocap body xy (world)
        self.top = torch.zeros(n, device=dev)  # platform top z (world)
        self.synced_height = torch.full((n,), -1.0, device=dev)  # command the mocap pose reflects
        self.prev_potential = torch.zeros(n, device=dev)
        self.potential_fresh = torch.ones(n, dtype=torch.bool, device=dev)
        self.prev_forward = torch.zeros(n, device=dev)
        self.forward_fresh = torch.ones(n, dtype=torch.bool, device=dev)
        self.prev_foot = torch.zeros(n, 2, device=dev)  # per-foot z potential
        self.foot_fresh = torch.ones(n, dtype=torch.bool, device=dev)
        self.success_latched = torch.zeros(n, dtype=torch.bool, device=dev)
        self.first_foot_latched = torch.zeros(n, dtype=torch.bool, device=dev)
        self.prev_vz = torch.zeros(n, device=dev)
        self.hold_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.hold_step = -1
        self.spawn_type = torch.zeros(n, dtype=torch.long, device=dev)  # SPAWN_*
        # Airborne/ballistic-curriculum accounting: per-episode flags plus
        # global counters (window resets on every curriculum stage change).
        self.episode_started = torch.zeros(n, dtype=torch.bool, device=dev)
        self.episode_landed = torch.zeros(n, dtype=torch.bool, device=dev)
        self.airborne_done = 0
        self.airborne_landed = 0
        self.ballistic_done = 0
        self.ballistic_landed = 0


def _jump_step_state(env: ManagerBasedRlEnv) -> _JumpStepState:
    state = getattr(env, "_jump_step_state", None)
    if state is None:
        state = _JumpStepState(env)
        env._jump_step_state = state
    return state


def spawn_class_outcomes(spawn_type: torch.Tensor, landed: torch.Tensor) -> dict[str, tuple[int, int]]:
    """Per-spawn-class episode accounting: ``{class_name: (episodes, landed)}``.

    Labels live NEXT to their masks on purpose — a positional bincount index
    (``bincount((type == AIRBORNE).long())``) silently swapped the airborne
    and floor buckets in the 2026-09-24 v9 eval.  Used by the eval script and
    unit-tested against known distributions.
    """
    out: dict[str, tuple[int, int]] = {}
    for name, cls in (
        ("floor", SPAWN_FLOOR),
        ("edge", SPAWN_EDGE),
        ("platform", SPAWN_PLATFORM),
        ("airborne", SPAWN_AIRBORNE),
        ("ballistic", SPAWN_BALLISTIC),
    ):
        mask = spawn_type == cls
        out[name] = (int(mask.sum()), int((mask & landed.bool()).sum()))
    return out


def _book_ended_episodes(state: _JumpStepState, env_ids: torch.Tensor, spawn_cls: int) -> tuple[int, int]:
    """Count ended episodes (and landed ones) whose PREVIOUS spawn type was
    ``spawn_cls`` — the curriculum drivers.  Must be called BEFORE the new
    spawn-type draw overwrites ``state.spawn_type``; envs on their first
    episode (``episode_started`` False) are skipped."""
    prev = state.episode_started[env_ids] & (state.spawn_type[env_ids] == spawn_cls)
    done = int(prev.sum())
    landed = int((prev & state.episode_landed[env_ids]).sum())
    return done, landed


def _ballistic_launch(
    top_rel: torch.Tensor,
    apex: torch.Tensor,
    clearance: torch.Tensor,
    distance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve the (vx, vz) launch velocity for the ballistic spawn.

    ``top_rel``: platform top above the ground (the twist command);
    ``apex``: arc apex above the top; ``clearance``: sole height at spawn;
    ``distance``: spawn root to the platform front face.  vz = sqrt(2 g
    (top + apex - clearance)) puts the arc apex exactly ``apex`` above the
    top; the flight time to come back down to top level is
    t = vz/g + sqrt(2 apex/g); vx covers (distance + half depth - the 10 mm
    standing offset) in that time so the robot comes down on the platform
    centre.  All tensors (N,), returns ((N,), (N,)).
    """
    rise = (top_rel + apex - clearance).clamp_min(0.02)
    vz = torch.sqrt(2.0 * GRAVITY * rise)
    t_land = vz / GRAVITY + torch.sqrt(2.0 * apex / GRAVITY)
    vx = (distance + 0.5 * PLATFORM_DEPTH_M - 0.010) / t_land
    return vx, vz


def _mocap_write(env: ManagerBasedRlEnv, state: _JumpStepState, env_ids: torch.Tensor) -> None:
    z = state.top[env_ids] - PLATFORM_HALF_THICKNESS_M
    pos = torch.stack((state.center_xy[env_ids, 0], state.center_xy[env_ids, 1], z), dim=-1)
    quat = torch.zeros(len(env_ids), 4, device=env.device)
    quat[:, 0] = 1.0
    env.scene[PLATFORM_ENTITY].write_mocap_pose_to_sim(torch.cat((pos, quat), dim=-1), env_ids=env_ids)


# --- events ---------------------------------------------------------------------


def reset_jump_step(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance_range: tuple[float, float] = PLATFORM_DISTANCE_RANGE,
    edge_distance_range: tuple[float, float] = EDGE_DISTANCE_RANGE,
    airborne_clearance_range: tuple[float, float] = AIRBORNE_CLEARANCE_RANGE,
    ballistic_apex_range: tuple[float, float] = BALLISTIC_APEX_RANGE,
    ballistic_distance_range: tuple[float, float] = BALLISTIC_DISTANCE_RANGE,
    spawn_z: float = 0.122,
    position_noise: float = 0.005,
    yaw_noise_deg: float = 5.0,
    tilt_noise_deg: float = 2.0,
    joint_noise: float = 0.03,
    platform_spawn_prob: float = 0.0,
    edge_spawn_prob: float = 0.15,
    airborne_spawn_prob: float = 0.2,
    ballistic_spawn_prob: float = 0.5,
) -> None:
    """Spawn the robot and place the platform.  Spawn types (v11 mix):

    * ``ballistic_spawn_prob`` (curriculum-driven, 0.5 -> 0): the "just
      jumped" state — HOME pose, upright, soles 1-2 cm off the ground
      ``ballistic_distance_range`` in front of the platform, with a SOLVED
      launch velocity (``_ballistic_launch``): vz puts the arc apex 5-15 cm
      above the platform top, vx brings the robot down on the platform
      centre.  Bulk experience of the flight + landing the floor spawns
      never discovered (the v4-v10 gap: the crouch exists, the landing
      exists, the coordinated takeoff does not).
    * ``airborne_spawn_prob`` (constant 0.2 in v11): HOME pose in mid-air
      above the platform centre, soles ``airborne_clearance_range`` above
      the top, zero velocity — keeps the landing skill alive (v10 showed
      retiring it dilutes the skill).  xy jitter anisotropic: ±5 mm in x
      (the sole spans -17/+31 mm of the ankle site on a 60 mm deep top),
      ±30 mm in y.
    * ``platform_spawn_prob`` (0 — superseded): standing on the platform top.
    * ``edge_spawn_prob``: on the floor only ``edge_distance_range`` from the
      platform's front face (takeoff-proximate, feet inside the foot gate).
    * the rest: on the floor ``distance_range`` ahead of the front face.

    Every spawn records its type in ``state.spawn_type`` (SPAWN_*) and closes
    the accounting of the ENDED episode (curriculum counters).

    The platform top follows the twist ``vx`` command.  Commands resample AFTER
    reset events (see module docstring), so the read here can be one episode
    stale; the ``step``-mode sync event re-pins the platform to the live
    command before the first physics step notices.  The read is clamped into
    the sampling range so the all-zero initial command still yields a valid
    platform.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.long)
    n, dev = len(env_ids), env.device
    state = _jump_step_state(env)
    asset: Entity = env.scene[asset_cfg.name]
    origins = env.scene.env_origins[env_ids]

    # Close the ended episodes' books for the curricula (the first reset
    # after startup has no previous episode).
    done_eps, landed_eps = _book_ended_episodes(state, env_ids, SPAWN_AIRBORNE)
    state.airborne_done += done_eps
    state.airborne_landed += landed_eps
    done_eps, landed_eps = _book_ended_episodes(state, env_ids, SPAWN_BALLISTIC)
    state.ballistic_done += done_eps
    state.ballistic_landed += landed_eps

    # Spawn-type draw (edge/ballistic spawns change where the platform goes).
    u = torch.rand(n, device=dev)
    on_ballistic = u < ballistic_spawn_prob
    on_airborne = (u >= ballistic_spawn_prob) & (u < ballistic_spawn_prob + airborne_spawn_prob)
    on_platform = (u >= ballistic_spawn_prob + airborne_spawn_prob) & (
        u < ballistic_spawn_prob + airborne_spawn_prob + platform_spawn_prob
    )
    on_edge = (u >= ballistic_spawn_prob + airborne_spawn_prob + platform_spawn_prob) & (
        u < ballistic_spawn_prob + airborne_spawn_prob + platform_spawn_prob + edge_spawn_prob
    )
    state.spawn_type[env_ids] = torch.where(
        on_ballistic,
        torch.full_like(u, SPAWN_BALLISTIC, dtype=torch.long),
        torch.where(
            on_airborne,
            torch.full_like(u, SPAWN_AIRBORNE, dtype=torch.long),
            torch.where(
                on_platform,
                torch.full_like(u, SPAWN_PLATFORM, dtype=torch.long),
                torch.where(on_edge, torch.full_like(u, SPAWN_EDGE, dtype=torch.long), torch.full_like(u, SPAWN_FLOOR, dtype=torch.long)),
            ),
        ),
    )

    # Platform: front face ``distance`` ahead of the robot root, top at the
    # commanded height (clamped: the pre-first-resample command is all zeros).
    command = env.command_manager.get_command("twist")[:, 0]
    height = command[env_ids].clamp(*PLATFORM_HEIGHT_RANGE)
    far = distance_range[0] + torch.rand(n, device=dev) * (distance_range[1] - distance_range[0])
    near = edge_distance_range[0] + torch.rand(n, device=dev) * (edge_distance_range[1] - edge_distance_range[0])
    bal_dist = ballistic_distance_range[0] + torch.rand(n, device=dev) * (
        ballistic_distance_range[1] - ballistic_distance_range[0]
    )
    distance = torch.where(on_edge, near, torch.where(on_ballistic, bal_dist, far))
    state.center_xy[env_ids, 0] = origins[:, 0] + distance + 0.5 * PLATFORM_DEPTH_M
    state.center_xy[env_ids, 1] = origins[:, 1]
    state.top[env_ids] = origins[:, 2] + height
    state.synced_height[env_ids] = height
    _mocap_write(env, state, env_ids)

    # Robot root: HOME stance, yaw ~0 (facing +x), small noise.  Floor spawns
    # stand at the env origin; platform spawns stand on the top; airborne
    # spawns drop onto it from ``clearance`` above; ballistic spawns launch
    # from just off the ground with a solved velocity.
    xy_noise = (torch.rand(n, 2, device=dev) * 2.0 - 1.0) * position_noise
    # Tighter jitter on the platform so both soles stay inside the 60 mm depth.
    plat_noise = (torch.rand(n, 2, device=dev) * 2.0 - 1.0) * 0.002
    air_noise = torch.stack(
        (
            (torch.rand(n, device=dev) * 2.0 - 1.0) * 0.005,
            (torch.rand(n, device=dev) * 2.0 - 1.0) * 0.03,
        ),
        dim=-1,
    )
    clearance = airborne_clearance_range[0] + torch.rand(n, device=dev) * (
        airborne_clearance_range[1] - airborne_clearance_range[0]
    )
    root_x = torch.where(
        on_platform,
        state.center_xy[env_ids, 0] - 0.010 + plat_noise[:, 0],
        origins[:, 0] + xy_noise[:, 0],
    )
    root_y = torch.where(
        on_platform,
        state.center_xy[env_ids, 1] + plat_noise[:, 1],
        origins[:, 1] + xy_noise[:, 1],
    )
    root_z = torch.where(on_platform, state.top[env_ids] + spawn_z, origins[:, 2] + spawn_z)
    root_x = torch.where(on_airborne, state.center_xy[env_ids, 0] - 0.010 + air_noise[:, 0], root_x)
    root_y = torch.where(on_airborne, state.center_xy[env_ids, 1] + air_noise[:, 1], root_y)
    # Root height above the soles at HOME is ~0.120 (spawn_z minus the 2 mm
    # sole clearance), so root z = top + 0.120 + clearance puts the soles
    # ``clearance`` above the top.
    root_z = torch.where(on_airborne, state.top[env_ids] + 0.120 + clearance, root_z)
    # Ballistic: soles just off the ground, launch velocity solved from the
    # sampled apex margin / distance (see _ballistic_launch).
    bal_apex = ballistic_apex_range[0] + torch.rand(n, device=dev) * (ballistic_apex_range[1] - ballistic_apex_range[0])
    bal_c = BALLISTIC_SOLE_CLEARANCE_RANGE[0] + torch.rand(n, device=dev) * (
        BALLISTIC_SOLE_CLEARANCE_RANGE[1] - BALLISTIC_SOLE_CLEARANCE_RANGE[0]
    )
    bal_vx, bal_vz = _ballistic_launch(height, bal_apex, bal_c, bal_dist)
    root_z = torch.where(on_ballistic, origins[:, 2] + 0.120 + bal_c, root_z)
    root_pos = torch.stack((root_x, root_y, root_z), dim=-1)
    yaw = (torch.rand(n, device=dev) * 2.0 - 1.0) * math.radians(yaw_noise_deg)
    pitch = (torch.rand(n, device=dev) * 2.0 - 1.0) * math.radians(tilt_noise_deg)
    roll = (torch.rand(n, device=dev) * 2.0 - 1.0) * math.radians(0.5 * tilt_noise_deg)
    quat = microduck_mdp._quat_from_yaw_pitch_roll(yaw, pitch, roll)
    asset.write_root_link_pose_to_sim(torch.cat((root_pos, quat), dim=-1), env_ids=env_ids)
    root_vel = torch.zeros(n, 6, device=dev)
    root_vel[:, 0] = torch.where(on_ballistic, bal_vx, root_vel[:, 0])
    root_vel[:, 2] = torch.where(on_ballistic, bal_vz, root_vel[:, 2])
    asset.write_root_link_velocity_to_sim(root_vel, env_ids=env_ids)

    # Joints: HOME + noise, clamped off the hard limits (same pattern as the ladder spawn).
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    servo_ids = microduck_mdp._servo_joint_ids(env, asset)
    joint_pos[:, servo_ids] += (torch.rand(n, len(servo_ids), device=dev) * 2.0 - 1.0) * joint_noise
    limits = asset.data.joint_pos_limits[env_ids][:, servo_ids]
    joint_pos[:, servo_ids] = torch.maximum(
        torch.minimum(joint_pos[:, servo_ids], limits[..., 1] - 0.02),
        limits[..., 0] + 0.02,
    )
    asset.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)

    # A custom pose reset must also clear the constraint warm-start, otherwise the
    # new contacts inherit whatever collision ended the previous episode.
    sim = getattr(env, "sim", None)
    if sim is not None and hasattr(sim.data, "qacc_warmstart"):
        sim.data.qacc_warmstart[env_ids] = 0.0

    state.potential_fresh[env_ids] = True
    state.forward_fresh[env_ids] = True
    state.foot_fresh[env_ids] = True
    state.success_latched[env_ids] = False
    state.first_foot_latched[env_ids] = False
    state.hold_steps[env_ids] = 0
    state.prev_vz[env_ids] = 0.0
    state.episode_started[env_ids] = True
    state.episode_landed[env_ids] = False


def sync_jump_step_platform(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None = None) -> None:
    """Re-pin the mocap platform top to the live twist ``vx`` command.

    Step-mode event (runs after ``command_manager.compute``): commands resample
    after reset-mode events, so this — not the reset event — is what guarantees
    the platform height equals the command the actor observes.  Only envs whose
    command changed are rewritten (once per episode in practice).
    """
    state = _jump_step_state(env)
    height = env.command_manager.get_command("twist")[:, 0].clamp_min(0.0)
    state.top = env.scene.env_origins[:, 2] + height
    changed = (state.synced_height - height).abs() > 1e-6
    state.synced_height = height.clone()
    if not bool(changed.any()):
        return
    env_ids = changed.nonzero(as_tuple=False).squeeze(-1)
    _mocap_write(env, state, env_ids)


# --- rewards --------------------------------------------------------------------


def jump_step_progress(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_delta: float = PROGRESS_MAX_DELTA_M,
    cap_above_top: float = PROGRESS_CAP_ABOVE_TOP_M,
) -> torch.Tensor:
    """Potential-based height progress: delta of min(trunk_z, top + cap), capped
    per step.  Rising pays, holding pays zero, falling pays back — unfarmable.
    The first call after a reset establishes the potential and pays nothing
    (the reset event cannot know the freshly resampled platform height)."""
    state = _jump_step_state(env)
    asset: Entity = env.scene[asset_cfg.name]
    trunk_z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2], nan=0.0)
    potential = torch.minimum(trunk_z, state.top + cap_above_top)
    delta = (potential - state.prev_potential).clamp(-max_delta, max_delta)
    delta = torch.where(state.potential_fresh, torch.zeros_like(delta), delta)
    state.prev_potential = potential
    state.potential_fresh = torch.zeros_like(state.potential_fresh)
    return torch.nan_to_num(delta, nan=0.0)


def jump_step_forward_progress(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
    max_delta: float = PROGRESS_MAX_DELTA_M,
) -> torch.Tensor:
    """Potential-based forward progress toward the platform centre: pays the
    delta of -(FEET xy distance to the platform centre), per-step capped and
    floored at half the platform depth + 1 cm, so walking through/past the
    platform or stepping back off cannot collect.

    v7: measured on the mean FOOT position, not the trunk.  The trunk version
    was collected by leaning chest/head over the platform with the feet
    planted (probe8: 88% edge-spawn "contacts" were toes on the front face
    during a head-first lean — the lean basin).  Feet don't advance during a
    lean, so the lean pays nothing here.  The pure-vertical progress alone
    pays nothing for the approach (blind hop discovery needs a forward
    component to find the 20 cm gap)."""
    state = _jump_step_state(env)
    asset: Entity = env.scene[asset_cfg.name]
    feet_xy = torch.nan_to_num(asset.data.site_pos_w[:, asset_cfg.site_ids, :2], nan=0.0)
    dist = (feet_xy.mean(dim=1) - state.center_xy).norm(dim=1)
    floor = 0.5 * PLATFORM_DEPTH_M + 0.01
    potential = -torch.maximum(dist, torch.full_like(dist, floor))
    delta = (potential - state.prev_forward).clamp(-max_delta, max_delta)
    delta = torch.where(state.forward_fresh, torch.zeros_like(delta), delta)
    state.prev_forward = potential
    state.forward_fresh = torch.zeros_like(state.forward_fresh)
    return torch.nan_to_num(delta, nan=0.0)


def jump_step_foot_progress(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
    max_delta: float = PROGRESS_MAX_DELTA_M,
    gate_radius: float = FOOT_PROGRESS_GATE_M,
) -> torch.Tensor:
    """Per-foot z progress toward the platform top, gated on proximity.

    Potential per foot = min(foot_z, platform_top); pays the per-step capped
    delta, but only while the foot is within ``gate_radius`` of the platform
    centre horizontally — the approach is already paid by
    ``jump_step_forward_progress``, this term teaches clearing the top with
    the FEET once close.  Unfarmable the same way as the trunk terms: bobbing
    a foot below the top nets zero, holding pays zero, above the top is
    capped, and the first step after a reset only establishes the potential.
    The two feet are MEAN-combined (not max): a two-footed lift collects in
    full, a single-foot lift collects half, so the lagging foot always keeps
    its gradient (a max would saturate on one foot and let the other dangle).
    """
    state = _jump_step_state(env)
    asset: Entity = env.scene[asset_cfg.name]
    feet = torch.nan_to_num(asset.data.site_pos_w[:, asset_cfg.site_ids], nan=0.0)  # (N, 2, 3)
    potential = torch.minimum(feet[:, :, 2], state.top[:, None])
    delta = (potential - state.prev_foot).clamp(-max_delta, max_delta)
    delta = torch.where(state.foot_fresh[:, None], torch.zeros_like(delta), delta)
    near = (feet[:, :, :2] - state.center_xy[:, None, :]).norm(dim=2) < gate_radius
    pay = (delta * near).mean(dim=1)
    state.prev_foot = potential
    state.foot_fresh = torch.zeros_like(state.foot_fresh)
    return torch.nan_to_num(pay, nan=0.0)


def _success_condition(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    sensor_name: str,
    foot_margin: float,
    upright_min: float,
) -> torch.Tensor:
    """Both feet above the platform top, both supported by platform contact, upright."""
    state = _jump_step_state(env)
    asset: Entity = env.scene[asset_cfg.name]
    feet_z = torch.nan_to_num(asset.data.site_pos_w[:, asset_cfg.site_ids, 2], nan=0.0)
    high = (feet_z > (state.top - foot_margin)[:, None]).all(dim=1)
    found = env.scene.sensors[sensor_name].data.found.reshape(env.num_envs, -1)[:, :2]
    supported = (found > 0).all(dim=1)
    upright = asset.data.projected_gravity_b[:, 2] < -upright_min
    return high & supported & upright


def jump_step_first_foot_bonus(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
    sensor_name: str = PLATFORM_CONTACT_SENSOR,
    ground_sensor_name: str = "feet_ground_contact",
    foot_margin: float = SUCCESS_FOOT_MARGIN_M,
    upright_min: float = 0.7,
) -> torch.Tensor:
    """One-shot (per episode) bonus for the FIRST foot on the platform top,
    paid only for a DYNAMIC mount transition: at the step the counted foot
    touches the platform top, the OTHER foot must be airborne (no ground and
    no platform contact).  Trunk upright-ish (relaxed vs success: -0.7).

    v8 anti-farm: the v7 policy latched this bonus by parking one foot on the
    platform edge and never bringing the second foot up — a static park that
    satisfied "foot on top + contact + upright" forever.  Requiring the
    trailing foot airborne makes a quasi-static one-foot park worth exactly
    zero; only a hop/step-up caught mid-flight pays.  Latched, so even a
    valid transition pays once per episode."""
    state = _jump_step_state(env)
    asset: Entity = env.scene[asset_cfg.name]
    feet_z = torch.nan_to_num(asset.data.site_pos_w[:, asset_cfg.site_ids, 2], nan=0.0)
    found_plat = env.scene.sensors[sensor_name].data.found.reshape(env.num_envs, -1)[:, :2] > 0
    found_ground = env.scene.sensors[ground_sensor_name].data.found.reshape(env.num_envs, -1)[:, :2] > 0
    on_top = (feet_z > (state.top - foot_margin)[:, None]) & found_plat
    airborne = ~(found_plat | found_ground)
    transition = (on_top[:, 0] & airborne[:, 1]) | (on_top[:, 1] & airborne[:, 0])
    upright = asset.data.projected_gravity_b[:, 2] < -upright_min
    condition = transition & upright
    pay = (condition & ~state.first_foot_latched).float()
    state.first_foot_latched |= condition
    return pay


def jump_step_body_contact_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Self-negating cost (<= 0) on head/trunk contact with the platform.

    v7 anti-lean term (probe8: the v6 policy mounts the platform head-first —
    chest/beak on the top edge, feet planted — instead of stepping up).  Any
    contact between the head/trunk bodies (see the sensor's primary pattern)
    and the platform pays per step; feet and legs are free (a shin brushing
    the face during a mount is fine, same tolerance as the ladder)."""
    found = env.scene.sensors[sensor_name].data.found
    return -torch.nan_to_num(found.reshape(env.num_envs, -1).any(dim=1), nan=0.0).float()


def jump_step_success(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_name: str = PLATFORM_CONTACT_SENSOR,
    foot_margin: float = SUCCESS_FOOT_MARGIN_M,
    upright_min: float = SUCCESS_UPRIGHT_MIN,
    platform_spawn_scale: float = SUCCESS_PLATFORM_SPAWN_SCALE,
    ballistic_spawn_scale: float = SUCCESS_BALLISTIC_SPAWN_SCALE,
) -> torch.Tensor:
    """One-shot (per episode) bonus for landing on the platform, upright.
    Rate-limited by construction (latched), so it is not a jackpot.

    Spawn-typed (v6, extended v9/v11): floor-born episodes (far or edge) pay
    in full; nearly-done spawns pay a fraction (kept at absolute 3 points for
    platform-top/airborne via ``platform_spawn_scale``; ballistic pays
    ``ballistic_spawn_scale`` = 6 — farther from done than a free fall, the
    flight still has to be managed).  v5 paid full price for stand-still-
    on-spawn success and the policy stopped approaching the platform
    entirely — goal-state spawn experience is kept, but it must not out-earn
    the real maneuver."""
    state = _jump_step_state(env)
    condition = _success_condition(env, asset_cfg, sensor_name, foot_margin, upright_min)
    pay = (condition & ~state.success_latched).float()
    goal_spawn = (state.spawn_type == SPAWN_PLATFORM) | (state.spawn_type == SPAWN_AIRBORNE)
    scale = torch.where(goal_spawn, torch.full_like(pay, platform_spawn_scale), torch.ones_like(pay))
    scale = torch.where(state.spawn_type == SPAWN_BALLISTIC, torch.full_like(pay, ballistic_spawn_scale), scale)
    state.success_latched |= condition
    return pay * scale


def jump_step_vertical_impact_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    clip: float = 0.5,
) -> torch.Tensor:
    """Self-negating cost (<= 0) on trunk vertical-velocity jumps (landing
    impacts) — the ladder recipe's vertical_impact without the ladder state."""
    state = _jump_step_state(env)
    asset: Entity = env.scene[asset_cfg.name]
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    jump = (vz - state.prev_vz).abs().clamp_max(clip)
    state.prev_vz = vz.clone()
    return -jump


# --- terminations -----------------------------------------------------------------


def jump_step_landed(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_name: str = PLATFORM_CONTACT_SENSOR,
    foot_margin: float = SUCCESS_FOOT_MARGIN_M,
    upright_min: float = SUCCESS_UPRIGHT_MIN,
    hold_s: float = SUCCESS_HOLD_S,
) -> torch.Tensor:
    """Success: the success condition held for ``hold_s`` (time_out=True)."""
    state = _jump_step_state(env)
    condition = _success_condition(env, asset_cfg, sensor_name, foot_margin, upright_min)
    step = int(env.common_step_counter)
    if state.hold_step != step:
        state.hold_steps = torch.where(condition, state.hold_steps + 1, torch.zeros_like(state.hold_steps))
        state.hold_step = step
    landed = state.hold_steps >= round(hold_s / env.step_dt)
    state.episode_landed |= landed  # airborne-curriculum accounting
    return landed


def jump_step_fallen(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    gravity_z: float = FALLEN_GRAVITY_Z,
) -> torch.Tensor:
    """Fallen: trunk tipped well past vertical (projected_gravity z > -0.5)."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b[:, 2] > gravity_z


# --- curriculum -------------------------------------------------------------------


def _spawn_rate_curriculum(
    env: ManagerBasedRlEnv,
    event_name: str,
    param_name: str,
    table: tuple[dict, ...],
    done: int,
    landed: int,
    min_episodes: int,
    label: str,
) -> tuple[float, bool]:
    """Rate-driven spawn-fraction schedule (shared by the airborne/ballistic
    curricula).  Highest satisfied rate threshold wins; rewrites the reset
    event's param through the live event manager (never ``env.cfg``).
    Returns (active fraction, changed?)."""
    term_cfg = env.event_manager.get_term_cfg(event_name)
    current = float(term_cfg.params.get(param_name, 0.0))
    if done < min_episodes:
        return current, False
    rate = landed / done
    prob = current
    for stage in table:  # descending rate thresholds
        if rate >= stage["rate"]:
            prob = stage["prob"]
            break
    if prob != current:
        term_cfg.params[param_name] = prob
        print(f"[jump_step] {label} curriculum: landed rate {rate:.3f} over {done} episodes -> prob {prob}")
        return prob, True
    return prob, False


def jump_step_airborne_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None = None,
    event_name: str = "reset_jump_step",
    min_episodes: int = AIRBORNE_CURRICULUM_MIN_EPISODES,
) -> float:
    """Airborne-fraction schedule (v10 thresholds).  Unregistered since v11
    (airborne spawns are a constant share now); kept for the stage tests."""
    state = _jump_step_state(env)
    prob, changed = _spawn_rate_curriculum(
        env, event_name, "airborne_spawn_prob", AIRBORNE_CURRICULUM,
        state.airborne_done, state.airborne_landed, min_episodes, "airborne",
    )
    if changed:
        state.airborne_done = 0
        state.airborne_landed = 0
    return prob


def jump_step_ballistic_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None = None,
    event_name: str = "reset_jump_step",
    min_episodes: int = AIRBORNE_CURRICULUM_MIN_EPISODES,
) -> float:
    """Retire the ballistic ("just jumped") spawns as the flight+landing is
    learned.  Driver: landed-success rate of BALLISTIC episodes only
    (floor-born success is 0 for most of training).  200-episode windows,
    bidirectional — see BALLISTIC_CURRICULUM for the threshold rationale."""
    state = _jump_step_state(env)
    prob, changed = _spawn_rate_curriculum(
        env, event_name, "ballistic_spawn_prob", BALLISTIC_CURRICULUM,
        state.ballistic_done, state.ballistic_landed, min_episodes, "ballistic",
    )
    if changed:
        state.ballistic_done = 0
        state.ballistic_landed = 0
    return prob


def jump_step_airborne_rate(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None = None) -> float:
    """Reporter-only curriculum term: the current airborne-episode landed
    rate of the ACTIVE measurement window (logged as Curriculum/airborne_rate
    so the driver signal itself is observable, not just the stage output)."""
    state = _jump_step_state(env)
    if state.airborne_done == 0:
        return 0.0
    return state.airborne_landed / state.airborne_done


def jump_step_ballistic_rate(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None = None) -> float:
    """Reporter-only curriculum term: the current ballistic-episode landed
    rate of the ACTIVE measurement window (Curriculum/ballistic_rate)."""
    state = _jump_step_state(env)
    if state.ballistic_done == 0:
        return 0.0
    return state.ballistic_landed / state.ballistic_done


# --- env cfg ----------------------------------------------------------------------


def make_microduck_jump_step_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velocity_env_cfg(play=play, rough=False)

    # --- scene: all-collision robot (a jump landing can clip the trunk/head on
    # the platform edge, same rationale as the ladder tasks) + mocap platform.
    cfg.scene.entities = {
        "robot": MICRODUCK_STANDUP_ROBOT_CFG,
        PLATFORM_ENTITY: _platform_entity_cfg(),
    }
    feet_platform_cfg = ContactSensorCfg(
        name=PLATFORM_CONTACT_SENSOR,
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",  # LEFT, RIGHT
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern=PLATFORM_ENTITY, entity=PLATFORM_ENTITY),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )
    cfg.scene.sensors = (*cfg.scene.sensors, feet_platform_cfg)
    # Head/trunk vs platform (v7 anti-lean): chest/beak on the platform pays,
    # feet and legs are free.  Body names from the allcollisions model:
    # trunk/neck/head only — hip/leg/ankle bodies are deliberately excluded.
    body_platform_cfg = ContactSensorCfg(
        name="body_platform_contact",
        primary=ContactMatch(
            mode="body",
            pattern=r"^(trunk_base|neck|neck_pitch|yaw_roll_motion|jaw_soft|mouth)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern=PLATFORM_ENTITY, entity=PLATFORM_ENTITY),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    cfg.scene.sensors = (*cfg.scene.sensors, body_platform_cfg)
    cfg.viewer.body_name = "trunk_base"
    cfg.viewer.distance = 0.7
    cfg.episode_length_s = EPISODE_LENGTH_S

    # --- commands: twist vx = platform height; the other two twist slots are
    # exact zero (deployment writes [height, 0, 0]).  The resampling interval
    # exceeds the episode length so the height is constant per episode.  The
    # head/body pose commands keep the velocity defaults (tiny alive ranges).
    twist = cfg.commands["twist"]
    twist.rel_standing_envs = 0.0
    twist.rel_turn_in_place_envs = 0.0
    twist.rel_heading_envs = 0.0
    # Base velocity template sets rel_forward_envs=0.2: those envs get
    # vx = |vx|.clamp(min=0.3) (mjlab velocity_command.py), i.e. a 30 cm
    # "platform" floating in the sky on ~20% of episodes.  Must be 0 here.
    twist.rel_forward_envs = 0.0
    twist.heading_command = False
    twist.ranges.heading = None
    twist.resampling_time_range = (8.0, 12.0)
    twist.ranges.lin_vel_x = PLATFORM_HEIGHT_RANGE
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (0.0, 0.0)
    twist.debug_vis = False
    if play:
        # Fixed mid-range platform for repeatable viewing.
        twist.ranges.lin_vel_x = (
            float(os.getenv("MICRODUCK_JUMP_STEP_HEIGHT_M", "0.045")),
        ) * 2

    # --- rewards ---------------------------------------------------------------
    for name in (
        "track_linear_velocity",
        "track_angular_velocity",
        "pose",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "head_pose_tracking",
        "head_pose_bias",
        "body_pose_tracking",
        "upright",
    ):
        cfg.rewards.pop(name, None)
    # Motion-blockers stay at the (light) velocity values; a two-footed hop
    # needs pitch rate (same values the ladder's hop uses).
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    # Smoothness: small from the start — any attempt-tax active while the jump
    # is being explored makes "do nothing" win (AGENTS.md).  The
    # action_rate_weight curriculum below ramps it in after discovery.
    cfg.rewards["action_rate_l2"].weight = -0.01

    robot = SceneEntityCfg("robot")
    feet = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))
    cfg.rewards["jump_step_progress"] = RewardTermCfg(
        func=jump_step_progress,
        weight=50.0,
        params={"asset_cfg": robot, "max_delta": PROGRESS_MAX_DELTA_M},
    )
    cfg.rewards["jump_step_forward_progress"] = RewardTermCfg(
        func=jump_step_forward_progress,
        weight=30.0,
        params={"asset_cfg": feet, "max_delta": PROGRESS_MAX_DELTA_M},
    )
    cfg.rewards["jump_step_foot_progress"] = RewardTermCfg(
        func=jump_step_foot_progress,
        weight=30.0,
        params={"asset_cfg": feet, "max_delta": PROGRESS_MAX_DELTA_M, "gate_radius": FOOT_PROGRESS_GATE_M},
    )
    cfg.rewards["jump_step_first_foot_bonus"] = RewardTermCfg(
        func=jump_step_first_foot_bonus,
        weight=5.0,
        params={"asset_cfg": feet, "sensor_name": PLATFORM_CONTACT_SENSOR},
    )
    # Self-negating cost: positive weight is intentional (it returns <= 0).
    cfg.rewards["body_platform_contact"] = RewardTermCfg(
        func=jump_step_body_contact_penalty,
        weight=0.05,
        params={"sensor_name": "body_platform_contact"},
    )
    cfg.rewards["jump_step_success"] = RewardTermCfg(
        func=jump_step_success,
        weight=40.0,  # v8: 20 -> 40, the full mount must out-earn every basin's dwell return
        params={"asset_cfg": feet, "sensor_name": PLATFORM_CONTACT_SENSOR},
    )
    # Self-negating cost: positive weight is intentional (it returns <= 0).
    cfg.rewards["vertical_impact"] = RewardTermCfg(
        func=jump_step_vertical_impact_penalty,
        weight=0.5,
        params={"asset_cfg": robot, "clip": 0.5},
    )

    # --- events: platform placement + HOME-stance spawn replaces the flat reset.
    cfg.events.pop("reset_base", None)
    cfg.events.pop("reset_robot_joints", None)
    reset_term = EventTermCfg(
        func=reset_jump_step,
        mode="reset",
        params={
            "asset_cfg": robot,
            # Reverse-curriculum spawns in training.  OFF in play/eval so the
            # eval success rate measures the actual behavior, not free states
            # (the eval script opts airborne spawns back in via its own flag).
            "platform_spawn_prob": 0.0,
            "edge_spawn_prob": 0.0 if play else 0.15,
            "airborne_spawn_prob": 0.0 if play else 0.2,
            "ballistic_spawn_prob": 0.0 if play else 0.5,
        },
    )
    # Run the spawn before every other reset event.
    cfg.events = {"reset_jump_step": reset_term, **cfg.events}
    # Re-pin the platform to the live command after every command resample
    # (commands resample AFTER reset-mode events — see the module docstring).
    cfg.events["sync_jump_step_platform"] = EventTermCfg(func=sync_jump_step_platform, mode="step")
    # No pushes while the jump is being discovered (keep the event for a later
    # curriculum; zero range = no-op).
    if "push_robot" in cfg.events:
        cfg.events["push_robot"].params["velocity_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0)}

    # --- terminations ------------------------------------------------------------
    cfg.terminations.pop("fell_over", None)
    cfg.terminations["fallen"] = TerminationTermCfg(
        func=jump_step_fallen,
        params={"asset_cfg": robot},
    )
    cfg.terminations["jump_step_landed"] = TerminationTermCfg(
        func=jump_step_landed,
        time_out=True,
        params={"asset_cfg": feet, "sensor_name": PLATFORM_CONTACT_SENSOR, "hold_s": SUCCESS_HOLD_S},
    )

    # --- curriculum ---------------------------------------------------------------
    for name in ("standing_envs", "head_pose_bias_weight"):
        cfg.curriculum.pop(name, None)
    # Smoothness after skill discovery, never before (AGENTS.md).
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.01},
                {"step": 1000 * NUM_STEPS_PER_ENV, "weight": -0.05},
                {"step": 1500 * NUM_STEPS_PER_ENV, "weight": -0.1},
            ],
        },
    )
    if not play:
        # v11: retire the ballistic ("just jumped") spawns as the
        # flight+landing success rate climbs (0.5 -> 0.3 -> 0.15 -> 0).
        cfg.curriculum["ballistic_spawn"] = CurriculumTermCfg(
            func=jump_step_ballistic_curriculum,
            params={"event_name": "reset_jump_step"},
        )
        # Reporters: the windowed landed rates themselves (diagnostics —
        # distinguishes "accounting broken" from "rate below threshold";
        # airborne_rate also tracks landing-skill retention, the v10 lesson).
        cfg.curriculum["airborne_rate"] = CurriculumTermCfg(func=jump_step_airborne_rate)
        cfg.curriculum["ballistic_rate"] = CurriculumTermCfg(func=jump_step_ballistic_rate)
    return cfg


MicroduckJumpStepRlCfg = deepcopy(MicroduckRlCfg)
MicroduckJumpStepRlCfg.experiment_name = "jump_step"
MicroduckJumpStepRlCfg.run_name = "jump_step"
MicroduckJumpStepRlCfg.save_interval = 100
MicroduckJumpStepRlCfg.max_iterations = 2_000
