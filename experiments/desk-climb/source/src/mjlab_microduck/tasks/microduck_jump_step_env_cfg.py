"""Microduck single-step jump task ("jump onto a low platform").

Built on the velocity recipe (BAM actuators, full DR + obs-noise + NaN-guard
stack, 61-D observation contract) with flat terrain plus one mocap platform.

Task
----
* The robot spawns in the HOME stance on flat ground, facing +x, with a
  60 mm deep x 230 mm wide platform 0.18-0.25 m ahead.  The platform top
  height IS the twist command: ``vx`` is sampled in 0.03-0.06 m and the reset
  event places the mocap platform so its top sits at that height.  Jump onto
  the platform and stand.
* Reward is episodic-style (AGENTS.md): potential-based trunk-height progress
  (paid on the DELTA of min(trunk_z, platform_top + 0.09), per-step capped —
  rising pays, holding pays zero, falling pays back, so it cannot be farmed),
  a one-shot success bonus (both feet above the platform top, feet supported
  by platform contact, trunk upright), an |a_z|-class landing-impact cost,
  and a light action-rate tax that stays small during skill discovery.
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
PLATFORM_HEIGHT_RANGE = (0.03, 0.06)  # m, sampled as the twist vx command
PLATFORM_DISTANCE_RANGE = (0.18, 0.25)  # m, robot root to the platform front face

EPISODE_LENGTH_S = 5.0
PROGRESS_CAP_ABOVE_TOP_M = 0.09  # pay min(trunk_z, top + this) deltas
PROGRESS_MAX_DELTA_M = 0.005  # per-step pay cap (anti-jackpot rate limit)
SUCCESS_FOOT_MARGIN_M = 0.005  # both feet above top - this
SUCCESS_UPRIGHT_MIN = 0.9  # -projected_gravity_b z
SUCCESS_HOLD_S = 0.4
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
        self.success_latched = torch.zeros(n, dtype=torch.bool, device=dev)
        self.prev_vz = torch.zeros(n, device=dev)
        self.hold_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.hold_step = -1


def _jump_step_state(env: ManagerBasedRlEnv) -> _JumpStepState:
    state = getattr(env, "_jump_step_state", None)
    if state is None:
        state = _JumpStepState(env)
        env._jump_step_state = state
    return state


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
    spawn_z: float = 0.122,
    position_noise: float = 0.005,
    yaw_noise_deg: float = 5.0,
    tilt_noise_deg: float = 2.0,
    joint_noise: float = 0.03,
) -> None:
    """Spawn the robot in HOME stance facing +x and place the platform ahead of it.

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

    # Platform: front face ``distance`` ahead of the robot root, top at the
    # commanded height (clamped: the pre-first-resample command is all zeros).
    command = env.command_manager.get_command("twist")[:, 0]
    height = command[env_ids].clamp(*PLATFORM_HEIGHT_RANGE)
    distance = distance_range[0] + torch.rand(n, device=dev) * (distance_range[1] - distance_range[0])
    state.center_xy[env_ids, 0] = origins[:, 0] + distance + 0.5 * PLATFORM_DEPTH_M
    state.center_xy[env_ids, 1] = origins[:, 1]
    state.top[env_ids] = origins[:, 2] + height
    state.synced_height[env_ids] = height
    _mocap_write(env, state, env_ids)

    # Robot root: HOME stance at the env origin, yaw ~0 (facing +x), small noise.
    xy_noise = (torch.rand(n, 2, device=dev) * 2.0 - 1.0) * position_noise
    root_pos = torch.stack(
        (origins[:, 0] + xy_noise[:, 0], origins[:, 1] + xy_noise[:, 1], origins[:, 2] + spawn_z),
        dim=-1,
    )
    yaw = (torch.rand(n, device=dev) * 2.0 - 1.0) * math.radians(yaw_noise_deg)
    pitch = (torch.rand(n, device=dev) * 2.0 - 1.0) * math.radians(tilt_noise_deg)
    roll = (torch.rand(n, device=dev) * 2.0 - 1.0) * math.radians(0.5 * tilt_noise_deg)
    quat = microduck_mdp._quat_from_yaw_pitch_roll(yaw, pitch, roll)
    asset.write_root_link_pose_to_sim(torch.cat((root_pos, quat), dim=-1), env_ids=env_ids)
    asset.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)

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
    state.success_latched[env_ids] = False
    state.hold_steps[env_ids] = 0
    state.prev_vz[env_ids] = 0.0


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


def jump_step_success(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_name: str = PLATFORM_CONTACT_SENSOR,
    foot_margin: float = SUCCESS_FOOT_MARGIN_M,
    upright_min: float = SUCCESS_UPRIGHT_MIN,
) -> torch.Tensor:
    """One-shot (per episode) bonus for landing on the platform, upright.
    Rate-limited by construction (latched), so it is not a jackpot."""
    state = _jump_step_state(env)
    condition = _success_condition(env, asset_cfg, sensor_name, foot_margin, upright_min)
    pay = (condition & ~state.success_latched).float()
    state.success_latched |= condition
    return pay


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
    return state.hold_steps >= round(hold_s / env.step_dt)


def jump_step_fallen(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    gravity_z: float = FALLEN_GRAVITY_Z,
) -> torch.Tensor:
    """Fallen: trunk tipped well past vertical (projected_gravity z > -0.5)."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b[:, 2] > gravity_z


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
    cfg.rewards["jump_step_success"] = RewardTermCfg(
        func=jump_step_success,
        weight=20.0,
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
        params={"asset_cfg": robot},
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
    return cfg


MicroduckJumpStepRlCfg = deepcopy(MicroduckRlCfg)
MicroduckJumpStepRlCfg.experiment_name = "jump_step"
MicroduckJumpStepRlCfg.run_name = "jump_step"
MicroduckJumpStepRlCfg.save_interval = 100
MicroduckJumpStepRlCfg.max_iterations = 2_000
