"""Microduck tug-CHAIN-3 task — 3v3 duck tug-of-war, learner = red middle duck.

v8 of the chain self-play line. v7 (1v1) matched the win condition but ducks
still spent 56% of match time on the floor, for two structural reasons this
task fixes:

  1. The match's middle duck is loaded from BOTH sides (chest ring pulled by
     its trailing teammate, butt ring by the enemy chain). 1v1 never produces
     that工况 — so here the learner IS the red middle duck, chained into a
     full 3v3 with the match's link geometry (robot/tug_of_war.py: teammate
     trunk spacing 0.24 m, inner-duck center gap 0.30 m, force path through
     the body: pulled at the butt ring, pulling with the chest ring).
  2. v7 ended the episode when the opponent fell (~3 s rounds — no grind
     experience). Here fallen FROZEN ducks stay in the sim as dead weight on
     the rope (match rules); only the LEARNER falling ends the episode, so
     rounds run to the 20 s time limit.

The five frozen ducks run the v7 chain policies (red pair: tugchain_steady_v7;
blue trio: tugchain_shuffle_v7 — the frozen mix is fixed for BOTH task styles;
the style flag only selects the learner's reward recipe). Each frozen duck is
driven by its own zero-dim FrozenTugOpponentAction term (independent
last_action buffers), so the learner keeps the exact 61D/14D contract.

DR / obs noise / delays: velocity-parity for the LEARNER only (inherited from
make_microduck_velocity_env_cfg); frozen ducks get default spawns, no DR, no
pushes — they represent deployed policies as-is.
"""

import math
from pathlib import Path

import mujoco as _mujoco
import numpy as np

# ── Toggles ──────────────────────────────────────────────────────────────────
ENABLE_GAP_DR = True            # per-episode per-cord spawn pre-tension noise
ENABLE_VELOCITY_PUSHES = True   # velocity-parity shove robustness (learner only)

# Symmetry — must stay OFF: pulling with a lean is not mirror-symmetric.
ENABLE_SYMMETRY = False

# ── Cord constants (match cord model, robot/tug_of_war.py) ──────────────────
ROPE_STIFFNESS = 200.0        # N/m
ROPE_SLACK = 0.005            # dead band past the nominal ring-to-ring distance
ROPE_LIMIT_MARGIN = 0.05      # soft upper safety catch past the taut length
ROPE_DAMPING = 2.0            # model.tendon_damping in the match demo
ROPE_WIDTH = 0.0025
ROPE_RGBA = (0.92, 0.87, 0.70, 1.0)

# Match link geometry: teammate trunk spacing 0.24 m, inner-duck center gap
# 0.30 m. Ring-to-ring distances derive from the collar geometry
# (TUG_RING_LOCAL / TUG_CHEST_LOCAL); each cord's taut length = ring distance
# + slack. All rings sit at trunk-local z=0.016 on equal-height trunks, so the
# cords are horizontal and the taut length equals the horizontal ring distance.
TEAMMATE_TRUNK_GAP = 0.24
CENTER_TRUNK_GAP = 0.30

# Frozen v7 policies: red frozen ducks run steady, blue run shuffle. Fixed for
# both task styles (the style flag selects the LEARNER's recipe only).
_POLICY_DIR = Path(__file__).resolve().parents[3] / "policies"
FROZEN_POLICY_RED = _POLICY_DIR / "tugchain_steady_v7.onnx"
FROZEN_POLICY_BLUE = _POLICY_DIR / "tugchain_shuffle_v7.onnx"

# Spawn: every cord starts 5–10 mm PAST its taut length (pre-loaded chain).
_SPAWN_PRETENSION = (0.005, 0.010)

# Task reward shaping
PROGRESS_MAX_PAID_RATE = 0.4   # m/s of midpoint drag that pays; faster pays no extra

# Style recipe parameters (identical to the v6/v7 recipes)
STEADY_LEAN_PITCH = math.radians(-18.0)  # butt toward the enemy chain
SHUFFLE_LEAN_PITCH = 0.0
LEAN_STD = 0.07
STEADY_CADENCE_HZ = 1.5
SHUFFLE_CADENCE_HZ = 3.5
CADENCE_STD_HZ = 0.75

# Bounds (plane terrain — the base out_of_terrain_bounds is a no-op here).
# Learner only; fallen frozen ducks are dead weight, not terminations.
MAX_DISTANCE = 2.0
FALLEN_TRUNK_Z = 0.055         # same fall criterion as the tug-of-war demo
OVERLEAN_LIMIT = math.radians(35.0)  # v6 hard posture gate on the LEARNER

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import terminations as mjlab_terminations
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_TUG_ROBOT_CFG,
    TUG_CHEST_LOCAL,
    TUG_RING_LOCAL,
    make_tug_duck_entity_cfg,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    NUM_STEPS_PER_ENV,
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG

# Derived cord lengths (after the imports — TUG_*_LOCAL live in constants).
_TEAMMATE_RING_GAP = TEAMMATE_TRUNK_GAP + TUG_RING_LOCAL[0] - TUG_CHEST_LOCAL[0]
TEAMMATE_TAUT_LENGTH = _TEAMMATE_RING_GAP + ROPE_SLACK      # 0.1304 m
_CENTER_RING_GAP = CENTER_TRUNK_GAP + 2.0 * TUG_RING_LOCAL[0]
CENTER_TAUT_LENGTH = _CENTER_RING_GAP + ROPE_SLACK          # 0.1678 m

# Spawn trunk gaps = nominal + slack + pre-tension (per-cord sampled).
TEAMMATE_GAP_RANGE = (
    TEAMMATE_TRUNK_GAP + ROPE_SLACK + _SPAWN_PRETENSION[0],
    TEAMMATE_TRUNK_GAP + ROPE_SLACK + _SPAWN_PRETENSION[1],
)
CENTER_GAP_RANGE = (
    CENTER_TRUNK_GAP + ROPE_SLACK + _SPAWN_PRETENSION[0],
    CENTER_TRUNK_GAP + ROPE_SLACK + _SPAWN_PRETENSION[1],
)

# Frozen ducks in chain order with their policy. Red = learner's team.
_FROZEN_DUCKS = {
    "red_inner": FROZEN_POLICY_RED,
    "red_outer": FROZEN_POLICY_RED,
    "blue_inner": FROZEN_POLICY_BLUE,
    "blue_mid": FROZEN_POLICY_BLUE,
    "blue_outer": FROZEN_POLICY_BLUE,
}

# Cord table: (name, site A, site B, taut length). Butt ring (rope_hook) faces
# the enemy/center, chest ring (rope_hook_chest) faces the trailing teammate —
# the force path goes through each duck's body, like the match harness.
_CORDS = (
    ("cord_red_outer", "red_outer/rope_hook", "robot/rope_hook_chest", TEAMMATE_TAUT_LENGTH),
    ("cord_red_mid", "robot/rope_hook", "red_inner/rope_hook_chest", TEAMMATE_TAUT_LENGTH),
    ("cord_center", "red_inner/rope_hook", "blue_inner/rope_hook", CENTER_TAUT_LENGTH),
    ("cord_blue_mid", "blue_inner/rope_hook_chest", "blue_mid/rope_hook", TEAMMATE_TAUT_LENGTH),
    ("cord_blue_outer", "blue_mid/rope_hook_chest", "blue_outer/rope_hook", TEAMMATE_TAUT_LENGTH),
)


def _add_chain3_ropes(spec: _mujoco.MjSpec) -> None:
    """Scene-level spec_fn: the 5 tension-only dead-band cords of the 3v3 chain.

    Same cross-entity tendon pattern and cord parameters as the 1v1 chain
    (which follows robot/tug_of_war.py), damping included — mujoco 3.10's
    MjSpec binds tendon damping as a (3,1) array whose element 0 compiles to
    the scalar tendon_damping (verified against spec.compile()).
    """
    for name, site_a, site_b, taut in _CORDS:
        cord = spec.add_tendon(
            name=name,
            stiffness=ROPE_STIFFNESS,
            springlength=(0.0, taut),
            limited=True,
            range=(0.0, taut + ROPE_LIMIT_MARGIN),
            width=ROPE_WIDTH,
            rgba=ROPE_RGBA,
            solref_limit=(0.02, 1.0),
            solimp_limit=(0.90, 0.95, 0.001, 0.5, 2.0),
        )
        cord.damping = np.array([[ROPE_DAMPING], [0.0], [0.0]])
        cord.wrap_site(site_a)
        cord.wrap_site(site_b)


def make_microduck_tug_chain3_env_cfg(
    style: str = "steady",
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create the Microduck 3v3 tug-chain environment configuration.

    ``style`` selects the LEARNER's gait-style recipe ("steady" lean-pull /
    "shuffle" quick-pull). The frozen-duck policy mix is identical for both.
    """
    assert style in ("steady", "shuffle")
    steady = style == "steady"

    cfg = make_microduck_velocity_env_cfg(play=play, rough=False)

    # Six ducks in contact range of each other overflow the velocity env's
    # default nconmax=35 (observed: narrowphase overflow at reset when the
    # pre-tensioned chain settles). Same headroom as the rough-terrain envs.
    cfg.sim.nconmax = 200

    # ── Scene: 6 harness ducks — learner "robot" (red middle) + 5 frozen ────
    cfg.scene.entities = {"robot": MICRODUCK_TUG_ROBOT_CFG}
    for name in _FROZEN_DUCKS:
        cfg.scene.entities[name] = make_tug_duck_entity_cfg()
    cfg.scene.spec_fn = _add_chain3_ropes

    # ── Actions: learner keeps the 14D joint_pos term; each frozen duck gets
    # its own zero-dim action term running its frozen v7 ONNX every env step.
    for name, policy_path in _FROZEN_DUCKS.items():
        cfg.actions[f"frozen_{name}"] = microduck_mdp.FrozenTugOpponentActionCfg(
            entity_name=name,
            policy_path=str(policy_path),
        )

    # ── Rewards: drop command-gated locomotion terms (same set as v7) ───────
    for name in [
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "upright",  # replaced by tug_chain3_trunk_lean (a puller leans)
        "pose",     # do-nothing jackpot vs the task stack (see tug task)
    ]:
        cfg.rewards.pop(name, None)

    # Anti-slip stays always-on (no command gate in this command-less task).
    cfg.rewards["foot_slip"].weight = -0.1
    cfg.rewards["foot_slip"].params["command_threshold"] = 0.0

    # ── Rewards: the task (identical in both styles) ─────────────────────────
    # Main objective: potential-based rope-chain MIDPOINT drag — the match's
    # actual win condition. Same weight/rate-cap math as v7's chain_progress.
    cfg.rewards["chain3_progress"] = RewardTermCfg(
        func=microduck_mdp.tug_chain3_progress,
        weight=500.0,
        params={"max_paid_rate": PROGRESS_MAX_PAID_RATE},
    )
    # Survival under load: pays per step ONLY while the learner's butt cord is
    # taut and the trunk is up — a 20 s grind must out-pay a kamikaze yank.
    cfg.rewards["tug_taut_alive"] = RewardTermCfg(
        func=microduck_mdp.tug_taut_alive,
        weight=0.3,
        params={
            "taut_length": TEAMMATE_TAUT_LENGTH,
            "robot_site": "rope_hook",
            "cart_site": "rope_hook_chest",
            "cart_asset": "red_inner",
        },
    )

    # ── Rewards: gait style (the ONLY differences between the recipes) ──────
    # Gated on (midpoint moving toward red OR learner's butt cord taut): the
    # stance must pay through a standstill grind but never as free posing.
    cfg.rewards["tug_chain3_trunk_lean"] = RewardTermCfg(
        func=microduck_mdp.tug_chain3_trunk_lean_tracking,
        weight=1.5,
        params={
            "target_pitch": STEADY_LEAN_PITCH if steady else SHUFFLE_LEAN_PITCH,
            "std": LEAN_STD,
            "taut_length": TEAMMATE_TAUT_LENGTH,
        },
    )
    cfg.rewards["tug_chain3_step_cadence"] = RewardTermCfg(
        func=microduck_mdp.tug_chain3_step_cadence_tracking,
        weight=2.0,
        params={
            "sensor_name": "feet_ground_contact",
            "target_hz": STEADY_CADENCE_HZ if steady else SHUFFLE_CADENCE_HZ,
            "std_hz": CADENCE_STD_HZ,
            "taut_length": TEAMMATE_TAUT_LENGTH,
        },
    )
    # Action smoothness: same stage-0 values and ramps as v6/v7.
    cfg.rewards["action_rate_l2"].weight = -0.1 if steady else -0.05

    # ── Commands: twist tiny zero-pad (slot parity only; no tracking reward) ─
    command = cfg.commands["twist"]
    command.rel_turn_in_place_envs = 0.0
    command.ranges.lin_vel_x = (-0.01, 0.01)
    command.ranges.lin_vel_y = (-0.01, 0.01)
    command.ranges.ang_vel_z = (-0.05, 0.05)
    command.debug_vis = False
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))
    # head_pose / body_pose commands + obs terms stay as the velocity factory
    # wired them (tiny ranges keep the slots' input neurons alive; head
    # tracking at weight 2.0 keeps the 280 g counterweight head near HOME).

    # ── Terminations ─────────────────────────────────────────────────────────
    # Only the LEARNER's fall ends the episode: tilt, low trunk, hard 35°
    # overlean gate. Fallen frozen ducks stay in the sim as dead weight on the
    # rope (match rules) — this is the v8 fix for v7's ~3 s mean rounds.
    cfg.terminations["trunk_low"] = TerminationTermCfg(
        func=microduck_mdp.root_height_below,
        params={"min_height": FALLEN_TRUNK_Z},
    )
    cfg.terminations["overlean"] = TerminationTermCfg(
        func=mjlab_terminations.bad_orientation,
        params={"limit_angle": OVERLEAN_LIMIT},
    )
    # NaN guards apply to the WHOLE field: a NaN duck corrupts the shared sim
    # state even when it isn't the learner.
    for name in _FROZEN_DUCKS:
        cfg.terminations[f"nan_state_{name}"] = TerminationTermCfg(
            func=microduck_mdp.robot_state_is_nan,
            time_out=False,
            params={"asset_cfg": SceneEntityCfg(name)},
        )
    # Bounds: learner only (a frozen duck dragged out of range is the LEARNER's
    # problem to solve via the midpoint reward, not a termination).
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.terminations["out_of_bounds"] = TerminationTermCfg(
        func=microduck_mdp.tug_out_of_bounds,
        time_out=True,
        params={
            "max_distance": MAX_DISTANCE,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # ── Events ───────────────────────────────────────────────────────────────
    # Standing-start joint noise on the LEARNER only; frozen ducks spawn at
    # exact DEFAULT_POSE.
    cfg.events["reset_robot_joints"].params["position_range"] = (-0.05, 0.05)

    # Team layout — MUST come after reset_base (insertion order): all frozen
    # duck poses derive from the final robot pose. Also freezes the per-env
    # pull direction (robot heading at reset).
    if ENABLE_GAP_DR:
        teammate_gap, center_gap = TEAMMATE_GAP_RANGE, CENTER_GAP_RANGE
    else:
        teammate_gap = (TEAMMATE_TRUNK_GAP, TEAMMATE_TRUNK_GAP)
        center_gap = (CENTER_TRUNK_GAP, CENTER_TRUNK_GAP)
    cfg.events["reset_tug_chain3_team"] = EventTermCfg(
        func=microduck_mdp.reset_tug_chain3_team,
        mode="reset",
        params={
            "teammate_gap_range": teammate_gap,
            "center_gap_range": center_gap,
        },
    )

    if not ENABLE_VELOCITY_PUSHES:
        cfg.events.pop("push_robot", None)

    # ── Curriculum ───────────────────────────────────────────────────────────
    # Same as v7: velocity factory's com/head-com/head-pose curricula stay;
    # only the action_rate ramp is style-specific.
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": (
                [
                    {"step": 0, "weight": -0.1},
                    {"step": 500 * NUM_STEPS_PER_ENV, "weight": -0.2},
                    {"step": 750 * NUM_STEPS_PER_ENV, "weight": -0.4},
                    {"step": 1000 * NUM_STEPS_PER_ENV, "weight": -0.6},
                    {"step": 1250 * NUM_STEPS_PER_ENV, "weight": -0.8},
                    {"step": 1500 * NUM_STEPS_PER_ENV, "weight": -1.0},
                ]
                if steady
                else [
                    {"step": 0, "weight": -0.05},
                    {"step": 750 * NUM_STEPS_PER_ENV, "weight": -0.15},
                    {"step": 1500 * NUM_STEPS_PER_ENV, "weight": -0.3},
                    {"step": 2500 * NUM_STEPS_PER_ENV, "weight": -0.5},
                ]
            ),
        },
    )

    return cfg


# ── RL runner configs ─────────────────────────────────────────────────────────

def _make_tug_chain3_rl_cfg(experiment_name: str) -> RslRlOnPolicyRunnerCfg:
    """Velocity-shaped PPO config with a per-style experiment name."""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,  # normalizer MUST be baked into ONNX by export.py
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=PpoWithSymmetryCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
        ),
        wandb_project="mjlab_microduck",
        experiment_name=experiment_name,
        run_name=experiment_name,
        save_interval=250,
        num_steps_per_env=NUM_STEPS_PER_ENV,
        max_iterations=10_000,
    )


MicroduckTugChain3SteadyRlCfg = _make_tug_chain3_rl_cfg("tugchain3_steady")
MicroduckTugChain3ShuffleRlCfg = _make_tug_chain3_rl_cfg("tugchain3_shuffle")
