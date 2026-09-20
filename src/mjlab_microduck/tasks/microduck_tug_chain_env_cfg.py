"""Microduck tug-CHAIN task — 1v1 duck-vs-duck tug-of-war, frozen-ONNX opponent.

Two harness ducks stand back-to-back, linked butt-ring to butt-ring by the
match's tension-only cord (parameters from robot/tug_of_war.py: 200 N/m,
0.005 m dead band past the ~0.103 m nominal ring distance, damping 2.0). The
learner ("robot" entity) trains against the "opponent" entity driven by a
FROZEN sled-tug ONNX policy rebuilt in pure torch (see mdp.FrozenTugPolicy /
FrozenTugOpponentAction). Motivation: sled-trained tug policies fall 67-90% of
the time when the cart is replaced by a live duck that pulls back — the
opponent has to be in the training loop.

Cross-adversary pairing (each style trains against the OTHER style's frozen
policy, so neither side sees a mirror of itself):
  - "steady"  learner (lean-back pull)   vs frozen tug_shuffle_v6
  - "shuffle" learner (high-cadence pull) vs frozen tug_steady_v6

DR / obs noise / delays: velocity-parity for the LEARNER, inherited by
building on make_microduck_velocity_env_cfg — exactly like the sled tug task.
The opponent duck gets NO DR and no push events (it represents the deployed
frozen policy as-is); the per-step opponent drive is a zero-dim action term
(mdp.FrozenTugOpponentActionCfg), so the learner's action contract stays 14D
and the opponent runs every env step, batched on the env device.
"""

import math
from pathlib import Path

import mujoco as _mujoco
import numpy as np

# ── Toggles ──────────────────────────────────────────────────────────────────
# Robot-side DR / noise / delays follow the velocity recipe's own module flags
# (照搬 velocity, same as the sled tug task); these toggles cover what THIS
# task adds on top.
ENABLE_OPPONENT_GAP_DR = True   # per-episode spawn gap noise around the pre-tensioned link
ENABLE_VELOCITY_PUSHES = True   # velocity-parity shove robustness (learner only)

# Symmetry — must stay OFF: pulling with a lean is not mirror-symmetric.
ENABLE_SYMMETRY = False

# ── Rope / chain constants (match cord model, robot/tug_of_war.py) ───────────
ROPE_STIFFNESS = 200.0        # N/m
ROPE_SLACK = 0.005            # dead band past the nominal ring-to-ring distance
ROPE_LIMIT_MARGIN = 0.05      # soft upper safety catch past the taut length
ROPE_DAMPING = 2.0            # model.tendon_damping in the match demo
ROPE_WIDTH = 0.0025
ROPE_RGBA = (0.92, 0.87, 0.70, 1.0)

# 1v1 link geometry: butt ring to butt ring ≈ 0.103 m nominal + slack. Both
# rings sit at trunk-local TUG_RING_LOCAL on same-height trunks, so the rope
# is horizontal and the taut length is the horizontal ring distance + slack.
RING_GAP_NOMINAL = 0.103
ROPE_TAUT_LENGTH = RING_GAP_NOMINAL + ROPE_SLACK  # 0.108

# Frozen opponent policies (cross-adversary), resolved against the repo root.
_POLICY_DIR = Path(__file__).resolve().parents[3] / "policies"
OPPONENT_POLICY_BY_STYLE = {
    "steady": _POLICY_DIR / "tug_shuffle_v6.onnx",
    "shuffle": _POLICY_DIR / "tug_steady_v6.onnx",
}

# Spawn: ring distance starts 5–10 mm PAST the taut length (rope pre-loaded,
# no free slack phase). Trunk gap = ring gap + 2 × |TUG_RING_LOCAL.x| —
# computed after the imports below (TUG_RING_LOCAL lives in
# microduck_constants).
_SPAWN_PRETENSION = (0.005, 0.010)

# Task reward shaping
PROGRESS_MAX_PAID_RATE = 0.4   # m/s of opponent drag that pays; faster pays no extra

# Style recipe parameters (identical to the sled tug task's two recipes)
# v11: butt-to-rope means the puller leans FORWARD (head into the pull
# direction, sled-dog stance) — leaning back toward the rope was the
# face-the-rope pose, which reads wrong in our back-to-back setup.
STEADY_LEAN_PITCH = math.radians(8.0)  # v12: +15° 前倾让 38% 头重的鸭子前栽，降到 +8° 保平衡
SHUFFLE_LEAN_PITCH = 0.0
LEAN_STD = 0.07
STEADY_CADENCE_HZ = 1.5
SHUFFLE_CADENCE_HZ = 3.5
CADENCE_STD_HZ = 0.75

# Bounds (plane terrain — the base out_of_terrain_bounds is a no-op here)
MAX_DISTANCE = 2.0             # both ducks, from their env origin
FALLEN_TRUNK_Z = 0.055         # same fall criterion as the tug-of-war demo
OVERLEAN_LIMIT = math.radians(35.0)  # v6 hard posture gate on the LEARNER (the
                                     # opponent's posture is its own policy's
                                     # problem — it terminates via fell_over /
                                     # trunk_low like any fall)

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
    MICRODUCK_TUG_CHAIN_OPPONENT_CFG,
    MICRODUCK_TUG_ROBOT_CFG,
    TUG_RING_LOCAL,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    NUM_STEPS_PER_ENV,
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG

# Trunk-to-trunk spawn gap: nominal ring distance + slack + pre-tension.
_TRUNK_GAP_NOMINAL = RING_GAP_NOMINAL - 2.0 * TUG_RING_LOCAL[0]  # 0.2402 m
OPPONENT_TRUNK_GAP_RANGE = (
    _TRUNK_GAP_NOMINAL + ROPE_SLACK + _SPAWN_PRETENSION[0],
    _TRUNK_GAP_NOMINAL + ROPE_SLACK + _SPAWN_PRETENSION[1],
)


def _add_chain_rope(spec: _mujoco.MjSpec) -> None:
    """Scene-level spec_fn: tension-only dead-band rope, butt ring ↔ butt ring.

    Entities are attached with "<name>/" prefixes before spec_fn runs, so the
    wrap sites are "robot/rope_hook" and "opponent/rope_hook" (same
    cross-entity tendon pattern as the sled tug's robot→cart rope). Parameters
    follow robot/tug_of_war.py's cord model, including the runtime
    tendon_damping=2.0 — set at spec level here (mujoco 3.10's MjSpec binds
    tendon damping as a (3,1) array whose element 0 compiles to the scalar
    tendon_damping; verified against spec.compile()).
    """
    rope = spec.add_tendon(
        name="tug_rope",
        stiffness=ROPE_STIFFNESS,
        springlength=(0.0, ROPE_TAUT_LENGTH),
        limited=True,
        range=(0.0, ROPE_TAUT_LENGTH + ROPE_LIMIT_MARGIN),
        width=ROPE_WIDTH,
        rgba=ROPE_RGBA,
        solref_limit=(0.02, 1.0),
        solimp_limit=(0.90, 0.95, 0.001, 0.5, 2.0),
    )
    rope.damping = np.array([[ROPE_DAMPING], [0.0], [0.0]])
    rope.wrap_site("robot/rope_hook")
    rope.wrap_site("opponent/rope_hook")


def make_microduck_tug_chain_env_cfg(
    style: str = "steady",
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create the Microduck 1v1 tug-chain environment configuration.

    ``style`` selects the LEARNER's gait-style recipe ("steady" lean-pull /
    "shuffle" quick-pull); the frozen opponent is the other style's v6 policy.
    """
    assert style in ("steady", "shuffle")
    steady = style == "steady"

    cfg = make_microduck_velocity_env_cfg(play=play, rough=False)

    # ── Scene: two harness ducks (learner + frozen-policy opponent) ──────────
    cfg.scene.entities = {
        "robot": MICRODUCK_TUG_ROBOT_CFG,
        "opponent": MICRODUCK_TUG_CHAIN_OPPONENT_CFG,
    }
    cfg.scene.spec_fn = _add_chain_rope

    # ── Actions: learner keeps the 14D joint_pos term; the opponent is driven
    # by a zero-dim action term running the frozen ONNX policy every env step.
    cfg.actions["opponent_policy"] = microduck_mdp.FrozenTugOpponentActionCfg(
        entity_name="opponent",
        policy_path=str(OPPONENT_POLICY_BY_STYLE[style]),
    )

    # ── Rewards: drop command-gated locomotion terms (same set as the sled
    # tug task — the pull is self-directed, twist stays tiny/zero-padded) ─────
    for name in [
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "upright",  # replaced by tug_chain_trunk_lean (a puller leans)
        "pose",     # do-nothing jackpot vs the task stack (see tug task)
    ]:
        cfg.rewards.pop(name, None)

    # Anti-slip stays always-on (no command gate in this command-less task).
    cfg.rewards["foot_slip"].weight = -0.1
    cfg.rewards["foot_slip"].params["command_threshold"] = 0.0

    # ── Rewards: the task (identical in both styles) ─────────────────────────
    # Main objective: potential-based OPPONENT drag. Same weight/rate-cap math
    # as the sled task's tug_cart_progress (Δ 0.004 m/step at 0.2 m/s × 500 ≈
    # 2/step; the 0.4 m/s cap keeps yanks from out-paying steady pulls).
    cfg.rewards["chain_progress"] = RewardTermCfg(
        func=microduck_mdp.tug_chain_progress,
        weight=500.0,
        params={"max_paid_rate": PROGRESS_MAX_PAID_RATE},
    )
    # Survival under load: pays per step ONLY while the rope is taut and the
    # trunk is up — same term as the sled task, rope length now measured
    # ring-to-ring against the opponent.
    cfg.rewards["tug_taut_alive"] = RewardTermCfg(
        func=microduck_mdp.tug_taut_alive,
        weight=0.3,
        params={
            "taut_length": ROPE_TAUT_LENGTH,
            "robot_site": "rope_hook",
            "cart_site": "rope_hook",
            "cart_asset": "opponent",
        },
    )

    # ── Rewards: gait style (the ONLY differences between the recipes) ──────
    # Both style terms are GATED on (opponent moving your way OR rope taut) —
    # the sled task's cart-moving gate would go silent in a standstill grind
    # and teach rope-hanging against a live opponent.
    cfg.rewards["tug_chain_trunk_lean"] = RewardTermCfg(
        func=microduck_mdp.tug_chain_trunk_lean_tracking,
        weight=1.5,
        params={
            "target_pitch": STEADY_LEAN_PITCH if steady else SHUFFLE_LEAN_PITCH,
            "std": LEAN_STD,
            "taut_length": ROPE_TAUT_LENGTH,
        },
    )
    cfg.rewards["tug_chain_step_cadence"] = RewardTermCfg(
        func=microduck_mdp.tug_chain_step_cadence_tracking,
        weight=2.0,
        params={
            "sensor_name": "feet_ground_contact",
            "target_hz": STEADY_CADENCE_HZ if steady else SHUFFLE_CADENCE_HZ,
            "std_hz": CADENCE_STD_HZ,
            "taut_length": ROPE_TAUT_LENGTH,
        },
    )
    # v13: hold the spawn heading — self-directed pullers curve (v7 spun
    # -265°/15s in the match); the yaw coupling/damping fixes all blocked
    # the bout-deciding pivot, so straightness must be TRAINED.
    cfg.rewards["tug_chain_heading_hold"] = RewardTermCfg(
        func=microduck_mdp.tug_chain_heading_hold,
        weight=0.5,
        params={"std": 0.35},
    )
    # Action smoothness: same stage-0 values and ramps as the sled tug task.
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
    # Learner fall = tilt OR low trunk, plus the v6 hard 35° overlean gate.
    cfg.terminations["trunk_low"] = TerminationTermCfg(
        func=microduck_mdp.root_height_below,
        params={"min_height": FALLEN_TRUNK_Z},
    )
    cfg.terminations["overlean"] = TerminationTermCfg(
        func=mjlab_terminations.bad_orientation,
        params={"limit_angle": OVERLEAN_LIMIT},
    )
    # Opponent fall ends the round too — a downed opponent is a won (or broken)
    # exchange, and dragging a corpse along the floor is not the task.
    cfg.terminations["opponent_fell_over"] = TerminationTermCfg(
        func=mjlab_terminations.bad_orientation,
        params={
            "limit_angle": math.radians(70.0),
            "asset_cfg": SceneEntityCfg("opponent"),
        },
    )
    cfg.terminations["opponent_trunk_low"] = TerminationTermCfg(
        func=microduck_mdp.root_height_below,
        params={
            "min_height": FALLEN_TRUNK_Z,
            "asset_cfg": SceneEntityCfg("opponent"),
        },
    )
    # NaN guards: the robot's is inherited from the velocity wiring (with the
    # feet_ground_contact sensor check); the opponent gets its own state guard.
    cfg.terminations["nan_state_opponent"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"asset_cfg": SceneEntityCfg("opponent")},
    )
    # Base out_of_terrain_bounds is a no-op on plane terrain — replace with
    # explicit distance-from-origin bounds for BOTH ducks.
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.terminations["out_of_bounds"] = TerminationTermCfg(
        func=microduck_mdp.tug_out_of_bounds,
        time_out=True,
        params={
            "max_distance": MAX_DISTANCE,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.terminations["opponent_out_of_bounds"] = TerminationTermCfg(
        func=microduck_mdp.tug_out_of_bounds,
        time_out=True,
        params={
            "max_distance": MAX_DISTANCE,
            "asset_cfg": SceneEntityCfg("opponent"),
        },
    )

    # ── Events ───────────────────────────────────────────────────────────────
    # Standing-start joint noise on the LEARNER only (deployment hands off
    # from a settled stand); the opponent spawns at exact DEFAULT_POSE.
    cfg.events["reset_robot_joints"].params["position_range"] = (-0.05, 0.05)

    # Opponent placement — MUST come after reset_base (insertion order): the
    # opponent pose derives from the final robot pose. Also freezes the
    # per-env pull direction (robot heading at reset).
    gap_range = OPPONENT_TRUNK_GAP_RANGE if ENABLE_OPPONENT_GAP_DR else (
        sum(OPPONENT_TRUNK_GAP_RANGE) / 2.0,
        sum(OPPONENT_TRUNK_GAP_RANGE) / 2.0,
    )
    cfg.events["reset_tug_chain_opponent"] = EventTermCfg(
        func=microduck_mdp.reset_tug_chain_opponent,
        mode="reset",
        params={"trunk_gap_range": gap_range},
    )

    if not ENABLE_VELOCITY_PUSHES:
        cfg.events.pop("push_robot", None)

    # ── Curriculum ───────────────────────────────────────────────────────────
    # Same replacement as the sled tug task: the velocity factory's
    # com/head-com/head-pose curricula stay valid; only the action_rate ramp
    # is style-specific.
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

def _make_tug_chain_rl_cfg(experiment_name: str) -> RslRlOnPolicyRunnerCfg:
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


MicroduckTugChainSteadyRlCfg = _make_tug_chain_rl_cfg("tugchain_steady")
MicroduckTugChainShuffleRlCfg = _make_tug_chain_rl_cfg("tugchain_shuffle")
