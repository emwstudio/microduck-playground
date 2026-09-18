"""Microduck tug task — drag a weighted sled on a slack rope, two style recipes.

The duck spawns facing +x with a 3 kg low-slung cart behind it (-x); a
tension-only spatial tendon (dead-band cord, same model as the NvN
tug-of-war demo) links the butt tow eye to the cart's front eye with a few cm
of slack. Walking forward with the rope taut drags the cart; that dragged
displacement is the task.

Two style recipes share the SAME task/anti-fall terms and differ only in gait
style shaping, so their comparison is fair:

  - "steady"  (red team): lean-back pull — trunk pitch tracks ~-18° (butt
    toward the load, classic tug-of-war stance), low step cadence, tight
    action-rate smoothing (velocity's full ramp to -1.0).
  - "shuffle" (blue team): high-frequency shuffle pull — upright trunk (pitch
    target 0), high step cadence, higher pull-speed weight, action-rate cap
    relaxed (ramps to -0.5 only).

DR / obs noise / delays: velocity-parity, inherited by building on
make_microduck_velocity_env_cfg (BAM friction fields, joint-friction DR via
friction_scale, armature/mass/CoM DR, IMU misalignment, encoder bias, pushes).
The cart is a separate mjlab entity (like the ball-kick ball); the rope tendon
is added by a scene-level spec_fn because mjlab attaches entities with
"<name>/" prefixes and a tendon can only wrap sites within one composed spec.

Cart physics notes (measured, CPU MuJoCo): geom friction combines by
element-wise MAX and the terrain floor carries mu=1.0, so the cart geom uses
priority=1 to let its own mu=0.15 win the pair — slip threshold ≈ 4.4 N at
3 kg, ~55% of the duck's static foot-grip budget: pullable, but it costs.
"""

import math

import mujoco as _mujoco

# ── Toggles ──────────────────────────────────────────────────────────────────
# Robot-side DR / noise / delays follow the velocity recipe's own module flags
# (照搬 velocity); these toggles cover what THIS task adds on top.
ENABLE_CART_MASS_RANDOMIZATION = True      # startup pseudo_inertia on the cart body
ENABLE_CART_FRICTION_RANDOMIZATION = True  # per-episode scale on the cart geom mu
ENABLE_CART_GAP_DR = True                  # spawn gap noise == rope slack DR
ENABLE_VELOCITY_PUSHES = True              # velocity-parity shove robustness

# Symmetry — must stay OFF: pulling with a lean is not mirror-symmetric.
ENABLE_SYMMETRY = False

# ── Rope / cart constants ────────────────────────────────────────────────────
ROPE_STIFFNESS = 200.0        # N/m, duck-to-duck cord value from tug_of_war.py
ROPE_SLACK = 0.03             # dead band past the nominal hook-to-hook distance
ROPE_LIMIT_MARGIN = 0.05      # soft upper safety catch past the taut length
ROPE_WIDTH = 0.0025
ROPE_RGBA = (0.92, 0.87, 0.70, 1.0)

# Nominal hook-to-hook spawn geometry: butt hook sits at trunk-local
# (-0.0686, 0, 0.016), trunk spawns at z≈0.12-0.13, the cart eye at z=0.04 —
# the rope leaves the butt sloping ~0.10 m down over the horizontal gap.
CART_GAP_NOMINAL = 0.16       # horizontal hook-to-hook distance at spawn
CART_GAP_DR = 0.03            # ± spawn-gap noise (this IS the slack DR)
CART_GAP_RANGE = (CART_GAP_NOMINAL - CART_GAP_DR, CART_GAP_NOMINAL + CART_GAP_DR)
_HOOK_DZ = 0.10               # butt-hook height (~0.14) minus cart-eye height (0.04)
ROPE_TAUT_LENGTH = math.hypot(CART_GAP_NOMINAL, _HOOK_DZ) + ROPE_SLACK

CART_MASS_SCALE_RANGE = (0.8, 2.0)      # 2.4–6.0 kg — learn to pull a load that does NOT give
CART_FRICTION_SCALE_RANGE = (0.7, 4.0)  # mu 0.105–0.6: 低端好拉，高端 6kg 时 ~35N
                                        # 拖不动 — 模拟比赛中不肯动的对手鸭子
                                        # (v3 上限 8.8N，比赛遇到 15.7N+ 硬锚点就翻)

# Task reward shaping
PROGRESS_MAX_PAID_RATE = 0.4   # m/s of cart drag that pays; faster pays no extra
PULL_SPEED_CAP_STEADY = 0.25   # taut-rope pull-speed reward saturates here
PULL_SPEED_CAP_SHUFFLE = 0.35

# Style recipe parameters
STEADY_LEAN_PITCH = math.radians(-18.0)  # head toward -x = backward lean (butt to load)
SHUFFLE_LEAN_PITCH = 0.0
LEAN_STD = 0.07                # rad (~±4°) — v5 收紧: v4 挂到 -50° 都没被扣分
STEADY_CADENCE_HZ = 1.5        # slow deliberate steps
SHUFFLE_CADENCE_HZ = 3.5       # fast alternating shuffle
CADENCE_STD_HZ = 0.75

# Bounds (plane terrain — the base out_of_terrain_bounds is a no-op here)
ROBOT_MAX_DISTANCE = 2.0
CART_MAX_DISTANCE = 2.5
FALLEN_TRUNK_Z = 0.055         # same fall criterion as the tug-of-war demo

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_TUG_CART_CFG,
    MICRODUCK_TUG_ROBOT_CFG,
    TUG_CART_HALF_X,
    TUG_CART_HALF_Z,
    TUG_RING_LOCAL,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    NUM_STEPS_PER_ENV,
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


def _add_tug_rope(spec: _mujoco.MjSpec) -> None:
    """Scene-level spec_fn: tension-only dead-band rope, butt eye → cart eye.

    Entities are attached with "<name>/" prefixes before spec_fn runs, so the
    wrap sites are "robot/rope_hook" and "cart/tug_hook". Parameters follow
    robot/tug_of_war.py's cord model (stiffness / springlength / limited /
    range / solref_limit / solimp_limit); springlength's upper end is the taut
    length = nominal spawn distance + slack.
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
    rope.wrap_site("robot/rope_hook")
    rope.wrap_site("cart/tug_hook")


def make_microduck_tug_env_cfg(
    style: str = "steady",
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create the Microduck tug environment configuration.

    ``style`` selects the gait-style recipe ("steady" lean-pull / "shuffle"
    quick-pull); the task, anti-fall and sim2real regularizer terms are
    identical between the two.
    """
    assert style in ("steady", "shuffle")
    steady = style == "steady"

    cfg = make_microduck_velocity_env_cfg(play=play, rough=False)

    # ── Scene: walk robot + butt tow eye, plus the sled cart ────────────────
    cfg.scene.entities = {
        "robot": MICRODUCK_TUG_ROBOT_CFG,
        "cart": MICRODUCK_TUG_CART_CFG,
    }
    cfg.scene.spec_fn = _add_tug_rope

    # ── Rewards: drop command-gated locomotion terms (no velocity command
    # here — the pull is self-directed, twist stays tiny/zero-padded) ────────
    for name in [
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "upright",  # replaced by tug_trunk_lean_tracking (a puller leans)
        # pose pays joints-near-HOME whenever the (here ~zero) command is zero:
        # a ~0.76/step do-nothing jackpot that dwarfs the task stack AND taxes
        # the walking legs a pull needs — the v1 trials farmed exactly this
        # (leaned/posed, cart never moved). Remove it; head_pose_tracking at
        # weight 2.0 still pins the counterweight head.
        "pose",
    ]:
        cfg.rewards.pop(name, None)

    # Anti-slip stays always-on (no command gate in this command-less task).
    cfg.rewards["foot_slip"].weight = -0.1  # velocity's deliberate weak value
    cfg.rewards["foot_slip"].params["command_threshold"] = 0.0

    # ── Rewards: the task (identical in both styles) ─────────────────────────
    # Main objective: potential-based cart drag. At 0.2 m/s the Δ per step is
    # 0.004 m → ×500 pays ~2/step, matching velocity's main-term mass; the
    # 0.4 m/s rate cap keeps jerks from out-paying steady pulls (anti-jackpot).
    cfg.rewards["tug_cart_progress"] = RewardTermCfg(
        func=microduck_mdp.tug_cart_progress,
        weight=500.0,
        params={"max_paid_rate": PROGRESS_MAX_PAID_RATE},
    )
    # Taut-rope pull speed = the CART's speed under a taut rope (v2: v1 paid
    # robot speed and was farmed by stretching the stiff rope while the cart
    # sat still). Small on top of progress; shuffle pays it more.
    cfg.rewards["tug_taut_pull_speed"] = RewardTermCfg(
        func=microduck_mdp.tug_taut_rope_pull_speed,
        weight=2.0 if steady else 4.0,
        params={
            "taut_length": ROPE_TAUT_LENGTH,
            "max_speed": PULL_SPEED_CAP_STEADY if steady else PULL_SPEED_CAP_SHUFFLE,
        },
    )
    # Survival under load (v3): pays per step ONLY while the rope is taut and
    # the trunk is up — v2's burst-pull-then-die rounds lasted ~2 s; a 10 s
    # grind must out-pay a 1.5 s sprint. Weight 0.3/step ≈ progress mass, so
    # staying alive through the pull beats a kamikaze yank.
    cfg.rewards["tug_taut_alive"] = RewardTermCfg(
        func=microduck_mdp.tug_taut_alive,
        weight=0.3,
        params={"taut_length": ROPE_TAUT_LENGTH},
    )

    # ── Rewards: gait style (the ONLY differences between the recipes) ──────
    # Both style terms are GATED on cart motion inside the mdp funcs (×0 when
    # the cart sits still) — ungated style rewards get farmed as pure posing.
    cfg.rewards["tug_trunk_lean"] = RewardTermCfg(
        func=microduck_mdp.tug_trunk_lean_tracking,
        weight=1.5,
        params={
            "target_pitch": STEADY_LEAN_PITCH if steady else SHUFFLE_LEAN_PITCH,
            "std": LEAN_STD,
        },
    )
    cfg.rewards["tug_step_cadence"] = RewardTermCfg(
        func=microduck_mdp.tug_step_cadence_tracking,
        weight=2.0,  # v4 was 1.0 — drowned by the 500-weight progress term, so
                     # hanging on the rope out-paid stepping; v5 makes legs pay
        params={
            "sensor_name": "feet_ground_contact",
            "target_hz": STEADY_CADENCE_HZ if steady else SHUFFLE_CADENCE_HZ,
            "std_hz": CADENCE_STD_HZ,
        },
    )
    # Action smoothness: stage-0 value; the curriculum below ramps it tighter
    # (steady → -1.0, velocity's proven schedule) or stays relaxed (shuffle →
    # -0.5 — a fast shuffle needs quicker action changes).
    cfg.rewards["action_rate_l2"].weight = -0.1 if steady else -0.05

    # ── Commands: twist tiny zero-pad (slot parity only; no tracking reward) ─
    command = cfg.commands["twist"]
    command.rel_turn_in_place_envs = 0.0  # meaningless without tracking rewards
    command.ranges.lin_vel_x = (-0.01, 0.01)
    command.ranges.lin_vel_y = (-0.01, 0.01)
    command.ranges.ang_vel_z = (-0.05, 0.05)
    command.debug_vis = False
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))
    # head_pose / body_pose commands + obs terms stay as the velocity factory
    # wired them (tiny ranges keep the slots' input neurons alive; head tracking
    # at weight 2.0 keeps the 280 g head near HOME — it's a counterweight).

    # ── Terminations ─────────────────────────────────────────────────────────
    # Fall = tilt OR low trunk (same dual criterion as the tug-of-war demo):
    # base fell_over (bad_orientation 70°) + a trunk-height guard.
    cfg.terminations["trunk_low"] = TerminationTermCfg(
        func=microduck_mdp.root_height_below,
        params={"min_height": FALLEN_TRUNK_Z},
    )
    # NaN guard for the robot is inherited from the velocity wiring (with the
    # feet_ground_contact sensor check); the cart gets its own state guard.
    cfg.terminations["nan_state_cart"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"asset_cfg": SceneEntityCfg("cart")},
    )
    # Base out_of_terrain_bounds is a no-op on plane terrain — replace with
    # explicit distance-from-origin bounds for both the duck and the cart.
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.terminations["out_of_bounds"] = TerminationTermCfg(
        func=microduck_mdp.tug_out_of_bounds,
        time_out=True,
        params={
            "max_distance": ROBOT_MAX_DISTANCE,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.terminations["cart_out_of_bounds"] = TerminationTermCfg(
        func=microduck_mdp.tug_out_of_bounds,
        time_out=True,
        params={
            "max_distance": CART_MAX_DISTANCE,
            "asset_cfg": SceneEntityCfg("cart"),
        },
    )

    # ── Events ───────────────────────────────────────────────────────────────
    # Standing-start joint noise (deployment hands off from a settled stand,
    # not exact HOME) — same as ball_kick.
    cfg.events["reset_robot_joints"].params["position_range"] = (-0.05, 0.05)

    # Cart placement — MUST come after reset_base (insertion order): the cart
    # position derives from the final robot pose. Also freezes the per-env
    # pull direction (robot heading at reset).
    if ENABLE_CART_GAP_DR:
        gap_range = CART_GAP_RANGE
    else:
        gap_range = (CART_GAP_NOMINAL, CART_GAP_NOMINAL)
    cfg.events["reset_tug_cart"] = EventTermCfg(
        func=microduck_mdp.reset_tug_cart,
        mode="reset",
        params={
            "gap_range": gap_range,
            "hook_local_x": TUG_RING_LOCAL[0],
            "cart_half_x": TUG_CART_HALF_X,
            "cart_half_z": TUG_CART_HALF_Z,
            "asset_name": "cart",
        },
    )

    if ENABLE_CART_MASS_RANDOMIZATION:
        # Physics-consistent mass+inertia scale on the cart body (startup =
        # fixed per env for the whole run, non-accumulating — same mechanism
        # as the robot's randomize_mass_inertia).
        _cm_lo, _cm_hi = CART_MASS_SCALE_RANGE
        cfg.events["randomize_cart_mass"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("cart", body_names=("tug_cart",)),
                "alpha_range": (math.log(_cm_lo) / 2.0, math.log(_cm_hi) / 2.0),
            },
        )

    if ENABLE_CART_FRICTION_RANDOMIZATION:
        # Per-episode scale on the cart geom's tangential mu around the 0.15
        # base; dr.geom_friction re-reads compile-time defaults (scale op), so
        # this is non-accumulating.
        cfg.events["randomize_cart_friction"] = EventTermCfg(
            func=dr.geom_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("cart", geom_names=("tug_cart_geom",)),
                "operation": "scale",
                "ranges": CART_FRICTION_SCALE_RANGE,
            },
        )

    if not ENABLE_VELOCITY_PUSHES:
        cfg.events.pop("push_robot", None)

    # ── Curriculum ───────────────────────────────────────────────────────────
    # The velocity factory's com/head-com/head-pose curricula stay valid
    # (their events/commands are untouched). Replace only the action_rate ramp
    # with the style-specific schedule.
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

def _make_tug_rl_cfg(experiment_name: str) -> RslRlOnPolicyRunnerCfg:
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


MicroduckTugSteadyRlCfg = _make_tug_rl_cfg("tug_steady")
MicroduckTugShuffleRlCfg = _make_tug_rl_cfg("tug_shuffle")
