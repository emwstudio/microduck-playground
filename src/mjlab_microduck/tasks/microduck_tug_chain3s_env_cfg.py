"""Microduck tug-CHAIN-3s task — 3v3 FULL self-play, all 6 ducks learning.

v9 of the chain self-play line. v8 (one learner + 5 frozen v7 ducks) hit its
training metrics but collapsed in the 5v5 match (94% floor time — worse than
v7): a frozen duck's rigid dynamics taught the learner to lean on teammates
that hold like anchors, while in a real match everyone leans on everyone and
the chain falls together. v9 removes ALL frozen ducks: every duck in the 3v3
match is driven by ONE shared learning policy (symmetric game; the two task
styles produce the two teams' match policies).

Scene, cord geometry and spawn layout reuse the v8 3v3 task
(microduck_tug_chain3_env_cfg): 6 harness ducks, 5 match-spec cords, the red
middle duck keeps the name "robot" (velocity-recipe wiring and the ONNX
export metadata patch key off it). The differences are structural:

  - actions: 6 per-duck joint_pos terms (14D each, BAM path identical to the
    single-duck tasks);
  - obs: ONE custom term (mdp.tug_chain3s_duck_obs) producing (N, 6·61) —
    per-duck 61D deployment-contract rows, clean reads like the deployment
    rehearsal (no obs noise; BAM actuator delay still applies to all ducks);
  - rewards/terminations: computed per duck (see tug_chain3s_vecenv.py for the
    (N-match) → (N·6 duck-row) flattening);
  - match-level done: ANY duck falling (tilt 70° / trunk low / 35° overlean),
    leaving bounds, or NaN ends the round for all 6 — a fall loses the match;
  - domain randomization (friction / mass / CoM / armature / BAM friction) is
    applied to ALL SIX ducks — they are all real robots. Velocity pushes and
    encoder-bias DR are dropped (no external pushes in a match; obs are clean).
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import terminations as mjlab_terminations
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    EventTermCfg,
    ObservationTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_TUG_ROBOT_CFG,
    make_tug_duck_entity_cfg,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_tug_chain3_env_cfg import (
    CENTER_GAP_RANGE,
    CENTER_TAUT_LENGTH,
    TEAMMATE_GAP_RANGE,
    TEAMMATE_TAUT_LENGTH,
    _add_chain3_ropes,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    HEAD_BODY_NAMES,
    NUM_STEPS_PER_ENV,
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG

# ── Toggles ──────────────────────────────────────────────────────────────────
ENABLE_GAP_DR = True   # per-episode per-cord spawn pre-tension noise

# Symmetry — must stay OFF: pulling with a lean is not mirror-symmetric.
ENABLE_SYMMETRY = False

# Task reward shaping
PROGRESS_MAX_PAID_RATE = 0.4   # m/s of midpoint drag that pays; faster pays no extra

# Style recipe parameters (identical to the v6–v8 recipes)
STEADY_LEAN_PITCH = math.radians(-18.0)  # butt toward the enemy chain
SHUFFLE_LEAN_PITCH = 0.0
LEAN_STD = 0.07
STEADY_CADENCE_HZ = 1.5
SHUFFLE_CADENCE_HZ = 3.5
CADENCE_STD_HZ = 0.75

# Bounds / fall criteria (per duck — any duck down ends the match)
MAX_DISTANCE = 2.0
FALLEN_TRUNK_Z = 0.055
OVERLEAN_LIMIT = math.radians(35.0)  # v6 hard posture gate, per duck in v9

_FOOT_GEOMS = ("left_foot_collision", "right_foot_collision")


def _feet_sensor_cfg(duck: str) -> ContactSensorCfg:
    return ContactSensorCfg(
        name=f"feet_ground_contact_{duck}",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",
            entity=duck,
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )


def _self_collision_cfg(duck: str) -> ContactSensorCfg:
    return ContactSensorCfg(
        name=f"self_collision_{duck}",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity=duck),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity=duck),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )


def _clone_dr_event(base: EventTermCfg, duck: str, **asset_kwargs) -> EventTermCfg:
    """Clone one of the velocity recipe's DR events for another duck entity."""
    params = dict(base.params)
    params["asset_cfg"] = SceneEntityCfg(duck, **asset_kwargs)
    return EventTermCfg(func=base.func, mode=base.mode, params=params)


def make_chain3s_reward_spec(style: str = "steady") -> dict:
    """Per-duck reward/curriculum spec consumed by TugChainSelfPlayVecEnv.

    This is a standalone builder (not only a cfg attribute) because mjlab's
    train CLI reconstructs the env cfg through tyro, which drops non-dataclass
    attributes — the runner rebuilds the spec from the experiment name when
    the attribute didn't survive (see TugChainSelfPlayRunner).
    """
    assert style in ("steady", "shuffle")
    steady = style == "steady"
    taut = {"teammate_taut": TEAMMATE_TAUT_LENGTH, "center_taut": CENTER_TAUT_LENGTH}
    return {
        "style": style,
        "terms": [
            # Team term: midpoint drag toward your side (potential-based Δ).
            (
                "chain3s_team_progress",
                microduck_mdp.tug_chain3s_team_progress,
                500.0,
                {"max_paid_rate": PROGRESS_MAX_PAID_RATE},
            ),
            # Per-duck survival under load (own butt cord taut + upright).
            ("tug_taut_alive", microduck_mdp.tug_chain3s_taut_alive, 0.3, taut),
            # Style terms (gated: cord taut OR midpoint moving your way).
            (
                "trunk_lean",
                microduck_mdp.tug_chain3s_trunk_lean,
                1.5,
                {
                    "target_pitch": STEADY_LEAN_PITCH if steady else SHUFFLE_LEAN_PITCH,
                    "std": LEAN_STD,
                    **taut,
                },
            ),
            (
                "step_cadence",
                microduck_mdp.tug_chain3s_step_cadence,
                2.0,
                {
                    "target_hz": STEADY_CADENCE_HZ if steady else SHUFFLE_CADENCE_HZ,
                    "std_hz": CADENCE_STD_HZ,
                    **taut,
                },
            ),
            # Head pinned near HOME (counterweight; no head command slot in v9).
            ("head_home", microduck_mdp.tug_chain3s_head_home, 2.0, {"std": 0.5}),
            # Regularizers (penalties — negative weights, cost-style ≥ 0 funcs).
            ("action_rate_l2", microduck_mdp.tug_chain3s_action_rate, None, {}),  # curriculum
            ("body_ang_vel", microduck_mdp.tug_chain3s_body_ang_vel, -0.05, {}),
            ("dof_pos_limits", microduck_mdp.tug_chain3s_joint_limits, -1.0, {}),
            ("self_collisions", microduck_mdp.tug_chain3s_self_collision, -1.0, {}),
            ("foot_slip", microduck_mdp.tug_chain3s_foot_slip, -0.1, {}),
        ],
        # Same ramps as v7/v8 (wrapper-side, off env.common_step_counter).
        "action_rate_stages": (
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
    }


def make_microduck_tug_chain3s_env_cfg(
    style: str = "steady",
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create the Microduck 3v3 full-self-play tug-chain environment.

    ``style`` selects the shared policy's gait-style recipe ("steady"
    lean-pull / "shuffle" quick-pull) — every duck in the match plays the same
    style; the match itself pits this run's policy against the other style's.
    """
    assert style in ("steady", "shuffle")
    steady = style == "steady"
    ducks = microduck_mdp.CHAIN3S_DUCKS

    cfg = make_microduck_velocity_env_cfg(play=play, rough=False)

    # Six ducks in contact range of each other overflow the velocity env's
    # default nconmax=35 (same fix as the v8 3v3 task).
    cfg.sim.nconmax = 200
    # The wrapper (tug_chain3s_vecenv.py) handles resets itself so terminal
    # rewards are computed from the true terminal state.
    cfg.auto_reset = False

    # ── Scene: 6 harness ducks + 5 match cords (v8 geometry verbatim) ───────
    cfg.scene.entities = {"robot": MICRODUCK_TUG_ROBOT_CFG}
    for duck in ducks:
        if duck != "robot":
            cfg.scene.entities[duck] = make_tug_duck_entity_cfg()
    cfg.scene.spec_fn = _add_chain3_ropes
    cfg.scene.sensors = tuple(
        s
        for duck in ducks
        for s in (_feet_sensor_cfg(duck), _self_collision_cfg(duck))
    )

    # ── Actions: one 14D joint_pos term per duck (BAM path identical to the
    # single-duck tasks; scale=1.0 + default-offset, like the velocity recipe).
    # Dict order = CHAIN3S_DUCKS order — the wrapper's flattening depends on it.
    base_action = cfg.actions["joint_pos"]
    assert isinstance(base_action, JointPositionActionCfg)
    base_action.scale = 1.0
    cfg.actions = {}
    for duck in ducks:
        name = "joint_pos" if duck == "robot" else f"joint_pos_{duck}"
        cfg.actions[name] = JointPositionActionCfg(
            entity_name=duck,
            actuator_names=base_action.actuator_names,
            scale=1.0,
        )

    # ── Observations: one custom term producing all six 61D duck rows ───────
    for group in ("actor", "critic"):
        cfg.observations[group].terms = {
            "ducks61": ObservationTermCfg(func=microduck_mdp.tug_chain3s_duck_obs),
        }

    # ── Rewards: none in the inner manager — per-duck rewards are computed by
    # the wrapper (mjlab's RewardManager only understands per-env rows). ─────
    cfg.rewards = {}

    # ── Commands / curriculum: none (zero-padded command slots baked into the
    # obs; the action-rate ramp runs wrapper-side off common_step_counter). ──
    cfg.commands = {}
    cfg.curriculum = {}

    # ── Terminations: match ends if ANY duck falls / leaves / NaNs ──────────
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.terminations.pop("fell_over", None)  # replaced by per-duck terms
    cfg.terminations.pop("nan_state", None)
    for duck in ducks:
        cfg.terminations[f"fell_over_{duck}"] = TerminationTermCfg(
            func=mjlab_terminations.bad_orientation,
            params={
                "limit_angle": math.radians(70.0),
                "asset_cfg": SceneEntityCfg(duck),
            },
        )
        cfg.terminations[f"overlean_{duck}"] = TerminationTermCfg(
            func=mjlab_terminations.bad_orientation,
            params={
                "limit_angle": OVERLEAN_LIMIT,
                "asset_cfg": SceneEntityCfg(duck),
            },
        )
        cfg.terminations[f"trunk_low_{duck}"] = TerminationTermCfg(
            func=microduck_mdp.root_height_below,
            params={"min_height": FALLEN_TRUNK_Z, "asset_cfg": SceneEntityCfg(duck)},
        )
        cfg.terminations[f"nan_state_{duck}"] = TerminationTermCfg(
            func=microduck_mdp.robot_state_is_nan,
            time_out=False,
            params={"asset_cfg": SceneEntityCfg(duck)},
        )
        cfg.terminations[f"out_of_bounds_{duck}"] = TerminationTermCfg(
            func=microduck_mdp.tug_out_of_bounds,
            time_out=True,
            params={"max_distance": MAX_DISTANCE, "asset_cfg": SceneEntityCfg(duck)},
        )

    # ── Events ───────────────────────────────────────────────────────────────
    # Keep: reset_scene_to_default (default), expand_bam_friction_fields (BAM
    # field expansion), reset_action_history (clears last_action for all six
    # ducks at reset). Drop or replace the robot-only velocity events:
    #   - reset_base / reset_robot_joints → reset_tug_chain3s_match (all six)
    #   - encoder_bias (clean obs — nothing reads the bias) and push_robot
    #     (no external pushes in a match工况) → dropped
    #   - the DR stack → cloned per duck (all six are real robots)
    for name in ("reset_base", "reset_robot_joints", "encoder_bias", "push_robot"):
        cfg.events.pop(name, None)

    dr_clones = {
        "randomize_com": {"body_names": ("trunk_base",)},
        "randomize_head_com": {"body_names": HEAD_BODY_NAMES},
        "randomize_mass_inertia": {"body_names": ("trunk_base",)},
        "randomize_joint_friction": {},
        "randomize_armature": {"joint_names": (r".*",)},
        "foot_friction": {"geom_names": _FOOT_GEOMS},
    }
    for name, asset_kwargs in dr_clones.items():
        base = cfg.events.pop(name, None)
        if base is None:
            continue
        for duck in ducks:
            cfg.events[f"{name}_{duck}"] = _clone_dr_event(base, duck, **asset_kwargs)

    gap_teammate = TEAMMATE_GAP_RANGE if ENABLE_GAP_DR else (0.24, 0.24)
    gap_center = CENTER_GAP_RANGE if ENABLE_GAP_DR else (0.30, 0.30)
    cfg.events["reset_tug_chain3s_match"] = EventTermCfg(
        func=microduck_mdp.reset_tug_chain3s_match,
        mode="reset",
        params={
            "teammate_gap_range": gap_teammate,
            "center_gap_range": gap_center,
        },
    )

    # ── Self-play spec consumed by the wrapper (tug_chain3s_vecenv.py) ──────
    cfg.chain3s_spec = make_chain3s_reward_spec(style)

    return cfg


# ── RL runner configs ─────────────────────────────────────────────────────────

def _make_tug_chain3s_rl_cfg(experiment_name: str) -> RslRlOnPolicyRunnerCfg:
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


MicroduckTugChain3sSteadyRlCfg = _make_tug_chain3s_rl_cfg("tugchain3s_steady")
MicroduckTugChain3sShuffleRlCfg = _make_tug_chain3s_rl_cfg("tugchain3s_shuffle")
