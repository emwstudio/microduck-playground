"""Cfg-invariant and scene-structure tests for the tug task.

CPU-only: compiles the composed scene spec (robot + cart + rope tendon) and
checks the reward/termination wiring of both style recipes — no env build.
"""

import math

import mujoco
import pytest
from mjlab.scene import Scene
from mjlab.tasks.registry import list_tasks

from mjlab_microduck.robot.microduck_constants import (
    TUG_CART_FRICTION,
    TUG_CART_MASS,
    TUG_RING_LOCAL,
)
from mjlab_microduck.tasks import microduck_tug_env_cfg as tug_cfg_mod
from mjlab_microduck.tasks.microduck_tug_env_cfg import (
    MicroduckTugShuffleRlCfg,
    MicroduckTugSteadyRlCfg,
    make_microduck_tug_env_cfg,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)


def _compiled_tug_model(style: str = "steady"):
    cfg = make_microduck_tug_env_cfg(style=style)
    scene = Scene(cfg.scene, device="cpu")
    return scene.compile()


def _id(model, obj, name):
    return mujoco.mj_name2id(model, obj, name)


# ── Scene structure ──────────────────────────────────────────────────────────

def test_scene_has_cart_rope_and_hook_sites():
    model = _compiled_tug_model()
    assert model.ntendon == 1
    rope_id = _id(model, mujoco.mjtObj.mjOBJ_TENDON, "tug_rope")
    assert rope_id >= 0
    assert _id(model, mujoco.mjtObj.mjOBJ_SITE, "robot/rope_hook") >= 0
    assert _id(model, mujoco.mjtObj.mjOBJ_SITE, "cart/tug_hook") >= 0
    cart_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, "cart/tug_cart")
    assert cart_body >= 0
    assert model.body_mass[cart_body] == pytest.approx(TUG_CART_MASS)


def test_rope_uses_tug_of_war_cord_model():
    model = _compiled_tug_model()
    rope_id = _id(model, mujoco.mjtObj.mjOBJ_TENDON, "tug_rope")
    assert model.tendon_stiffness[rope_id] == pytest.approx(tug_cfg_mod.ROPE_STIFFNESS)
    # Tension-only dead band: zero-length lower spring, taut length = nominal
    # spawn distance + slack, soft safety catch beyond.
    assert model.tendon_lengthspring[rope_id, 0] == pytest.approx(0.0)
    assert model.tendon_lengthspring[rope_id, 1] == pytest.approx(
        tug_cfg_mod.ROPE_TAUT_LENGTH
    )
    assert model.tendon_limited[rope_id] == 1
    assert model.tendon_range[rope_id, 1] == pytest.approx(
        tug_cfg_mod.ROPE_TAUT_LENGTH + tug_cfg_mod.ROPE_LIMIT_MARGIN
    )
    # The tendon wraps exactly the two hook sites.
    wrap_site_rows = [
        model.wrap_objid[i]
        for i in range(model.nwrap)
        if model.wrap_type[i] == mujoco.mjtWrap.mjWRAP_SITE
    ]
    assert len(wrap_site_rows) == 2


def test_cart_geom_friction_wins_the_floor_pair():
    # MuJoCo combines geom frictions by element-wise max and the floor carries
    # mu=1.0 — the cart geom must take priority=1 for its own low mu to apply.
    model = _compiled_tug_model()
    geom_id = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "cart/tug_cart_geom")
    assert geom_id >= 0
    assert model.geom_friction[geom_id, 0] == pytest.approx(TUG_CART_FRICTION)
    assert model.geom_priority[geom_id] == 1


def test_butt_hook_site_matches_harness_collar_eye():
    model = _compiled_tug_model()
    site_id = _id(model, mujoco.mjtObj.mjOBJ_SITE, "robot/rope_hook")
    assert model.site_pos[site_id] == pytest.approx(TUG_RING_LOCAL)


def test_servo_layout_untouched_by_cart():
    # The walk model's 14-actuator contract must survive the added entities.
    model = _compiled_tug_model()
    assert model.nu == 14
    free_joints = sum(
        1 for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
    )
    assert free_joints == 2  # robot trunk + cart


# ── Rope physics smoke (pure CPU MuJoCo) ─────────────────────────────────────

def _set_free_qpos(model, data, joint_name, pos):
    jid = _id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    adr = model.jnt_qposadr[jid]
    data.qpos[adr : adr + 3] = pos
    data.qpos[adr + 3 : adr + 7] = (1.0, 0.0, 0.0, 0.0)


def test_rope_is_slack_at_nominal_spawn_and_taut_when_stretched():
    model = _compiled_tug_model()
    data = mujoco.MjData(model)
    _set_free_qpos(model, data, "robot/trunk_base_freejoint", (0.0, 0.0, 0.13))
    # Nominal spawn: cart center 0.16 + 0.10 m behind the butt hook.
    _set_free_qpos(model, data, "cart/tug_cart_free", (-0.0686 - 0.26, 0.0, 0.02))
    mujoco.mj_forward(model, data)
    rope_id = _id(model, mujoco.mjtObj.mjOBJ_TENDON, "tug_rope")
    cart_dof = model.jnt_dofadr[
        _id(model, mujoco.mjtObj.mjOBJ_JOINT, "cart/tug_cart_free")
    ]
    length_slack = data.ten_length[rope_id]
    assert length_slack < tug_cfg_mod.ROPE_TAUT_LENGTH
    assert abs(data.qfrc_spring[cart_dof]) < 1e-9
    # Drag the cart 0.3 m further out: the rope stretches past the dead band
    # and pulls the cart toward the duck (+x) at duck-strength scale
    # (200 N/m × stretch ≈ a few N). Tendon spring force lands in qfrc_spring.
    _set_free_qpos(model, data, "cart/tug_cart_free", (-0.0686 - 0.56, 0.0, 0.02))
    mujoco.mj_forward(model, data)
    stretch = data.ten_length[rope_id] - tug_cfg_mod.ROPE_TAUT_LENGTH
    assert stretch > 0.0
    assert data.qfrc_spring[cart_dof] == pytest.approx(
        tug_cfg_mod.ROPE_STIFFNESS * stretch, rel=0.05
    )


# ── Reward / termination wiring ──────────────────────────────────────────────

@pytest.mark.parametrize("style", ["steady", "shuffle"])
def test_penalty_weights_are_negative_task_weights_positive(style):
    cfg = make_microduck_tug_env_cfg(style=style)
    # mjlab-base cost functions return >= 0 → negative weight.
    for name in (
        "foot_slip",
        "action_rate_l2",
        "self_collisions",
        "dof_pos_limits",
        "body_ang_vel",
        "angular_momentum",
    ):
        assert cfg.rewards[name].weight < 0.0, name
    # Task + style terms return >= 0 (progress is potential-based Δ, clamped)
    # → positive weight. No self-negating *_penalty terms exist in this task.
    # "pose" is deliberately REMOVED (do-nothing jackpot, see cfg comment).
    assert "pose" not in cfg.rewards
    for name in (
        "tug_cart_progress",
        "tug_taut_pull_speed",
        "tug_trunk_lean",
        "tug_step_cadence",
        "head_pose_tracking",
    ):
        assert cfg.rewards[name].weight > 0.0, name


def test_shared_terms_identical_between_styles():
    steady = make_microduck_tug_env_cfg(style="steady")
    shuffle = make_microduck_tug_env_cfg(style="shuffle")
    # Everything except the four style knobs must match for a fair A/B.
    shared = (
        "tug_cart_progress",
        "head_pose_tracking",
        "foot_slip",
        "self_collisions",
        "dof_pos_limits",
        "body_ang_vel",
        "angular_momentum",
    )
    for name in shared:
        assert steady.rewards[name].weight == shuffle.rewards[name].weight, name
        assert steady.rewards[name].func is shuffle.rewards[name].func, name
    # Style differences: lean target, cadence target, pull-speed weight/cap,
    # action-rate ramp.
    assert steady.rewards["tug_trunk_lean"].params["target_pitch"] == pytest.approx(
        math.radians(-18.0)
    )
    assert shuffle.rewards["tug_trunk_lean"].params["target_pitch"] == 0.0
    assert (
        shuffle.rewards["tug_step_cadence"].params["target_hz"]
        > steady.rewards["tug_step_cadence"].params["target_hz"]
    )
    assert (
        shuffle.rewards["tug_taut_pull_speed"].weight
        > steady.rewards["tug_taut_pull_speed"].weight
    )
    assert (
        shuffle.curriculum["action_rate_weight"].params["weight_stages"][-1]["weight"]
        > steady.curriculum["action_rate_weight"].params["weight_stages"][-1]["weight"]
    )


def test_command_gated_locomotion_terms_removed():
    cfg = make_microduck_tug_env_cfg()
    for name in (
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "upright",
    ):
        assert name not in cfg.rewards, name
    # Anti-slip stays but with no command gate (there is no velocity command).
    assert cfg.rewards["foot_slip"].params["command_threshold"] == 0.0


def test_terminations_cover_fall_nan_and_bounds():
    cfg = make_microduck_tug_env_cfg()
    assert "fell_over" in cfg.terminations  # tilt guard from the base recipe
    assert cfg.terminations["trunk_low"].params["min_height"] == pytest.approx(0.055)
    assert "nan_state" in cfg.terminations
    assert cfg.terminations["nan_state"].params["sensor_names"] == (
        "feet_ground_contact",
    )
    assert "nan_state_cart" in cfg.terminations
    assert cfg.terminations["out_of_bounds"].params["asset_cfg"].name == "robot"
    assert cfg.terminations["cart_out_of_bounds"].params["asset_cfg"].name == "cart"
    assert (
        cfg.terminations["cart_out_of_bounds"].params["max_distance"]
        > cfg.terminations["out_of_bounds"].params["max_distance"]
    )


# ── 61D obs contract ─────────────────────────────────────────────────────────

def test_obs_layout_matches_velocity_term_for_term():
    tug = make_microduck_tug_env_cfg()
    vel = make_microduck_velocity_env_cfg()
    for group in ("actor", "critic"):
        assert list(tug.observations[group].terms) == list(
            vel.observations[group].terms
        ), group


def test_command_slots_kept_alive_with_tiny_ranges():
    cfg = make_microduck_tug_env_cfg()
    twist = cfg.commands["twist"]
    assert twist.ranges.lin_vel_x == (-0.01, 0.01)
    assert twist.rel_turn_in_place_envs == 0.0
    # head/body pose commands keep tiny non-zero ranges (no dead weights).
    for name, dim in (("head_pose", 4), ("body_pose", 6)):
        cmd = cfg.commands[name]
        assert len(cmd.ranges) == dim
        assert all(lo < hi for lo, hi in cmd.ranges)


# ── Toggles ──────────────────────────────────────────────────────────────────

def test_cart_dr_toggles(monkeypatch):
    monkeypatch.setattr(tug_cfg_mod, "ENABLE_CART_MASS_RANDOMIZATION", False)
    monkeypatch.setattr(tug_cfg_mod, "ENABLE_CART_FRICTION_RANDOMIZATION", False)
    monkeypatch.setattr(tug_cfg_mod, "ENABLE_VELOCITY_PUSHES", False)
    cfg = make_microduck_tug_env_cfg()
    assert "randomize_cart_mass" not in cfg.events
    assert "randomize_cart_friction" not in cfg.events
    assert "push_robot" not in cfg.events


def test_gap_dr_toggle_freezes_slack(monkeypatch):
    monkeypatch.setattr(tug_cfg_mod, "ENABLE_CART_GAP_DR", False)
    cfg = make_microduck_tug_env_cfg()
    lo, hi = cfg.events["reset_tug_cart"].params["gap_range"]
    assert lo == hi == tug_cfg_mod.CART_GAP_NOMINAL


def test_cart_dr_events_present_by_default():
    cfg = make_microduck_tug_env_cfg()
    assert cfg.events["randomize_cart_mass"].mode == "startup"
    assert cfg.events["randomize_cart_friction"].mode == "reset"
    # Cart events target the cart entity, never the robot.
    assert cfg.events["randomize_cart_mass"].params["asset_cfg"].name == "cart"
    assert cfg.events["randomize_cart_friction"].params["asset_cfg"].name == "cart"
    # Cart placement must run after the robot root reset (insertion order).
    keys = list(cfg.events)
    assert keys.index("reset_tug_cart") > keys.index("reset_base")


# ── Registration ─────────────────────────────────────────────────────────────

def test_task_registration():
    tasks = list_tasks()
    assert "Mjlab-Microduck-Tug-Steady" in tasks
    assert "Mjlab-Microduck-Tug-Shuffle" in tasks
    assert MicroduckTugSteadyRlCfg.experiment_name == "tug_steady"
    assert MicroduckTugShuffleRlCfg.experiment_name == "tug_shuffle"
    # Leaning pull is not mirror-symmetric.
    assert MicroduckTugSteadyRlCfg.algorithm.symmetry_cfg is None
    assert MicroduckTugShuffleRlCfg.algorithm.symmetry_cfg is None


def test_play_cfg_builds():
    cfg = make_microduck_tug_env_cfg(style="shuffle", play=True)
    assert cfg.scene.entities["cart"] is not None
    assert cfg.scene.spec_fn is not None
