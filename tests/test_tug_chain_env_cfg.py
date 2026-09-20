"""Cfg-invariant and scene-structure tests for the 1v1 tug-chain task.

CPU-only: compiles the composed scene spec (two harness ducks + cross-entity
rope tendon), validates the frozen-opponent torch rebuild against onnxruntime,
and checks the reward/termination wiring of both style recipes — no env build.
"""

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import Scene
from mjlab.tasks.registry import list_tasks

from mjlab_microduck.robot.microduck_constants import TUG_RING_LOCAL
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import microduck_tug_chain_env_cfg as chain_cfg_mod
from mjlab_microduck.tasks.microduck_tug_chain_env_cfg import (
    MicroduckTugChainShuffleRlCfg,
    MicroduckTugChainSteadyRlCfg,
    make_microduck_tug_chain_env_cfg,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)


def _compiled_chain_model(style: str = "steady"):
    cfg = make_microduck_tug_chain_env_cfg(style=style)
    scene = Scene(cfg.scene, device="cpu")
    return scene.compile()


def _id(model, obj, name):
    return mujoco.mj_name2id(model, obj, name)


# ── Scene structure ──────────────────────────────────────────────────────────

def test_scene_has_two_ducks_and_one_rope():
    model = _compiled_chain_model()
    assert model.ntendon == 1
    assert _id(model, mujoco.mjtObj.mjOBJ_TENDON, "tug_rope") >= 0
    assert _id(model, mujoco.mjtObj.mjOBJ_SITE, "robot/rope_hook") >= 0
    assert _id(model, mujoco.mjtObj.mjOBJ_SITE, "opponent/rope_hook") >= 0
    # Both ducks are full walk models with harness collars.
    assert _id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/trunk_base") >= 0
    assert _id(model, mujoco.mjtObj.mjOBJ_BODY, "opponent/trunk_base") >= 0


def test_servo_layout_doubled_not_altered():
    # Two ducks × 14 servos; the per-duck 14-actuator contract must survive.
    model = _compiled_chain_model()
    assert model.nu == 28
    free_joints = sum(
        1 for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
    )
    assert free_joints == 2  # robot trunk + opponent trunk


def test_rope_uses_match_cord_model():
    model = _compiled_chain_model()
    rope_id = _id(model, mujoco.mjtObj.mjOBJ_TENDON, "tug_rope")
    assert model.tendon_stiffness[rope_id] == pytest.approx(chain_cfg_mod.ROPE_STIFFNESS)
    assert model.tendon_damping[rope_id] == pytest.approx(chain_cfg_mod.ROPE_DAMPING)
    # Tension-only dead band: zero-length lower spring, taut = nominal + slack.
    assert model.tendon_lengthspring[rope_id, 0] == pytest.approx(0.0)
    assert model.tendon_lengthspring[rope_id, 1] == pytest.approx(
        chain_cfg_mod.ROPE_TAUT_LENGTH
    )
    assert model.tendon_limited[rope_id] == 1
    assert model.tendon_range[rope_id, 1] == pytest.approx(
        chain_cfg_mod.ROPE_TAUT_LENGTH + chain_cfg_mod.ROPE_LIMIT_MARGIN
    )
    wrap_site_rows = [
        model.wrap_objid[i]
        for i in range(model.nwrap)
        if model.wrap_type[i] == mujoco.mjtWrap.mjWRAP_SITE
    ]
    assert len(wrap_site_rows) == 2


def test_hook_sites_match_harness_collar_eye():
    model = _compiled_chain_model()
    for prefix in ("robot", "opponent"):
        site_id = _id(model, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}/rope_hook")
        assert model.site_pos[site_id] == pytest.approx(TUG_RING_LOCAL)


# ── Rope physics smoke (pure CPU MuJoCo) ─────────────────────────────────────

def _set_free_qpos(model, data, joint_name, pos, quat=(1.0, 0.0, 0.0, 0.0)):
    jid = _id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    adr = model.jnt_qposadr[jid]
    data.qpos[adr : adr + 3] = pos
    data.qpos[adr + 3 : adr + 7] = quat


def test_rope_preloaded_at_spawn_gap():
    model = _compiled_chain_model()
    data = mujoco.MjData(model)
    trunk_z = 0.125
    gap = sum(chain_cfg_mod.OPPONENT_TRUNK_GAP_RANGE) / 2.0
    # Learner at origin facing +x; opponent behind, facing -x (yaw 180°).
    _set_free_qpos(model, data, "robot/trunk_base_freejoint", (0.0, 0.0, trunk_z))
    _set_free_qpos(
        model, data, "opponent/trunk_base_freejoint", (-gap, 0.0, trunk_z),
        quat=(0.0, 0.0, 0.0, 1.0),
    )
    mujoco.mj_forward(model, data)
    rope_id = _id(model, mujoco.mjtObj.mjOBJ_TENDON, "tug_rope")
    robot_dof = model.jnt_dofadr[
        _id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot/trunk_base_freejoint")
    ]
    opp_dof = model.jnt_dofadr[
        _id(model, mujoco.mjtObj.mjOBJ_JOINT, "opponent/trunk_base_freejoint")
    ]
    # Spawn pre-tension: ring distance sits 5–10 mm past the taut length and
    # the rope pulls both ducks toward each other at ~1–2 N.
    stretch = data.ten_length[rope_id] - chain_cfg_mod.ROPE_TAUT_LENGTH
    assert 0.004 < stretch < 0.012
    assert data.qfrc_spring[robot_dof] == pytest.approx(
        -chain_cfg_mod.ROPE_STIFFNESS * stretch, rel=0.05
    )
    assert data.qfrc_spring[opp_dof] == pytest.approx(
        chain_cfg_mod.ROPE_STIFFNESS * stretch, rel=0.05
    )
    # Slacked ducks (closer than nominal): no force.
    _set_free_qpos(
        model, data, "opponent/trunk_base_freejoint", (-0.20, 0.0, trunk_z),
        quat=(0.0, 0.0, 0.0, 1.0),
    )
    mujoco.mj_forward(model, data)
    assert data.ten_length[rope_id] < chain_cfg_mod.ROPE_TAUT_LENGTH
    assert abs(data.qfrc_spring[opp_dof]) < 1e-9


# ── Frozen opponent policy: torch rebuild vs onnxruntime ─────────────────────

@pytest.mark.parametrize("style", ["steady", "shuffle"])
def test_frozen_policy_torch_matches_onnxruntime(style):
    onnxruntime = pytest.importorskip("onnxruntime")
    path = chain_cfg_mod.OPPONENT_POLICY_BY_STYLE[style]
    assert Path(path).exists(), path
    policy = microduck_mdp.FrozenTugPolicy(path, device="cpu")
    assert policy.obs_dim == 61
    assert policy.action_dim == 14

    sess = onnxruntime.InferenceSession(str(path))
    rng = np.random.default_rng(0)
    # Plausible obs scale plus wide outliers: parity must hold off-distribution.
    obs = np.concatenate(
        [rng.normal(size=(256, 61)).astype(np.float32),
         (5.0 * rng.normal(size=(16, 61))).astype(np.float32)]
    )
    # The exported graph has a fixed batch dim of 1 — run row by row.
    ref = np.concatenate(
        [sess.run(["actions"], {"obs": obs[i : i + 1]})[0] for i in range(len(obs))]
    )
    with torch.no_grad():
        got = policy(torch.from_numpy(obs)).numpy()
    assert np.abs(got - ref).max() < 1e-4


# ── Reward / termination wiring ──────────────────────────────────────────────

@pytest.mark.parametrize("style", ["steady", "shuffle"])
def test_penalty_weights_are_negative_task_weights_positive(style):
    cfg = make_microduck_tug_chain_env_cfg(style=style)
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
    assert "pose" not in cfg.rewards
    for name in (
        "chain_progress",
        "tug_taut_alive",
        "tug_chain_trunk_lean",
        "tug_chain_step_cadence",
        "head_pose_tracking",
    ):
        assert cfg.rewards[name].weight > 0.0, name


def test_shared_terms_identical_between_styles():
    steady = make_microduck_tug_chain_env_cfg(style="steady")
    shuffle = make_microduck_tug_chain_env_cfg(style="shuffle")
    # Everything except the style knobs must match for a fair A/B.
    shared = (
        "chain_progress",
        "tug_taut_alive",
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
    # Style differences: lean target, cadence target, action-rate ramp.
    assert steady.rewards["tug_chain_trunk_lean"].params[
        "target_pitch"
    ] == pytest.approx(math.radians(15.0))  # v11: 前倾（背绳负重犬式）
    assert shuffle.rewards["tug_chain_trunk_lean"].params["target_pitch"] == 0.0
    assert (
        shuffle.rewards["tug_chain_step_cadence"].params["target_hz"]
        > steady.rewards["tug_chain_step_cadence"].params["target_hz"]
    )
    assert (
        shuffle.curriculum["action_rate_weight"].params["weight_stages"][-1]["weight"]
        > steady.curriculum["action_rate_weight"].params["weight_stages"][-1]["weight"]
    )


def test_command_gated_locomotion_terms_removed():
    cfg = make_microduck_tug_chain_env_cfg()
    for name in (
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "upright",
    ):
        assert name not in cfg.rewards, name
    assert cfg.rewards["foot_slip"].params["command_threshold"] == 0.0


def test_terminations_cover_falls_nan_overlean_and_bounds():
    cfg = make_microduck_tug_chain_env_cfg()
    # Learner: tilt, low trunk, hard 35° overlean gate, NaN.
    assert "fell_over" in cfg.terminations
    assert cfg.terminations["trunk_low"].params["min_height"] == pytest.approx(0.055)
    assert "overlean" in cfg.terminations
    assert cfg.terminations["overlean"].params["limit_angle"] == pytest.approx(
        math.radians(35.0)
    )
    assert cfg.terminations["overlean"].params.get(
        "asset_cfg", SceneEntityCfg("robot")
    ).name == "robot"
    assert "nan_state" in cfg.terminations
    # Opponent: fall (tilt OR low trunk) ends the round; NaN guarded.
    assert cfg.terminations["opponent_fell_over"].params[
        "asset_cfg"
    ].name == "opponent"
    assert cfg.terminations["opponent_fell_over"].params[
        "limit_angle"
    ] == pytest.approx(math.radians(70.0))
    assert cfg.terminations["opponent_trunk_low"].params[
        "asset_cfg"
    ].name == "opponent"
    assert cfg.terminations["nan_state_opponent"].params[
        "asset_cfg"
    ].name == "opponent"
    # Bounds on both ducks; base out_of_terrain_bounds replaced.
    assert "out_of_terrain_bounds" not in cfg.terminations
    assert cfg.terminations["out_of_bounds"].params["asset_cfg"].name == "robot"
    assert (
        cfg.terminations["opponent_out_of_bounds"].params["asset_cfg"].name
        == "opponent"
    )


# ── Opponent driver wiring ───────────────────────────────────────────────────

@pytest.mark.parametrize("style", ["steady", "shuffle"])
def test_opponent_action_term_is_zero_dim_and_cross_adversary(style):
    cfg = make_microduck_tug_chain_env_cfg(style=style)
    opp = cfg.actions["opponent_policy"]
    assert opp.entity_name == "opponent"
    assert Path(opp.policy_path).exists()
    # Cross-adversary: the steady learner fights the frozen SHUFFLE policy.
    expected = "tug_shuffle_v6.onnx" if style == "steady" else "tug_steady_v6.onnx"
    assert opp.policy_path.endswith(expected)
    # The learner's own joint action term is untouched.
    assert "joint_pos" in cfg.actions


# ── 61D obs contract ─────────────────────────────────────────────────────────

def test_obs_layout_matches_velocity_term_for_term():
    chain = make_microduck_tug_chain_env_cfg()
    vel = make_microduck_velocity_env_cfg()
    for group in ("actor", "critic"):
        assert list(chain.observations[group].terms) == list(
            vel.observations[group].terms
        ), group


def test_command_slots_kept_alive_with_tiny_ranges():
    cfg = make_microduck_tug_chain_env_cfg()
    twist = cfg.commands["twist"]
    assert twist.ranges.lin_vel_x == (-0.01, 0.01)
    assert twist.rel_turn_in_place_envs == 0.0
    for name, dim in (("head_pose", 4), ("body_pose", 6)):
        cmd = cfg.commands[name]
        assert len(cmd.ranges) == dim
        assert all(lo < hi for lo, hi in cmd.ranges)


# ── Events ───────────────────────────────────────────────────────────────────

def test_opponent_reset_event_after_reset_base_and_targets_opponent():
    cfg = make_microduck_tug_chain_env_cfg()
    keys = list(cfg.events)
    assert keys.index("reset_tug_chain_opponent") > keys.index("reset_base")
    gap = cfg.events["reset_tug_chain_opponent"].params["trunk_gap_range"]
    # Spawn ring distance = trunk gap - 2|ring_x| sits past the taut length.
    for g in gap:
        ring_gap = g - 2.0 * abs(TUG_RING_LOCAL[0])
        assert ring_gap > chain_cfg_mod.ROPE_TAUT_LENGTH
        assert ring_gap < chain_cfg_mod.ROPE_TAUT_LENGTH + 0.015


def test_gap_dr_toggle_freezes_spawn(monkeypatch):
    monkeypatch.setattr(chain_cfg_mod, "ENABLE_OPPONENT_GAP_DR", False)
    cfg = make_microduck_tug_chain_env_cfg()
    lo, hi = cfg.events["reset_tug_chain_opponent"].params["trunk_gap_range"]
    assert lo == hi
    monkeypatch.setattr(chain_cfg_mod, "ENABLE_VELOCITY_PUSHES", False)
    cfg = make_microduck_tug_chain_env_cfg()
    assert "push_robot" not in cfg.events


# ── Registration ─────────────────────────────────────────────────────────────

def test_task_registration():
    tasks = list_tasks()
    assert "Mjlab-Microduck-TugChain-Steady" in tasks
    assert "Mjlab-Microduck-TugChain-Shuffle" in tasks
    assert MicroduckTugChainSteadyRlCfg.experiment_name == "tugchain_steady"
    assert MicroduckTugChainShuffleRlCfg.experiment_name == "tugchain_shuffle"
    # Leaning pull is not mirror-symmetric.
    assert MicroduckTugChainSteadyRlCfg.algorithm.symmetry_cfg is None
    assert MicroduckTugChainShuffleRlCfg.algorithm.symmetry_cfg is None


def test_play_cfg_builds():
    cfg = make_microduck_tug_chain_env_cfg(style="shuffle", play=True)
    assert cfg.scene.entities["opponent"] is not None
    assert cfg.scene.spec_fn is not None
