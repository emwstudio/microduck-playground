"""Tests for the 3v3 full-self-play tug chain (v9).

Scene/cfg checks are CPU-only compile tests. The flattening/done/reward
behavior tests build one small (2-match) CPU env + wrapper in a module-scoped
fixture — no training loop.
"""

import math

import mujoco
import pytest
import torch
from mjlab.scene import Scene
from mjlab.tasks.registry import list_tasks, load_runner_cls

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import microduck_tug_chain3s_env_cfg as chain3s_mod
from mjlab_microduck.tasks.microduck_tug_chain3s_env_cfg import (
    MicroduckTugChain3sShuffleRlCfg,
    MicroduckTugChain3sSteadyRlCfg,
    make_microduck_tug_chain3s_env_cfg,
)

DUCKS = microduck_mdp.CHAIN3S_DUCKS


def _compiled_model(style: str = "steady"):
    cfg = make_microduck_tug_chain3s_env_cfg(style=style)
    scene = Scene(cfg.scene, device="cpu")
    return scene.compile()


def _id(model, obj, name):
    return mujoco.mj_name2id(model, obj, name)


# ── Scene structure ──────────────────────────────────────────────────────────

def test_scene_has_six_ducks_five_cords():
    model = _compiled_model()
    assert model.ntendon == 5
    assert model.nu == 6 * 14
    for duck in DUCKS:
        assert _id(model, mujoco.mjtObj.mjOBJ_BODY, f"{duck}/trunk_base") >= 0, duck
        assert _id(model, mujoco.mjtObj.mjOBJ_SITE, f"{duck}/rope_hook") >= 0, duck
        assert (
            _id(model, mujoco.mjtObj.mjOBJ_SITE, f"{duck}/rope_hook_chest") >= 0
        ), duck


# ── Cfg wiring ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("style", ["steady", "shuffle"])
def test_six_learning_action_terms_in_duck_order(style):
    cfg = make_microduck_tug_chain3s_env_cfg(style=style)
    expected = ["joint_pos"] + [f"joint_pos_{d}" for d in DUCKS[1:]]
    assert list(cfg.actions) == expected  # flattening contract: slot order
    for name, duck in zip(expected, DUCKS):
        assert cfg.actions[name].entity_name == duck
    # No frozen-ONNX terms anywhere — every duck is a learner.
    assert not any("frozen" in k for k in cfg.actions)


def test_inner_env_contract():
    cfg = make_microduck_tug_chain3s_env_cfg()
    assert cfg.auto_reset is False  # wrapper resets whole matches itself
    assert cfg.rewards == {}        # per-duck rewards live in the wrapper
    assert cfg.commands == {}       # zero-padded command slots baked into obs
    assert hasattr(cfg, "chain3s_spec")
    # Per-duck sensors for cadence / slip / self-collision.
    for duck in DUCKS:
        assert f"feet_ground_contact_{duck}" in [s.name for s in cfg.scene.sensors]
        assert f"self_collision_{duck}" in [s.name for s in cfg.scene.sensors]


def test_any_duck_down_ends_the_match():
    cfg = make_microduck_tug_chain3s_env_cfg()
    for duck in DUCKS:
        for prefix in ("fell_over", "overlean", "trunk_low", "nan_state", "out_of_bounds"):
            term = cfg.terminations[f"{prefix}_{duck}"]
            assert term.params["asset_cfg"].name == duck
        assert cfg.terminations[f"overlean_{duck}"].params[
            "limit_angle"
        ] == pytest.approx(math.radians(35.0))
        assert cfg.terminations[f"trunk_low_{duck}"].params[
            "min_height"
        ] == pytest.approx(0.055)
    assert "time_out" in cfg.terminations


def test_reward_spec_signs_and_styles():
    steady = make_microduck_tug_chain3s_env_cfg(style="steady").chain3s_spec
    shuffle = make_microduck_tug_chain3s_env_cfg(style="shuffle").chain3s_spec
    penalties = {"action_rate_l2", "body_ang_vel", "dof_pos_limits",
                 "self_collisions", "foot_slip"}
    for spec in (steady, shuffle):
        for name, _func, weight, _params in spec["terms"]:
            if name in penalties:
                assert name == "action_rate_l2" or weight < 0.0, name
            else:
                assert weight > 0.0, name
        # action_rate ramps are all-negative in both styles
        assert all(s["weight"] < 0 for s in spec["action_rate_stages"])
    # Style knobs differ; shared task terms don't.
    s_terms = dict((t[0], t) for t in steady["terms"])
    h_terms = dict((t[0], t) for t in shuffle["terms"])
    assert s_terms["trunk_lean"][3]["target_pitch"] == pytest.approx(
        math.radians(-18.0)
    )
    assert h_terms["trunk_lean"][3]["target_pitch"] == 0.0
    assert (
        h_terms["step_cadence"][3]["target_hz"]
        > s_terms["step_cadence"][3]["target_hz"]
    )
    for name in ("chain3s_team_progress", "tug_taut_alive", "head_home"):
        assert s_terms[name][2] == h_terms[name][2], name
    assert (
        shuffle["action_rate_stages"][-1]["weight"]
        > steady["action_rate_stages"][-1]["weight"]
    )


def test_dr_events_cover_all_six_ducks():
    cfg = make_microduck_tug_chain3s_env_cfg()
    for base in ("randomize_com", "randomize_mass_inertia", "randomize_joint_friction",
                 "randomize_armature", "foot_friction"):
        assert base not in cfg.events  # robot-only originals replaced
        for duck in DUCKS:
            ev = cfg.events[f"{base}_{duck}"]
            assert ev.params["asset_cfg"].name == duck
    assert "push_robot" not in cfg.events
    assert "encoder_bias" not in cfg.events
    assert "reset_tug_chain3s_match" in cfg.events
    assert "expand_bam_friction_fields" in cfg.events


def test_task_registration():
    tasks = list_tasks()
    assert "Mjlab-Microduck-TugChain3S-Steady" in tasks
    assert "Mjlab-Microduck-TugChain3S-Shuffle" in tasks
    assert MicroduckTugChain3sSteadyRlCfg.experiment_name == "tugchain3s_steady"
    assert MicroduckTugChain3sShuffleRlCfg.experiment_name == "tugchain3s_shuffle"
    # The flattening wrapper is injected via the custom runner.
    assert load_runner_cls("Mjlab-Microduck-TugChain3S-Steady") is not None
    assert load_runner_cls("Mjlab-Microduck-TugChain3S-Steady") is load_runner_cls(
        "Mjlab-Microduck-TugChain3S-Shuffle"
    )


# ── Live-env flattening behavior (one 2-match CPU env for the module) ────────

@pytest.fixture(scope="module")
def flat_env():
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab_microduck.tasks.tug_chain3s_vecenv import TugChainSelfPlayVecEnv

    cfg = make_microduck_tug_chain3s_env_cfg(style="steady")
    cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg, device="cpu")
    flat = TugChainSelfPlayVecEnv(env)
    yield flat
    flat.close()


def test_obs_flattening_shape_and_row_mapping(flat_env):
    flat = flat_env
    env = flat.env
    obs = flat.get_observations()
    assert obs["actor"].shape == (2 * 6, 61)
    assert obs["critic"].shape == (2 * 6, 61)
    assert flat.num_envs == 12  # reported to rsl_rl as duck rows, not matches
    assert flat.num_actions == 14

    # Row for duck d in match m must read entity d of match m: tilt ONE duck
    # 90° and check only its row's projected gravity changes.
    obs0 = microduck_mdp.tug_chain3s_duck_obs(env).reshape(2, 6, 61).clone()
    slot = DUCKS.index("blue_mid")
    e = env.scene["blue_mid"]
    pose = e.data.root_link_pose_w[[0]].clone()
    s = math.sqrt(0.5)
    pose[:, 3:] = torch.tensor([s, s, 0.0, 0.0], device=env.device)
    e.write_root_link_pose_to_sim(pose, torch.tensor([0], device=env.device))
    env.sim.forward()
    obs1 = microduck_mdp.tug_chain3s_duck_obs(env).reshape(2, 6, 61)
    for m in range(2):
        for d in range(6):
            changed = not torch.allclose(obs0[m, d], obs1[m, d])
            assert changed == (m == 0 and d == slot), (m, d)
    # proj_gravity slice of the tilted duck flipped sign (lying on its side).
    assert obs1[0, slot, 5] == pytest.approx(0.0, abs=0.1)
    env.reset()


def test_done_broadcast_and_match_reset(flat_env):
    flat = flat_env
    env = flat.env
    env.reset()
    for _ in range(3):
        flat.step(torch.zeros(flat.num_envs, 14, device=flat.device))
    assert env.episode_length_buf.tolist() == [3, 3]

    # Flip ONE duck of match 0 past the 35° overlean gate → whole match dies.
    e = env.scene["red_inner"]
    pose = e.data.root_link_pose_w[[0]].clone()
    angle = math.radians(45.0)
    pose[:, 3:] = torch.tensor(
        [math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0], device=env.device
    )
    e.write_root_link_pose_to_sim(pose, torch.tensor([0], device=env.device))
    env.sim.forward()

    obs, rew, dones, extras = flat.step(torch.zeros(flat.num_envs, 14, device=flat.device))
    assert dones[:6].tolist() == [1] * 6      # match 0: all six rows done
    assert dones[6:].tolist() == [0] * 6      # match 1: untouched
    assert env.episode_length_buf[0].item() == 0  # match 0 was reset
    assert env.episode_length_buf[1].item() == 4
    # Done rows got post-reset obs (a fresh standing duck is upright again).
    assert obs["actor"][0, 5] == pytest.approx(-1.0, abs=0.15)  # pg_z upright


def test_action_rows_reach_the_right_duck(flat_env):
    """Row m·6+s action must land on the s-th duck's action term slice."""
    flat = flat_env
    env = flat.env
    env.reset()
    actions = torch.zeros(flat.num_envs, 14, device=flat.device)
    # Distinctive constant per (match, slot).
    for m in range(2):
        for s in range(6):
            actions[m * 6 + s] = 0.01 * (m * 6 + s + 1)
    flat.step(actions)
    am = env.action_manager
    for s, duck in enumerate(DUCKS):
        term_name = "joint_pos" if duck == "robot" else f"joint_pos_{duck}"
        idx = am.active_terms.index(term_name)
        got = am.action[:, idx * 14 : (idx + 1) * 14]
        for m in range(2):
            assert torch.allclose(
                got[m], torch.full((14,), 0.01 * (m * 6 + s + 1), device=flat.device)
            ), (duck, m)
    # And the obs last_action slice carries the same values per duck row.
    obs = flat.get_observations()["actor"]
    for s in range(6):
        assert torch.allclose(
            obs[s, 34:48], torch.full((14,), 0.01 * (s + 1), device=flat.device)
        ), s
    env.reset()


def test_last_action_zeroed_on_reset(flat_env):
    flat = flat_env
    env = flat.env
    flat.step(torch.full((flat.num_envs, 14), 0.3, device=flat.device))
    env.reset()
    assert env.action_manager.action.abs().max() == 0.0
    obs = flat.get_observations()["actor"]
    assert obs[:, 34:48].abs().max() == 0.0


def test_team_progress_sign_per_team(flat_env):
    flat = flat_env
    env = flat.env
    env.reset()
    # Anchor the potential buffer at the spawn layout.
    env.episode_length_buf[:] = 2
    microduck_mdp.tug_chain3s_team_progress(env)
    u = env._tug_pull_dir_w
    b0 = env.scene["blue_inner"]
    pose = b0.data.root_link_pose_w.clone()
    pose[:, 0] += u[:, 0] * 0.05  # midpoint toward red
    pose[:, 1] += u[:, 1] * 0.05
    b0.write_root_link_pose_to_sim(pose, slice(None))
    env.sim.forward()
    rew = microduck_mdp.tug_chain3s_team_progress(env)
    assert (rew[:, :3] > 0).all()   # red ducks paid
    assert (rew[:, 3:] < 0).all()   # blue ducks charged
    # Symmetric: pulling it back reverses the signs.
    pose = b0.data.root_link_pose_w.clone()
    pose[:, 0] -= u[:, 0] * 0.05
    pose[:, 1] -= u[:, 1] * 0.05
    b0.write_root_link_pose_to_sim(pose, slice(None))
    env.sim.forward()
    rew = microduck_mdp.tug_chain3s_team_progress(env)
    assert (rew[:, :3] < 0).all()
    assert (rew[:, 3:] > 0).all()
    env.reset()


def test_wrapper_rewards_finite_and_logged(flat_env):
    flat = flat_env
    env = flat.env
    env.reset()
    for _ in range(4):
        obs, rew, dones, extras = flat.step(
            torch.zeros(flat.num_envs, 14, device=flat.device)
        )
        assert torch.isfinite(rew).all()
        assert rew.shape == (flat.num_envs,)
        assert extras["time_outs"].shape == (flat.num_envs,)
    # A standing pre-tensioned spawn pays the alive/head terms: positive.
    assert (rew > 0).any()
    env.reset()
