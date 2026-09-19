"""Cfg-invariant, scene-structure and termination-rule tests for the 3v3
tug-chain task.

Scene/cfg tests are CPU-only compile checks. The termination/progress tests
build one small (2-env) CPU env in a module-scoped fixture and probe the
managers directly — no training loop.
"""

import math
from pathlib import Path

import mujoco
import pytest
import torch
from mjlab.scene import Scene
from mjlab.tasks.registry import list_tasks

from mjlab_microduck.robot.microduck_constants import (
    TUG_CHEST_LOCAL,
    TUG_RING_LOCAL,
)
from mjlab_microduck.tasks import microduck_tug_chain3_env_cfg as chain3_mod
from mjlab_microduck.tasks.microduck_tug_chain3_env_cfg import (
    MicroduckTugChain3ShuffleRlCfg,
    MicroduckTugChain3SteadyRlCfg,
    make_microduck_tug_chain3_env_cfg,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)


def _compiled_chain3_model(style: str = "steady"):
    cfg = make_microduck_tug_chain3_env_cfg(style=style)
    scene = Scene(cfg.scene, device="cpu")
    return scene.compile()


def _id(model, obj, name):
    return mujoco.mj_name2id(model, obj, name)


_ALL_DUCKS = ("robot",) + tuple(chain3_mod._FROZEN_DUCKS)

# ── Scene structure ──────────────────────────────────────────────────────────

def test_scene_has_six_ducks_and_five_cords():
    model = _compiled_chain3_model()
    assert model.ntendon == 5
    for duck in _ALL_DUCKS:
        assert _id(model, mujoco.mjtObj.mjOBJ_BODY, f"{duck}/trunk_base") >= 0, duck
        # Every duck carries both collar rings (butt + chest tow eyes).
        assert _id(model, mujoco.mjtObj.mjOBJ_SITE, f"{duck}/rope_hook") >= 0, duck
        assert (
            _id(model, mujoco.mjtObj.mjOBJ_SITE, f"{duck}/rope_hook_chest") >= 0
        ), duck
    for name, _, _, _ in chain3_mod._CORDS:
        assert _id(model, mujoco.mjtObj.mjOBJ_TENDON, name) >= 0, name


def test_servo_layout_scaled_not_altered():
    # Six ducks × 14 servos; the per-duck 14-actuator contract must survive.
    model = _compiled_chain3_model()
    assert model.nu == 6 * 14
    free_joints = sum(
        1 for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
    )
    assert free_joints == 6


def test_cords_use_match_cord_model():
    model = _compiled_chain3_model()
    for name, site_a, site_b, taut in chain3_mod._CORDS:
        cid = _id(model, mujoco.mjtObj.mjOBJ_TENDON, name)
        assert model.tendon_stiffness[cid] == pytest.approx(chain3_mod.ROPE_STIFFNESS)
        assert model.tendon_damping[cid] == pytest.approx(chain3_mod.ROPE_DAMPING)
        assert model.tendon_lengthspring[cid, 0] == pytest.approx(0.0)
        assert model.tendon_lengthspring[cid, 1] == pytest.approx(taut)
        assert model.tendon_limited[cid] == 1
        assert model.tendon_range[cid, 1] == pytest.approx(
            taut + chain3_mod.ROPE_LIMIT_MARGIN
        )
        # The cord wraps exactly the two named sites.
        adr, num = model.tendon_adr[cid], model.tendon_num[cid]
        wraps = [
            i
            for i in range(adr, adr + num)
            if model.wrap_type[i] == mujoco.mjtWrap.mjWRAP_SITE
        ]
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, model.wrap_objid[i]) for i in wraps]
        assert names == [site_a, site_b], name
    # Center cord is longer than teammate cords (0.30 m gap vs 0.24 m spacing).
    assert chain3_mod.CENTER_TAUT_LENGTH > chain3_mod.TEAMMATE_TAUT_LENGTH


def test_hook_sites_match_harness_collar_eyes():
    model = _compiled_chain3_model()
    for duck in _ALL_DUCKS:
        butt = _id(model, mujoco.mjtObj.mjOBJ_SITE, f"{duck}/rope_hook")
        chest = _id(model, mujoco.mjtObj.mjOBJ_SITE, f"{duck}/rope_hook_chest")
        assert model.site_pos[butt] == pytest.approx(TUG_RING_LOCAL)
        assert model.site_pos[chest] == pytest.approx(TUG_CHEST_LOCAL)


# ── Chain physics smoke (pure CPU MuJoCo) ────────────────────────────────────

def _set_free_qpos(model, data, joint_name, pos, quat=(1.0, 0.0, 0.0, 0.0)):
    jid = _id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    adr = model.jnt_qposadr[jid]
    data.qpos[adr : adr + 3] = pos
    data.qpos[adr + 3 : adr + 7] = quat


def test_cords_preloaded_at_spawn_layout():
    """Lay the 6 ducks out at the nominal spawn spacing (yaw=0, pull axis +x):
    every cord must sit 5–10 mm past its taut length and pull its pair
    together at ~1–2 N."""
    model = _compiled_chain3_model()
    data = mujoco.MjData(model)
    trunk_z = 0.125
    tm = sum(chain3_mod.TEAMMATE_GAP_RANGE) / 2.0
    cg = sum(chain3_mod.CENTER_GAP_RANGE) / 2.0
    # Learner (red middle) at origin facing +x; reds face +x, blues -x.
    yaw_180 = (0.0, 0.0, 0.0, 1.0)
    layout = {
        "robot": (0.0, (1.0, 0.0, 0.0, 0.0)),
        "red_outer": (tm, (1.0, 0.0, 0.0, 0.0)),
        "red_inner": (-tm, (1.0, 0.0, 0.0, 0.0)),
        "blue_inner": (-tm - cg, yaw_180),
        "blue_mid": (-tm - cg - tm, yaw_180),
        "blue_outer": (-tm - cg - 2 * tm, yaw_180),
    }
    for duck, (x, quat) in layout.items():
        _set_free_qpos(
            model, data, f"{duck}/trunk_base_freejoint", (x, 0.0, trunk_z), quat
        )
    mujoco.mj_forward(model, data)
    for name, _, _, taut in chain3_mod._CORDS:
        cid = _id(model, mujoco.mjtObj.mjOBJ_TENDON, name)
        stretch = data.ten_length[cid] - taut
        assert 0.004 < stretch < 0.012, (name, stretch)


# ── Frozen driver wiring ─────────────────────────────────────────────────────

@pytest.mark.parametrize("style", ["steady", "shuffle"])
def test_five_frozen_drivers_registered(style):
    cfg = make_microduck_tug_chain3_env_cfg(style=style)
    # Learner's own term untouched; total action dim stays 14 (frozen terms 0D).
    assert "joint_pos" in cfg.actions
    for name in chain3_mod._FROZEN_DUCKS:
        term = cfg.actions[f"frozen_{name}"]
        assert term.entity_name == name
        assert Path(term.policy_path).exists()
    # Team policy assignment: red frozen ducks run steady_v7, blue shuffle_v7.
    for name in ("red_inner", "red_outer"):
        assert cfg.actions[f"frozen_{name}"].policy_path.endswith(
            "tugchain_steady_v7.onnx"
        )
    for name in ("blue_inner", "blue_mid", "blue_outer"):
        assert cfg.actions[f"frozen_{name}"].policy_path.endswith(
            "tugchain_shuffle_v7.onnx"
        )


# ── Reward / termination wiring ──────────────────────────────────────────────

@pytest.mark.parametrize("style", ["steady", "shuffle"])
def test_penalty_weights_are_negative_task_weights_positive(style):
    cfg = make_microduck_tug_chain3_env_cfg(style=style)
    for name in (
        "foot_slip",
        "action_rate_l2",
        "self_collisions",
        "dof_pos_limits",
        "body_ang_vel",
        "angular_momentum",
    ):
        assert cfg.rewards[name].weight < 0.0, name
    assert "pose" not in cfg.rewards
    for name in (
        "chain3_progress",
        "tug_taut_alive",
        "tug_chain3_trunk_lean",
        "tug_chain3_step_cadence",
        "head_pose_tracking",
    ):
        assert cfg.rewards[name].weight > 0.0, name


def test_shared_terms_identical_between_styles():
    steady = make_microduck_tug_chain3_env_cfg(style="steady")
    shuffle = make_microduck_tug_chain3_env_cfg(style="shuffle")
    shared = (
        "chain3_progress",
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
    assert steady.rewards["tug_chain3_trunk_lean"].params[
        "target_pitch"
    ] == pytest.approx(math.radians(-18.0))
    assert shuffle.rewards["tug_chain3_trunk_lean"].params["target_pitch"] == 0.0
    assert (
        shuffle.rewards["tug_chain3_step_cadence"].params["target_hz"]
        > steady.rewards["tug_chain3_step_cadence"].params["target_hz"]
    )
    assert (
        shuffle.curriculum["action_rate_weight"].params["weight_stages"][-1]["weight"]
        > steady.curriculum["action_rate_weight"].params["weight_stages"][-1]["weight"]
    )
    # The frozen policy mix is style-independent.
    for name in chain3_mod._FROZEN_DUCKS:
        assert (
            steady.actions[f"frozen_{name}"].policy_path
            == shuffle.actions[f"frozen_{name}"].policy_path
        )


def test_termination_rules_learner_falls_only():
    cfg = make_microduck_tug_chain3_env_cfg()
    # Learner: tilt, low trunk, hard 35° overlean gate, NaN, bounds.
    assert "fell_over" in cfg.terminations  # base recipe, 70° on robot
    assert cfg.terminations["trunk_low"].params["min_height"] == pytest.approx(0.055)
    assert cfg.terminations["overlean"].params["limit_angle"] == pytest.approx(
        math.radians(35.0)
    )
    assert "nan_state" in cfg.terminations
    assert cfg.terminations["out_of_bounds"].params["asset_cfg"].name == "robot"
    assert "out_of_terrain_bounds" not in cfg.terminations
    # NaN covers the whole field — one termination per frozen duck.
    for name in chain3_mod._FROZEN_DUCKS:
        term = cfg.terminations[f"nan_state_{name}"]
        assert term.params["asset_cfg"].name == name
    # NO fall/bounds terminations on frozen ducks: fallen ducks are dead
    # weight on the rope (match rules), not episode enders.
    for tname, term in cfg.terminations.items():
        asset = term.params.get("asset_cfg")
        if asset is None or tname.startswith("nan_state"):
            continue
        assert asset.name == "robot", tname


# ── 61D obs contract ─────────────────────────────────────────────────────────

def test_obs_layout_matches_velocity_term_for_term():
    chain3 = make_microduck_tug_chain3_env_cfg()
    vel = make_microduck_velocity_env_cfg()
    for group in ("actor", "critic"):
        assert list(chain3.observations[group].terms) == list(
            vel.observations[group].terms
        ), group


# ── Events ───────────────────────────────────────────────────────────────────

def test_team_reset_event_after_reset_base():
    cfg = make_microduck_tug_chain3_env_cfg()
    keys = list(cfg.events)
    assert keys.index("reset_tug_chain3_team") > keys.index("reset_base")
    params = cfg.events["reset_tug_chain3_team"].params
    for gap, nominal in (
        (params["teammate_gap_range"], chain3_mod.TEAMMATE_TRUNK_GAP),
        (params["center_gap_range"], chain3_mod.CENTER_TRUNK_GAP),
    ):
        for g in gap:
            assert g > nominal  # pre-tensioned past the cord's taut length


def test_gap_dr_toggle_freezes_spawn(monkeypatch):
    monkeypatch.setattr(chain3_mod, "ENABLE_GAP_DR", False)
    cfg = make_microduck_tug_chain3_env_cfg()
    params = cfg.events["reset_tug_chain3_team"].params
    assert params["teammate_gap_range"][0] == params["teammate_gap_range"][1]
    assert params["center_gap_range"][0] == params["center_gap_range"][1]
    monkeypatch.setattr(chain3_mod, "ENABLE_VELOCITY_PUSHES", False)
    cfg = make_microduck_tug_chain3_env_cfg()
    assert "push_robot" not in cfg.events


# ── Registration ─────────────────────────────────────────────────────────────

def test_task_registration():
    tasks = list_tasks()
    assert "Mjlab-Microduck-TugChain3-Steady" in tasks
    assert "Mjlab-Microduck-TugChain3-Shuffle" in tasks
    assert MicroduckTugChain3SteadyRlCfg.experiment_name == "tugchain3_steady"
    assert MicroduckTugChain3ShuffleRlCfg.experiment_name == "tugchain3_shuffle"
    assert MicroduckTugChain3SteadyRlCfg.algorithm.symmetry_cfg is None
    assert MicroduckTugChain3ShuffleRlCfg.algorithm.symmetry_cfg is None


def test_play_cfg_builds():
    cfg = make_microduck_tug_chain3_env_cfg(style="shuffle", play=True)
    assert cfg.scene.entities["blue_outer"] is not None
    assert cfg.scene.spec_fn is not None


# ── Live-env behavior (one small CPU env for the whole module) ───────────────

@pytest.fixture(scope="module")
def chain3_env():
    from mjlab.envs import ManagerBasedRlEnv

    cfg = make_microduck_tug_chain3_env_cfg(style="steady")
    cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg, device="cpu")
    yield env
    env.close()


def _flip_onto_side(env, duck: str, env_ids=(0, 1)):
    """Rotate a duck's trunk 90° about x (lying on its side), in place."""
    e = env.scene[duck]
    ids = torch.tensor(list(env_ids), device=env.device)
    pose = e.data.root_link_pose_w[ids].clone()
    s = math.sqrt(0.5)
    pose[:, 3:] = torch.tensor([s, s, 0.0, 0.0], device=env.device)
    e.write_root_link_pose_to_sim(pose, ids)
    e.write_root_link_velocity_to_sim(torch.zeros(len(ids), 6, device=env.device), ids)
    env.sim.forward()


def test_frozen_duck_fall_does_not_terminate(chain3_env):
    env = chain3_env
    env.reset()
    _flip_onto_side(env, "blue_mid")  # a fallen frozen duck = dead weight
    terminated = env.termination_manager.compute()
    assert not terminated.any()
    # Sanity: the same pose on the LEARNER must terminate (checked next test).


def test_learner_fall_terminates(chain3_env):
    env = chain3_env
    env.reset()
    _flip_onto_side(env, "robot")
    terminated = env.termination_manager.compute()
    assert terminated.all()


def test_frozen_drivers_run_every_step(chain3_env):
    env = chain3_env
    env.reset()
    terms = {
        name: env.action_manager.get_term(f"frozen_{name}")
        for name in chain3_mod._FROZEN_DUCKS
    }
    before = {n: t._last_action.clone() for n, t in terms.items()}
    env.step(torch.zeros(env.num_envs, 14, device=env.device))
    for n, t in terms.items():
        assert t._last_action.shape == (env.num_envs, 14)
        # A live frozen policy reacts to a fresh state: buffer must change.
        assert not torch.allclose(before[n], t._last_action), n


def test_frozen_last_action_cleared_on_reset(chain3_env):
    env = chain3_env
    env.reset()
    for _ in range(3):
        env.step(torch.zeros(env.num_envs, 14, device=env.device))
    env.reset()
    for name in chain3_mod._FROZEN_DUCKS:
        term = env.action_manager.get_term(f"frozen_{name}")
        assert term._last_action.abs().max() == 0.0, name


def test_midpoint_progress_sign(chain3_env):
    """Dragging the chain midpoint toward the red (learner) side pays;
    letting it slide toward blue charges."""
    from mjlab_microduck.tasks import mdp as microduck_mdp

    env = chain3_env
    env.reset()
    for _ in range(2):
        env.step(torch.zeros(env.num_envs, 14, device=env.device))
    u = env._tug_pull_dir_w  # frozen at reset by reset_tug_chain3_team
    assert u.abs().sum(dim=1).min() > 0.9  # unit pull axis per env
    b0 = env.scene["blue_inner"]

    def _shift(delta_m: float) -> torch.Tensor:
        pose = b0.data.root_link_pose_w.clone()
        pose[:, 0] += u[:, 0] * delta_m
        pose[:, 1] += u[:, 1] * delta_m
        b0.write_root_link_pose_to_sim(pose, slice(None))
        env.sim.forward()
        return microduck_mdp.tug_chain3_progress(env)

    # episode_length_buf > 1 after the two steps, so no fresh re-anchor.
    gained = _shift(+0.05)
    assert (gained > 0).all()
    lost = _shift(-0.05)
    assert (lost < 0).all()
