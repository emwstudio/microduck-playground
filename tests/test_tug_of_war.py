"""Structural tests for the NvN tug-of-war demo scene (scripts/tug_of_war.py)."""

import mujoco
import numpy as np
import pytest

from mjlab_microduck.robot.tug_of_war import (
    DEFAULT_POSE,
    JOINT_NAMES,
    MIN_FALLEN,
    WIN_X,
    build_tug_spec,
    check_winner,
    compute_obs,
    duck_spawns,
    find_duck_rigs,
    resolve_rope_visuals,
    team_prefixes,
)


@pytest.fixture(scope="module")
def tug_model():
    spec = build_tug_spec(n_per_team=5)
    return spec.compile()


def test_spec_has_ten_robots_140_actuators_18_cords(tug_model) -> None:
    model = tug_model
    assert model.nu == 10 * 14
    assert model.ntendon == 9 * 2
    free_joints = sum(1 for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE)
    assert free_joints == 10


def test_all_rope_sites_resolve(tug_model) -> None:
    red, blue = team_prefixes(5)
    for prefix in red + blue:
        for side in ("left", "right", "chest"):
            assert mujoco.mj_name2id(
                tug_model, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}rope_{side}") >= 0


def test_hemp_visuals_span_the_chain(tug_model) -> None:
    # 9 strands x 4 sag sub-segments of mocap; physics tendons hidden.
    assert tug_model.nmocap == 9 * 4
    assert tug_model.ntendon == 18
    visuals = resolve_rope_visuals(tug_model, 5)
    assert len(visuals) == 36
    assert all(v.body_id >= 0 and v.geom_id >= 0 for v in visuals)
    assert all(v.trunk_a_id >= 0 and v.trunk_b_id >= 0 for v in visuals)


def test_every_trunk_wears_a_wrap(tug_model) -> None:
    red, blue = team_prefixes(5)
    for prefix in red + blue:
        geom_id = mujoco.mj_name2id(
            tug_model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}tug_wrap")
        assert geom_id >= 0
        assert tug_model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH


def test_team_shell_colors_differ(tug_model) -> None:
    red_id = mujoco.mj_name2id(tug_model, mujoco.mjtObj.mjOBJ_MATERIAL, "left_shell_material")
    blue_id = mujoco.mj_name2id(tug_model, mujoco.mjtObj.mjOBJ_MATERIAL, "b0_left_shell_material")
    assert red_id >= 0 and blue_id >= 0
    assert not np.allclose(tug_model.mat_rgba[red_id], tug_model.mat_rgba[blue_id])


def test_rigs_cover_all_actuators_in_joint_order(tug_model) -> None:
    rigs = find_duck_rigs(tug_model, 5)
    assert len(rigs) == 10
    all_ids = np.concatenate([r.actuator_ids for r in rigs])
    assert sorted(all_ids.tolist()) == list(range(tug_model.nu))
    for rig in rigs:
        for k, jn in enumerate(JOINT_NAMES):
            joint_id = tug_model.actuator_trnid[rig.actuator_ids[k], 0]
            assert mujoco.mj_id2name(tug_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) == f"{rig.prefix}{jn}"


def test_obs_batch_is_61d_and_finite(tug_model) -> None:
    model = tug_model
    data = mujoco.MjData(model)
    rigs = find_duck_rigs(model, 5)
    for rig in rigs:
        data.qpos[rig.joint_qpos_idx] = DEFAULT_POSE
    mujoco.mj_forward(model, data)
    obs = compute_obs(model, data, rigs, np.full(len(rigs), 0.25, dtype=np.float32))
    assert obs.shape == (10, 61)
    assert np.isfinite(obs).all()
    # Freshly reset with DEFAULT_POSE: joint_pos_rel ≈ 0, command slot = pull speed.
    assert np.abs(obs[:, 6:20]).max() < 1e-5
    assert np.allclose(obs[:, 48], 0.25)


def test_check_winner_line_cross() -> None:
    spec = build_tug_spec(n_per_team=1)
    model = spec.compile()
    data = mujoco.MjData(model)
    rigs = find_duck_rigs(model, 1)
    mujoco.mj_forward(model, data)
    assert check_winner(model, data, rigs)[0] is None
    red = next(r for r in rigs if r.team == "red")
    blue = next(r for r in rigs if r.team == "blue")
    # Drag both ducks so the midpoint sits past the red win line.
    data.qpos[red.free_qpos_adr] = -WIN_X - 0.2
    data.qpos[blue.free_qpos_adr] = -WIN_X - 0.1
    mujoco.mj_forward(model, data)
    assert check_winner(model, data, rigs) == ("red", check_winner(model, data, rigs)[1])
    winner, _ = check_winner(model, data, rigs)
    assert winner == "red"


def test_check_winner_fallen_team() -> None:
    spec = build_tug_spec(n_per_team=2)
    model = spec.compile()
    data = mujoco.MjData(model)
    rigs = find_duck_rigs(model, 2)
    mujoco.mj_forward(model, data)
    # Knock every blue duck to the floor.
    for rig in rigs:
        if rig.team == "blue":
            data.qpos[rig.free_qpos_adr + 2] = 0.02
    mujoco.mj_forward(model, data)
    winner, reason = check_winner(model, data, rigs, min_fallen=2)
    assert winner == "red"
    assert "down" in reason


def test_spawns_face_outward_and_straddle_center() -> None:
    spawns = duck_spawns(5)
    red, blue = team_prefixes(5)
    assert all(spawns[p][0] < 0 for p in red)
    assert all(spawns[p][0] > 0 for p in blue)
    assert spawns[red[0]][1] == (0.0, 0.0, 0.0, 1.0)  # red faces -x
    assert spawns[blue[0]][1] == (1.0, 0.0, 0.0, 0.0)  # blue faces +x
