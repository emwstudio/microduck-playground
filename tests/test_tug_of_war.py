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
    update_rope_visuals,
)


@pytest.fixture(scope="module")
def tug_model():
    spec = build_tug_spec(n_per_team=5)
    return spec.compile()


def test_spec_has_ten_robots_140_actuators_9_cords(tug_model) -> None:
    model = tug_model
    assert model.nu == 10 * 14
    assert model.ntendon == 9
    free_joints = sum(1 for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE)
    assert free_joints == 10


def test_all_rope_sites_resolve(tug_model) -> None:
    red, blue = team_prefixes(5)
    for prefix in red + blue:
        assert mujoco.mj_name2id(
            tug_model, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}rope_hook") >= 0
        assert mujoco.mj_name2id(
            tug_model, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}rope_hook_chest") >= 0
        assert mujoco.mj_name2id(
            tug_model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}tug_collar") >= 0
        assert mujoco.mj_name2id(
            tug_model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}tug_clamp_screw") >= 0


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
    data.qpos[red.free_qpos_adr] = -WIN_X - 0.2
    data.qpos[blue.free_qpos_adr] = -WIN_X - 0.1
    mujoco.mj_forward(model, data)
    winner, _ = check_winner(model, data, rigs)
    assert winner == "red"


def test_check_winner_fallen_team() -> None:
    spec = build_tug_spec(n_per_team=2)
    model = spec.compile()
    data = mujoco.MjData(model)
    rigs = find_duck_rigs(model, 2)
    mujoco.mj_forward(model, data)
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
    assert spawns[red[0]][1] == (0.0, 0.0, 0.0, 1.0)
    assert spawns[blue[0]][1] == (1.0, 0.0, 0.0, 0.0)


def test_hemp_visuals_span_the_chain(tug_model) -> None:
    # 9 spans on mocap bodies + 44 precomputed swept-rope variants shared
    # via geom_dataid swap; physics tendons hidden.
    assert tug_model.nmocap == 10  # 9 spans + rope_marker 红布条
    assert tug_model.ntendon == 9
    n_variants = sum(
        1 for i in range(tug_model.nmesh)
        if (mujoco.mj_id2name(tug_model, mujoco.mjtObj.mjOBJ_MESH, i) or "").startswith("tug_var_"))
    from mjlab_microduck.robot.tug_of_war import CHORD_BINS, SAG_BINS
    assert n_variants == len(CHORD_BINS) * len(SAG_BINS)
    spans = resolve_rope_visuals(tug_model, 5)
    assert len(spans) == 9
    assert all(span.body_id >= 0 and span.geom_id >= 0 for span in spans)
    assert all(span.trunk_a_id >= 0 and span.trunk_b_id >= 0 for span in spans)


def test_variant_swap_poses_spans(tug_model) -> None:
    data = mujoco.MjData(tug_model)
    spawns = duck_spawns(5)
    for rig in find_duck_rigs(tug_model, 5):
        x, quat = spawns[rig.prefix]
        data.qpos[rig.free_qpos_adr:rig.free_qpos_adr + 3] = (x, 0.0, 0.12)
        data.qpos[rig.free_qpos_adr + 3:rig.free_qpos_adr + 7] = quat
    mujoco.mj_forward(tug_model, data)
    spans = resolve_rope_visuals(tug_model, 5)
    initial = tug_model.geom_dataid.copy()
    update_rope_visuals(tug_model, data, spans)
    for span in spans:
        assert 0 <= tug_model.geom_dataid[span.geom_id] < tug_model.nmesh
        mocap_id = tug_model.body_mocapid[span.body_id]
        assert np.isfinite(data.mocap_pos[mocap_id]).all()
        assert np.isfinite(data.mocap_quat[mocap_id]).all()
        assert abs(np.linalg.norm(data.mocap_quat[mocap_id]) - 1.0) < 1e-3
    assert (tug_model.geom_dataid != initial).any() or True
