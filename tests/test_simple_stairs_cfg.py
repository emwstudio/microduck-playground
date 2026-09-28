"""Cfg-invariant tests for simple_stairs v12 (open-riser 25 mm design).

The stair/ladder code lives in the desk-climb snapshot tree
(experiments/desk-climb/source/src), not the main src tree.  APPEND the
snapshot dirs to the already-loaded package __path__ lists: snapshot-only
modules (robot/ladder, tasks/microduck_*stair*, tasks/mdp shadowing is NOT
needed at import time) resolve from the snapshot, everything else keeps
resolving from the main tree — no sys.modules swapping, no contamination.
"""

import importlib.util
import math
from pathlib import Path

import pytest
import torch

import mjlab_microduck.robot as _robot
import mjlab_microduck.tasks as _tasks

_SNAPSHOT_SRC = Path(__file__).resolve().parents[1] / "experiments/desk-climb/source/src/mjlab_microduck"
for _pkg, _sub in ((_robot, "robot"), (_tasks, "tasks")):
    _dir = str(_SNAPSHOT_SRC / _sub)
    if _dir not in _pkg.__path__:
        _pkg.__path__.append(_dir)

from mjlab_microduck.robot import ladder as lad  # noqa: E402
from mjlab_microduck.tasks import microduck_simple_stairs_env_cfg as ss  # noqa: E402


def _f64(*xs):
    return tuple(torch.tensor([x], dtype=torch.float64) for x in xs)


def test_levels_table_is_open_riser_compliant():
    g = ss.SIMPLE_STAIRS_GEOMETRY
    assert len(ss.SIMPLE_STAIRS_LEVELS) == 5
    # v12 target: converge on 24-25.5 mm, not 27-30 mm.
    assert ss.SIMPLE_STAIRS_LEVELS[-1]["riser"] == (0.024, 0.0255)
    for level in ss.SIMPLE_STAIRS_LEVELS:
        r_lo, _ = level["riser"]
        _, a_hi = level["angle"]
        run = r_lo / math.tan(math.radians(a_hi))
        # every band satisfies the open gap at its own LOWER riser
        assert run >= g.tread_depth_m + 0.002 - 1e-3, (level, run)


def test_clamp_open_riser_keeps_25mm_and_clamps_angle_to_gap():
    g = ss.SIMPLE_STAIRS_GEOMETRY
    # 25 mm riser must NOT be clamped up to the 29 mm under-tread minimum.
    riser, angle = lad.clamp_riser_angle(g, *_f64(0.025, 19.0), open_riser=True)
    assert float(riser) == pytest.approx(0.025)
    assert float(angle) == pytest.approx(19.0)  # inside the gap bound: untouched
    # run < depth + 2 mm (25 deg -> run 53.6 mm): angle clamped down to the gap bound.
    riser, angle = lad.clamp_riser_angle(g, *_f64(0.025, 25.0), open_riser=True)
    assert float(angle) == pytest.approx(math.degrees(math.atan(0.025 / 0.062)), abs=1e-4)
    # 24 mm at the level-4 band top stays put (the v12 top level draws).
    riser, angle = lad.clamp_riser_angle(g, *_f64(0.024, 20.0), open_riser=True)
    assert float(angle) == pytest.approx(20.0)


def test_clamp_default_path_unchanged_under_tread():
    g = ss.SIMPLE_STAIRS_GEOMETRY  # full-width: same_side_spacing = 1
    # 20 mm is clamped UP to the 29 mm toe-under-tread minimum (v2's discovery).
    riser, angle = lad.clamp_riser_angle(g, *_f64(0.020, 20.0))
    assert float(riser) == pytest.approx(0.029)
    assert float(angle) == pytest.approx(20.0)  # below the ankle-shell bound
    # Alternating ladder geometry: 16 mm stays, steep angle clamped to the
    # ankle-shell bound atan(2*0.016/0.022) = 55.5 deg — legacy behaviour.
    alt = lad.LADDER_GEOMETRY
    riser, angle = lad.clamp_riser_angle(alt, *_f64(0.016, 60.0))
    assert float(riser) == pytest.approx(0.016)
    assert float(angle) == pytest.approx(math.degrees(math.atan(2 * 0.016 / 0.022)), abs=1e-3)
    # And a shallow one passes through untouched.
    riser, angle = lad.clamp_riser_angle(alt, *_f64(0.016, 47.0))
    assert float(angle) == pytest.approx(47.0)


def test_validate_tread_clearance_open_riser_mode():
    g = ss.SIMPLE_STAIRS_GEOMETRY
    # 25 mm / 20 deg -> run 68.7 mm >= 62 mm: legal in open-riser mode...
    lad.validate_tread_clearance(g, 0.025, 20.0, open_riser=True)
    # ...but ILLEGAL under the legacy under-tread rule (below 29 mm).
    with pytest.raises(ValueError, match="toe clearance minimum"):
        lad.validate_tread_clearance(g, 0.025, 20.0)
    # run too small for the gap: 25 mm / 25 deg -> 53.6 mm < 62 mm.
    with pytest.raises(ValueError, match="open-riser gap"):
        lad.validate_tread_clearance(g, 0.025, 25.0, open_riser=True)
    # Legacy callers unchanged: the old 30 mm / 26.6 deg design still passes,
    # and an alternating-ladder level passes.
    lad.validate_tread_clearance(g, 0.030, 26.6)
    lad.validate_tread_clearance(lad.LADDER_GEOMETRY, 0.024, 60.0)


def _load_snapshot_mdp():
    """Exec the snapshot tasks/mdp.py under a scratch name (no sys.modules
    registration) and hand it to the snapshot ladder cfg module, whose
    ``microduck_mdp`` global otherwise binds the main tree's ladder-less mdp."""
    path = _SNAPSHOT_SRC / "tasks" / "mdp.py"
    spec = importlib.util.spec_from_file_location("_desk_climb_snapshot_mdp", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_factory_wiring_open_riser_only_for_simple_stairs():
    from mjlab_microduck.tasks import microduck_ladder_env_cfg as ladder_mod

    snap_mdp = _load_snapshot_mdp()
    ladder_mod.microduck_mdp = snap_mdp
    ss.microduck_mdp = snap_mdp
    cfg = ss.make_microduck_simple_stairs_env_cfg()
    assert cfg.events["reset_stair_ladder"].params["open_riser"] is True
    # The alternating ladder family must be zero-change: default stays False.
    ladder_cfg = ladder_mod.make_microduck_ladder_env_cfg()
    assert ladder_cfg.events["reset_stair_ladder"].params["open_riser"] is False


def test_v13_stall_penalty_and_overshoot_clearance():
    from mjlab_microduck.tasks import microduck_ladder_env_cfg as ladder_mod

    snap_mdp = _load_snapshot_mdp()
    ladder_mod.microduck_mdp = snap_mdp
    ss.microduck_mdp = snap_mdp  # the simple_stairs module bound the main mdp at import
    cfg = ss.make_microduck_simple_stairs_env_cfg()
    # F1 (v14 dose): the anti-park stall penalty is wired, self-negating ->
    # POSITIVE weight; 4 s window (only true parks), 2x-kill weight 1.0.
    assert "tread_stall" in cfg.rewards
    assert cfg.rewards["tread_stall"].weight == 1.0
    assert cfg.rewards["tread_stall"].params["stall_s"] == 4.0
    # F2: the overshoot ceiling now covers the official gait family's foot apex
    assert cfg.rewards["swing_overshoot"].params["clearance"] == 0.05
    l0_riser_hi = ss.SIMPLE_STAIRS_LEVELS[0]["riser"][1]  # 0.017
    l4_riser_hi = ss.SIMPLE_STAIRS_LEVELS[-1]["riser"][1]  # 0.0255
    assert l0_riser_hi + 0.003 + 0.05 >= 0.0666  # official median foot apex
    assert l4_riser_hi + 0.003 + 0.05 >= 0.0724  # official p90 foot apex
    # The ladder family reward table is zero-change (F1/F2 live only in simple_stairs).
    ladder_cfg = ladder_mod.make_microduck_ladder_env_cfg()
    assert "tread_stall" not in ladder_cfg.rewards
    assert ladder_cfg.rewards["swing_overshoot"].params["clearance"] == 0.025


def _patch_snapshot_mdp():
    from mjlab_microduck.tasks import microduck_ladder_env_cfg as ladder_mod

    snap_mdp = _load_snapshot_mdp()
    ladder_mod.microduck_mdp = snap_mdp
    ss.microduck_mdp = snap_mdp
    return snap_mdp


def test_v15_top_approach_spawn_wiring():
    _patch_snapshot_mdp()
    train = ss.make_microduck_simple_stairs_env_cfg(play=False)
    assert train.events["reset_stair_ladder"].params["top_approach_prob"] == 0.35
    assert "top_approach_spawn" in train.curriculum
    stages = train.curriculum["top_approach_spawn"].params["param_stages"]
    assert [s["params"]["top_approach_prob"] for s in stages] == [0.35, 0.15]
    assert stages[1]["step"] == 1000 * 24
    play = ss.make_microduck_simple_stairs_env_cfg(play=True)
    assert play.events["reset_stair_ladder"].params["top_approach_prob"] == 0.0
    assert "top_approach_spawn" not in play.curriculum
    # ladder family zero-change: no near-top spawn anywhere else.
    from mjlab_microduck.tasks import microduck_ladder_env_cfg as ladder_mod

    ladder_cfg = ladder_mod.make_microduck_ladder_env_cfg()
    assert ladder_cfg.events["reset_stair_ladder"].params["top_approach_prob"] == 0.0


def test_v15_start_level_seed(monkeypatch):
    _patch_snapshot_mdp()
    # Default: no seed event (from-scratch behaviour unchanged).
    cfg = ss.make_microduck_simple_stairs_env_cfg()
    assert "seed_start_level" not in cfg.events
    # Warm start: MICRODUCK_SIMPLE_STAIRS_START_LEVEL=3 seeds the first reset.
    monkeypatch.setenv("MICRODUCK_SIMPLE_STAIRS_START_LEVEL", "3")
    cfg = ss.make_microduck_simple_stairs_env_cfg()
    assert list(cfg.events)[0] == "seed_start_level"  # runs before the spawn
    assert cfg.events["seed_start_level"].params["level"] == 3

    class _StubEnv:
        num_envs = 4
        device = "cpu"

    env = _StubEnv()
    seed = cfg.events["seed_start_level"].func
    seed(env, None, **cfg.events["seed_start_level"].params)
    assert env._stair.level.tolist() == [3, 3, 3, 3]
    env._stair.level[0] = 1  # simulate curriculum movement...
    seed(env, None, **cfg.events["seed_start_level"].params)  # one-shot: no re-pin
    assert env._stair.level[0] == 1


def test_warm_start_patch_resets_counters(monkeypatch):
    """MICRODUCK_WARM_START=1 zeroes the env step counter and the iteration
    after a checkpoint load; unset, the restored values are kept."""
    snap_mdp = _load_snapshot_mdp()
    from mjlab.rl.runner import MjlabOnPolicyRunner

    class _Unwrapped:
        common_step_counter = 96000

    class _Env:
        unwrapped = _Unwrapped()

    class _Runner:
        env = _Env()
        current_learning_iteration = 3998

    monkeypatch.setattr(
        snap_mdp, "_orig_runner_load",
        lambda self, path, *a, **k: {"env_state": {"common_step_counter": 96000}},
    )
    monkeypatch.setenv(snap_mdp.WARM_START_ENV, "1")
    r = _Runner()
    MjlabOnPolicyRunner.load(r, "model_3998.pt")
    assert r.env.unwrapped.common_step_counter == 0
    assert r.current_learning_iteration == 0

    monkeypatch.delenv(snap_mdp.WARM_START_ENV)
    r = _Runner()
    r.env = _Env()
    r.env.unwrapped = _Unwrapped()
    MjlabOnPolicyRunner.load(r, "model_3998.pt")
    assert r.env.unwrapped.common_step_counter == 96000
    assert r.current_learning_iteration == 3998


# --- v16: tumble deficit + top-exempt stall -------------------------------------


class _StubStairEnv:
    """Just enough env for the v16 state-driven reward functions."""

    def __init__(self, last_tread, episode_buf, level=2):
        import torch as _t

        n = len(last_tread)
        self.num_envs = n
        self.device = "cpu"
        self.episode_length_buf = _t.tensor(episode_buf)
        self._stair = type(
            "Stair",
            (),
            {
                "foot_last_tread": _t.tensor(last_tread),
                "geometry": ss.SIMPLE_STAIRS_GEOMETRY,
                "level": _t.full((n,), level, dtype=_t.long),
            },
        )()


def test_v16_tumble_deficit_logic():
    _patch_snapshot_mdp()  # _stair_state lives in the snapshot mdp
    f = ss.simple_stairs_tumble_deficit_penalty
    # Episode start on tread 5: latch at the spawn, no charge.
    env = _StubStairEnv([[5, 4]], [1])
    assert f(env).tolist() == [0.0]
    # Mid-episode from here on (fresh re-latch only fires at episode start).
    env.episode_length_buf = torch.tensor([10])
    # Climbing to 8 never pays a deficit.
    env._stair.foot_last_tread = torch.tensor([[8, 7]])
    assert f(env).tolist() == [0.0]
    # Crouch / bobbing on the same treads: free.
    assert f(env).tolist() == [0.0]
    # One foot still holding the high point: free (max over feet).
    env._stair.foot_last_tread = torch.tensor([[8, 6]])
    assert f(env).tolist() == [0.0]
    # Both feet down a tread: bleed exactly one tread per step.
    env._stair.foot_last_tread = torch.tensor([[7, 6]])
    assert f(env).tolist() == [-1.0]
    # Tumbling to the floor from a best of 8: -(8 - (-1)) = -9.
    env._stair.foot_last_tread = torch.tensor([[-1, -1]])
    assert f(env).tolist() == [-9.0]
    # Next episode re-latches: no bleed from the old high-water mark.
    env.episode_length_buf = torch.tensor([1])
    env._stair.foot_last_tread = torch.tensor([[0, -1]])
    assert f(env).tolist() == [0.0]


def test_v18_tumble_deficit_highwater_gate():
    _patch_snapshot_mdp()
    assert ss.TUMBLE_DEFICIT_MIN_TREAD == 8
    f = ss.simple_stairs_tumble_deficit_penalty
    # Same tumble trajectory, three episode shapes:
    # (a) never got past tread 7 (incl. the farm ceiling) -> always free,
    # even after sliding all the way down.
    env = _StubStairEnv([[0, -1]], [1])
    assert f(env).tolist() == [0.0]
    env.episode_length_buf = torch.tensor([10])
    env._stair.foot_last_tread = torch.tensor([[7, 6]])
    assert f(env).tolist() == [0.0]
    env._stair.foot_last_tread = torch.tensor([[1, -1]])
    assert f(env).tolist() == [0.0]  # frontier/farm slide below 8: free
    # (b) reached tread 8 mid-episode -> tax engages at once for its remainder.
    env2 = _StubStairEnv([[0, -1]], [1])
    assert f(env2).tolist() == [0.0]
    env2.episode_length_buf = torch.tensor([10])
    env2._stair.foot_last_tread = torch.tensor([[7, 6]])
    assert f(env2).tolist() == [0.0]  # still below the gate
    env2._stair.foot_last_tread = torch.tensor([[8, 7]])
    assert f(env2).tolist() == [0.0]  # gate opens HERE, at the best itself
    env2._stair.foot_last_tread = torch.tensor([[7, 6]])
    assert f(env2).tolist() == [-1.0]  # one tread below: bleeds
    # (c) high-water 9, tumble to the floor -> the full water-mark deficit.
    env3 = _StubStairEnv([[5, 4]], [1])
    assert f(env3).tolist() == [0.0]
    env3.episode_length_buf = torch.tensor([10])
    env3._stair.foot_last_tread = torch.tensor([[9, 8]])
    assert f(env3).tolist() == [0.0]
    env3._stair.foot_last_tread = torch.tensor([[-1, -1]])
    assert f(env3).tolist() == [-10.0]  # -(9 - (-1))


def test_v16_tread_stall_top_exemption(monkeypatch):
    import torch as _t
    from types import SimpleNamespace

    fake_mdp = SimpleNamespace(
        ladder_tread_stall_penalty=lambda env, stall_s=4.0: _t.full((env.num_envs,), -1.0),
        _stair_state=lambda env: env._stair,
    )
    monkeypatch.setattr(ss, "microduck_mdp", fake_mdp)
    wrapper = ss._tread_stall_top_exempt
    # v20 zone: only the deck itself (num_treads - 1 = 11) is exempt.
    # treads (7, 8) and (9, 10): below the deck -> the v14 dose fires.
    assert wrapper(_StubStairEnv([[7, 8]], [100])).tolist() == [-1.0]
    assert wrapper(_StubStairEnv([[9, 10]], [100])).tolist() == [-1.0]
    assert wrapper(_StubStairEnv([[10, 10]], [100])).tolist() == [-1.0]
    # both feet on the deck (11): exempt.
    assert wrapper(_StubStairEnv([[11, 11]], [100])).tolist() == [0.0]
    # one foot back on a tread or the floor: not exempt.
    assert wrapper(_StubStairEnv([[11, 10]], [100])).tolist() == [-1.0]
    assert wrapper(_StubStairEnv([[11, -1]], [100])).tolist() == [-1.0]


def test_v16_cfg_terms_and_ladder_unchanged():
    _patch_snapshot_mdp()
    cfg = ss.make_microduck_simple_stairs_env_cfg()
    assert cfg.rewards["tumble_deficit"].weight == 2.0
    assert cfg.rewards["tumble_deficit"].func is ss.simple_stairs_tumble_deficit_penalty
    assert cfg.rewards["tread_stall"].func is ss._tread_stall_top_exempt
    assert cfg.rewards["tread_stall"].weight == 1.0
    assert cfg.rewards["tread_stall"].params["stall_s"] == 4.0
    from mjlab_microduck.tasks import microduck_ladder_env_cfg as ladder_mod

    ladder_cfg = ladder_mod.make_microduck_ladder_env_cfg()
    assert "tumble_deficit" not in ladder_cfg.rewards
    assert "tread_stall" not in ladder_cfg.rewards


# --- beveled landing nose (2026-09-26) ------------------------------------------


def test_landing_nose_bevel_geometry():
    import dataclasses

    import mujoco

    g_plain = lad.StairLadderGeometry(num_treads=12, alternating=False, tread_depth_m=0.060, landing_every=12)
    # Family default: no bevel, and the landing spec is exactly today's plain box.
    assert g_plain.landing_nose_bevel_m == 0.0
    spec = lad.make_tread_spec(g_plain, 11)
    assert len(spec.body("tread").geoms) == 1
    g_bevel = dataclasses.replace(g_plain, landing_nose_bevel_m=0.025)
    spec = lad.make_tread_spec(g_bevel, 11)
    geoms = spec.body("tread").geoms
    assert len(geoms) == 2
    assert geoms[1].type == mujoco.mjtGeom.mjGEOM_MESH
    assert geoms[1].name == "nose_bevel"  # contact-classifies as the landing tread
    model = spec.compile()  # builds cleanly
    assert model.ngeom == 2
    # Non-landing treads are untouched by the bevel field.
    assert len(lad.make_tread_spec(g_bevel, 5).body("tread").geoms) == 1
    # Clearance logic is unaffected by the new field.
    lad.validate_tread_clearance(g_bevel, 0.025, 20.0, open_riser=True)
    riser, angle = lad.clamp_riser_angle(g_bevel, *_f64(0.025, 19.0), open_riser=True)
    assert float(riser) == pytest.approx(0.025)
    assert float(angle) == pytest.approx(19.0)
    # The alternating ladder family's landing spec is unchanged (default 0).
    from mjlab_microduck.tasks import microduck_ladder_env_cfg as _ladder_cfg_mod

    assert len(lad.make_tread_spec(_ladder_cfg_mod.STAIRCASE_GEOMETRY, 7).body("tread").geoms) == 1


def test_landing_flush_geometry():
    from mjlab_microduck.robot import ladder as lad
    import torch

    g_plain = lad.StairLadderGeometry(num_treads=12, alternating=False, landing_every=12)
    import dataclasses
    g_flush = dataclasses.replace(g_plain, landing_flush=True)
    riser = torch.full((2,), 0.025)
    top_plain = lad.tread_top_heights_geom(g_plain, riser)
    top_flush = lad.tread_top_heights_geom(g_flush, riser)
    # Ladder family default: flush off — landing top stays one riser up.
    assert top_plain[0, 11].item() == pytest.approx(12 * 0.025)
    # Flush: landing top continues the previous tread's level; others untouched.
    assert top_flush[0, 11].item() == pytest.approx(top_flush[0, 10].item())
    assert top_flush[0, 10].item() == pytest.approx(top_plain[0, 10].item())
    # (mesh placement is mocap/reset-time; the height function above is what
    # drives both the box z and the nose-x pullback via (top - z_base)/tan.)


def test_simple_stairs_bevel_switch(monkeypatch):
    _patch_snapshot_mdp()
    # Default (2026-09-28): bevel off — the flush landing join was the real
    # fix (first on-deck stand, relay seed 19927, bevel-free A/B).
    for play in (False, True):
        cfg = ss.make_microduck_simple_stairs_env_cfg(play=play)
        geo = cfg.events["reset_stair_ladder"].params["geometry"]
        assert geo.landing_nose_bevel_m == 0.0
    # Env override re-enables the wedge; the seed event carries the same geometry.
    monkeypatch.setenv("SIMPLE_STAIRS_NOSE_BEVEL_M", "0.025")
    monkeypatch.setenv("MICRODUCK_SIMPLE_STAIRS_START_LEVEL", "3")
    cfg = ss.make_microduck_simple_stairs_env_cfg()
    assert cfg.events["reset_stair_ladder"].params["geometry"].landing_nose_bevel_m == pytest.approx(0.025)
    assert cfg.events["seed_start_level"].params["geometry"].landing_nose_bevel_m == pytest.approx(0.025)


def test_table_legs_and_rail_overhang(monkeypatch):
    import dataclasses

    g = lad.StairLadderGeometry(num_treads=12, alternating=False, tread_depth_m=0.060, landing_every=12)
    # Family defaults: everything off.
    assert g.landing_table_leg_radius_m == 0.0
    assert g.rail_overhang_m == 0.0
    assert len(lad.make_tread_spec(g, 11).body("tread").geoms) == 1
    # Legs on: box + 4 non-collidable corner cylinders; they coexist with the bevel.
    g_legs = dataclasses.replace(g, landing_table_leg_radius_m=0.0075)
    geoms = lad.make_tread_spec(g_legs, 11).body("tread").geoms
    assert len(geoms) == 5
    legs = [x for x in geoms if x.name and x.name.startswith("table_leg")]
    assert len(legs) == 4
    assert all(int(x.contype) == 0 and int(x.conaffinity) == 0 for x in legs)
    g_both = dataclasses.replace(g_legs, landing_nose_bevel_m=0.025)
    assert len(lad.make_tread_spec(g_both, 11).body("tread").geoms) == 6
    # Rail overhang: stock incline poses/spec identical at any overhang (the
    # extension is a separate horizontal resting segment, tested in
    # test_rails_rest_on_deck_not_embedded).
    angle = torch.tensor([math.radians(20.0)])
    x0 = torch.zeros(1)
    y0 = torch.zeros(1)
    p0, q0 = lad.rail_poses(g, angle, x0, y0)
    g_oh = dataclasses.replace(g, rail_overhang_m=0.12)
    p1, q1 = lad.rail_poses(g_oh, angle, x0, y0)
    assert torch.allclose(p0, p1)
    assert torch.allclose(q0, q1)
    s0 = float(lad.make_rail_spec(g).body("rail").geoms[0].size[2])
    s1 = float(lad.make_rail_spec(g_oh).body("rail").geoms[0].size[2])
    assert s1 == pytest.approx(s0)
    # simple_stairs switches: table legs default ON, rail overhang default OFF
    # (user 2026-09-28: the two desk-top sticks read as broken); env toggles invert.
    _patch_snapshot_mdp()
    cfg = ss.make_microduck_simple_stairs_env_cfg()
    geo = cfg.events["reset_stair_ladder"].params["geometry"]
    assert geo.landing_table_leg_radius_m == pytest.approx(0.0075)
    assert geo.rail_overhang_m == 0.0
    monkeypatch.setenv("SIMPLE_STAIRS_TABLE_LEGS", "0")
    monkeypatch.setenv("SIMPLE_STAIRS_RAIL_OVERHANG_M", "0.12")
    geo = ss.make_microduck_simple_stairs_env_cfg().events["reset_stair_ladder"].params["geometry"]
    assert geo.landing_table_leg_radius_m == 0.0
    assert geo.rail_overhang_m == pytest.approx(0.12)


# --- walk-on relay helpers --------------------------------------------------------


def _load_relay_script():
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments/desk-climb/source/scripts/render_stairs_getup.py"
    )
    spec = importlib.util.spec_from_file_location("_render_stairs_getup", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_walk_on_helpers():
    import numpy as np

    mod = _load_relay_script()
    # advance: climb speed per step, clamped at the deck's far edge
    assert mod.walk_on_advance(0.30, 0.40, 0.04, 0.02) == pytest.approx(0.3008)
    assert mod.walk_on_advance(0.3995, 0.40, 0.04, 0.02) == 0.40
    assert mod.walk_on_advance(0.40, 0.40, 0.04, 0.02) == 0.40
    # obs format: per-foot [fwd, up] of (target - foot) in the yaw frame,
    # scaled 1/0.05, per-side lateral targets, clamped to +/-5.
    feet = np.array([[0.90, 0.04, 0.30], [0.90, -0.04, 0.30]], dtype=np.float64)
    out = mod.walk_on_obs_targets(1.00, 0.06, 0.33, feet, 0.0)
    assert out[0] == pytest.approx(2.0)  # fwd left = (1.00 - 0.90) / 0.05
    assert out[1] == pytest.approx(0.6)  # up = (0.33 - 0.30) / 0.05
    assert out[2] == pytest.approx(2.0)  # fwd right (lateral term vanishes at yaw 0)
    assert out[3] == pytest.approx(0.6)
    # yaw = +90 deg: fwd comes from the lateral (y) difference, per-side signed
    out = mod.walk_on_obs_targets(1.00, 0.06, 0.33, feet, math.pi / 2)
    assert out[0] == pytest.approx((0.06 - 0.04) / 0.05)   # left: +0.4
    assert out[2] == pytest.approx((-0.06 + 0.04) / 0.05)  # right: -0.4
    # far target clamps to +/-5
    out = mod.walk_on_obs_targets(3.00, 0.06, 0.33, feet, 0.0)
    assert out[0] == 5.0 and out[2] == 5.0
    # deck footprint check
    assert mod.on_deck_xy(np.array([0.30, 0.10]), np.array([0.30, 0.0]), 0.20, 0.30)
    assert not mod.on_deck_xy(np.array([0.55, 0.10]), np.array([0.30, 0.0]), 0.20, 0.30)
    assert not mod.on_deck_xy(np.array([0.30, 0.31]), np.array([0.30, 0.0]), 0.20, 0.30)


# --- flush-landing foot_tread classification (xy, not z) -------------------------


def _flush_state(flush: bool):
    """Real tread layout at riser 25 mm / 20 deg for the simple_stairs
    geometry, with landing_flush on/off — nothing hardcoded."""
    import dataclasses
    from types import SimpleNamespace

    g = ss.SIMPLE_STAIRS_GEOMETRY
    # Mirror the factory's production geometry (flush + zero landing setback).
    g = dataclasses.replace(g, landing_setback_m=0.0)
    if flush:
        g = dataclasses.replace(g, landing_flush=True)
    riser = torch.tensor([0.025])
    angle = torch.tensor([math.radians(20.0)])
    x0 = torch.zeros(1)
    y0 = torch.zeros(1)
    centres, tops, nose_u, target_xy, tread_yaw = lad.tread_layout(g, riser, angle, x0, y0)
    return SimpleNamespace(
        geometry=g,
        tread_side=torch.tensor([float(g.tread_side(i)) for i in range(g.num_treads)]),
        tread_top=tops,
        tread_centre=centres,
        tread_target_xy=target_xy,
        tread_yaw=tread_yaw,
        _nose_u=nose_u,
    )


class _StubAsset:
    def __init__(self, feet_xyz):
        self.data = type(
            "Data",
            (),
            {
                "site_pos_w": torch.tensor(feet_xyz, dtype=torch.float32),
                "root_link_pos_w": torch.zeros(len(feet_xyz), 3),
                "projected_gravity_b": torch.tensor([[0.0, 0.0, -1.0]]),
                "root_link_lin_vel_w": torch.zeros(len(feet_xyz), 3),
                "root_link_ang_vel_w": torch.zeros(len(feet_xyz), 3),
            },
        )()

    def find_sites(self, name):
        return ([0 if name == "left_foot" else 1], None)


def _target_index(state, feet_xyz):
    env = type("E", (), {"num_envs": 1, "device": "cpu", "common_step_counter": 0, "_stair": state})()
    snap_mdp = _load_snapshot_mdp()
    info = snap_mdp._stair_foot_target_info(env, _StubAsset(feet_xyz))
    return int(info["index"][0, 0]), int(info["index"][0, 1])


def test_flush_landing_target_candidates():
    state = _flush_state(flush=True)
    g = state.geometry
    top_idx = g.num_treads - 1
    c10 = state.tread_centre[0, top_idx - 1]
    nose = float(state.tread_centre[0, top_idx, 0]) - 0.5 * g.landing_depth_m
    z = float(state.tread_top[0, top_idx]) + 0.003
    # Foot standing on the last mini tread (approach): the flush deck must be
    # offered as the next target (the z-test alone would offer nothing).
    feet = [[[float(c10[0]), 0.042, float(state.tread_top[0, top_idx - 1]) + 0.003],
             [float(c10[0]), -0.042, float(state.tread_top[0, top_idx - 1]) + 0.003]]]
    assert _target_index(state, feet) == (top_idx, top_idx)
    # Foot already ON the deck (past the nose): still no target (end of stairs).
    feet = [[[nose + 0.05, 0.0, z], [nose + 0.05, 0.0, z]]]
    idx = _target_index(state, feet)
    assert idx == (g.num_treads, g.num_treads)
    # Foot on the second-to-last mini tread: target is the last mini tread.
    c9 = state.tread_centre[0, top_idx - 2]
    feet = [[[float(c9[0]), 0.042, float(state.tread_top[0, top_idx - 2]) + 0.003],
             [float(c9[0]), -0.042, float(state.tread_top[0, top_idx - 2]) + 0.003]]]
    assert _target_index(state, feet) == (top_idx - 1, top_idx - 1)
    # Non-flush: byte-identical behaviour (the xy branch never runs).
    state_nf = _flush_state(flush=False)
    assert _target_index(state_nf, feet) == (top_idx - 1, top_idx - 1)
    feet10 = [[[float(c10[0]), 0.042, float(state.tread_top[0, top_idx - 1]) + 0.003],
               [float(c10[0]), -0.042, float(state.tread_top[0, top_idx - 1]) + 0.003]]]
    assert _target_index(state_nf, feet10) == (top_idx, top_idx)  # z covers it as before


def test_feet_on_landing_xy_classification():
    state = _flush_state(flush=True)
    g = state.geometry
    top_idx = g.num_treads - 1
    nose = float(state.tread_centre[0, top_idx, 0]) - 0.5 * g.landing_depth_m
    top = float(state.tread_top[0, top_idx])
    cx = float(state.tread_centre[0, top_idx, 0])
    snap_mdp = _load_snapshot_mdp()
    f = lambda feet: snap_mdp.feet_on_landing_xy(state, torch.tensor(feet)).tolist()
    assert f([[[nose + 0.05, 0.0, top + 0.003], [cx, 0.10, top + 0.002]]]) == [[True, True]]
    assert f([[[nose - 0.03, 0.0, top + 0.003], [nose + 0.05, 0.0, top + 0.003]]]) == [[False, True]]  # behind the nose
    assert f([[[nose + 0.05, 0.0, 0.0], [nose + 0.05, 0.0, top + 0.003]]]) == [[False, True]]  # floor under the deck
    assert f([[[nose + 0.05, 0.5 * g.landing_width_m + 0.05, top + 0.003], [nose + 0.05, 0.0, top + 0.003]]]) == [[False, True]]


def test_flush_reached_top_xy_gate(monkeypatch):
    state = _flush_state(flush=True)
    g = state.geometry
    top_idx = g.num_treads - 1
    top = float(state.tread_top[0, top_idx])
    nose = float(state.tread_centre[0, top_idx, 0]) - 0.5 * g.landing_depth_m
    snap_mdp = _load_snapshot_mdp()

    def _run(flush_state):
        # Both feet INSIDE the deck footprint but the contact geom still reads
        # the mini tread (the flush seam): success must latch via the xy gate
        # when flush, and must NOT when non-flush (behaviour unchanged).
        asset = _StubAsset(
            [[[nose + 0.06, 0.042, top + 0.003], [nose + 0.06, -0.042, top + 0.003]]]
        )
        asset.data.root_link_pos_w = torch.tensor([[nose + 0.06, 0.0, top + 0.115]])
        monkeypatch.setattr(
            snap_mdp, "_stair_contacts",
            lambda env: {
                "foot_tread": torch.tensor([[top_idx - 1, top_idx - 1]]),
                "foot_support": torch.tensor([[True, True]]),
            },
        )
        env = type(
            "E",
            (),
            {
                "num_envs": 1,
                "device": "cpu",
                "step_dt": 0.02,
                "_stair": flush_state,
                "scene": {"robot": asset},
                "common_step_counter": 0,
            },
        )()
        fired = False
        for i in range(30):
            env.common_step_counter = i
            fired |= bool(snap_mdp.ladder_reached_top(env)[0])
        return fired

    assert _run(state) is True
    assert _run(_flush_state(flush=False)) is False


def test_rails_rest_on_deck_not_embedded():
    import dataclasses

    g = lad.StairLadderGeometry(num_treads=12, alternating=False, tread_depth_m=0.060, landing_every=12)
    # Stock behaviour: no overhang -> no ext entities, incline spec/poses unchanged.
    assert not any(n.startswith("rail_ext") for n in lad.rail_entity_names(g))
    g_oh = dataclasses.replace(g, rail_overhang_m=0.12, landing_flush=True)
    names = lad.rail_entity_names(g_oh)
    assert names[-2:] == ("rail_ext_left", "rail_ext_right")
    # The inclined stock rail is byte-identical to no-overhang (poses + spec).
    angle = torch.tensor([math.radians(20.0)])
    x0 = torch.zeros(1)
    y0 = torch.zeros(1)
    p0, q0 = lad.rail_poses(g, angle, x0, y0)
    p1, q1 = lad.rail_poses(g_oh, angle, x0, y0)
    assert torch.allclose(p0, p1) and torch.allclose(q0, q1)
    s0 = lad.make_rail_spec(g).body("rail").geoms[0].size
    s1 = lad.make_rail_spec(g_oh).body("rail").geoms[0].size
    assert all(float(a) == pytest.approx(float(b)) for a, b in zip(s0, s1))
    # The resting segment: horizontal box, overhang long, same cross-section.
    ext_spec = lad._rail_spec_for(g_oh, "rail_ext_left")
    size = ext_spec.body("rail_ext").geoms[0].size
    assert float(size[0]) == pytest.approx(0.06)
    assert float(size[1]) == pytest.approx(0.5 * g_oh.rail_width_m)
    assert float(size[2]) == pytest.approx(0.5 * g_oh.rail_depth_m)
    # Resting pose: starts 1 cm past the nose, bottom floats 1.5 mm above the
    # deck top (RESTS on it, never inside the box), yaw-only quat, and clear
    # of the duck's central walk path on the deck.
    riser = torch.tensor([0.025])
    centres, tops, _n, _txy, _yaw = lad.tread_layout(g_oh, riser, angle, x0, y0)
    fyaw = torch.zeros(1)
    pos, quat = lad.rail_ext_poses(g_oh, centres[:, -1], tops[:, -1], fyaw)
    top = float(tops[0, -1])
    nose = float(centres[0, -1, 0]) - 0.5 * g_oh.landing_depth_m
    for k in range(2):
        bottom = float(pos[0, k, 2]) - 0.5 * g_oh.rail_depth_m
        assert bottom == pytest.approx(top + 0.0015)  # rests on the deck, not embedded
        assert float(pos[0, k, 0]) == pytest.approx(nose + 0.01 + 0.06)
        assert float(quat[0, k, 1]) == 0.0 and float(quat[0, k, 2]) == 0.0  # yaw-only
    v = 0.5 * g_oh.clear_width_m + 0.5 * g_oh.rail_width_m
    assert v < 0.5 * g_oh.landing_width_m  # lies on the deck surface (at the sides)
    assert v - 0.06 > 0.5 * g_oh.rail_width_m + 0.03  # clear of the duck's foot targets (y = ±0.06)


def test_stand_tracker_deck_gate():
    mod = _load_relay_script()
    dt = 0.02
    # (a) upright at deck height but standing on the LAST TREAD (feet off
    # deck) for 14 s -> never stood (the seed-19927 false positive).
    tr = mod.StandTracker()
    for _ in range(round(14.0 / dt)):
        tr.update(True, False, dt)
    assert not tr.stood(3.0)
    # (b) on the deck for 3.2 s continuous -> stood.
    tr = mod.StandTracker()
    for _ in range(round(3.2 / dt)):
        tr.update(True, True, dt)
    assert tr.stood(3.0)
    # (c) on-deck run with a brief getup foot lift (0.3 s off in 1.7 s =
    # 18% < 20%) -> the run survives, held counts on-deck time only.
    tr = mod.StandTracker()
    for _ in range(round(1.4 / dt)):
        tr.update(True, True, dt)
    for _ in range(round(0.3 / dt)):
        tr.update(True, False, dt)
    for _ in range(round(1.8 / dt)):
        tr.update(True, True, dt)
    assert tr.stood(3.0)
    # (d) a long off-deck stretch (0.8 s off in 3.0 s = 27% > 20%) -> invalid.
    tr = mod.StandTracker()
    for _ in range(round(1.2 / dt)):
        tr.update(True, True, dt)
    for _ in range(round(0.8 / dt)):
        tr.update(True, False, dt)
    for _ in range(round(2.5 / dt)):
        tr.update(True, True, dt)
    assert not tr.stood(3.0)
    # (e) any topple resets the run.
    tr = mod.StandTracker()
    for _ in range(round(2.0 / dt)):
        tr.update(True, True, dt)
    tr.update(False, True, dt)
    for _ in range(round(2.0 / dt)):
        tr.update(True, True, dt)
    assert not tr.stood(3.0)


def test_num_treads_env_var(monkeypatch):
    _patch_snapshot_mdp()
    # Default: byte-identical to today's 12-tread geometry.
    cfg = ss.make_microduck_simple_stairs_env_cfg()
    geo = cfg.events["reset_stair_ladder"].params["geometry"]
    assert geo.num_treads == 12 and geo.landing_every == 12
    assert geo.is_landing(11) and not geo.is_landing(10)
    # 25 treads: landing follows (only the top platform), all the adaptive
    # consumers see the new top index.
    monkeypatch.setenv("SIMPLE_STAIRS_NUM_TREADS", "25")
    cfg = ss.make_microduck_simple_stairs_env_cfg()
    geo = cfg.events["reset_stair_ladder"].params["geometry"]
    assert geo.num_treads == 25 and geo.landing_every == 25
    assert geo.is_landing(24) and not geo.is_landing(23)
    assert geo.num_flights == 1  # single flight, no walking paths
    # Entity set follows: 25 treads + stock rails (+ resting segments when
    # the overhang is explicitly enabled — it is off by default since 81d4704).
    monkeypatch.setenv("SIMPLE_STAIRS_RAIL_OVERHANG_M", "0.12")
    cfg = ss.make_microduck_simple_stairs_env_cfg()
    geo = cfg.events["reset_stair_ladder"].params["geometry"]
    cfgs = lad.make_stair_ladder_entity_cfgs(geo)
    assert sum(1 for k in cfgs if k.startswith("tread_")) == 25
    assert {"rail_left", "rail_right", "rail_ext_left", "rail_ext_right"} <= set(cfgs)
    # v20 stall exemption threshold follows (num_treads - 1 = 24).
    import torch as _t
    from types import SimpleNamespace

    fake_mdp = SimpleNamespace(
        ladder_tread_stall_penalty=lambda env, stall_s=4.0: _t.full((env.num_envs,), -1.0),
        _stair_state=lambda env: env._stair,
    )
    monkeypatch.setattr(ss, "microduck_mdp", fake_mdp)
    wrapper = ss._tread_stall_top_exempt
    env24 = _StubStairEnv([[24, 24]], [100])
    env24._stair.geometry = geo
    assert wrapper(env24).tolist() == [0.0]
    env23 = _StubStairEnv([[23, 24]], [100])
    env23._stair.geometry = geo
    assert wrapper(env23).tolist() == [-1.0]
    # Top-approach spawn band follows: num_treads-4 .. num_treads-2 = 21..23.
    assert (geo.num_treads - 4, geo.num_treads - 2) == (21, 23)
    # Ladder family: untouched by the new env var.
    from mjlab_microduck.tasks import microduck_ladder_env_cfg as ladder_mod

    ladder_cfg = ladder_mod.make_microduck_ladder_env_cfg()
    assert ladder_cfg.events["reset_stair_ladder"].params["geometry"].num_treads == 16
