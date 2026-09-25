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


def test_v17_tumble_deficit_level_gate():
    _patch_snapshot_mdp()
    assert ss.TUMBLE_DEFICIT_MIN_LEVEL == 2
    f = ss.simple_stairs_tumble_deficit_penalty
    # Same tumble (best 8, both feet on the floor), three level bands:
    # L0/L1 -> tax-free (exploration), L2 -> the full deficit.
    for level, expect in ((0, 0.0), (1, 0.0), (2, -9.0), (4, -9.0)):
        env = _StubStairEnv([[5, 4]], [1], level=level)
        assert f(env).tolist() == [0.0]  # latch at spawn
        env.episode_length_buf = torch.tensor([10])
        env._stair.foot_last_tread = torch.tensor([[8, 7]])
        assert f(env).tolist() == [0.0]  # climb to the high-water mark
        env._stair.foot_last_tread = torch.tensor([[-1, -1]])
        assert f(env).tolist() == [expect], level


def test_v16_tread_stall_top_exemption(monkeypatch):
    import torch as _t
    from types import SimpleNamespace

    fake_mdp = SimpleNamespace(
        ladder_tread_stall_penalty=lambda env, stall_s=4.0: _t.full((env.num_envs,), -1.0),
        _stair_state=lambda env: env._stair,
    )
    monkeypatch.setattr(ss, "microduck_mdp", fake_mdp)
    wrapper = ss._tread_stall_top_exempt
    # treads (7, 8): below the zone -> the v14 dose fires.
    assert wrapper(_StubStairEnv([[7, 8]], [100])).tolist() == [-1.0]
    # treads (9, 10): both within 3 of the top (num_treads - 3 = 9) -> exempt.
    assert wrapper(_StubStairEnv([[9, 10]], [100])).tolist() == [0.0]
    # one foot back on the floor: not exempt.
    assert wrapper(_StubStairEnv([[9, -1]], [100])).tolist() == [-1.0]


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
