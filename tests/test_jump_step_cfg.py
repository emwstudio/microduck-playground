"""Cfg-invariant tests for Mjlab-JumpStep-MicroDuck (v11 ballistic spawns).

The jump_step task lives in the desk-climb snapshot tree
(experiments/desk-climb/source/src), not the main src tree.  Extend the
already-imported mjlab_microduck.tasks package path so the module resolves
from the snapshot while its own imports keep resolving from the main tree
(no sys.modules swapping — zero contamination of the other test modules).
"""

from pathlib import Path

import torch

import mjlab_microduck.tasks as _tasks

_SNAPSHOT_TASKS = (
    Path(__file__).resolve().parents[1] / "experiments/desk-climb/source/src/mjlab_microduck/tasks"
)
if str(_SNAPSHOT_TASKS) not in _tasks.__path__:
    _tasks.__path__.append(str(_SNAPSHOT_TASKS))

from mjlab_microduck.tasks import microduck_jump_step_env_cfg as js  # noqa: E402


def test_factory_builds_and_spawn_probs():
    train = js.make_microduck_jump_step_env_cfg(play=False)
    play = js.make_microduck_jump_step_env_cfg(play=True)
    tp = train.events["reset_jump_step"].params
    assert tp["ballistic_spawn_prob"] == 0.5
    assert tp["airborne_spawn_prob"] == 0.2
    assert tp["edge_spawn_prob"] == 0.15
    assert tp["platform_spawn_prob"] == 0.0
    pp = play.events["reset_jump_step"].params
    for key in ("ballistic_spawn_prob", "airborne_spawn_prob", "edge_spawn_prob", "platform_spawn_prob"):
        assert pp[key] == 0.0, key
    assert "ballistic_spawn" in train.curriculum
    assert "ballistic_rate" in train.curriculum
    assert "airborne_rate" in train.curriculum
    assert "airborne_spawn" not in train.curriculum  # airborne share is constant since v11
    assert "ballistic_spawn" not in play.curriculum


def test_obs_layout_stays_61d():
    cfg = js.make_microduck_jump_step_env_cfg()
    names = list(cfg.observations["actor"].terms)
    assert names == [
        "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel",
        "actions", "command", "head_command", "body_command",
    ]
    dims = 3 + 3 + 14 + 14 + 14 + 3 + 4 + 6
    assert dims == 61


def test_reward_weights_and_signs():
    cfg = js.make_microduck_jump_step_env_cfg()
    w = {k: v.weight for k, v in cfg.rewards.items()}
    # Task terms
    assert w["jump_step_success"] == 40.0
    assert w["jump_step_first_foot_bonus"] == 5.0
    assert w["jump_step_progress"] == 50.0
    assert w["jump_step_forward_progress"] == 30.0
    assert w["jump_step_foot_progress"] == 30.0
    # Self-negating penalties must keep POSITIVE weights (they return <= 0)
    assert w["vertical_impact"] > 0
    assert w["body_platform_contact"] > 0
    # mjlab-base cost functions keep NEGATIVE weights
    for name in ("action_rate_l2", "body_ang_vel", "angular_momentum", "dof_pos_limits", "self_collisions"):
        assert w[name] < 0, name


def test_spawn_type_constants_and_stage_tables():
    assert js.SPAWN_BALLISTIC == 4
    assert js.SPAWN_BALLISTIC not in (js.SPAWN_FLOOR, js.SPAWN_EDGE, js.SPAWN_PLATFORM, js.SPAWN_AIRBORNE)
    assert js.AIRBORNE_CLEARANCE_RANGE == (0.08, 0.15)
    assert js.BALLISTIC_APEX_RANGE == (0.05, 0.15)
    assert js.BALLISTIC_DISTANCE_RANGE == (0.05, 0.15)
    for table, first_rung in ((js.AIRBORNE_CURRICULUM, 0.15), (js.BALLISTIC_CURRICULUM, 0.10)):
        rates = [s["rate"] for s in table]
        probs = [s["prob"] for s in table]
        assert rates == sorted(rates, reverse=True)
        assert probs == sorted(probs)
        assert probs[0] == 0.0
        assert first_rung in rates
    # v11: ballistic's first rung is more conservative than airborne's (harder skill)
    assert min(r["rate"] for r in js.BALLISTIC_CURRICULUM if r["rate"] > 0) < 0.15
    # Nearly-done success scales: ballistic farther-from-done than a free fall
    assert js.SUCCESS_BALLISTIC_SPAWN_SCALE > js.SUCCESS_PLATFORM_SPAWN_SCALE
    assert js.SUCCESS_PLATFORM_SPAWN_SCALE * 40.0 == 3.0
    assert js.SUCCESS_BALLISTIC_SPAWN_SCALE * 40.0 == 6.0


def test_reset_signature_defaults():
    import inspect

    sig = inspect.signature(js.reset_jump_step)
    assert sig.parameters["ballistic_spawn_prob"].default == 0.5
    assert sig.parameters["airborne_spawn_prob"].default == 0.2
    assert sig.parameters["platform_spawn_prob"].default == 0.0
    assert sig.parameters["edge_spawn_prob"].default == 0.15
    assert sig.parameters["airborne_clearance_range"].default == (0.08, 0.15)
    assert sig.parameters["ballistic_apex_range"].default == (0.05, 0.15)
    assert sig.parameters["ballistic_distance_range"].default == (0.05, 0.15)


def test_ballistic_launch_physics():
    """Launch solution: apex clears the top with margin; trajectory comes
    down on the platform centre at top level; velocities inside the anchors."""
    g = js.GRAVITY
    # Whole sampling cube: h in [0.025, 0.035], A in [0.05, 0.15],
    # c in [0.01, 0.02], d in [0.05, 0.15] — check corners.
    for h in (0.025, 0.035):
        for A in (0.05, 0.15):
            for c in (0.01, 0.02):
                for d in (0.05, 0.15):
                    top = torch.tensor([h])
                    vx, vz = js._ballistic_launch(
                        top, torch.tensor([A]), torch.tensor([c]), torch.tensor([d])
                    )
                    vx, vz = float(vx), float(vz)
                    # vz puts the apex exactly A above the top.
                    apex_abs = c + vz**2 / (2 * g)
                    assert abs(apex_abs - (h + A)) < 1e-6
                    # Flight back down to top level lands on the centre offset.
                    t = vz / g + (2 * A / g) ** 0.5
                    x_land = vx * t
                    z_land = c + vz * t - 0.5 * g * t**2
                    assert abs(z_land - h) < 1e-6
                    assert abs(x_land - (d + 0.5 * js.PLATFORM_DEPTH_M - 0.010)) < 1e-6
                    # Anchors from the v11 design: vz ~1.0-1.9, vx ~0.2-1.0.
                    assert 1.0 < vz < 1.9
                    assert 0.15 < vx < 1.0
                    # The arc apex strictly exceeds the platform top.
                    assert apex_abs > h


class _StubEventManager:
    def __init__(self, params: dict):
        self._term = type("Term", (), {"params": params})()

    def get_term_cfg(self, name):
        assert name == "reset_jump_step"
        return self._term


class _StubEnv:
    """Just enough surface for _jump_step_state + the curriculum functions."""

    def __init__(self, params: dict):
        self.num_envs = 4
        self.device = "cpu"
        self.event_manager = _StubEventManager(params)


def _run_airborne_curriculum(env, done, landed, min_episodes=200):
    state = js._jump_step_state(env)
    state.airborne_done = done
    state.airborne_landed = landed
    return js.jump_step_airborne_curriculum(env, min_episodes=min_episodes)


def _run_ballistic_curriculum(env, done, landed, min_episodes=200):
    state = js._jump_step_state(env)
    state.ballistic_done = done
    state.ballistic_landed = landed
    return js.jump_step_ballistic_curriculum(env, min_episodes=min_episodes)


def test_airborne_curriculum_below_min_episodes_noop():
    env = _StubEnv({"airborne_spawn_prob": 0.6})
    assert _run_airborne_curriculum(env, done=100, landed=100) == 0.6
    state = js._jump_step_state(env)
    assert state.airborne_done == 100  # window untouched


def test_airborne_curriculum_stage_transitions():
    # rate 0.8 >= 0.50 -> retire airborne spawns entirely
    env = _StubEnv({"airborne_spawn_prob": 0.6})
    assert _run_airborne_curriculum(env, done=250, landed=200) == 0.0
    state = js._jump_step_state(env)
    assert state.airborne_done == 0 and state.airborne_landed == 0  # window reset
    assert env.event_manager.get_term_cfg("reset_jump_step").params["airborne_spawn_prob"] == 0.0
    # rate 0.35 -> 0.2
    env = _StubEnv({"airborne_spawn_prob": 0.4})
    assert _run_airborne_curriculum(env, done=200, landed=70) == 0.2
    # rate 0.2 -> 0.4
    env = _StubEnv({"airborne_spawn_prob": 0.6})
    assert _run_airborne_curriculum(env, done=200, landed=40) == 0.4
    # rate 0.1 -> stays 0.6 (highest tier)
    env = _StubEnv({"airborne_spawn_prob": 0.6})
    assert _run_airborne_curriculum(env, done=200, landed=20) == 0.6


def test_airborne_curriculum_bidirectional_and_window():
    # Low rate at an advanced stage re-introduces airborne spawns.
    env = _StubEnv({"airborne_spawn_prob": 0.2})
    assert _run_airborne_curriculum(env, done=300, landed=10) == 0.6
    # Rate matching the CURRENT tier: no param write, window keeps accumulating.
    env = _StubEnv({"airborne_spawn_prob": 0.6})
    assert _run_airborne_curriculum(env, done=250, landed=20) == 0.6
    state = js._jump_step_state(env)
    assert state.airborne_done == 250  # not reset — same stage


def test_ballistic_curriculum_stage_transitions():
    # rate 0.4 >= 0.35 -> retire ballistic spawns entirely
    env = _StubEnv({"ballistic_spawn_prob": 0.5})
    assert _run_ballistic_curriculum(env, done=250, landed=100) == 0.0
    state = js._jump_step_state(env)
    assert state.ballistic_done == 0 and state.ballistic_landed == 0
    assert env.event_manager.get_term_cfg("reset_jump_step").params["ballistic_spawn_prob"] == 0.0
    # rate 0.25 -> 0.15
    env = _StubEnv({"ballistic_spawn_prob": 0.3})
    assert _run_ballistic_curriculum(env, done=200, landed=50) == 0.15
    # rate 0.12 -> 0.3
    env = _StubEnv({"ballistic_spawn_prob": 0.5})
    assert _run_ballistic_curriculum(env, done=200, landed=24) == 0.3
    # rate 0.05 -> stays 0.5
    env = _StubEnv({"ballistic_spawn_prob": 0.5})
    assert _run_ballistic_curriculum(env, done=200, landed=10) == 0.5
    # below the window: no-op
    env = _StubEnv({"ballistic_spawn_prob": 0.5})
    assert _run_ballistic_curriculum(env, done=50, landed=50) == 0.5


# --- spawn-class accounting (the 2026-09-24 label-swap regression) ----------------


def test_spawn_class_outcomes_no_cross_talk():
    spawn_type = torch.tensor([
        js.SPAWN_FLOOR, js.SPAWN_FLOOR, js.SPAWN_EDGE,
        js.SPAWN_PLATFORM, js.SPAWN_AIRBORNE, js.SPAWN_AIRBORNE, js.SPAWN_AIRBORNE,
        js.SPAWN_BALLISTIC, js.SPAWN_BALLISTIC,
    ])
    landed = torch.tensor([False, True, True, False, True, False, True, True, False])
    out = js.spawn_class_outcomes(spawn_type, landed)
    assert out["floor"] == (2, 1)
    assert out["edge"] == (1, 1)
    assert out["platform"] == (1, 0)
    assert out["airborne"] == (3, 2)
    assert out["ballistic"] == (2, 1)
    # Every episode is accounted exactly once.
    assert sum(v[0] for v in out.values()) == len(spawn_type)
    assert sum(v[1] for v in out.values()) == int(landed.sum())


def test_book_ended_episodes_counts_only_previous_class():
    from types import SimpleNamespace

    # 7 envs: airborne landed, airborne failed, floor LANDED, edge,
    # platform, ballistic landed, ballistic on its FIRST episode (skipped).
    state = SimpleNamespace(
        episode_started=torch.tensor([True, True, True, True, True, True, False]),
        spawn_type=torch.tensor([
            js.SPAWN_AIRBORNE, js.SPAWN_AIRBORNE, js.SPAWN_FLOOR,
            js.SPAWN_EDGE, js.SPAWN_PLATFORM, js.SPAWN_BALLISTIC, js.SPAWN_BALLISTIC,
        ]),
        episode_landed=torch.tensor([True, False, True, False, False, True, True]),
    )
    env_ids = torch.arange(7)
    # Only env 0 counts for airborne; the floor-born landing (env 2) must NOT
    # leak into the airborne driver.
    assert js._book_ended_episodes(state, env_ids, js.SPAWN_AIRBORNE) == (2, 1)
    # Only env 5 counts for ballistic; env 6 is on its first episode.
    assert js._book_ended_episodes(state, env_ids, js.SPAWN_BALLISTIC) == (1, 1)


def test_book_ended_episodes_empty_when_no_class():
    from types import SimpleNamespace

    state = SimpleNamespace(
        episode_started=torch.ones(3, dtype=torch.bool),
        spawn_type=torch.tensor([js.SPAWN_FLOOR, js.SPAWN_EDGE, js.SPAWN_PLATFORM]),
        episode_landed=torch.ones(3, dtype=torch.bool),
    )
    assert js._book_ended_episodes(state, torch.arange(3), js.SPAWN_AIRBORNE) == (0, 0)
    assert js._book_ended_episodes(state, torch.arange(3), js.SPAWN_BALLISTIC) == (0, 0)


def test_rate_reporters():
    env = _StubEnv({})
    state = js._jump_step_state(env)
    assert js.jump_step_airborne_rate(env) == 0.0
    assert js.jump_step_ballistic_rate(env) == 0.0
    state.airborne_done = 10
    state.airborne_landed = 4
    state.ballistic_done = 8
    state.ballistic_landed = 1
    assert js.jump_step_airborne_rate(env) == 0.4
    assert js.jump_step_ballistic_rate(env) == 0.125
