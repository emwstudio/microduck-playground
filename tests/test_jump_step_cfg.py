"""Cfg-invariant tests for Mjlab-JumpStep-MicroDuck (v9 airborne curriculum).

The jump_step task lives in the desk-climb snapshot tree
(experiments/desk-climb/source/src), not the main src tree.  Extend the
already-imported mjlab_microduck.tasks package path so the module resolves
from the snapshot while its own imports keep resolving from the main tree
(no sys.modules swapping — zero contamination of the other test modules).
"""

from pathlib import Path

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
    assert tp["airborne_spawn_prob"] == 0.6
    assert tp["edge_spawn_prob"] == 0.2
    assert tp["platform_spawn_prob"] == 0.0
    pp = play.events["reset_jump_step"].params
    assert pp["airborne_spawn_prob"] == 0.0
    assert pp["edge_spawn_prob"] == 0.0
    assert pp["platform_spawn_prob"] == 0.0
    assert "airborne_spawn" in train.curriculum
    assert "airborne_spawn" not in play.curriculum


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


def test_airborne_constants_and_stage_table():
    assert js.AIRBORNE_CLEARANCE_RANGE == (0.08, 0.15)
    assert js.SPAWN_AIRBORNE == 3
    assert js.SPAWN_AIRBORNE not in (js.SPAWN_FLOOR, js.SPAWN_EDGE, js.SPAWN_PLATFORM)
    rates = [s["rate"] for s in js.AIRBORNE_CURRICULUM]
    probs = [s["prob"] for s in js.AIRBORNE_CURRICULUM]
    assert rates == sorted(rates, reverse=True)
    assert probs == sorted(probs)
    assert probs[0] == 0.0 and probs[-1] == 0.6


def test_reset_signature_defaults():
    import inspect

    sig = inspect.signature(js.reset_jump_step)
    assert sig.parameters["airborne_spawn_prob"].default == 0.6
    assert sig.parameters["platform_spawn_prob"].default == 0.0
    assert sig.parameters["edge_spawn_prob"].default == 0.2
    assert sig.parameters["airborne_clearance_range"].default == (0.08, 0.15)


class _StubEventManager:
    def __init__(self, prob):
        self._term = type("Term", (), {"params": {"airborne_spawn_prob": prob}})()

    def get_term_cfg(self, name):
        assert name == "reset_jump_step"
        return self._term


class _StubEnv:
    """Just enough surface for _jump_step_state + the curriculum function."""

    def __init__(self, prob):
        self.num_envs = 4
        self.device = "cpu"
        self.event_manager = _StubEventManager(prob)


def _run_curriculum(env, done, landed, min_episodes=200):
    state = js._jump_step_state(env)
    state.airborne_done = done
    state.airborne_landed = landed
    return js.jump_step_airborne_curriculum(env, min_episodes=min_episodes)


def test_airborne_curriculum_below_min_episodes_noop():
    env = _StubEnv(0.6)
    assert _run_curriculum(env, done=100, landed=100) == 0.6
    state = js._jump_step_state(env)
    assert state.airborne_done == 100  # window untouched


def test_airborne_curriculum_stage_transitions():
    # rate 0.8 >= 0.7 -> retire airborne spawns entirely
    env = _StubEnv(0.6)
    assert _run_curriculum(env, done=250, landed=200) == 0.0
    state = js._jump_step_state(env)
    assert state.airborne_done == 0 and state.airborne_landed == 0  # window reset
    assert env.event_manager.get_term_cfg("reset_jump_step").params["airborne_spawn_prob"] == 0.0
    # rate 0.55 -> 0.2
    env = _StubEnv(0.4)
    assert _run_curriculum(env, done=200, landed=110) == 0.2
    # rate 0.4 -> 0.4
    env = _StubEnv(0.6)
    assert _run_curriculum(env, done=200, landed=80) == 0.4
    # rate 0.1 -> stays 0.6 (highest tier)
    env = _StubEnv(0.6)
    assert _run_curriculum(env, done=200, landed=20) == 0.6


def test_airborne_curriculum_bidirectional_and_window():
    # Low rate at an advanced stage re-introduces airborne spawns.
    env = _StubEnv(0.2)
    assert _run_curriculum(env, done=300, landed=10) == 0.6
    # Rate matching the CURRENT tier: no param write, window keeps accumulating.
    env = _StubEnv(0.6)
    assert _run_curriculum(env, done=250, landed=20) == 0.6
    state = js._jump_step_state(env)
    assert state.airborne_done == 250  # not reset — same stage


# --- spawn-class accounting (the 2026-09-24 label-swap regression) ----------------


def test_spawn_class_outcomes_no_cross_talk():
    import torch

    spawn_type = torch.tensor([
        js.SPAWN_FLOOR, js.SPAWN_FLOOR, js.SPAWN_EDGE,
        js.SPAWN_PLATFORM, js.SPAWN_AIRBORNE, js.SPAWN_AIRBORNE, js.SPAWN_AIRBORNE,
    ])
    landed = torch.tensor([False, True, True, False, True, False, True])
    out = js.spawn_class_outcomes(spawn_type, landed)
    assert out["floor"] == (2, 1)
    assert out["edge"] == (1, 1)
    assert out["platform"] == (1, 0)
    assert out["airborne"] == (3, 2)
    # Every episode is accounted exactly once.
    assert sum(v[0] for v in out.values()) == len(spawn_type)
    assert sum(v[1] for v in out.values()) == int(landed.sum())


def test_book_ended_episodes_counts_only_previous_airborne():
    import torch
    from types import SimpleNamespace

    # 6 envs: two airborne (one landed), one floor (landed!), one edge,
    # one platform, one airborne on its FIRST episode (must be skipped).
    state = SimpleNamespace(
        episode_started=torch.tensor([True, True, True, True, True, False]),
        spawn_type=torch.tensor([
            js.SPAWN_AIRBORNE, js.SPAWN_AIRBORNE, js.SPAWN_FLOOR,
            js.SPAWN_EDGE, js.SPAWN_PLATFORM, js.SPAWN_AIRBORNE,
        ]),
        episode_landed=torch.tensor([True, False, True, False, False, True]),
    )
    env_ids = torch.arange(6)
    done, landed = js._book_ended_episodes(state, env_ids)
    # Only env 0 counts as a landed airborne episode; the floor-born landing
    # (env 2) must NOT leak into the airborne driver, and the first-episode
    # env 5 is skipped entirely.
    assert (done, landed) == (2, 1)


def test_book_ended_episodes_empty_when_no_airborne():
    import torch
    from types import SimpleNamespace

    state = SimpleNamespace(
        episode_started=torch.ones(3, dtype=torch.bool),
        spawn_type=torch.tensor([js.SPAWN_FLOOR, js.SPAWN_EDGE, js.SPAWN_PLATFORM]),
        episode_landed=torch.ones(3, dtype=torch.bool),
    )
    done, landed = js._book_ended_episodes(state, torch.arange(3))
    assert (done, landed) == (0, 0)


def test_airborne_rate_reporter():
    env = _StubEnv(0.6)
    state = js._jump_step_state(env)
    assert js.jump_step_airborne_rate(env) == 0.0  # empty window
    state.airborne_done = 10
    state.airborne_landed = 4
    assert js.jump_step_airborne_rate(env) == 0.4
    train = js.make_microduck_jump_step_env_cfg(play=False)
    assert "airborne_rate" in train.curriculum
    play = js.make_microduck_jump_step_env_cfg(play=True)
    assert "airborne_rate" not in play.curriculum
