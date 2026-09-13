"""Structural and physics tests for the rigid-arm 360-degree swing task."""

import math
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from mjlab_microduck.robot.microduck_constants import (
    SWING360_PIVOT_DAMPING,
    SWING360_PIVOT_FRICTIONLOSS,
    SWING360_ROD_LENGTH,
    SWING_ANCHOR_HEIGHT,
    SWING_ATTACHMENT_Z,
    SWING_BOTTOM_TRUNK_Z,
    get_swing360_spec,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_swing360_env_cfg import (
    MicroduckSwing360RlCfg,
    make_microduck_swing360_env_cfg,
)

_PIVOT_TO_TRUNK = SWING360_ROD_LENGTH + SWING_ATTACHMENT_Z


def _trunk_pose_for_pivot(theta: float) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """World position and quaternion of the welded trunk at a pivot angle."""
    pos = (
        -_PIVOT_TO_TRUNK * math.sin(theta),
        0.0,
        SWING_ANCHOR_HEIGHT - _PIVOT_TO_TRUNK * math.cos(theta),
    )
    quat = (math.cos(theta / 2.0), 0.0, math.sin(theta / 2.0), 0.0)
    return pos, quat


def _set_hanging_state(model: mujoco.MjModel, data: mujoco.MjData, theta: float) -> None:
    """Place the freejoint trunk and the pivot hinge consistently at theta."""
    pivot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "passive_swing_pivot")
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    pos, quat = _trunk_pose_for_pivot(theta)
    root_adr = model.jnt_qposadr[root_id]
    data.qpos[root_adr : root_adr + 3] = pos
    data.qpos[root_adr + 3 : root_adr + 7] = quat
    data.qpos[model.jnt_qposadr[pivot_id]] = theta
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _pivot_angle(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    pivot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "passive_swing_pivot")
    return float(data.qpos[model.jnt_qposadr[pivot_id]])


def test_swing360_model_uses_hinged_rigid_arm_and_weld() -> None:
    model = get_swing360_spec().compile()
    assert model.ntendon == 0
    assert model.neq == 1
    assert model.eq_type[0] == mujoco.mjtEq.mjEQ_WELD
    obj1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.eq_obj1id[0])
    obj2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.eq_obj2id[0])
    assert (obj1, obj2) == ("swing360_arm", "trunk_base")

    pivot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "passive_swing_pivot")
    assert pivot_id >= 0
    assert model.jnt_type[pivot_id] == mujoco.mjtJoint.mjJNT_HINGE
    np.testing.assert_allclose(model.jnt_axis[pivot_id], (0.0, 1.0, 0.0), atol=1e-12)
    assert not model.jnt_limited[pivot_id]
    pivot_dof = model.jnt_dofadr[pivot_id]
    assert math.isclose(model.dof_damping[pivot_dof], SWING360_PIVOT_DAMPING)
    assert math.isclose(model.dof_frictionloss[pivot_dof], SWING360_PIVOT_FRICTIONLOSS)

    arm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "swing360_arm")
    np.testing.assert_allclose(model.body_pos[arm_id], (0.0, 0.0, SWING_ANCHOR_HEIGHT), atol=1e-12)
    for geom_name in ("swing360_arm_shaft", "swing360_arm_crossbar"):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        assert geom_id >= 0
        assert model.geom_contype[geom_id] == 0
        assert model.geom_conaffinity[geom_id] == 0


def test_hanging_spawn_is_still_and_welded() -> None:
    model = get_swing360_spec().compile()
    data = mujoco.MjData(model)
    _set_hanging_state(model, data, 0.0)
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    np.testing.assert_allclose(
        data.qpos[model.jnt_qposadr[root_id] : model.jnt_qposadr[root_id] + 3],
        (0.0, 0.0, SWING_BOTTOM_TRUNK_Z),
        atol=1e-9,
    )
    for _ in range(1000):  # two seconds
        mujoco.mj_step(model, data)
    # The seated CoM sits a few millimetres ahead of the pivot axis, so the
    # arm settles a fraction of a degree off vertical rather than at exactly
    # zero; what matters is that it stays hung and comes to rest.
    assert abs(_pivot_angle(model, data)) < 0.02
    pivot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "passive_swing_pivot")
    assert abs(float(data.qvel[model.jnt_dofadr[pivot_id]])) < 0.05


def test_rigid_pendulum_swings_through_bottom_and_weld_holds() -> None:
    model = get_swing360_spec().compile()
    data = mujoco.MjData(model)
    theta0 = 0.5
    _set_hanging_state(model, data, theta0)

    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    crossed_bottom = False
    max_return = 0.0
    for step in range(1500):  # three seconds
        mujoco.mj_step(model, data)
        theta = _pivot_angle(model, data)
        if theta < 0.0:
            crossed_bottom = True
        if crossed_bottom:
            max_return = max(max_return, theta)
        if step % 50 == 0:
            expected_pos, expected_quat = _trunk_pose_for_pivot(theta)
            np.testing.assert_allclose(
                data.qpos[model.jnt_qposadr[root_id] : model.jnt_qposadr[root_id] + 3],
                expected_pos,
                atol=3e-3,  # soft weld tolerance
            )
            actual_quat = data.qpos[model.jnt_qposadr[root_id] + 3 : model.jnt_qposadr[root_id] + 7]
            assert abs(float(np.dot(actual_quat, expected_quat))) > math.cos(math.radians(0.5))

    assert crossed_bottom, "pendulum never swung through the bottom"
    # Bearing loss is small: the return arc must recover most of the release.
    assert max_return > 0.6 * theta0


def test_swing360_frontier_pays_only_new_ground(monkeypatch) -> None:
    angle = torch.zeros(2)
    rate = torch.zeros(2)
    monkeypatch.setattr(
        microduck_mdp,
        "_swing360_kinematics",
        lambda env, asset_cfg: (angle, rate),
    )
    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        episode_length_buf=torch.tensor([2, 2]),
        step_dt=0.02,
    )

    def pay() -> torch.Tensor:
        return microduck_mdp.swing360_frontier_progress(env)

    # The rate cap (8 rad/s * 0.02 s = 0.16 rad per step) always applies, so
    # a fast jump is only paid its first 0.16 rad and the rest is forfeited.
    angle[:] = 0.5
    first = pay()
    torch.testing.assert_close(first, torch.full((2,), (0.16 / math.pi) ** 2 / 0.02))
    # Retracting and re-reaching an old frontier pays nothing.
    angle[:] = 0.3
    torch.testing.assert_close(pay(), torch.zeros(2))
    angle[:] = -0.5  # absolute value: the other side is also old ground
    torch.testing.assert_close(pay(), torch.zeros(2))
    # New ground pays the squared-frontier delta from the actual frontier.
    angle[:] = -0.7
    expected = ((0.66 / math.pi) ** 2 - (0.5 / math.pi) ** 2) / 0.02
    torch.testing.assert_close(pay(), torch.full((2,), expected))
    # A teleport is rate-capped: only max_paid_rate * step_dt is paid.
    angle[:] = 100.0
    capped = pay()
    expected_capped = ((0.86 / math.pi) ** 2 - (0.7 / math.pi) ** 2) / 0.02
    torch.testing.assert_close(capped, torch.full((2,), expected_capped))
    # A fresh episode resets the frontier.
    env.episode_length_buf[:] = 1
    angle[:] = 0.1
    torch.testing.assert_close(pay(), torch.full((2,), (0.1 / math.pi) ** 2 / 0.02))


def test_swing360_frontier_beyond_pi_is_linear_not_quadratic(monkeypatch) -> None:
    angle = torch.zeros(2)
    rate = torch.zeros(2)
    monkeypatch.setattr(
        microduck_mdp,
        "_swing360_kinematics",
        lambda env, asset_cfg: (angle, rate),
    )
    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        episode_length_buf=torch.tensor([2, 2]),
        step_dt=0.02,
    )
    # Jump the frontier past pi (paid cap forfeits the jump itself).
    angle[:] = 3.3
    microduck_mdp.swing360_frontier_progress(env)
    # One capped step of new ground past pi pays the tangent rate, not (f/pi)^2.
    angle[:] = 3.5
    paid = microduck_mdp.swing360_frontier_progress(env)
    expected_linear = 2.0 * (0.16 / math.pi) / 0.02
    torch.testing.assert_close(paid, torch.full((2,), expected_linear))
    # Far past pi the value function stays linear instead of exploding:
    # at f = 10 rad the quadratic would be ~10.1, the tangent caps at ~5.4.
    far = torch.tensor([10.0])
    value = microduck_mdp._swing360_frontier_value(far)
    torch.testing.assert_close(value, torch.tensor([1.0 + 2.0 * (10.0 / math.pi - 1.0)]))
    assert float(value) < 0.6 * float((far / math.pi) ** 2)


def test_swing360_frontier_spawn_floor_blocks_free_arc_pay(monkeypatch) -> None:
    angle = torch.zeros(2)
    rate = torch.zeros(2)
    monkeypatch.setattr(
        microduck_mdp,
        "_swing360_kinematics",
        lambda env, asset_cfg: (angle, rate),
    )
    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        episode_length_buf=torch.tensor([1, 1]),
        step_dt=0.02,
        _swing360_spawn_floor=torch.tensor([2.0, 0.0]),
    )
    # env0 spawned at 2.0 rad: swinging at or below the spawn arc pays zero.
    angle[:] = torch.tensor([1.5, 0.1])
    paid = microduck_mdp.swing360_frontier_progress(env)
    assert float(paid[0]) == 0.0
    torch.testing.assert_close(paid[1], torch.tensor((0.1 / math.pi) ** 2 / 0.02))
    # Progress BEYOND the spawn arc earns the delta from the floor.
    angle[:] = torch.tensor([2.2, 0.1])
    paid = microduck_mdp.swing360_frontier_progress(env)
    expected = ((2.16 / math.pi) ** 2 - (2.0 / math.pi) ** 2) / 0.02
    torch.testing.assert_close(paid[0], torch.tensor(expected))


def test_swing360_task_preserves_actor_contract_and_exact_reset() -> None:
    cfg = make_microduck_swing360_env_cfg()
    assert cfg.scene.env_spacing == 0.0
    assert cfg.viewer.origin_type == cfg.viewer.OriginType.WORLD
    actor_terms = cfg.observations["actor"].terms
    assert "base_lin_vel" not in actor_terms
    assert actor_terms["head_command"].params["dim"] == 4
    assert actor_terms["body_command"].params["dim"] == 6
    assert actor_terms["body_command"].func is microduck_mdp.zero_command_padding
    assert actor_terms["command"].func is microduck_mdp.swing_plane_heading_observation
    for group in (cfg.observations["actor"], cfg.observations["critic"]):
        for term_name in ("joint_pos", "joint_vel"):
            assert group.terms[term_name].params["asset_cfg"].joint_names == (
                r"^(?!passive_).*",
            )
    assert (
        cfg.observations["critic"].terms["body_command"].func
        is microduck_mdp.swing360_state_observation
    )
    assert (
        cfg.observations["critic"].terms["command"].func
        is microduck_mdp.swing360_frontier_observation
    )
    assert "swing360_state" not in actor_terms

    assert cfg.rewards["swing360_frontier_progress"].weight == 112.0
    assert cfg.rewards["swing360_frontier_progress"].params == {
        "max_paid_rate": 8.0,
    }
    assert cfg.rewards["swing360_height"].weight == 8.0
    assert cfg.rewards["swing360_energy"].weight == 0.5
    assert cfg.rewards["dof_pos_limits"].weight == -1.0
    assert cfg.rewards["action_rate_l2"].weight == -0.03
    assert cfg.rewards["joint_torques_l2"].weight == -0.001
    # No cord-era terms may leak into the rigid-arm task.
    for name in cfg.rewards:
        assert "string" not in name
        assert "lateral" not in name
        assert "alignment" not in name

    assert MicroduckSwing360RlCfg.experiment_name == "microduck_swing360"
    assert MicroduckSwing360RlCfg.algorithm.learning_rate == 1.0e-4
    assert MicroduckSwing360RlCfg.algorithm.class_name.endswith(
        ":SwingPlanarCorrectionPPO"
    )

    reset = cfg.events["reset_base"]
    assert reset.params["pose_range"] == {}
    assert reset.params["velocity_range"] == {}
    assert cfg.events["arc_spawn"].params == {
        "probability": 0.0,
        "max_angle": 2.97,
    }
    spawn_stages = cfg.curriculum["arc_spawn_probability"].params["probability_stages"]
    assert spawn_stages[0] == {"step": 0, "probability": 0.0}
    assert spawn_stages[-1]["probability"] == 0.25
    play_cfg = make_microduck_swing360_env_cfg(play=True)
    assert play_cfg.events["arc_spawn"].params["probability"] == 0.0
    command = cfg.commands["twist"]
    assert command.ranges.lin_vel_x == (0.0, 0.0)
    assert command.ranges.lin_vel_y == (0.0, 0.0)
    assert command.ranges.ang_vel_z == (0.0, 0.0)
    assert "fell_over" not in cfg.terminations
    assert make_microduck_swing360_env_cfg(play=True).seed == 72


def test_swing360_nominal_actuator_limit_mode_fixes_conservative_midpoints(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MICRODUCK_SWING_NOMINAL_ACTUATOR", "1")
    nominal = make_microduck_swing360_env_cfg()
    actuator = nominal.scene.entities["robot"].articulation.actuators[0]
    assert actuator.vin_range == (7.35, 7.35)
    assert actuator.vin_drop_gain_range == (0.10, 0.10)
    assert actuator.delay_min_lag == 5
    assert actuator.delay_max_lag == 5

    monkeypatch.delenv("MICRODUCK_SWING_NOMINAL_ACTUATOR")
    randomized = make_microduck_swing360_env_cfg()
    randomized_actuator = randomized.scene.entities["robot"].articulation.actuators[0]
    assert randomized_actuator.vin_range == (6.5, 8.2)
    assert randomized_actuator.vin_drop_gain_range == (0.0, 0.2)
    assert randomized_actuator.delay_min_lag == 3
    assert randomized_actuator.delay_max_lag == 6
