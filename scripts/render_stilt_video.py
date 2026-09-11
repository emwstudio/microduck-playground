#!/usr/bin/env python3
"""Render a trained stilt policy to MP4 — headless CPU MuJoCo rehearsal.

Reuses infer_policy.PolicyInference (61D obs contract, projected gravity,
new_cmd_obs) with the compile-time stilt morphology from stilt_constants.
Actuators are BAM M6 (voltage-controlled XL330), matching training — plain
XML PD falls over (verified: 10cm policy face-planted at 2.5 s without BAM).

BAM constants mirror _BAM_ACTUATOR_KWARGS in robot/microduck_constants.py
(same values as microduck_rl/scripts/infer_policy.py).

Usage (cwd = third_party/microduck-playground):
    PYTHONPATH=src uv run --no-sync python scripts/render_stilt_video.py \
        --policy ../../artifacts/stilts_repro/10cm/policy.onnx \
        --height-cm 10 --blend 0.5 --duration 5 \
        --out ../../artifacts/stilts_repro/videos/stilt_h10cm.mp4
"""

import argparse
import importlib.util
import math
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


infer = _load_module("infer_policy", REPO_ROOT / "scripts" / "infer_policy.py")

from mjlab_microduck.robot.stilt_constants import get_stilt_walk_spec  # noqa: E402

SPAWN_Z_BASE = 0.125  # walk-robot reset height midpoint; stilt length added on top
SIM_TIMESTEP = 0.005
DECIMATION = 4        # 0.005 * 4 = 0.02 s = 50 Hz control, as trained

BAM_MOTOR_NAME = "xl330"
BAM_MODEL = "m6"
BAM_KP_FW = 200.0
BAM_VIN = 7.4           # midpoint of the (6.5, 8.2) training DR
BAM_VIN_DROP_GAIN = 0.1  # midpoint of (0.0, 0.2)
BAM_VIN_MIN = 6.0
BAM_STIFF_SOLREF_FRICTION = (-5.0e4, -2.0e2)
BAM_STIFF_SOLIMP_FRICTION = (0.99, 0.9999, 0.001, 0.5, 2.0)


def build_model(height_cm: float, blend: float):
    """Compile the stilt scene with BAM torque motors (mirrors BamActuator.edit_spec)."""
    import mujoco
    from bam.model import load_model
    from bam.mujoco import MujocoController

    bam_model = load_model(motor_name=BAM_MOTOR_NAME, model=BAM_MODEL)
    bam_model.actuator.kp = BAM_KP_FW
    bam_model.actuator.vin = BAM_VIN
    bam_model.actuator.max_current = None  # training runs without the limiter

    spec = get_stilt_walk_spec(height_cm=height_cm, blend=blend)
    spec.worldbody.add_light(
        pos=[0, 0, 3.5], dir=[0, 0, -1],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
    )
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],
        rgba=[0.35, 0.35, 0.38, 1.0],
    )

    force_limit = bam_model.actuator.vin * bam_model.kt.value / bam_model.R.value
    names = []
    for act in spec.actuators:
        tgt = act.target
        tgt_name = tgt.name if hasattr(tgt, "name") else str(tgt)
        if tgt_name.startswith("passive_"):
            continue
        act.set_to_motor()
        act.forcelimited = True
        act.forcerange = (-force_limit, force_limit)
        act.ctrllimited = False
        act.gear = [1.0, 0, 0, 0, 0, 0]
        names.append(act.name)
        for joint in spec.joints:
            if joint.name == tgt_name:
                joint.damping = np.zeros((3, 1))
                joint.frictionloss = 0.0
                joint.solref_friction = BAM_STIFF_SOLREF_FRICTION
                joint.solimp_friction = BAM_STIFF_SOLIMP_FRICTION
                break

    model = spec.compile()
    model.opt.timestep = SIM_TIMESTEP
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 720
    data = mujoco.MjData(model)
    bam_ctrl = MujocoController(
        bam_model, names, model, data,
        vin_drop_gain=BAM_VIN_DROP_GAIN, vin_min=BAM_VIN_MIN,
    )
    print(f"BAM {BAM_MODEL} on {len(names)} joints: vin={BAM_VIN}V kp_fw={BAM_KP_FW:.0f} "
          f"forcerange=+/-{force_limit:.3f}Nm")
    return model, data, bam_ctrl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--height-cm", type=float, required=True)
    ap.add_argument("--blend", type=float, default=0.5)
    ap.add_argument("--speed", type=float, default=0.15)
    ap.add_argument("--duration", type=float, default=5.0, help="walking seconds (after 1 s settle)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import imageio.v2 as imageio
    import mujoco

    height_m = args.height_cm * 0.01
    model, data, bam_ctrl = build_model(args.height_cm, args.blend)

    policy = infer.PolicyInference(
        model,
        data,
        walking_onnx_path=str(args.policy),
        new_cmd_obs=True,
        use_projected_gravity=True,
    )

    fj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qa = int(model.jnt_qposadr[fj])
    data.qpos[qa + 0] = 0.0
    data.qpos[qa + 1] = 0.0
    data.qpos[qa + 2] = SPAWN_Z_BASE + height_m
    data.qpos[qa + 3 : qa + 7] = [1, 0, 0, 0]
    for i, idx in enumerate(policy.joint_qpos_indices):
        data.qpos[idx] = policy.default_pose[i]
    bam_ctrl.reset(data.qpos)
    bam_ctrl.q_target[:] = policy.default_pose
    mujoco.mj_forward(model, data)

    control_dt = DECIMATION * model.opt.timestep
    fps = int(round(1.0 / control_dt))
    renderer = mujoco.Renderer(model, height=720, width=1280)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = 90.0
    cam.elevation = -8.0
    cam.distance = max(0.55, 0.45 + 2.0 * height_m)  # same framing as env cfg viewer

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(args.out), fps=fps, macro_block_size=1)

    start_x = float(data.qpos[qa])
    fell_at = None

    def step(cmd_x):
        policy.vel_cmd = np.array([cmd_x, 0.0, 0.0], dtype=np.float32)
        policy._update_command()
        action = policy.infer()
        policy.apply_action(action)
        bam_ctrl.q_target[:] = data.ctrl
        for _ in range(DECIMATION):
            bam_ctrl.update()
            mujoco.mj_step(model, data)

    def render():
        cam.lookat[:] = [float(data.qpos[qa]), float(data.qpos[qa + 1]), 0.10 + 0.55 * height_m]
        renderer.update_scene(data, camera=cam)
        writer.append_data(renderer.render())

    def tilt_deg():
        w, x, y, z = data.qpos[qa + 3 : qa + 7]
        return math.degrees(math.acos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))

    for _ in range(int(round(1.0 / control_dt))):
        step(0.0)
        render()

    t = 0.0
    for _ in range(int(round(args.duration / control_dt))):
        step(args.speed)
        render()
        t += control_dt
        if fell_at is None and tilt_deg() > 60.0:
            fell_at = t
            print(f"  ** FELL at t={t:.2f}s (tilt>60 deg)")

    writer.close()
    dist = float(data.qpos[qa]) - start_x
    fell = "no" if fell_at is None else f"@{fell_at:.2f}s"
    print(
        f"h={args.height_cm:g}cm blend={args.blend}  avg_speed={dist / args.duration:.3f} m/s"
        f"  fell={fell}  -> {args.out}"
    )


if __name__ == "__main__":
    main()
