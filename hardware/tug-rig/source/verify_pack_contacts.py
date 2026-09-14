#!/usr/bin/env python3
"""Tow-pack clearance verification using MuJoCo's own collision engine.

The pack is welded to trunk_base with contype=1 and a 3 mm contact margin;
every sampled pose is forward-kinematicsed and real mj_contacts are counted.
No homebrew mesh proximity, no frame bookkeeping — the numbers are exactly
what the simulator sees.

Contacts are split into:
  MOUNTING  pack geoms touching trunk_base (the intended rail-feet interface)
  BLOCKED   pack geoms touching any other body (collision)
  TIGHT     same but only inside the 3 mm margin (near miss)

Sweeps: per-joint (14 servos x 9 values) + 400 LHS samples of leg space.
Tug-of-war envelope: hip_pitch in [-0.5, 0.35], |hip_roll| <= 0.4 rad
(walking is ±0.3 / small roll; fold-back arcs are the documented budget).

Output: ../meshes/pack_contact_report.json
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[3]
ROBOT_XML = REPO / "src/mjlab_microduck/robot/microduck/robot_allcollisions.xml"
PACK_STL = REPO / "src/mjlab_microduck/robot/microduck/assets/tug_towpack.stl"
REPORT = Path(__file__).resolve().parent.parent / "meshes" / "pack_contact_report.json"

LEG_JOINTS = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
              "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]
HEAD_JOINTS = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]
DEFAULT_POSE = np.array([
    0.0, -0.0873, -0.4579, -0.0049, 0.4530,
    0.3491, 0.3491, 0.0, 0.0,
    0.0, 0.0873, 0.4579, 0.0049, -0.4530,
])
PITCH_ENV = (-0.5, 0.35)
ROLL_ENV = 0.4
MARGIN = 0.003


def build_model() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
    spec.add_mesh(name="verify_pack", file=str(PACK_STL))
    trunk = next(b for b in spec.bodies if b.name == "trunk_base")
    trunk.add_geom(
        name="verify_pack_geom",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="verify_pack",
        contype=1,
        conaffinity=0,          # matches nothing by default; pair via condim below
        density=0.0,
        margin=MARGIN,
    )
    # robot geoms: let them collide with the pack (condim-3 vs condim-3)
    for geom in spec.geoms:
        if geom.name != "verify_pack_geom":
            geom.condim = 3
    spec.geoms[-1].condim = 3
    return spec.compile()


def classify_contacts(model, data, pack_geom_id, trunk_id):
    mounting = blocked = tight = 0
    for i in range(data.ncon):
        c = data.contact[i]
        g = {int(c.geom1), int(c.geom2)}
        if pack_geom_id not in g:
            continue
        other = (g - {pack_geom_id}).pop()
        if model.geom_bodyid[other] == trunk_id:
            mounting += 1
        elif c.dist < 0.0002:
            blocked += 1
        else:
            tight += 1
    return mounting, blocked, tight


def main() -> None:
    model = build_model()
    data = mujoco.MjData(model)
    pack_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "verify_pack_geom")
    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    ids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
           for n in LEG_JOINTS + HEAD_JOINTS}
    ranges = {n: model.jnt_range[j].tolist() for n, j in ids.items()}

    def evaluate(samples):
        out = {"samples": len(samples), "mounting": 0, "blocked": 0, "tight": 0,
               "blocked_poses": 0}
        for qpos in samples:
            data.qpos[0:7] = (0.0, 0.0, 0.12, 1.0, 0.0, 0.0, 0.0)
            for i, n in enumerate(LEG_JOINTS + HEAD_JOINTS):
                data.qpos[model.jnt_qposadr[ids[n]]] = DEFAULT_POSE[i]
            for jid, val in qpos:
                data.qpos[model.jnt_qposadr[jid]] = val
            mujoco.mj_forward(model, data)
            m, b, t = classify_contacts(model, data, pack_geom_id, trunk_id)
            out["mounting"] += m
            out["blocked"] += b
            out["tight"] += t
            if b:
                out["blocked_poses"] += 1
        out["blocked_pose_pct"] = round(100.0 * out["blocked_poses"] / max(1, len(samples)), 1)
        return out

    report = {"pack": PACK_STL.name, "margin_mm": MARGIN * 1000,
              "envelope": {"hip_pitch": PITCH_ENV, "hip_roll_abs": ROLL_ENV},
              "sweeps": {}}

    for name in LEG_JOINTS + HEAD_JOINTS:
        lo, hi = ranges[name]
        samples = [[(ids[name], v)] for v in np.linspace(lo, hi, 9)]
        res = evaluate(samples)
        report["sweeps"][f"per_joint/{name}"] = res
        print(f"{name:16s} blocked_poses={res['blocked_poses']} contacts(b/t)={res['blocked']}/{res['tight']}")

    rng = np.random.default_rng(7)
    unit = rng.random((400, len(LEG_JOINTS)))
    lhs = [[(ids[n], lo + u * (hi - lo))
            for n, u, (lo, hi) in zip(LEG_JOINTS, row, [ranges[n] for n in LEG_JOINTS])]
           for row in unit]
    res = evaluate(lhs)
    report["sweeps"]["lhs_legspace_400"] = res
    print(f"{'LHS x400':16s} blocked_poses={res['blocked_poses']} ({res['blocked_pose_pct']}%)")

    env = [q for q in lhs
           if all(PITCH_ENV[0] <= v <= PITCH_ENV[1] for j, v in q
                  if "hip_pitch" in next(n for n, jj in ids.items() if jj == j))
           and all(abs(v) <= ROLL_ENV for j, v in q
                   if "hip_roll" in next(n for n, jj in ids.items() if jj == j))]
    res_env = evaluate(env)
    report["sweeps"]["lhs_tug_envelope"] = res_env
    print(f"{'LHS envelope':16s} blocked_poses={res_env['blocked_poses']} ({res_env['blocked_pose_pct']}%)")

    report["note"] = ("MuJoCo-native contacts. MOUNTING counts pack rail-feet vs "
                      "trunk_base — the intended attachment interface, not a fault. "
                      "No insertion-force, pull-out, wear or load testing performed.")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))
    print("wrote", REPORT)


if __name__ == "__main__":
    main()
