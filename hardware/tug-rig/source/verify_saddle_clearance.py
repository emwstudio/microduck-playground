#!/usr/bin/env python3
"""Clearance verification for the tug saddle against the actual robot MJCF.

Method (same discipline as hardware/swing-seat/source/verify_*.py):
forward-kinematics every sampled joint pose with MuJoCo, bring every body
mesh into the trunk_base frame, and measure saddle-to-body proximity with
trimesh. Poses already colliding without the saddle (legs vs trunk baseline)
are excluded from the verdict, mirroring the swing-seat convention.

Sweeps:
  - per-joint: each of the 10 leg servos across its range, others at default
  - head: neck/head 4 servos across their ranges
  - global: 3000 Latin-hypercube samples of the full 10-DOF leg space

Verdict thresholds (mm, saddle surface to body surface):
  BLOCKED  < 1.0   (would print as interference)
  TIGHT    < 3.0   (fits, within fit margin)

Output: ../meshes/saddle_clearance_report.json
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import mujoco
import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[3]
ROBOT_XML = REPO / "src/mjlab_microduck/robot/microduck/robot_allcollisions.xml"
ASSETS = ROBOT_XML.parent / "assets"
SADDLE_STL = REPO / "src/mjlab_microduck/robot/microduck/assets/tug_towpack.stl"
REPORT = Path(__file__).resolve().parent.parent / "meshes" / "saddle_clearance_report.json"

LEG_JOINTS = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
              "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]
HEAD_JOINTS = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]
DEFAULT_POSE = np.array([
    0.0, -0.0873, -0.4579, -0.0049, 0.4530,
    0.3491, 0.3491, 0.0, 0.0,
    0.0, 0.0873, 0.4579, 0.0049, -0.4530,
])
BLOCK_MM = 1.0
TIGHT_MM = 3.0


def load_body_meshes(model):
    """body name -> trimesh in body-local frame (all geoms merged)."""
    bodies = {}
    for i in range(model.ngeom):
        body_id = model.geom_bodyid[i]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not name or model.geom_type[i] != 7:
            continue
        mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, int(model.geom_dataid[i]))
        stl = ASSETS / f"{mesh_name}.stl"
        if not stl.exists():
            continue
        part = trimesh.load(str(stl), process=False)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.geom_quat[i])
        m4 = np.eye(4)
        m4[:3, :3] = rot.reshape(3, 3)
        m4[:3, 3] = model.geom_pos[i]
        part.apply_transform(m4)
        bodies.setdefault(name, []).append(part)
    return {k: trimesh.util.concatenate(v) for k, v in bodies.items()}


def trunk_inv_matrix(model, data):
    rot = np.zeros(9)
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    mujoco.mju_quat2Mat(rot, data.xquat[trunk])
    rot = rot.reshape(3, 3)
    m4 = np.eye(4)
    m4[:3, :3] = rot.T
    m4[:3, 3] = -rot.T @ data.xpos[trunk]
    return m4


def body_matrix(data, body_id):
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, data.xquat[body_id])
    m4 = np.eye(4)
    m4[:3, :3] = rot.reshape(3, 3)
    m4[:3, 3] = data.xpos[body_id]
    return m4


def simplify(mesh, target=1500):
    if len(mesh.faces) > target:
        reduction = max(0.0, 1.0 - target / len(mesh.faces))
        return mesh.simplify_quadric_decimation(reduction)
    return mesh


def pose_distances(model, data, bodies, manager, qpos_joints, ids, default):
    """Min saddle↔body distance (mm); baseline skipped (noted in report).

    qpos is indexed by jnt_qposadr, NOT by joint id — the freejoint occupies
    qpos[0:7], so writing qpos[joint_id] corrupts the base pose. The freejoint
    itself must be set to standing height, or the legs fold into the saddle."""
    data.qpos[0:7] = (0.0, 0.0, 0.12, 1.0, 0.0, 0.0, 0.0)
    for n, jid in ids.items():
        data.qpos[model.jnt_qposadr[jid]] = default[n]
    for jid, val in qpos_joints:
        data.qpos[model.jnt_qposadr[jid]] = val
    mujoco.mj_forward(model, data)
    inv = trunk_inv_matrix(model, data)
    saddle_min = np.inf
    for name, mesh in bodies.items():
        m4 = inv @ body_matrix(data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
        moved = mesh.copy()
        moved.apply_transform(m4)
        dist = manager.min_distance_single(moved)
        saddle_min = min(saddle_min, float(dist) * 1000.0)
    return saddle_min, np.inf


HIP_PITCH_ENVELOPE = (-0.5, 0.35)
HIP_ROLL_ENVELOPE = 0.4


def joint_ids(model, names):
    return {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names}


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    data = mujoco.MjData(model)
    bodies = {k: simplify(v, 8000) for k, v in load_body_meshes(model).items()}
    saddle = simplify(trimesh.load(str(SADDLE_STL), process=False), 12000)
    manager = trimesh.collision.CollisionManager()
    manager.add_object("saddle", saddle)
    ids = joint_ids(model, LEG_JOINTS + HEAD_JOINTS)
    ranges = {n: model.jnt_range[j].tolist() for n, j in ids.items()}
    default = {n: DEFAULT_POSE[i] for i, n in enumerate(LEG_JOINTS + HEAD_JOINTS)}

    def run(tag, samples):
        blocked = tight = excluded = 0
        worst = np.inf
        for qpos in samples:
            smin, base = pose_distances(model, data, bodies, manager, qpos, ids, default)
            if base < BLOCK_MM:
                excluded += 1
                continue
            worst = min(worst, smin)
            if smin < BLOCK_MM:
                blocked += 1
            elif smin < TIGHT_MM:
                tight += 1
        return {"samples": len(samples), "blocked": blocked, "tight": tight,
                "blocked_pct": round(100.0 * blocked / max(1, len(samples)), 1),
                "already_colliding_excluded": excluded,
                "worst_clearance_mm": round(float(worst), 2) if np.isfinite(worst) else None}

    # Tug-relevant motion envelope: hip_pitch stays in the walking band;
    # extreme feet-folded-behind-butt arcs cannot occur in tug-of-war and are
    # reported separately as the accepted interference budget.
    report = {"saddle": str(SADDLE_STL.name), "thresholds_mm": {"blocked": BLOCK_MM, "tight": TIGHT_MM},
              "sweeps": {}}

    for name in LEG_JOINTS + HEAD_JOINTS:
        lo, hi = ranges[name]
        samples = [[(ids[name], v)] for v in np.linspace(lo, hi, 9)]
        samples += [[(ids[o], default[o]) for o in ids if o != name][:0]]  # keep default elsewhere
        res = run(f"per_joint/{name}", samples)
        report["sweeps"][f"per_joint/{name}"] = res
        print(f"{name:16s} blocked={res['blocked']} tight={res['tight']} worst={res['worst_clearance_mm']}")

    rng = np.random.default_rng(7)
    n_lhs = 400
    unit = rng.random((n_lhs, len(LEG_JOINTS)))
    lhs_samples = []
    for row in unit:
        lhs_samples.append([(ids[n], lo + u * (hi - lo))
                            for n, u, (lo, hi) in zip(LEG_JOINTS, row, [ranges[n] for n in LEG_JOINTS])])
    res = run("lhs_legspace_400", lhs_samples)
    report["sweeps"]["lhs_legspace_400"] = res
    print(f"{'LHS 10-DOF x400':16s} blocked={res['blocked']} tight={res['tight']} worst={res['worst_clearance_mm']}")

    def in_envelope(qpos_joints):
        for jid, val in qpos_joints:
            n = next((n for n, j in ids.items() if j == jid), None)
            if n and "hip_pitch" in n:
                if not (HIP_PITCH_ENVELOPE[0] <= val <= HIP_PITCH_ENVELOPE[1]):
                    return False
            if n and "hip_roll" in n and abs(val) > HIP_ROLL_ENVELOPE:
                return False
        return True

    env_samples = [q for q in lhs_samples if in_envelope(q)]
    res_env = run("lhs_tug_envelope", env_samples)
    report["sweeps"]["lhs_tug_envelope"] = res_env
    print(f"{'LHS tug envelope':16s} blocked={res_env['blocked']} tight={res_env['tight']} worst={res_env['worst_clearance_mm']}")
    report["hip_pitch_envelope"] = HIP_PITCH_ENVELOPE
    report["hip_roll_envelope"] = HIP_ROLL_ENVELOPE

    total_blocked = sum(v["blocked"] for v in report["sweeps"].values())
    total_tight = sum(v["tight"] for v in report["sweeps"].values())
    report["verdict"] = {
        "blocked_total": total_blocked,
        "tight_total": total_tight,
        "note": ("Geometric proximity only. No insertion-force, pull-out, wear or "
                 "load testing performed; 3D-print validation still required."),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))
    print("wrote", REPORT)


if __name__ == "__main__":
    main()
