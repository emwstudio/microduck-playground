#!/usr/bin/env python3
"""Measure the Microduck torso cross-section for the clamp collar.

Cuts triangle-plane sections through EVERY mesh attached to trunk_base in
robot_allcollisions.xml (the upstream Onshape export), across the band's
height, and reduces each 5° angular bin to the OUTERMOST radius — the
envelope the collar band must clear. Result: ../torso_contour.json
(72-bin polar contour in trunk-local coordinates, metres).

Lesson baked in: mesh VERTICES are too sparse to measure a section (STL
triangles are large); slice the TRIANGLES. And unreferenced vertices lie —
only face-referenced geometry is real.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

ROBOT_XML = Path(__file__).resolve().parents[3] / "src/mjlab_microduck/robot/microduck/robot_allcollisions.xml"
OUT = Path(__file__).resolve().parent.parent / "torso_contour.json"

Z_C = 0.018          # band centre height (mid purple shell: z 0.0003..0.042)
BAND_H = 0.014       # band height — sections cover [Z_C-H/2, Z_C+H/2]
N_BINS = 72


def trunk_meshes() -> list[tuple[np.ndarray, np.ndarray]]:
    model = mujoco.MjSpec.from_file(str(ROBOT_XML)).compile()
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    out = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != trunk or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[g]
        va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        v = model.mesh_vert[va:va + 3 * vn].reshape(-1, 3)
        f = model.mesh_face[fa:fa + 3 * fn].reshape(-1, 3)
        if len(f) == 0 or f.max() >= len(v):
            continue
        m = np.zeros(9)
        mujoco.mju_quat2Mat(m, model.geom_quat[g])
        vw = v @ m.reshape(3, 3).T + model.geom_pos[g]
        out.append((vw, f))
    return out


def section_points(meshes, z0: float) -> np.ndarray:
    pts = []
    for v, f in meshes:
        tri = v[f]
        z = tri[:, :, 2]
        for i, j in ((0, 1), (1, 2), (2, 0)):
            zi, zj = z[:, i], z[:, j]
            cross = (zi < z0) != (zj < z0)
            if cross.any():
                t = (z0 - zi[cross]) / (zj[cross] - zi[cross])
                p = tri[cross, i] + t[:, None] * (tri[cross, j] - tri[cross, i])
                pts.append(p[:, :2])
    return np.vstack(pts) if pts else np.zeros((0, 2))


def main() -> None:
    meshes = trunk_meshes()
    zs = np.linspace(Z_C - BAND_H / 2 + 0.0005, Z_C + BAND_H / 2 - 0.0005, 7)
    P = np.vstack([section_points(meshes, z0) for z0 in zs])
    ang = np.arctan2(P[:, 1], P[:, 0])
    rad = np.hypot(P[:, 0], P[:, 1])
    bins = np.linspace(-np.pi, np.pi, N_BINS + 1)
    cont = []
    for i in range(N_BINS):
        m = (ang >= bins[i]) & (ang < bins[i + 1])
        cont.append(float(rad[m].max()) if m.any() else None)
    centers = (bins[:-1] + bins[1:]) / 2
    OUT.write_text(json.dumps({"angle": centers.tolist(), "r": cont, "z_c": Z_C}))
    r = np.array([c for c in cont if c is not None])
    print(f"wrote {OUT.name}: {len(P)} section pts, "
          f"r front={cont[0]:.4f} back={cont[N_BINS // 2]:.4f} "
          f"min={r.min():.4f} max={r.max():.4f}")


if __name__ == "__main__":
    main()
