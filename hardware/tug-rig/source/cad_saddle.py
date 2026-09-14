#!/usr/bin/env python3
"""Form-fitted butt saddle for the Microduck tug rig (trimesh pipeline).

Unlike the generic elliptical band, this saddle is a NEGATIVE FORM of the
robot's actual back shell: the trunk mesh is voxelised, dilated outward to
make the 2.5 mm-fit inner face and the 3 mm wall, cut open at the front and
bottom, and capped with an integral tow-eye lug at the hip-back corner.
It cups the butt and wraps the hip flanks, the way the swing seat cradles
the robot. Output: meshes/tug_saddle_mm.stl (+ metre-scale asset).
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from scipy import ndimage

PITCH = 0.0012          # voxel size, m (robot STLs are metre-scale)
GAP_VOX = 6             # fit clearance ≈ 7.2 mm (smoothing eats ~3 mm)
WALL_VOX = 3            # wall ≈ 3.6 mm
Z_LO, Z_HI = -0.040, 0.030   # saddle height band (trunk-local, m)
X_CUT = 0.004           # open face: keep everything behind this x
EYE = np.array([-0.065, 0.0, -0.030])

OUT = Path(__file__).resolve().parent.parent / "meshes"
ASSETS = Path(__file__).resolve().parents[3] / "src/mjlab_microduck/robot/microduck/assets"
ROBOT_XML = (Path(__file__).resolve().parents[3]
             / "src/mjlab_microduck/robot/microduck/robot_allcollisions.xml")


def trunk_shell() -> trimesh.Trimesh:
    """The robot's trunk_base meshes merged, in trunk-local coordinates."""
    model = mujoco.MjSpec.from_file(str(ROBOT_XML)).compile()
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    parts = []
    for i in range(model.ngeom):
        if model.geom_bodyid[i] != trunk or model.geom_type[i] != 7:
            continue
        mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH,
                                      int(model.geom_dataid[i]))
        stl = ROBOT_XML.parent / "assets" / f"{mesh_name}.stl"
        if not stl.exists():
            continue
        part = trimesh.load(str(stl), process=False)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.geom_quat[i])
        # apply geom local transform: rotate then translate
        m4 = np.eye(4)
        m4[:3, :3] = rot.reshape(3, 3)
        m4[:3, 3] = model.geom_pos[i]
        part.apply_transform(m4)
        parts.append(part)
    return trimesh.util.concatenate(parts)


def voxel_shell(mesh: trimesh.Trimesh, dilate: int) -> tuple[np.ndarray, np.ndarray]:
    vox = mesh.voxelized(PITCH)
    grid = vox.matrix.copy()
    if dilate:
        grid = ndimage.binary_dilation(grid, iterations=dilate)
    return grid, vox.transform


def from_voxels(grid: np.ndarray, transform) -> trimesh.Trimesh:
    vg = trimesh.voxel.VoxelGrid(grid, transform)
    mesh = vg.marching_cubes.copy()
    mesh.apply_transform(transform)   # this trimesh returns index-space verts
    return mesh


def main() -> None:
    shell = trunk_shell()
    region = shell.copy()
    region.apply_translation([0, 0, 0])
    outer_grid, tf = voxel_shell(region, GAP_VOX + WALL_VOX)
    inner_grid, _ = voxel_shell(region, GAP_VOX)
    wall = outer_grid & ~inner_grid

    # open the front face and the bottom (legs), keep hip band only
    coords = np.indices(wall.shape).reshape(3, -1).T
    world = trimesh.voxel.ops.multibox  # noqa — just to keep import obvious
    pts = coords * PITCH + tf[:3, 3]
    # Keep only the back cup: |y| within the hip shell, behind x=-8 mm.
    # Wrapping further forward wraps the hip-leg attachment and blocks the
    # legs (the swing seat's "open leg corridors" lesson).
    keep = ((pts[:, 0] < -0.008) & (np.abs(pts[:, 1]) < 0.045)
            & (pts[:, 2] > Z_LO) & (pts[:, 2] < Z_HI))
    mask = np.zeros(wall.shape, dtype=bool)
    flat = mask.reshape(-1)
    flat[:] = keep
    wall &= mask

    saddle = from_voxels(wall, tf)
    saddle = trimesh.smoothing.filter_laplacian(saddle, iterations=1, lamb=0.2)

    # open-leg corridors (swing-seat lesson): slots over both hip pivots so
    # the leg roots, hip bearings and ankles swing free.
    corridors = []
    for side in (1.0, -1.0):
        # slot covers the hip pivot cone: rear of the flank, from below the
        # pivot to the top rim — measured from blocking contact points
        corridor = trimesh.creation.box(extents=(0.066, 0.040, 0.080))
        corridor.apply_translation((-0.019, side * 0.022, -0.009))
        corridors.append(corridor)
    saddle = trimesh.boolean.difference([saddle] + corridors, engine="manifold")

    # integral tow-eye lug at the hip-back corner
    lug = trimesh.creation.cylinder(radius=0.010, height=0.010, sections=24)
    lug.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], [1, 0, 0]))
    lug.apply_translation(EYE)
    saddle = trimesh.boolean.union([saddle, lug], engine="manifold")

    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    mm = saddle.copy()
    mm.apply_scale(1000.0)          # printable millimetre version
    mm.export(OUT / "tug_saddle_mm.stl")
    saddle.export(ASSETS / "tug_saddle.stl")   # metres, as generated
    print(f"saddle: {len(saddle.vertices)} verts, {len(saddle.faces)} tris, "
          f"watertight={saddle.is_watertight}, "
          f"bounds={np.round(saddle.bounds, 3).tolist()}")


if __name__ == "__main__":
    main()
