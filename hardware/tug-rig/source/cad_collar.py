#!/usr/bin/env python3
"""Clamp-collar tow harness for the Microduck tug rig (trimesh pipeline).

The user's design: ONE collar band wrapped around the torso shell, split
on one side (+y) with a screw through two lugs that tightens the band's
grip on the shell — a hose-clamp / shaft-collar principle. A closed tow
eye is integrated at the FRONT (chest pull point) and one at the BACK
(butt pull point); the whole tug force path flows through the collar, so
no back panel, standoffs or webbing strap are needed.

The band follows the measured shell contour (rounded box: front +30 mm,
back -47 mm, hips ±47 mm) with ~1.5 mm clamping allowance at band height
(z = -10 mm, clears the hip joints and the legs' swing).

Outputs: meshes/tug_collar_mm.stl (one piece: band + lugs + 2 eyes),
meshes/tug_clamp_screw_mm.stl (M3 screw + nut), plus metre copies in the
sim assets dir.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

# --- measured duck dimensions (metres, trunk-local) ---
XF, XB, HW = 0.034, 0.048, 0.049   # band centreline: front / back / side reach
Z_C = -0.010                        # band centre height (mid torso)
BAND_H, BAND_T = 0.014, 0.0022     # band cross-section: vertical × radial
GAP_HALF = 0.11                     # split half-angle at the +y side (~9 mm gap)

# --- clamp hardware ---
LUG_W, LUG_OUT, LUG_H = 0.008, 0.007, 0.012
SCREW_R, SCREW_LEN = 0.0015, 0.020
HEAD_R, HEAD_H = 0.0028, 0.003
NUT_R, NUT_H = 0.0028, 0.0024

# --- tow eyes ---
EYE_MAJOR, EYE_TUBE = 0.005, 0.0023
EYE_FRONT = np.array([XF + BAND_T / 2 + 0.004, 0.0, Z_C])
EYE_BACK = np.array([-(XB + BAND_T / 2 + 0.004), 0.0, Z_C])

OUT = Path(__file__).resolve().parent.parent / "meshes"
ASSETS = Path(__file__).resolve().parents[3] / "src/mjlab_microduck/robot/microduck/assets"


def band_path(n: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Open loop around the shell, ends facing each other across the +y gap."""
    ang = np.linspace(np.pi / 2 + GAP_HALF, np.pi / 2 + 2 * np.pi - GAP_HALF, n)
    a = (XF + XB) / 2 + (XF - XB) / 2 * np.cos(ang)
    x, y = a * np.cos(ang), HW * np.sin(ang)
    tang = np.gradient(np.stack([x, y], axis=1), ang, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    outward = np.stack([tang[:, 1], -tang[:, 0]], axis=1)
    pts = np.stack([x, y, np.full_like(x, Z_C)], axis=1)
    n3 = np.stack([outward[:, 0], outward[:, 1], np.zeros(n)], axis=1)
    return pts, n3


def band_mesh() -> trimesh.Trimesh:
    pts, n3 = band_path()
    n = len(pts)
    zhat = np.tile(np.array([[0.0, 0.0, 1.0]]), (n, 1))
    rings = np.concatenate([
        pts + (BAND_T / 2) * n3 + (BAND_H / 2) * zhat,
        pts + (BAND_T / 2) * n3 - (BAND_H / 2) * zhat,
        pts - (BAND_T / 2) * n3 - (BAND_H / 2) * zhat,
        pts - (BAND_T / 2) * n3 + (BAND_H / 2) * zhat,
    ])
    faces = []
    for i in range(n - 1):
        for k in range(4):
            k2 = (k + 1) % 4
            faces.append((k * n + i, k * n + i + 1, k2 * n + i + 1))
            faces.append((k * n + i, k2 * n + i + 1, k2 * n + i))
    for end in (0, n - 1):   # end caps
        faces.append((end, n + end, 2 * n + end))
        faces.append((end, 2 * n + end, 3 * n + end))
    band = trimesh.Trimesh(rings, np.array(faces), process=False)
    trimesh.repair.fix_normals(band)
    return band


def lugs() -> trimesh.Trimesh:
    """Two ears at the split ends, protruding +y, screw hole along x."""
    pts, _ = band_path(2)
    blocks = []
    for end in pts:
        lug = trimesh.creation.box(extents=(LUG_W, LUG_OUT + BAND_T, LUG_H))
        lug.apply_translation((end[0], end[1] + LUG_OUT / 2, Z_C))
        blocks.append(lug)
    return trimesh.boolean.union(blocks, engine="manifold")


def tow_eye(center: np.ndarray) -> trimesh.Trimesh:
    ring = trimesh.creation.torus(major_radius=EYE_MAJOR, minor_radius=EYE_TUBE,
                                  major_sections=40, minor_sections=12)
    ring.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], [1, 0, 0]))
    ring.apply_translation(center)
    return ring


def clamp_screw() -> trimesh.Trimesh:
    """M3 screw (head + shaft) and hex nut through the lugs, axis along x."""
    _, n3 = band_path(2)
    y = (band_path(2)[0][0, 1]) + LUG_OUT / 2
    shaft = trimesh.creation.cylinder(radius=SCREW_R, height=SCREW_LEN, sections=16)
    head = trimesh.creation.cylinder(radius=HEAD_R, height=HEAD_H, sections=16)
    head.apply_translation((0, 0, SCREW_LEN / 2 + HEAD_H / 2))
    nut = trimesh.creation.cylinder(radius=NUT_R, height=NUT_H, sections=6)
    nut.apply_translation((0, 0, -(SCREW_LEN / 2 + NUT_H / 2)))
    screw = trimesh.boolean.union([shaft, head, nut], engine="manifold")
    screw.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], [1, 0, 0]))
    screw.apply_translation((0.0, y, Z_C))
    return screw


def export_both(mesh: trimesh.Trimesh, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    mm = mesh.copy()
    mm.apply_scale(1000.0)
    mm.export(OUT / f"{name}_mm.stl")
    mesh.export(ASSETS / f"{name}.stl")
    print(f"{name}: {len(mesh.vertices)} verts, watertight={mesh.is_watertight}")


def main() -> None:
    collar = trimesh.boolean.union(
        [band_mesh(), lugs(), tow_eye(EYE_FRONT), tow_eye(EYE_BACK)],
        engine="manifold")
    trimesh.repair.fix_normals(collar)
    export_both(collar, "tug_collar")
    export_both(clamp_screw(), "tug_clamp_screw")
    print(f"one-piece collar: {len(collar.vertices)} verts, "
          f"{len(collar.faces)} faces, watertight={collar.is_watertight}, "
          f"volume={abs(collar.volume) * 1e3:.1f} cm^3")
    print("front eye at", EYE_FRONT.tolist(), " back eye at", EYE_BACK.tolist())


if __name__ == "__main__":
    main()
