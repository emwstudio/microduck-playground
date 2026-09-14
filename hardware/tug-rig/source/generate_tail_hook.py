#!/usr/bin/env python3
"""Parametric Microduck tug-of-war tail-hook assembly → STL assets.

Dimensions are measured off the robot (see src/mjlab_microduck/robot/
tug_of_war.py for the rope-side constants): butt shell face at local
x ≈ -41 mm, rope line at world z ≈ 85 mm (trunk z 115 mm → local z -30 mm),
16 mm rope with clearance → 30×20 mm carabiner opening.

Outputs (trunk-local coordinates):
  meshes/tug_hook_steel_mm.stl       base plate + standoff tube, millimetres
  meshes/tug_hook_carabiner_mm.stl   carabiner ring + gate, millimetres
  ../../../src/mjlab_microduck/robot/microduck/assets/tug_hook_steel.stl
  ../../../src/mjlab_microduck/robot/microduck/assets/tug_hook_carabiner.stl
  (same geometry in metres, for MuJoCo)
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

# --- duck measurements (local frame, millimetres) ---
BUTT_SHELL_X = -41.0          # butt shell face
PLATE_Z = -18.0               # plate centre height on the shell
EYE_X = -65.0                 # carabiner eye centre (3 cm aft of the shell)
EYE_Z = -30.0                 # eye height → rope line at world z ≈ 85 mm

# --- carabiner dimensions ---
OPENING_W = 30.0              # opening across (y)
OPENING_H = 20.0              # opening tall (z)
TUBE_R = 1.75                 # ring tube radius
GATE_R = 1.2                  # gate bar radius
STUD_R = 2.5                  # standoff tube radius

OUT_DIR = Path(__file__).resolve().parent.parent / "meshes"
ASSET_DIR = (Path(__file__).resolve().parents[3]
             / "src/mjlab_microduck/robot/microduck/assets")


def write_stl(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normals /= norms
    with open(path, "wb") as fh:
        fh.write(b"\0" * 80)
        fh.write(struct.pack("<I", len(faces)))
        for i in range(len(faces)):
            fh.write(struct.pack("<12fH", *normals[i], *v0[i], *v1[i], *v2[i], 0))


def tube_along(path_pts: np.ndarray, radius: float, sides: int = 10):
    """Sweep a circular tube along a polyline; returns (verts, faces)."""
    n = len(path_pts)
    tangents = np.gradient(path_pts, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    ref = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    flip = np.abs((tangents * ref).sum(axis=1)) > 0.9
    ref[flip] = np.array([1.0, 0.0, 0.0])
    u = ref - (ref * tangents).sum(axis=1, keepdims=True) * tangents
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    w = np.cross(tangents, u)
    betas = 2.0 * np.pi * np.arange(sides) / sides
    verts = []
    for i in range(n):
        for j in range(sides):
            verts.append(path_pts[i] + radius * (np.cos(betas[j]) * u[i] + np.sin(betas[j]) * w[i]))
    faces = []
    for i in range(n - 1):
        for j in range(sides):
            j2 = (j + 1) % sides
            a, b = i * sides + j, (i + 1) * sides + j
            c, d = (i + 1) * sides + j2, i * sides + j2
            faces += [(a, b, c), (a, c, d)]
    return np.array(verts), np.array(faces)


def box(center, size):
    cx, cy, cz = center
    sx, sy, sz = (s_ / 2.0 for s_ in size)
    corners = np.array([[cx + dx * sx, cy + dy * sy, cz + dz * sz]
                        for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)])
    idx = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    faces = []
    for a, b, c, d in idx:
        faces += [(a, b, c), (a, c, d)]
    return corners, np.array(faces)


def carabiner_mesh():
    """D-ring (obround) in the y-z plane + gate bar along the spine."""
    seg = 56
    hw, hh = OPENING_W / 2.0, OPENING_H / 2.0
    ang = 2.0 * np.pi * np.arange(seg) / seg
    # stadium outline: straight sides at ±(hw-hh) along y, semicircle caps
    cy = (hw - hh) * np.tanh(8.0 * np.cos(ang))
    outline = np.stack([np.zeros(seg), cy + hh * np.cos(ang), hh * np.sin(ang)], axis=1)
    outline += np.array([EYE_X, 0.0, EYE_Z])
    verts_r, faces_r = tube_along(outline, TUBE_R, sides=8)
    parts = [(verts_r, faces_r)]
    # gate: bar across the spine side of the opening (y = -(hw-hh) .. +(hw-hh))
    gate = np.stack([
        np.array([EYE_X, -(hw - hh), EYE_Z]),
        np.array([EYE_X, (hw - hh), EYE_Z]),
    ])
    verts_g, faces_g = tube_along(gate, GATE_R, sides=6)
    parts.append((verts_g, faces_g))
    return parts


# Waist frame: rigid steel loop clamped around the torso by the harness.
# Torso cross-section at hip height: back x ≈ -47, front x ≈ +14, y ±47 mm.
FRAME_CENTER = np.array([-12.0, 0.0, -5.0])   # local, frame plane at hip height
FRAME_HALF_Y = 58.0                            # 5 mm clearance off the shell
FRAME_HALF_X = 36.0                            # covers back -48 .. front +24
FRAME_TUBE_R = 1.5
BUTT_EYE = np.array([EYE_X, 0.0, EYE_Z])       # (-65, 0, -30)
CHEST_EYE = np.array([30.0, 0.0, EYE_Z])


def frame_mesh():
    """Rigid oval waist frame (stadium loop in x-y at hip height) with
    front/rear stems carrying the two carabiner eyes."""
    seg = 56
    ang = 2.0 * np.pi * np.arange(seg) / seg
    cx = FRAME_CENTER[0] + (FRAME_HALF_X - FRAME_HALF_Y) * 0.0 * np.cos(ang)
    x = FRAME_CENTER[0] + FRAME_HALF_X * np.cos(ang)
    y = FRAME_HALF_Y * np.sin(ang)
    outline = np.stack([x, y, np.full(seg, FRAME_CENTER[2])], axis=1)
    parts = [tube_along(outline, FRAME_TUBE_R, sides=8)]
    # rear stem: frame back edge → butt carabiner eye
    back_x = FRAME_CENTER[0] - FRAME_HALF_X
    parts.append(tube_along(np.stack([
        np.array([back_x, 0.0, FRAME_CENTER[2]]),
        BUTT_EYE,
    ]), STUD_R, sides=8))
    # front stem: frame front edge → chest carabiner eye
    front_x = FRAME_CENTER[0] + FRAME_HALF_X
    parts.append(tube_along(np.stack([
        np.array([front_x, 0.0, FRAME_CENTER[2]]),
        CHEST_EYE,
    ]), STUD_R, sides=8))
    return parts


def steel_mesh():
    return frame_mesh()


def merge(parts):
    verts, faces, offset = [], [], 0
    for v, f in parts:
        verts.append(v)
        faces.append(f + offset)
        offset += len(v)
    return np.concatenate(verts), np.concatenate(faces)


def chest_carabiner_mesh():
    parts = carabiner_mesh()
    return [(v - np.array([EYE_X - CHEST_EYE[0], 0.0, 0.0]), f) for v, f in parts]


def main() -> None:
    for name, parts in (("tug_hook_steel", steel_mesh()),
                        ("tug_hook_carabiner", carabiner_mesh()),
                        ("tug_chest_carabiner", chest_carabiner_mesh()),):
        verts, faces = merge(parts)
        write_stl(OUT_DIR / f"{name}_mm.stl", verts, faces)
        write_stl(ASSET_DIR / f"{name}.stl", verts * 0.001, faces)
        print(f"wrote {name}: {len(verts)} verts, {len(faces)} tris "
              f"(mm → {OUT_DIR}, m → {ASSET_DIR})")


if __name__ == "__main__":
    main()
