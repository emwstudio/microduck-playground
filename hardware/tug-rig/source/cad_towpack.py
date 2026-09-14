#!/usr/bin/env python3
"""Standoff tow-pack for the Microduck tug rig (trimesh pipeline).

Design language: a jetpack-style tow-pack, not a shrink-wrapped shell. A
crowned back panel floats behind the butt on two standoff rails — the legs'
fold-back swing space (measured: leg tips reach x ≈ -50 mm) sits entirely
INSIDE the 25 mm gap between shell (x=-47) and panel (x=-75), so clearance
is guaranteed by construction, no corridors, no shrink-wrap compromises.

The panel's back face carries the carabiner eye (the pulled end).

Outputs: meshes/tug_towpack_mm.stl, assets/tug_towpack.stl (metres),
plus the carabiner positioned at the pack eye.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

# --- measured duck dimensions (metres, trunk-local) ---
SHELL_BACK_X = -0.047
PANEL_X = -0.075              # 28 mm standoff; leg tips stop at -50 mm
PANEL_W, PANEL_H, PANEL_T = 0.058, 0.048, 0.004
RAIL_R = 0.003
RAIL_Y, RAIL_Z = 0.026, -0.020
EYE = np.array([PANEL_X - PANEL_T / 2 - 0.006, 0.0, -0.010])
EYE_D, EYE_TUBE = 0.009, 0.0022

OUT = Path(__file__).resolve().parent.parent / "meshes"
ASSETS = Path(__file__).resolve().parents[3] / "src/mjlab_microduck/robot/microduck/assets"


def rounded_panel() -> trimesh.Trimesh:
    from shapely.geometry import Polygon
    r = 0.010
    w, h = PANEL_W / 2, PANEL_H / 2
    pts = []
    for cx, cy, a0 in ((w - r, h - r, 0), (-(w - r), h - r, 90),
                       (-(w - r), -(h - r), 180), (w - r, -(h - r), 270)):
        for a in np.linspace(np.radians(a0), np.radians(a0 + 90), 12):
            pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    panel = trimesh.creation.extrude_polygon(Polygon(pts), PANEL_T)
    panel.apply_translation((PANEL_X, 0.0, -0.010 - PANEL_T / 2))
    return panel


STRAP_W, STRAP_T = 0.020, 0.002     # 20 mm velcro strap cross-section


def rail(side: float) -> trimesh.Trimesh:
    rail = trimesh.creation.cylinder(radius=RAIL_R, height=abs(PANEL_X - SHELL_BACK_X) + 0.006, sections=16)
    rail.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], [1, 0, 0]))
    rail.apply_translation(((SHELL_BACK_X + PANEL_X) / 2, side * RAIL_Y, RAIL_Z))
    # contoured locating pad: a stubby disc keyed to the hip-shell corner,
    # keeps the pack from rotating on the smooth shell
    pad = trimesh.creation.cylinder(radius=0.010, height=0.004, sections=24)
    pad.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], [1, 0, 0]))
    pad.apply_translation((SHELL_BACK_X - 0.002, side * RAIL_Y, RAIL_Z))
    return trimesh.boolean.union([rail, pad], engine="manifold")


def strap_slots(pack: trimesh.Trimesh) -> trimesh.Trimesh:
    """Vertical slots at the panel's side edges for the 20 mm velcro strap."""
    cutters = []
    for side in (1.0, -1.0):
        slot = trimesh.creation.box(extents=(PANEL_T + 0.004, STRAP_T + 0.001, STRAP_W + 0.001))
        slot.apply_translation((PANEL_X, side * (PANEL_W / 2 - 0.006), -0.010))
        cutters.append(slot)
    return trimesh.boolean.difference([pack] + cutters, engine="manifold")


def carabiner(eye: np.ndarray) -> trimesh.Trimesh:
    ring_r = EYE_D / 2 + EYE_TUBE
    ring = trimesh.creation.torus(major_radius=ring_r, minor_radius=EYE_TUBE,
                                  major_sections=40, minor_sections=12)
    gate = trimesh.creation.cylinder(radius=EYE_TUBE * 0.55, height=EYE_D, sections=12)
    gate.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], [0, 1, 0]))
    gate.apply_translation([0, -ring_r + EYE_TUBE, 0])
    cb = trimesh.boolean.union([ring, gate], engine="manifold")
    cb.apply_transform(trimesh.geometry.align_vectors([1, 0, 0], [0, 0, 1]))
    cb.apply_translation(eye)
    return cb


def export_both(mesh: trimesh.Trimesh, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    mm = mesh.copy()
    mm.apply_scale(1000.0)
    mm.export(OUT / f"{name}_mm.stl")
    mesh.export(ASSETS / f"{name}.stl")
    print(f"{name}: {len(mesh.vertices)} verts, watertight={mesh.is_watertight}")


def strap_mesh() -> trimesh.Trimesh:
    """Flat 20 mm velcro loop around the waist, threaded through the slots.

    Belt geometry: 20 mm width runs VERTICAL (matches the slots, which are
    cut for a belt threaded through them), 2 mm thick radially. The back
    arc passes through the slot plane (x = PANEL_X), front and sides hug
    the torso shell (back -47 mm, front +30 mm, hips ±47 mm).
    """
    xf, xb, hw, zc = 0.035, abs(PANEL_X) + 0.003, 0.050, -0.010
    n = 96
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    a = (xf + xb) / 2 + (xf - xb) / 2 * np.cos(ang)
    x, y = a * np.cos(ang), hw * np.sin(ang)
    pts = np.stack([x, y, np.full_like(x, zc)], axis=1)
    tang = np.gradient(np.stack([x, y], axis=1), ang, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    outward = np.stack([tang[:, 1], -tang[:, 0]], axis=1)   # radial, horizontal
    zhat = np.tile(np.array([[0.0, 0.0, 1.0]]), (n, 1))
    n3 = np.stack([outward[:, 0], outward[:, 1], np.zeros(n)], axis=1)
    verts = np.concatenate([
        pts + (STRAP_T / 2) * n3 + (STRAP_W / 2) * zhat,
        pts + (STRAP_T / 2) * n3 - (STRAP_W / 2) * zhat,
        pts - (STRAP_T / 2) * n3 - (STRAP_W / 2) * zhat,
        pts - (STRAP_T / 2) * n3 + (STRAP_W / 2) * zhat,
    ])
    faces = []
    for i in range(n):
        j = (i + 1) % n
        for k in range(4):
            k2 = (k + 1) % 4
            faces.append((k * n + i, k * n + j, k2 * n + j))
            faces.append((k * n + i, k2 * n + j, k2 * n + i))
    loop = trimesh.Trimesh(verts, np.array(faces), process=False)
    trimesh.repair.fix_normals(loop)
    return loop


def main() -> None:
    parts = [rounded_panel(), rail(1.0), rail(-1.0), carabiner(EYE)]
    pack = trimesh.boolean.union(parts, engine="manifold")
    pack = strap_slots(pack)
    trimesh.repair.fix_normals(pack)
    export_both(pack, "tug_towpack")
    export_both(strap_mesh(), "tug_strap")
    print(f"one-piece pack: {len(pack.vertices)} verts, "
          f"{len(pack.faces)} faces, watertight={pack.is_watertight}, "
          f"volume={abs(pack.volume) * 1e3:.1f} cm^3")
    print("pack eye at", EYE.tolist())


if __name__ == "__main__":
    main()
