#!/usr/bin/env python3
"""CAD-quality Microduck tug-of-war harness rig (trimesh/manifold pipeline).

Design language follows hardware/swing-seat: a single moulded part with
smooth transitions, not welded primitives.

Part: an elliptical waist band (stadium cross-section with a rounded
profile) clamped on the torso by the webbing harness, carrying a tapered
rear boss (butt carabiner eye) and a front boss (chest carabiner eye).
All dimensions are measured off the robot, see tug_of_war.py constants:
torso hip section back x=-47 / front x=+14 / y=±47 mm, rope line z≈85 mm
world (trunk local z=-30 mm).

Outputs (millimetres, trunk-local):
  meshes/tug_frame_mm.stl        waist band + bosses (painted steel look)
  meshes/tug_carabiner_mm.stl    D-ring carabiner with gate (anodized)
and metre-scale copies for MuJoCo in src/.../assets/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import Point, Polygon
from trimesh.creation import extrude_polygon, sweep_polygon

# --- measured duck dimensions (mm, trunk-local) ---
FRAME_Z = -5.0                 # band centre height (hip)
BAND_HALF_W = 56.0             # y half-width (shell ±47 + 9 mm clearance)
BAND_HALF_D = 34.0             # x half-depth around frame centre
FRAME_CX = -11.0               # band centre x (covers back -45 / front +23)
BAND_H = 9.0                   # band height
BAND_WALL = 3.2                # band wall thickness
BUTT_EYE = np.array([-65.0, 0.0, -30.0])
CHEST_EYE = np.array([30.0, 0.0, -30.0])
EYE_D = 9.0                    # carabiner eye inner diameter
EYE_TUBE = 2.2                 # carabiner tube radius
BOSS_TAPER = 0.55              # boss narrows toward the eye

OUT = Path(__file__).resolve().parent.parent / "meshes"
ASSETS = Path(__file__).resolve().parents[3] / "src/mjlab_microduck/robot/microduck/assets"


def stadium(half_w: float, half_d: float, r: float, n: int = 64) -> Polygon:
    """Stadium (obround) outline in x-y: two semicircles joined by straights."""
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = []
    for a in ang:
        ca, sa = np.cos(a), np.sin(a)
        cx = np.clip(ca * half_d, -(half_d - r), (half_d - r))
        pts.append((cx + np.sign(ca) * (half_d - r) + r * ca - np.sign(ca) * (half_d - r),
                    sa * half_w))
    return Polygon(pts)


def band() -> trimesh.Trimesh:
    """Elliptical waist band: stadium outline extruded, hollowed, edges eased."""
    outer = stadium(BAND_HALF_W, BAND_HALF_D, BAND_HALF_D * 0.8)
    inner = stadium(BAND_HALF_W - BAND_WALL, BAND_HALF_D - BAND_WALL, (BAND_HALF_D - BAND_WALL) * 0.8)
    shell = extrude_polygon(outer, BAND_H)
    hole = extrude_polygon(inner, BAND_H * 1.2)
    hole.apply_translation([0, 0, -BAND_H * 0.1])
    band_mesh = trimesh.boolean.difference([shell, hole], engine="manifold")
    band_mesh.apply_translation([FRAME_CX, 0, FRAME_Z - BAND_H / 2])
    return band_mesh


def boss(base_xy: tuple[float, float], eye: np.ndarray, base_r: float) -> trimesh.Trimesh:
    """Tapered boss blending from the band to a carabiner eye lug."""
    base = np.array([base_xy[0], base_xy[1], FRAME_Z])
    eye_pt = np.array(eye)
    vec = eye_pt - base
    length = float(np.linalg.norm(vec))
    cone = trimesh.creation.cone(radius=base_r, height=length, sections=24)
    # cone axis is +z from origin; aim it base → eye
    cone.apply_translation([0, 0, 0])
    z = np.array([0.0, 0.0, 1.0])
    axis = vec / length
    rot = trimesh.geometry.align_vectors(z, axis)
    cone.apply_transform(rot)
    cone.apply_translation(base)
    # lug: flattened sphere at the eye with a hole (subtract later)
    lug = trimesh.creation.icosphere(subdivisions=2, radius=base_r * 1.05)
    lug.apply_scale([1.0, 0.9, 0.9])
    lug.apply_translation(eye_pt)
    return trimesh.boolean.union([cone, lug], engine="manifold")


def carabiner() -> trimesh.Trimesh:
    """D-ring carabiner: rounded-rect ring swept with a circular profile +
    gate bar, as a single solid (hole via a subtraction cylinder)."""
    ring_r = EYE_D / 2 + EYE_TUBE
    ring = trimesh.creation.torus(major_radius=ring_r, minor_radius=EYE_TUBE,
                                  major_sections=40, minor_sections=12)
    # gate across the opening on one side
    gate = trimesh.creation.cylinder(radius=EYE_TUBE * 0.55, height=EYE_D, sections=12)
    gate.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], [0, 1, 0]))
    gate.apply_translation([0, -ring_r + EYE_TUBE, 0])
    return trimesh.boolean.union([ring, gate], engine="manifold")


def export_both(mesh: trimesh.Trimesh, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    mesh.export(OUT / f"{name}_mm.stl")
    m = mesh.copy()
    m.apply_scale(0.001)
    m.export(ASSETS / f"{name}.stl")
    print(f"{name}: {len(mesh.vertices)} verts, {len(mesh.faces)} tris, watertight={mesh.is_watertight}")


def frame_mesh() -> trimesh.Trimesh:
    parts = [band()]
    parts.append(boss((FRAME_CX - BAND_HALF_D + 2, 0.0), BUTT_EYE, 6.0))
    parts.append(boss((FRAME_CX + BAND_HALF_D - 2, 0.0), CHEST_EYE, 5.0))
    frame = trimesh.boolean.union(parts, engine="manifold")
    trimesh.repair.fix_normals(frame)
    return frame


def main() -> None:
    frame = frame_mesh()
    export_both(frame, "tug_frame")
    cb = carabiner()
    butt_cb = cb.copy()
    butt_cb.apply_translation(BUTT_EYE)
    export_both(butt_cb, "tug_hook_carabiner")
    chest_cb = cb.copy()
    chest_cb.apply_translation(CHEST_EYE)
    export_both(chest_cb, "tug_chest_carabiner")


if __name__ == "__main__":
    main()
