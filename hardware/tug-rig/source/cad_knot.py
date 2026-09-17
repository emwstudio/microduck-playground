#!/usr/bin/env python3
"""Prusik-style ring knot for the tug rope (trimesh/manifold pipeline).

Per the user's hand sketch (IMG_3701): the rope wraps ONE full turn around
the ring's bar (the bight captures the ring), the tail runs back PARALLEL
to the standing rope (two clean legs), then coils twice around it. The
standing stub (x 4..14 mm) is Ø5.2 so the sim rope (Ø5) slides INTO it —
the rope tip hides inside the solid, no junction seam.

Knot frame (mm): rim point at origin, ring centre at (-8, 0, 0), standing
rope along +x, ring plate in the xz plane (hole axis = y).
Outputs: meshes/tug_knot_mm.stl + metre copies (and an x-mirrored pair).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

OUT = Path(__file__).resolve().parent.parent / "meshes"
ASSETS = (Path(__file__).resolve().parents[3]
          / "src/mjlab_microduck/robot/microduck/assets")

R = 2.6   # tube radius (mm) — Ø5.2, swallows the Ø5 rope


def knot_path() -> np.ndarray:
    segs = []
    phi = np.linspace(np.pi, np.pi + 4.0 * np.pi, 25)          # tail coils
    segs.append(np.stack([6.0 + (phi - np.pi) / (4.0 * np.pi) * 4.5,
                          4.2 * np.cos(phi), 4.2 * np.sin(phi)], axis=1))
    segs += [
        np.array([[10.8, -1.6, -0.5], [11.5, -2.4, -1.0]]),
        np.array([[10.0, -2.5, -0.5], [6.0, -2.5, -0.5],
                  [2.5, -2.4, -1.0]]),                          # tail leg
        np.array([[0.8, -2.2, -1.0], [-0.5, -3.2, 0.0]]),
    ]
    t = np.linspace(-0.5 * np.pi + 0.3, 1.5 * np.pi + 0.3, 40)  # the full turn
    segs.append(np.stack([-1.2 + 4.8 * np.cos(t),
                          4.5 * np.sin(t), np.zeros_like(t)], axis=1))
    segs.append(np.array([[0.5, 3.0, 0.0], [2.0, 1.5, 0.0], [4.0, 0.0, 0.0],
                          [7.0, 0.0, 0.0], [10.0, 0.0, 0.0], [14.0, 0.0, 0.0]]))
    pts = np.vstack(segs)
    # densify to ~0.8 mm steps so capsule unions are crease-free
    dense = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        d = float(np.linalg.norm(b - a))
        for s in np.linspace(0.0, 1.0, max(2, int(d / 0.6)), endpoint=False)[1:]:
            dense.append(a + s * (b - a))
        dense.append(b)
    return np.array(dense)


def build() -> trimesh.Trimesh:
    pts = knot_path()
    parts = [trimesh.creation.capsule(radius=R, height=float(np.linalg.norm(b - a)),
                                      count=[20, 12],
                                      transform=trimesh.transformations
                                      .translation_matrix((a + b) / 2.0)
                                      @ trimesh.geometry.align_vectors(
                                          [0, 0, 1], b - a))
             for a, b in zip(pts[:-1], pts[1:])]
    knot = trimesh.boolean.union(parts, engine="manifold")
    assert knot.is_volume and knot.is_watertight, "knot must be one solid"
    return knot


def clear_collar(knot: trimesh.Trimesh, clearance: float = 0.25) -> trimesh.Trimesh:
    """Subtract the collar (inflated by `clearance` mm) from the knot so the
    assembled pair can NEVER interpenetrate (the orange plate used to bleed
    through the rope — 穿模). The collar is placed in the knot frame for BOTH
    rim sides (+x and -x) so the base solid and its mirror are both safe.
    Trimmed spots read as the rope pressed against the hardware."""
    collar = trimesh.load(ASSETS / "tug_collar.stl")
    collar.apply_scale(1000.0)                      # assets are in metres
    eye = np.array([-68.6, 0.0, 16.0])              # back-eye centre (mm)
    shells = []
    for rim in (eye + np.array([8.0, 0, 0]), eye - np.array([8.0, 0, 0])):
        base = collar.copy()
        base.apply_translation(-rim)
        shell = base
        for d in ([clearance, 0, 0], [-clearance, 0, 0], [0, clearance, 0],
                  [0, -clearance, 0], [0, 0, clearance], [0, 0, -clearance]):
            t = base.copy()
            t.apply_translation(d)
            shell = trimesh.boolean.union([shell, t], engine="manifold")
        shells.append(shell)
    knot = trimesh.boolean.difference([knot] + shells, engine="manifold")
    knot = max(knot.split(only_watertight=False), key=lambda m: m.volume)
    assert knot.is_volume and knot.is_watertight
    return knot


def export_both(mesh: trimesh.Trimesh, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    mesh.export(OUT / f"{name}_mm.stl")
    m = mesh.copy()
    m.apply_scale(0.001)
    m.export(ASSETS / f"{name}.stl")
    # OBJ with cylindrical UVs so the sim can use the rope's own twist
    # texture on the knot — STL carries no UVs and the untextured solid
    # read as flat doodle-brown next to the hemp rope.
    uv = np.stack([m.vertices[:, 0] / 0.0175,   # twist pitch ≈ 3.5 rope
                   np.arctan2(m.vertices[:, 2], m.vertices[:, 1])  # diameters,
                   / (2.0 * np.pi)], axis=1)                      # as the span
    m.visual = trimesh.visual.TextureVisuals(uv=uv)
    m.export(ASSETS / f"{name}.obj", include_texture=True)


def main() -> None:
    knot = clear_collar(build())
    export_both(knot, "tug_knot")
    mir = knot.copy()
    mir.apply_transform(np.diag([-1.0, 1.0, 1.0, 1.0]))
    mir.invert()   # mirror flips winding — restore outward normals
    export_both(mir, "tug_knot_mir")
    print("knot:", knot.extents.round(2), "mm, watertight:", knot.is_watertight)


if __name__ == "__main__":
    main()
