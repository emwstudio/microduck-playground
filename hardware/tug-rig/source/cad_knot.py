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


def knot_path(mirror: bool = False) -> np.ndarray:
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
    if mirror:
        pts = pts.copy()
        pts[:, 0] *= -1.0
    # densify to ~0.3 mm steps, then moving-average the polyline — capsule
    # unions ripple at every micro-kink otherwise (结面毛刺)
    dense = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        d = float(np.linalg.norm(b - a))
        for s_ in np.linspace(0.0, 1.0, max(2, int(d / 0.3)), endpoint=False)[1:]:
            dense.append(a + s_ * (b - a))
        dense.append(b)
    pts = np.array(dense)
    kernel = np.ones(9) / 9.0
    padded = np.vstack([np.repeat(pts[[0]], 4, axis=0), pts,
                        np.repeat(pts[[-1]], 4, axis=0)])
    pts = np.apply_along_axis(lambda c: np.convolve(c, kernel, mode="valid"),
                              0, padded)
    return pts


def build(mirror: bool = False) -> trimesh.Trimesh:
    pts = knot_path(mirror)
    parts = [trimesh.creation.capsule(radius=R, height=float(np.linalg.norm(b - a)),
                                      count=[28, 20],
                                      transform=trimesh.transformations
                                      .translation_matrix((a + b) / 2.0)
                                      @ trimesh.geometry.align_vectors(
                                          [0, 0, 1], b - a))
             for a, b in zip(pts[:-1], pts[1:])]
    knot = trimesh.boolean.union(parts, engine="manifold")
    assert knot.is_volume and knot.is_watertight, "knot must be one solid"
    return knot


def clear_collar(knot: trimesh.Trimesh, side: float,
                 clearance: float = 0.25) -> trimesh.Trimesh:
    """Subtract the collar (inflated by `clearance` mm) from the knot so the
    assembled pair can NEVER interpenetrate (the orange plate used to bleed
    through the rope — 穿模). `side` selects the rim the knot sits on: the
    shell is the collar expressed in THIS knot's own frame (rim at origin),
    no mirror fudge. Trimmed spots read as rope pressed against hardware."""
    collar = trimesh.load(ASSETS / "tug_collar.stl")
    collar.apply_scale(1000.0)                      # assets are in metres
    eye = np.array([-68.6, 0.0, 16.0])              # back-eye centre (mm)
    base = collar.copy()
    base.apply_translation(-(eye + side * np.array([8.0, 0.0, 0.0])))
    shell = base
    for d in ([clearance, 0, 0], [-clearance, 0, 0], [0, clearance, 0],
              [0, -clearance, 0], [0, 0, clearance], [0, 0, -clearance]):
        t = base.copy()
        t.apply_translation(d)
        shell = trimesh.boolean.union([shell, t], engine="manifold")
    knot = trimesh.boolean.difference([knot, shell], engine="manifold")
    knot = max(knot.split(only_watertight=False), key=lambda m: m.volume)
    assert knot.is_volume and knot.is_watertight
    return knot


def path_uvs(mesh: trimesh.Trimesh, pts: np.ndarray) -> np.ndarray:
    """SPATIAL UVs at CONSTANT SURFACE DENSITY: u = vertex x / 17.5 mm (the
    rope's twist pitch); v = angle about the rope axis × local radius /
    rope radius — on the fat coil blob the plain angular v stretched the
    texture's grooves thin and washed the knot out pale (用户: 纹理不对).
    Now one tile is always 17.5 mm × rope-circumference of REAL surface."""
    v = mesh.vertices                  # knot is built in mm
    u = v[:, 0] / 17.5
    r_local = np.hypot(v[:, 1], v[:, 2])
    ang = np.arctan2(v[:, 2], v[:, 1]) / (2.0 * np.pi) * (r_local / 2.5)
    return np.stack([u, ang], axis=1)


def export_both(mesh: trimesh.Trimesh, name: str, uv: np.ndarray) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    # Weld identical vertices so MuJoCo smooth-shades across facet edges —
    # the boolean output kept duplicated verts and rendered as visible
    # pixel-block facets next to the smooth rope.
    mesh.merge_vertices(merge_tex=False)
    mesh = mesh.smooth_shaded   # property: returns the smooth-shaded copy

    mesh.export(OUT / f"{name}_mm.stl")
    m = mesh.copy()
    m.apply_scale(0.001)
    m.export(ASSETS / f"{name}.stl")
    m.visual = trimesh.visual.TextureVisuals(uv=uv)
    m.export(ASSETS / f"{name}.obj", include_texture=True)


def main() -> None:
    # Build each facing from a mirrored PATH — a fresh union is watertight;
    # mirroring the solid breaks the volume flag that manifold needs.
    for mirror, side, name in ((False, +1.0, "tug_knot"), (True, -1.0, "tug_knot_mir")):
        knot = clear_collar(build(mirror), side=side)
        export_both(knot, name, path_uvs(knot, knot_path(mirror)))
        print(name, knot.extents.round(2), "mm, watertight:", knot.is_watertight)


if __name__ == "__main__":
    main()
