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
    # densify to ~0.8 mm steps so capsule unions are crease-free
    dense = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        d = float(np.linalg.norm(b - a))
        for s in np.linspace(0.0, 1.0, max(2, int(d / 0.3)), endpoint=False)[1:]:
            dense.append(a + s * (b - a))
        dense.append(b)
    return np.array(dense)


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
    """Sweep-style UVs: u = arclength along the knot path (one texture tile
    per 17.5 mm = the rope's twist pitch), v = angle around the local path
    tangent — the same mapping the span sweep uses, so knot and rope carry
    the SAME twist flow (圆柱映射的轴向条纹一眼假)."""
    # arclength table and per-segment tangents
    seg = np.diff(pts, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    tan = seg / seglen[:, None]
    arc = np.concatenate([[0.0], np.cumsum(seglen)])
    v = mesh.vertices                  # knot is built in mm, as is the path
    u_out = np.zeros(len(v))
    ang_out = np.zeros(len(v))
    # nearest point on the path per vertex (chunked to bound memory)
    # parallel-transport frames along the path — a fixed reference flips on
    # the helix and scrambles the v angle into ribs (线圈束竖条纹)
    tangents = np.gradient(pts, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(tangents[0] @ ref) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    e1_path = np.zeros_like(pts)
    e1_path[0] = ref - (ref @ tangents[0]) * tangents[0]
    e1_path[0] /= np.linalg.norm(e1_path[0])
    for i in range(1, len(pts)):
        e1_path[i] = e1_path[i - 1] - (e1_path[i - 1] @ tangents[i]) * tangents[i]
        un = np.linalg.norm(e1_path[i])
        e1_path[i] = e1_path[i - 1] if un < 1e-9 else e1_path[i] / un
    e2_path = np.cross(tangents, e1_path)
    for lo in range(0, len(v), 20000):
        p = v[lo:lo + 20000]
        # distance to each segment: project, clamp, measure
        ap = p[:, None, :] - pts[None, :-1, :]                  # (P, S, 3)
        tpar = np.einsum("psj,sj->ps", ap, tan) / seglen[None, :]
        tpar = np.clip(tpar, 0.0, 1.0)
        closest = pts[None, :-1, :] + tpar[:, :, None] * seg[None, :, :]
        d2 = ((p[:, None, :] - closest) ** 2).sum(axis=2)
        best = d2.argmin(axis=1)
        u_out[lo:lo + 20000] = (arc[best] + tpar[np.arange(len(p)), best]
                                * seglen[best])
        off = p - closest[np.arange(len(p)), best]
        # angle around the tangent in the path's parallel-transport frame
        a = np.arctan2((off * e2_path[best]).sum(axis=1),
                       (off * e1_path[best]).sum(axis=1))
        ang_out[lo:lo + 20000] = a
    return np.stack([u_out / 17.5, ang_out / (2.0 * np.pi)], axis=1)


def export_both(mesh: trimesh.Trimesh, name: str, uv: np.ndarray) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
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
