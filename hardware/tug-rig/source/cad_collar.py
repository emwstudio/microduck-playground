#!/usr/bin/env python3
"""Clamp-collar tow harness for the Microduck tug rig (trimesh pipeline).

The user's design: ONE collar band wrapped around the torso shell, split
on one side (+y) with a screw through two lugs that tightens the band's
grip on the shell — a hose-clamp / shaft-collar principle. A closed tow
eye is integrated at the FRONT (chest pull point) and one at the BACK
(butt pull point); each eye is fused to the band by a cast neck boss, so
the whole collar prints as ONE connected piece. The lugs are drilled
(Ø3.2 mm) for the M3 socket-head screw (modelled with hex socket and
thread) and its hex nut.

The band follows the measured shell contour (rounded box: front +30 mm,
back -47 mm, hips ±47 mm) with ~1-2 mm clamping allowance at band height
(z = -10 mm, clears the hip joints and the legs' swing).

Outputs: meshes/tug_collar_mm.stl (one connected piece),
meshes/tug_clamp_screw_mm.stl (screw + nut assembly), plus metre copies
in the sim assets dir.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

# --- measured duck dimensions (metres, trunk-local) ---
XF, XB, HW = 0.034, 0.048, 0.049   # band centreline: front / back / side reach
Z_C = -0.010                        # band centre height (mid torso)
BAND_H, BAND_T = 0.014, 0.0022     # band cross-section: vertical × radial
GAP_HALF = 0.34                     # split half-angle at the +y side (~28 mm gap)

# --- clamp hardware ---
LUG_W, LUG_OUT, LUG_H = 0.006, 0.007, 0.012   # ~6 mm between the lug faces: screw travel to clamp
HOLE_R = 0.0016                     # Ø3.2 mm clearance hole through the lugs
SCREW_R, SCREW_LEN = 0.0015, 0.024
HEAD_R, HEAD_H = 0.00275, 0.003    # M3 socket head: Ø5.5 × 3 mm
SOCKET_AF, SOCKET_D = 0.0025, 0.0015   # 2.5 mm hex key, 1.5 mm deep
NUT_AF, NUT_H = 0.0055, 0.0024     # M3 hex nut, 5.5 mm across flats
THREAD_PITCH, THREAD_R = 0.0005, 0.00035

# --- tow eyes ---
# Eyebolt principle: hole axis ALONG the pull direction (x). Inner hole
# Ø9 mm so a Ø4 mm rope passes doubled (lark's head) or a Ø6 mm rope single
# (bowline). The tube stands fully clear of the band face so the hole
# channel is through; a cast neck cradles the ring from BELOW the channel.
EYE_MAJOR, EYE_TUBE = 0.007, 0.0025
EYE_INNER_R = EYE_MAJOR - EYE_TUBE           # Ø9 mm clear hole
FACE_FRONT = XF + BAND_T / 2
FACE_BACK = XB + BAND_T / 2
EYE_FRONT = np.array([FACE_FRONT + 0.002 + EYE_MAJOR + EYE_TUBE, 0.0, Z_C])
EYE_BACK = np.array([-(FACE_BACK + 0.002 + EYE_MAJOR + EYE_TUBE), 0.0, Z_C])

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
    """Two ears at the split ends, protruding +y, drilled Ø3.2 along x."""
    pts, _ = band_path(2)
    blocks = []
    for end in pts:
        lug = trimesh.creation.box(extents=(LUG_W, LUG_OUT + BAND_T, LUG_H))
        lug.apply_translation((end[0], end[1] + LUG_OUT / 2, Z_C))
        blocks.append(lug)
    pair = trimesh.boolean.union(blocks, engine="manifold")
    y_mid = pts[0, 1] + LUG_OUT / 2
    hole = trimesh.creation.cylinder(radius=HOLE_R, height=LUG_W * 3, sections=24)
    hole.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], [1, 0, 0]))
    hole.apply_translation((0.0, y_mid, Z_C))
    return trimesh.boolean.difference([pair, hole], engine="manifold")


def tow_eye(center: np.ndarray) -> trimesh.Trimesh:
    """Closed D-ring fused to the band by a cast neck boss (ONE solid).

    D-ring orientation: ring plane VERTICAL and fore-aft (xz plane, hole
    axis along y) — the rope threads from the side and the pull stays in
    the ring's plane, the way a leash D-ring is loaded. The neck boss
    bridges from inside the band to the ring's band-side arc, stopping at
    the hole boundary, so the Ø9 mm side-hole is unobstructed.
    """
    ring = trimesh.creation.torus(major_radius=EYE_MAJOR, minor_radius=EYE_TUBE,
                                  major_sections=48, minor_sections=14)
    ring.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], [0, 1, 0]))
    ring.apply_translation(center)
    sign = 1.0 if center[0] > 0 else -1.0
    band_face = sign * (abs(center[0]) - 0.002 - EYE_MAJOR - EYE_TUBE)
    neck_x0 = band_face - sign * BAND_T        # inside the band
    neck_x1 = center[0] - sign * (EYE_MAJOR - EYE_TUBE / 2)   # into the tube, clear of the hole
    neck = trimesh.creation.box(extents=(abs(neck_x1 - neck_x0), 0.006, 0.007))
    neck.apply_translation(((neck_x0 + neck_x1) / 2, 0.0, Z_C))
    return trimesh.boolean.union([ring, neck], engine="manifold")


def hole_gauge_ok(collar: trimesh.Trimesh) -> bool:
    """Push a Ø8 mm gauge pin SIDEWAYS (along y) through each eye centre:
    the collar must not intersect it (hole channel clear for the rope)."""
    span = 2 * (EYE_MAJOR + EYE_TUBE) + 0.02   # ring tube extent + margin
    for center in (EYE_FRONT, EYE_BACK):
        pin = trimesh.creation.cylinder(radius=EYE_INNER_R - 0.0005, height=span, sections=24)
        pin.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], [0, 1, 0]))
        pin.apply_translation(center)
        if trimesh.boolean.intersection([collar, pin], engine="manifold").volume > 1e-12:
            return False
    return True


def _thread_helix(z0: float, z1: float) -> trimesh.Trimesh:
    """M3x0.5 thread ridge: helical sweep around the shaft (axis = z)."""
    from shapely.geometry import Point
    turns = (z1 - z0) / THREAD_PITCH
    n = max(int(turns * 10), 8)
    t = np.linspace(0, 2 * np.pi * turns, n)
    path = np.stack([
        (SCREW_R + THREAD_R / 2) * np.cos(t),
        (SCREW_R + THREAD_R / 2) * np.sin(t),
        np.linspace(z0, z1, n),
    ], axis=1)
    profile = Point(0, 0).buffer(THREAD_R, resolution=6)
    return trimesh.creation.sweep_polygon(profile, path)


def clamp_screw() -> trimesh.Trimesh:
    """M3 socket-head cap screw + hex nut through the lug holes (axis = x).

    Every part is modelled along z, rotated z→x and placed at its final x:
    head snug on the +x lug face, nut snug on the -x lug face, thread on
    the shaft between them. Separate shells on purpose — screw and nut are
    different physical parts (the STL is an assembly).
    """
    pts, _ = band_path(2)
    y_mid = pts[0, 1] + LUG_OUT / 2
    x2 = pts[1, 0] + LUG_W / 2                     # +x lug outer face
    x1 = pts[0, 0] - LUG_W / 2                     # -x lug outer face
    rot = trimesh.geometry.align_vectors([0, 0, 1], [1, 0, 0])

    def place(part: trimesh.Trimesh, x: float) -> trimesh.Trimesh:
        part.apply_transform(rot)
        part.apply_translation((x, y_mid, Z_C))
        return part

    head = trimesh.creation.cylinder(radius=HEAD_R, height=HEAD_H, sections=32)
    socket = trimesh.creation.cylinder(
        radius=SOCKET_AF / np.sqrt(3), height=SOCKET_D * 2, sections=6)
    socket.apply_translation((0, 0, HEAD_H / 2 - SOCKET_D / 2 + 0.0002))
    head = trimesh.boolean.difference([head, socket], engine="manifold")

    shaft_len = (x2 + 0.0005) - (x1 - NUT_H - 0.002)
    shaft = trimesh.creation.cylinder(radius=SCREW_R, height=shaft_len, sections=24)
    thread = _thread_helix(-shaft_len / 2, shaft_len / 2)

    nut = trimesh.creation.cylinder(radius=NUT_AF / np.sqrt(3), height=NUT_H, sections=6)
    nut_hole = trimesh.creation.cylinder(radius=SCREW_R, height=NUT_H * 3, sections=20)
    nut = trimesh.boolean.difference([nut, nut_hole], engine="manifold")

    x_shaft = (x2 + 0.0005 + x1 - NUT_H - 0.002) / 2
    return trimesh.util.concatenate([
        place(head, x2 + HEAD_H / 2),
        place(shaft, x_shaft),
        place(thread, x_shaft),
        place(nut, x1 - NUT_H / 2),
    ])


def export_both(mesh: trimesh.Trimesh, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    mm = mesh.copy()
    mm.apply_scale(1000.0)
    mm.export(OUT / f"{name}_mm.stl")
    mesh.export(ASSETS / f"{name}.stl")
    bodies = len(mesh.split(only_watertight=False)) if hasattr(mesh, "split") else "?"
    print(f"{name}: {len(mesh.vertices)} verts, bodies={bodies}")


def main() -> None:
    collar = trimesh.boolean.union(
        [band_mesh(), lugs(), tow_eye(EYE_FRONT), tow_eye(EYE_BACK)],
        engine="manifold")
    trimesh.repair.fix_normals(collar)
    export_both(collar, "tug_collar")
    export_both(clamp_screw(), "tug_clamp_screw")
    bodies = collar.split(only_watertight=False)
    print(f"one-piece collar: {len(collar.vertices)} verts, "
          f"{len(collar.faces)} faces, watertight={collar.is_watertight}, "
          f"connected_bodies={len(bodies)} (must be 1)")
    assert len(bodies) == 1, "collar is not one connected piece"
    assert hole_gauge_ok(collar), "tow-eye hole channel is blocked"
    print("hole gauge (Ø8 mm pin through each eye): CLEAR")
    print("front eye at", EYE_FRONT.tolist(), " back eye at", EYE_BACK.tolist())


if __name__ == "__main__":
    main()
