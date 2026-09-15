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

import json

import numpy as np
import trimesh

# --- measured duck dimensions (metres, trunk-local) ---
# Torso cross-section measured from robot_allcollisions.xml by
# measure_torso_contour.py (triangle sections through every trunk mesh):
# front r=22.8 mm, back r=46 mm, sides r=32 mm at band height.
Z_C = 0.016                         # band centre height — mid purple shell, clear of the neck servo
BAND_H, BAND_T = 0.014, 0.0022     # band cross-section: vertical × radial
ALLOW = 0.0012                      # clamping allowance: band inner face = shell + this
GAP_HALF = 0.18                     # split half-angle at the +y side (~12 mm gap)

CONTOUR_JSON = Path(__file__).resolve().parent.parent / "torso_contour.json"


def _shell_radius() -> tuple[np.ndarray, np.ndarray]:
    """Measured polar shell contour, y-symmetrized, rolling-MAX smoothed
    (never shrinks below the measured envelope — a mean filter once dipped
    the band into the shell, 0.24 mm graze)."""
    from scipy.ndimage import maximum_filter1d
    data = json.loads(CONTOUR_JSON.read_text())
    ang = np.array(data["angle"])
    r = np.array([np.nan if v is None else v for v in data["r"]])
    ok = ~np.isnan(r)
    r = np.interp(ang, ang[ok], r[ok])                 # fill gaps
    r = np.maximum(r, r[::-1])                         # symmetrize left/right
    r = maximum_filter1d(r, size=3, mode="wrap")       # smooth without shrinking
    return ang, r


_SHELL_ANG, _SHELL_R = _shell_radius()


def shell_r(theta: np.ndarray) -> np.ndarray:
    """Band CENTRELINE radius at angle theta: shell + allowance + half thickness."""
    wrapped = (theta + np.pi) % (2 * np.pi) - np.pi
    return np.interp(wrapped, _SHELL_ANG, _SHELL_R, period=2 * np.pi) + ALLOW + BAND_T / 2

# --- clamp hardware ---
LUG_W, LUG_OUT, LUG_H = 0.003, 0.010, 0.008   # slim 3 mm ears, ~7 mm of screw platform outside the ring
SCREW_OUT = 0.007                   # screw axis ~6 mm outside the band surface — hex key still unobstructed
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
EYE_MAJOR, EYE_TUBE = 0.008, 0.0025
EYE_INNER_R = EYE_MAJOR - EYE_TUBE           # Ø11 mm clear hole — Ø8 mm rope
                                             # enters at a sag angle without clipping
FACE_FRONT = float(shell_r(np.array([0.0]))[0]) + BAND_T / 2      # band outer surface, front
FACE_BACK = float(shell_r(np.array([np.pi]))[0]) + BAND_T / 2     # band outer surface, back
EYE_STANDOFF = 0.008   # eye centre this far past the band face — the whole
                        # eye head clears the band rim (it was hidden behind it)
EYE_FRONT = np.array([FACE_FRONT + EYE_STANDOFF + EYE_MAJOR + EYE_TUBE, 0.0, Z_C])
EYE_BACK = np.array([-(FACE_BACK + EYE_STANDOFF + EYE_MAJOR + EYE_TUBE), 0.0, Z_C])

OUT = Path(__file__).resolve().parent.parent / "meshes"
ASSETS = Path(__file__).resolve().parents[3] / "src/mjlab_microduck/robot/microduck/assets"


def band_path(n: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Open loop around the shell, ends facing each other across the +y gap.
    Radius comes from the MEASURED torso contour (torso_contour.json)."""
    ang = np.linspace(np.pi / 2 + GAP_HALF, np.pi / 2 + 2 * np.pi - GAP_HALF, n)
    r = shell_r(ang)
    x, y = r * np.cos(ang), r * np.sin(ang)
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


def split_ends() -> tuple[np.ndarray, np.ndarray]:
    """The two split-end points with their TRUE local outward normals.

    Never use band_path(2) for this: with only two points the gradient
    degenerates to the secant between the ends, which flips the outward
    normal INTO the duck's body (the 20 mm arms once speared the torso).
    """
    pts, n3 = band_path()
    return np.array([pts[0], pts[-1]]), np.array([n3[0], n3[-1]])


def lug_centre() -> np.ndarray:
    """Point on the screw axis: outboard on the lug arms, outside the ring."""
    pts, n3 = split_ends()
    ends = pts + n3 * SCREW_OUT
    return np.array([(ends[0, 0] + ends[1, 0]) / 2,
                     (ends[0, 1] + ends[1, 1]) / 2, Z_C])


def lugs() -> trimesh.Trimesh:
    """Two ears at the split ends. Both are axis-aligned and PARALLEL with
    their faces perpendicular to the screw axis (x) — the screw head and
    nut seat flat on the faces, so clamping force transfers squarely.
    (Tangent-rotated lugs splay ±12° and the head/nut would bear on an
    edge.) Drilled Ø3.2 along x."""
    pts, n3 = split_ends()
    blocks = []
    for end, outward in zip(pts, n3):
        lug = trimesh.creation.box(extents=(LUG_W, LUG_OUT + BAND_T, LUG_H))
        lug.apply_translation(end + outward * (LUG_OUT / 2))
        blocks.append(lug)
    pair = trimesh.boolean.union(blocks, engine="manifold")
    centre = lug_centre()
    span = abs(pts[1, 0] - pts[0, 0]) + LUG_W * 2 + 0.01
    hole = trimesh.creation.cylinder(radius=HOLE_R, height=span, sections=24)
    hole.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], [1, 0, 0]))
    hole.apply_translation(centre)
    return trimesh.boolean.difference([pair, hole], engine="manifold")


def tow_eye(center: np.ndarray) -> trimesh.Trimesh:
    """One-piece pad-eye plate fused to the band (lollipop profile, hole
    axis along y — the D-ring the user wants).

    The fore-aft span rope does NOT pass this plate: it ties off on the
    hole's near arc (knot coil wraps the rim on the pull side), exactly
    how a real rope knots onto a D-ring. Only a short working-end stub
    pokes through the hole (along y) for the threaded look.
    """
    from shapely.geometry import Point, box as shapely_box
    from shapely.ops import unary_union
    sign = 1.0 if center[0] > 0 else -1.0
    band_face = sign * (abs(center[0]) - EYE_STANDOFF - EYE_MAJOR - EYE_TUBE)
    head = Point(float(center[0]), Z_C).buffer(EYE_MAJOR + EYE_TUBE, resolution=48)
    stem = shapely_box(min(band_face - sign * BAND_T, center[0] + sign * 0.001),
                       Z_C - 0.005,
                       max(band_face - sign * BAND_T, center[0] + sign * 0.001),
                       Z_C + 0.005)
    outline = unary_union([head, stem]).buffer(0.001).buffer(-0.001)  # fillet the shoulders
    hole = Point(float(center[0]), Z_C).buffer(EYE_INNER_R, resolution=40)
    outline = outline.difference(hole)
    plate = trimesh.creation.extrude_polygon(outline, EYE_TUBE * 2)
    plate.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    plate.apply_translation((0.0, EYE_TUBE, 0.0))   # centre the plate on y=0
    return plate


def hole_gauge_ok(collar: trimesh.Trimesh) -> bool:
    """Push a Ø8 mm gauge pin SIDEWAYS (along y) through each eye centre:
    the collar must not intersect it (hole channel clear for the rope)."""
    span = 2 * (EYE_MAJOR + EYE_TUBE) + 0.004   # ring tube extent + small margin
    # (longer pins would sweep the neighbouring band at ±20° — the rope
    # never goes there; the channel that matters is through the ring)
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
    pts, _ = split_ends()
    centre = lug_centre()
    x2 = pts[1, 0] + LUG_W / 2 + 0.001           # +x lug outer face
    x1 = pts[0, 0] - LUG_W / 2 - 0.001           # -x lug outer face
    rot = trimesh.geometry.align_vectors([0, 0, 1], [1, 0, 0])

    def place(part: trimesh.Trimesh, x: float) -> trimesh.Trimesh:
        part.apply_transform(rot)
        part.apply_translation((x, centre[1], Z_C))
        return part

    head = trimesh.creation.cylinder(radius=HEAD_R, height=HEAD_H, sections=32)
    socket = trimesh.creation.cylinder(
        radius=SOCKET_AF / np.sqrt(3), height=SOCKET_D * 2, sections=6)
    socket.apply_translation((0, 0, HEAD_H / 2 - SOCKET_D / 2 + 0.0002))
    head = trimesh.boolean.difference([head, socket], engine="manifold")

    TAIL = 0.006                     # thread tail protruding past the nut
    shaft_len = (x2 + 0.0005) - (x1 - NUT_H - TAIL)
    shaft = trimesh.creation.cylinder(radius=SCREW_R, height=shaft_len, sections=24)
    thread = _thread_helix(-shaft_len / 2, shaft_len / 2)

    nut = trimesh.creation.cylinder(radius=NUT_AF / np.sqrt(3), height=NUT_H, sections=6)
    nut_hole = trimesh.creation.cylinder(radius=SCREW_R, height=NUT_H * 3, sections=20)
    nut = trimesh.boolean.difference([nut, nut_hole], engine="manifold")

    x_shaft = (x2 + 0.0005 + x1 - NUT_H - TAIL) / 2
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
    ends, outs = split_ends()
    for e, o in zip(ends, outs):
        tip = e + o * LUG_OUT
        assert np.hypot(*tip[:2]) > np.hypot(*e[:2]) + LUG_OUT * 0.9, \
            "lug arms point INWARD (into the duck's body)"
    print("hole gauge (Ø8 mm pin through each eye): CLEAR; lug arms point OUTWARD")
    print("front eye at", EYE_FRONT.tolist(), " back eye at", EYE_BACK.tolist())


if __name__ == "__main__":
    main()
