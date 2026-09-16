"""NvN tug-of-war scene builder (deployment-side demo, no training task).

Two teams of Microducks stand back-to-back along the world x-axis, chained
by tension-only spatial tendons wrapped on waist sites — the same dead-band
cord model as the swing strings. Every duck runs the same walking ONNX with
a constant forward velocity command; since each duck faces away from the
center line, forward walking IS pulling. Used by scripts/tug_of_war.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

_ROBOT_DIR: Path = Path(os.path.dirname(__file__)) / "microduck"
_SCENE_XML: Path = _ROBOT_DIR / "scene.xml"
_ROBOT_XML: Path = _ROBOT_DIR / "robot_allcollisions.xml"

# STAND2 default pose, identical to scripts/infer_policy.py and the ONNX
# metadata default_joint_pos (joint order = ONNX joint_names).
JOINT_NAMES = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)
DEFAULT_POSE = np.array([
    0.0, -0.0873, -0.4579, -0.0049, 0.4530,
    0.3491, 0.3491, 0.0, 0.0,
    0.0, 0.0873, 0.4579, 0.0049, -0.4530,
], dtype=np.float32)

# Team tint targets. bottom_head_shell_material (orange beak) is left alone so
# both teams keep their beaks; the body/head shells carry the team color.
SHELL_MATERIALS = ("left_shell_material", "right_shell_material", "top_head_shell_material")
RED_RGBA = (0.80, 0.15, 0.12, 1.0)
BLUE_RGBA = (0.15, 0.30, 0.85, 1.0)

# Rope: tension-only dead-band cord like the swing strings, but MUCH softer:
# 2000 N/m is fine anchored to the world (swing), between ducks a single mm
# of misalignment already means newtons of sideways yank. 200 N/m transmits
# a duck-strength pull (~10 N) at ~5 cm of stretch and tolerates spawn noise.
ROPE_STIFFNESS = 200.0
ROPE_SLACK = 0.005          # dead band extends this far past the spawn distance
ROPE_LIMIT_MARGIN = 0.05    # soft upper safety catch past the dead band
ROPE_WIDTH = 0.0025
ROPE_RGBA = (0.92, 0.87, 0.70, 1.0)
# Waist-wrap sites sit ON the torso shell (trunk half-width ~0.047, front
# face ~x+0.015, mid-torso z~0.035) — the rope visibly hugs the body instead
# of floating beside it.
ROPE_SITE_Y = 0.052
ROPE_SITE_Z = 0.017   # belly height, inside the visual wrap band
ROPE_CHEST_X = 0.022

DUCK_SPACING = 0.24         # trunk-to-trunk between teammates; duck x-extent is 0.21 m, tighter collides
CENTER_GAP = 0.30           # trunk distance between the two inner ducks (tails face center)
TRUNK_Z0 = 0.12             # spawn trunk height (scene.xml INIT keyframe)

# Win conditions
WIN_X = 0.45                # rope midpoint past ±WIN_X decides
FALLEN_TRUNK_Z = 0.055      # trunk below this = down (ducks dip to ~0.06 while stumbling)
FALLEN_GRAVITY_Z = -0.35    # projected-gravity z above this = not upright
MIN_FALLEN = 3              # this many ducks down loses the round for the team


def team_prefixes(n_per_team: int) -> tuple[list[str], list[str]]:
    """Red prefixes (the unprefixed scene robot is red duck 0) and blue."""
    red = ["" if i == 0 else f"r{i}_" for i in range(n_per_team)]
    blue = [f"b{j}_" for j in range(n_per_team)]
    return red, blue


def duck_spawns(n_per_team: int, spacing: float = DUCK_SPACING,
                gap: float = CENTER_GAP) -> dict[str, tuple[float, tuple[float, float, float, float]]]:
    """prefix -> (spawn x, spawn quat wxyz). Red faces -x, blue faces +x."""
    red, blue = team_prefixes(n_per_team)
    spawns: dict[str, tuple[float, tuple[float, float, float, float]]] = {}
    yaw_180 = (0.0, 0.0, 0.0, 1.0)   # wxyz, facing -x
    identity = (1.0, 0.0, 0.0, 0.0)  # facing +x
    for i, p in enumerate(red):
        spawns[p] = (-(gap / 2.0 + i * spacing), yaw_180)
    for j, p in enumerate(blue):
        spawns[p] = (gap / 2.0 + j * spacing, identity)
    return spawns


def _find_body(spec: mujoco.MjSpec, name: str):
    for body in spec.bodies:
        if body.name == name:
            return body
    raise KeyError(f"body {name!r} not found in spec")


def _tint_shells(spec: mujoco.MjSpec, rgba: tuple[float, float, float, float]) -> None:
    for mat in spec.materials:
        if mat.name in SHELL_MATERIALS:
            mat.rgba = rgba


ROPE_PLY_CENTER_R = 0.0038
ROPE_PLY_TUBE_R = 0.0043
ROPE_PLY_TWISTS_PER_TURN = 4

CHORD_BINS = np.round(np.arange(0.10, 0.38, 0.01), 2)
SAG_BINS = (0.0, 0.33, 0.67, 1.0)   # fraction of SAG_MAX
SPAN_T_SEGMENTS = 26
SPAN_ALPHA_SEGMENTS = 6


# Harness geometry: clamp collar (hardware/tug-rig/cad_collar.py) — one band
# around the torso shell, screw-tightened on the side, tow eyes integrated
# front (chest pull) and back (butt pull). Sites sit at the eye centres.
TEAM_Y = 0.0
RING_LOCAL_X = -0.0686
RING_LOCAL_Z = 0.016
RING_LOCAL_Y = 0.0
CHEST_LOCAL = np.array([0.0460, 0.0, 0.016])
SPAWN_Y = TEAM_Y


def ring_local(team: str) -> np.ndarray:
    return np.array([RING_LOCAL_X, RING_LOCAL_Y, RING_LOCAL_Z])


def chest_local(team: str) -> np.ndarray:
    return CHEST_LOCAL.copy()


def _add_rope_sites(spec: mujoco.MjSpec, team: str) -> None:
    trunk = _find_body(spec, "trunk_base")
    # alpha=0: sites are physics anchors, not decoration — no marker balls
    trunk.add_site(
        name="rope_hook",
        pos=tuple(ring_local(team)),
        size=(0.004,),
        rgba=(0.0, 0.0, 0.0, 0.0),
    )
    trunk.add_site(
        name="rope_hook_chest",
        pos=tuple(chest_local(team)),
        size=(0.004,),
        rgba=(0.0, 0.0, 0.0, 0.0),
    )


def _link_side_pairs(pa: str, pb: str, red: list[str]) -> tuple[tuple[str, str], ...]:
    """Teammates face the same way: left site to left site. The center pair
    faces away from each other, so r0's left sits at b0's right — same world
    side — and left-left would cross the cords diagonally."""
    same_facing = (pa in red) == (pb in red)
    return (("left", "left"), ("right", "right")) if same_facing \
        else (("left", "right"), ("right", "left"))


SAG_PER_SLACK = 0.6       # parabola depth per metre of slack
SAG_MAX = 0.045


# Rope texture. Spans use our procedural twist tile (soft 3-band diagonal
# helix, colours measured off the reference video's strand); coils use
# ambientCG Rope001 (CC0, https://ambientcg.com/view?id=Rope001).
_ROPE_COLOR = _ROBOT_DIR / "assets" / "rope_hemp_color.png"
_ROPE_NORMAL = _ROBOT_DIR / "assets" / "rope_hemp_normal.png"
_ROPE_ROUGHNESS = _ROBOT_DIR / "assets" / "rope_hemp_roughness.png"
_TWIST_CROWN = np.array([0.93, 0.85, 0.66])
_TWIST_GROOVE = np.array([0.72, 0.62, 0.44])


def _twist_texture(size: int = 256, bands: int = 3) -> tuple[bytes, bytes]:
    """Seamless diagonal-band twist tile + matching normal map.

    On a cylinder's unwrapped UV a helix is a diagonal line; the tile repeats
    `bands` soft ridges on the 45-degree diagonal so the rope reads as long-
    pitch twisted strands, with clean silky shading (no fibre speckle —
    minification noise was what made previous versions look ragged)."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    phase = ((xx + yy) / size * bands) % 1.0
    ridge = 0.5 + 0.5 * np.cos(2.0 * np.pi * phase)          # 1 crown, 0 groove
    shade = 0.87 + 0.13 * ridge                              # gentle contrast
    fibre = 0.015 * np.sin(xx * 0.9) + 0.01 * np.sin(yy * 2.3 + xx * 0.2)
    shade = np.clip(shade + fibre, 0, 1)
    rgb = shade[..., None] * (_TWIST_GROOVE + (_TWIST_CROWN - _TWIST_GROOVE) * ridge[..., None])
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)

    eps = 1.0 / size
    dhdx = (np.roll(ridge, -1, 1) - np.roll(ridge, 1, 1)) / (2 * eps)
    dhdy = (np.roll(ridge, -1, 0) - np.roll(ridge, 1, 0)) / (2 * eps)
    strength = 0.002
    nx, ny = -dhdx * strength, -dhdy * strength
    nz = np.ones_like(nx)
    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    nrm = np.stack([(nx / norm + 1) / 2, (ny / norm + 1) / 2, (nz / norm + 1) / 2], axis=-1)
    return rgb.tobytes(), (nrm * 255).astype(np.uint8).tobytes()


def _add_twist_material(spec: mujoco.MjSpec) -> None:
    rgb, normal = _twist_texture()
    tex = spec.add_texture(name="tug_twist_tex")
    tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    tex.width, tex.height, tex.nchannel = 256, 256, 3
    tex.data = rgb
    ntex = spec.add_texture(name="tug_twist_nrm")
    ntex.type = mujoco.mjtTexture.mjTEXTURE_2D
    ntex.width, ntex.height, ntex.nchannel = 256, 256, 3
    ntex.data = normal
    mat = spec.add_material(name="tug_twist")
    mat.rgba = (1.0, 1.0, 1.0, 1.0)   # MuJoCo's default 0.5 grey halves the texture
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "tug_twist_tex"
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_NORMAL] = "tug_twist_nrm"
    mat.texrepeat = (1.0, 1.0)
    mat.texuniform = False
    mat.specular = 0.6
    mat.shininess = 0.6




def _add_hemp_material(spec: mujoco.MjSpec) -> None:
    for name, path, role in (("tug_hemp_tex", _ROPE_COLOR, mujoco.mjtTextureRole.mjTEXROLE_RGB),
                             ("tug_hemp_nrm", _ROPE_NORMAL, mujoco.mjtTextureRole.mjTEXROLE_NORMAL),
                             ("tug_hemp_rgh", _ROPE_ROUGHNESS, mujoco.mjtTextureRole.mjTEXROLE_ROUGHNESS)):
        tex = spec.add_texture(name=name)
        tex.type = mujoco.mjtTexture.mjTEXTURE_2D
        tex.file = str(path)
    mat = spec.add_material(name="tug_hemp")
    mat.rgba = (1.0, 1.0, 1.0, 1.0)   # MuJoCo's default 0.5 grey halves the texture
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "tug_hemp_tex"
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_NORMAL] = "tug_hemp_nrm"
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_ROUGHNESS] = "tug_hemp_rgh"
    mat.texrepeat = (8.0, 1.0)
    mat.texuniform = True
    mat.specular = 0.35
    mat.shininess = 0.4


def _add_rig_materials(spec: mujoco.MjSpec) -> None:
    webbing = spec.add_material(name="tug_webbing")
    webbing.rgba = (0.16, 0.16, 0.18, 1.0)   # dark nylon harness
    webbing.specular = 0.15
    webbing.shininess = 0.1
    paint = spec.add_material(name="tug_frame_paint")
    paint.rgba = (0.62, 0.65, 0.70, 1.0)   # satin-machined saddle, reads on the body
    paint.specular = 0.8
    paint.shininess = 0.6
    steel = spec.add_material(name="tug_steel")
    steel.rgba = (0.10, 0.10, 0.12, 1.0)   # black-oxide screw: pops against the alu collar
    steel.specular = 0.9
    steel.shininess = 0.7
    orange = spec.add_material(name="tug_carabiner")
    orange.rgba = (0.92, 0.42, 0.08, 1.0)   # anodized-orange carabiner
    orange.specular = 0.85
    orange.shininess = 0.6
    knot = spec.add_material(name="tug_knot")
    knot.rgba = (0.83, 0.74, 0.55, 1.0)   # rope-average gold — reads continuous with the span
    knot.specular = 0.3
    knot.shininess = 0.3


def _load_hook_stls(spec: mujoco.MjSpec) -> None:
    """Parametric rig parts from hardware/tug-rig (STL, trunk-local)."""
    for mesh_name, filename in (("tug_collar_stl", "tug_collar.stl"),
                                ("tug_clamp_screw_stl", "tug_clamp_screw.stl")):
        spec.add_mesh(name=mesh_name, file=str(_ROBOT_DIR / "assets" / filename))


def _add_harness_rings(spec: mujoco.MjSpec, n_per_team: int) -> None:
    red, blue = team_prefixes(n_per_team)
    for prefix, team in [(p_, "red") for p_ in red] + [(p_, "blue") for p_ in blue]:
        trunk = _find_body(spec, f"{prefix}trunk_base")
        # Clamp collar (hardware/tug-rig/cad_collar.py), trunk-local: one
        # band around the torso, tightened by the side screw, with the tow
        # eyes integrated front and back.
        trunk.add_geom(
            name=f"{prefix}tug_collar",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="tug_collar_stl",
            material="tug_carabiner",
            contype=0,
            conaffinity=0,
            density=0.0,
        )
        trunk.add_geom(
            name=f"{prefix}tug_clamp_screw",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="tug_clamp_screw_stl",
            material="tug_steel",
            contype=0,
            conaffinity=0,
            density=0.0,
        )


def _span_hook_sites(pa: str, pb: str, red: list[str]) -> tuple[str, str]:
    """Force path through the duck: pulled at the butt, pulls with the
    chest. Each span uses the hooks facing each other — butt on the
    centre side, chest on the away side."""
    pa_red, pb_red = pa in red, pb in red
    if pa_red and pb_red:
        return "rope_hook", "rope_hook_chest"
    if pa_red and not pb_red:
        return "rope_hook", "rope_hook"
    return "rope_hook_chest", "rope_hook"


@dataclass
class SpanVisual:
    body_id: int
    geom_id: int
    trunk_a_id: int
    trunk_b_id: int
    local_a: np.ndarray
    local_b: np.ndarray
    nominal: float
    variant_mesh_ids: np.ndarray


def resolve_rope_visuals(model: mujoco.MjModel, n_per_team: int,
                         spacing: float = DUCK_SPACING,
                         gap: float = CENTER_GAP) -> list[SpanVisual]:
    red, blue = team_prefixes(n_per_team)
    chain = list(reversed(red)) + blue
    spans = []
    for idx, (pa, pb) in enumerate(zip(chain[:-1], chain[1:])):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"tug_span_{idx}")
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"tug_span_{idx}")
        variant_ids = np.array([
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, f"tug_var_{ci}_{si}")
             for si in range(len(SAG_BINS))]
            for ci in range(len(CHORD_BINS))
        ])
        site_a, site_b = _span_hook_sites(pa, pb, red)
        team_a = "red" if pa in red else "blue"
        team_b = "red" if pb in red else "blue"
        spans.append(SpanVisual(
            body_id=body_id,
            geom_id=geom_id,
            trunk_a_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{pa}trunk_base"),
            trunk_b_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{pb}trunk_base"),
            local_a=(ring_local(team_a) if site_a == "rope_hook" else chest_local(team_a)).copy(),
            local_b=(ring_local(team_b) if site_b == "rope_hook" else chest_local(team_b)).copy(),
            nominal=(gap if pa == red[0] else spacing),
            variant_mesh_ids=variant_ids,
        ))
    return spans


def _site_world(data: mujoco.MjData, body_id: int, local: np.ndarray,
                out: np.ndarray) -> None:
    mujoco.mju_rotVecQuat(out, local, data.xquat[body_id])
    out += data.xpos[body_id]


def _local_span_curve(chord: float, sag: float) -> np.ndarray:
    """Sagging rope in its local frame: x from -chord/2..+chord/2, ends AT
    the wrap sites with ZERO slope (sin² profile) — the rope leaves the
    knot exactly along the pull direction, so it passes the eye's tunnel
    straight, no rim contact. A parabola leaves at ±4·sag slope and clips."""
    t = np.linspace(0.0, 1.0, SPAN_T_SEGMENTS)
    points = np.zeros((SPAN_T_SEGMENTS, 3))
    points[:, 0] = (t - 0.5) * chord
    points[:, 2] = -sag * np.sin(np.pi * t) ** 2
    points[:, 1] = min(0.006, 0.5 * sag + 0.001) * np.sin(np.pi * t)
    return points


SPAN_RADIUS = 0.004   # Ø8 mm rope — threads the Ø9 mm pad-eye holes cleanly

EYE_RING_MAJOR = 0.008   # pad-eye ring radius (mirrors hardware/tug-rig/cad_collar.py EYE_MAJOR)
EYE_INNER_R = 0.0055     # pad-eye hole radius (EYE_MAJOR - EYE_TUBE)
KNOT_MAJOR = 0.0045      # rope coil cinching the eye's rim on the pull side (lark's head)


def _knot_mesh(spec: mujoco.MjSpec, name: str = "tug_knot") -> None:
    """Lark's-head bight on the eye's near rim: ONE clean loop of rope in
    the xy plane wrapping the rim section — inner strand threads the hole
    along y (out one face, back the other), outer strand wraps the plate's
    outside. The span rope ends into the loop. No coil blob — the two
    strands through the tunnel must READ as rope in the hole.
    """
    t = np.linspace(0.0, 2.0 * np.pi, 41)
    a, b = 0.007, 0.0075
    loop_pts = np.stack([a * np.cos(t), b * np.sin(t), np.zeros_like(t)], axis=1)
    verts, normals, uvs, faces = _sweep_smooth(loop_pts)
    uvs[:, 0] *= (2 * np.pi * a) / (3.5 * 2.0 * SPAN_RADIUS)
    mesh = spec.add_mesh(name=name)
    mesh.uservert = verts.flatten().astype(np.float32)
    mesh.usernormal = normals.flatten().astype(np.float32)
    mesh.userface = faces.flatten()
    mesh.usertexcoord = uvs.flatten().astype(np.float32)


def _add_knots(spec: mujoco.MjSpec, knot_use: list[tuple[str, str, float]], red: list[str]) -> None:
    """A wound knot on every pad eye that carries a rope — coil on the rim
    FACING the incoming rope (sign = direction of the other duck)."""
    for prefix, site, sign in knot_use:
        trunk = _find_body(spec, f"{prefix}trunk_base")
        team = "red" if prefix in red else "blue"
        eye = ring_local(team) if site == "rope_hook" else chest_local(team)
        trunk.add_geom(
            name=f"{prefix}tug_knot_{site}",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="tug_knot",
            pos=(eye[0] + sign * EYE_RING_MAJOR, 0.0, eye[2]),
            material="tug_knot",
            contype=0,
            conaffinity=0,
            density=0.0,
        )
SPAN_SMOOTH_ALPHA = 16


def _sweep_smooth(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Smooth cylinder swept along `points` (clean silhouette — the
    reference rope's twist is texture, not geometry; a 3-ply silhouette
    aliases into ragged fuzz at video resolution). (verts, normals, uvs, faces)"""
    n = len(points)
    tangents = np.gradient(points, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    ref = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    flip = np.abs((tangents * ref).sum(axis=1)) > 0.9
    ref[flip] = np.array([1.0, 0.0, 0.0])
    u = ref - (ref * tangents).sum(axis=1, keepdims=True) * tangents
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    w = np.cross(tangents, u)
    alphas = 2.0 * np.pi * np.arange(SPAN_SMOOTH_ALPHA) / SPAN_SMOOTH_ALPHA
    cos_a = np.cos(alphas)[None, :, None]
    sin_a = np.sin(alphas)[None, :, None]
    normals = cos_a * u[:, None, :] + sin_a * w[:, None, :]      # (n, A, 3)
    verts = points[:, None, :] + SPAN_RADIUS * normals
    verts = verts.reshape(-1, 3)
    normals = normals.reshape(-1, 3)
    uvs = np.zeros((n * SPAN_SMOOTH_ALPHA, 2))
    t_frac = np.linspace(0.0, 1.0, n)
    uvs[:, 0] = np.repeat(t_frac, SPAN_SMOOTH_ALPHA)
    uvs[:, 1] = np.tile(alphas / (2.0 * np.pi), n)
    faces = []
    for i in range(n - 1):
        for j in range(SPAN_SMOOTH_ALPHA):
            j2 = (j + 1) % SPAN_SMOOTH_ALPHA
            a = i * SPAN_SMOOTH_ALPHA + j
            b = (i + 1) * SPAN_SMOOTH_ALPHA + j
            c = (i + 1) * SPAN_SMOOTH_ALPHA + j2
            d = i * SPAN_SMOOTH_ALPHA + j2
            faces.append((a, b, c))
            faces.append((a, c, d))
    # End caps: an open tube end reads as a SEVERED rope (dark hollow ring).
    for ring, tip_i, flip in ((0, 0, True), (n - 1, n - 1, False)):
        tip = len(verts)
        verts = np.vstack([verts, points[tip_i][None, :]])
        normals = np.vstack([normals, -tangents[tip_i] if flip else tangents[tip_i]])
        uvs = np.vstack([uvs, [tip_i / (n - 1), 0.5]])
        for j in range(SPAN_SMOOTH_ALPHA):
            j2 = (j + 1) % SPAN_SMOOTH_ALPHA
            a, b = ring * SPAN_SMOOTH_ALPHA + j, ring * SPAN_SMOOTH_ALPHA + j2
            faces.append((tip, a, b) if flip else (tip, b, a))
    return verts, normals, uvs, np.array(faces, dtype=np.int32)


def _sweep_twisted(points: np.ndarray, twists: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """3-ply twisted tube swept along `points`: (verts, normals, faces);
    topology/UVs are constant for a given resolution."""
    n = len(points)
    tangents = np.gradient(points, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    ref = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    flip = np.abs((tangents * ref).sum(axis=1)) > 0.9
    ref[flip] = np.array([1.0, 0.0, 0.0])
    u = ref - (ref * tangents).sum(axis=1, keepdims=True) * tangents
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    w = np.cross(tangents, u)
    alphas = 2.0 * np.pi * np.arange(SPAN_ALPHA_SEGMENTS) / SPAN_ALPHA_SEGMENTS
    cos_a = np.cos(alphas)[None, :, None]
    sin_a = np.sin(alphas)[None, :, None]
    verts = []
    t_frac = np.linspace(0.0, 1.0, n)
    for k in range(3):
        phi = 2.0 * np.pi * (k / 3 + twists * t_frac)
        u2 = np.cos(phi)[:, None] * u + np.sin(phi)[:, None] * w
        centers = points + ROPE_PLY_CENTER_R * u2
        side = np.cross(tangents, u2)
        rings = (centers[:, None, :] + ROPE_PLY_TUBE_R
                 * (cos_a * u2[:, None, :] + sin_a * side[:, None, :]))
        verts.append(rings.reshape(-1, 3))
    verts = np.concatenate(verts)
    faces = []
    for k in range(3):
        base = k * n * SPAN_ALPHA_SEGMENTS
        for i in range(n - 1):
            for j in range(SPAN_ALPHA_SEGMENTS):
                j2 = (j + 1) % SPAN_ALPHA_SEGMENTS
                a = base + i * SPAN_ALPHA_SEGMENTS + j
                b = base + (i + 1) * SPAN_ALPHA_SEGMENTS + j
                c = base + (i + 1) * SPAN_ALPHA_SEGMENTS + j2
                d = base + i * SPAN_ALPHA_SEGMENTS + j2
                faces.append((a, b, c))
                faces.append((a, c, d))
    faces = np.array(faces, dtype=np.int32)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_n = np.cross(v1 - v0, v2 - v0)
    normals = np.zeros_like(verts)
    for col in range(3):
        np.add.at(normals, faces[:, col], face_n)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return verts, normals / norms, faces


def span_uvs() -> np.ndarray:
    """Constant UVs matching the sweep topology (v wraps plies)."""
    uvs = []
    n = SPAN_T_SEGMENTS
    for k in range(3):
        for i in range(n):
            for j in range(SPAN_ALPHA_SEGMENTS):
                uvs.append((i / (n - 1) * 4.0, (j / SPAN_ALPHA_SEGMENTS + k / 3) % 1.0))
    return np.array(uvs, dtype=np.float32)


def _matte_floor(spec: mujoco.MjSpec) -> None:
    """The reference video's checker floor is matte; scene.xml's groundplane
    material carries reflectance 0.2 which mirror-images every duck."""
    for mat in spec.materials:
        if mat.name == "groundplane":
            mat.reflectance = 0.0


def _add_tug_lighting(spec: mujoco.MjSpec) -> None:
    """Warm key + cool fill on top of the scene's single directional light —
    the rope crowns need a highlight direction and the grooves need a soft
    counter-light to read depth without true AO."""
    spec.visual.quality.offsamples = 8
    spec.visual.quality.shadowsize = 4096
    spec.worldbody.add_light(
        name="tug_key",
        pos=(0.0, 1.8, 2.0),
        dir=(0.0, -0.62, -0.78),
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=(0.9, 0.82, 0.68),
        specular=(0.5, 0.45, 0.35),
    )
    spec.worldbody.add_light(
        name="tug_fill",
        pos=(0.0, -1.5, 1.0),
        dir=(0.0, 0.55, -0.84),
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=(0.25, 0.30, 0.38),
        specular=(0.1, 0.12, 0.15),
    )


def _build_span_variants(spec: mujoco.MjSpec) -> None:
    _add_twist_material(spec)
    for ci, chord in enumerate(CHORD_BINS):
        for si, sag_frac in enumerate(SAG_BINS):
            sag = sag_frac * SAG_MAX
            points = _local_span_curve(float(chord), sag)
            verts, normals, uvs, faces = _sweep_smooth(points)
            # Twist pitch ≈ 3.5 rope diameters, measured off the reference.
            uvs[:, 0] *= float(chord) / (3.5 * 2.0 * SPAN_RADIUS)
            mesh = spec.add_mesh(name=f"tug_var_{ci}_{si}")
            mesh.uservert = verts.flatten().astype(np.float32)
            mesh.usernormal = normals.flatten().astype(np.float32)
            mesh.userface = faces.flatten()
            mesh.usertexcoord = uvs.flatten().astype(np.float32)


def update_rope_visuals(model: mujoco.MjModel, data: mujoco.MjData,
                        spans: list[SpanVisual], mjr_context=None) -> None:
    """Point each span geom at the nearest swept-rope variant and pose its
    mocap body on the chord between the two wrap endpoints."""
    p0 = np.zeros(3)
    p1 = np.zeros(3)
    quat = np.zeros(4)
    rot = np.zeros(9)
    for span in spans:
        _site_world(data, span.trunk_a_id, span.local_a, p0)
        _site_world(data, span.trunk_b_id, span.local_b, p1)
        chord_vec = p1 - p0
        chord = float(np.linalg.norm(chord_vec))
        if chord < 1e-6:
            continue
        # The rope's tip lands exactly ON the bight loop's outer arc
        # (eye + EYE_MAJOR + loop_a = eye + 15 mm, past the plate's outer
        # edge) and merges into the knot — no free end anywhere.
        back = EYE_RING_MAJOR + 0.007
        chord_vec /= chord
        e0 = p0 + chord_vec * back
        e1 = p1 - chord_vec * back
        span_vec = e1 - e0
        span_len = float(np.linalg.norm(span_vec))
        slack = max(0.0, span.nominal - chord)
        sag = min(SAG_MAX, SAG_PER_SLACK * slack + 0.002)
        # largest bin whose RENDERED half-length (bin/2 + rope tube) does
        # not exceed the span — a longer rope pokes its capped tip out of
        # the knot (reads as severed); a shorter one tucks inside
        below = np.nonzero(CHORD_BINS <= span_len - 2 * SPAN_RADIUS)[0]
        ci = int(below[-1]) if len(below) else 0
        si = int(np.argmin(np.abs(np.array(SAG_BINS) * SAG_MAX - sag)))
        model.geom_dataid[span.geom_id] = span.variant_mesh_ids[ci, si]
        # Mocap frame: x along the chord; local +z as close to world-UP as
        # the chord allows, so the baked sag (local -z) points DOWN.
        # (The previous mapping put z_axis = down, which flipped the baked
        # sag skyward — invisible in taut match footage, obvious at slack.)
        x_axis = span_vec / span_len
        up = np.array([0.0, 0.0, 1.0])
        z_axis = up - np.dot(up, x_axis) * x_axis
        zn = np.linalg.norm(z_axis)
        z_axis = z_axis / zn if zn > 1e-6 else np.array([0.0, 0.0, 1.0])
        y_axis = np.cross(z_axis, x_axis)
        rot[0::3] = x_axis
        rot[1::3] = y_axis
        rot[2::3] = z_axis
        mujoco.mju_mat2Quat(quat, rot)
        mocap_id = model.body_mocapid[span.body_id]
        data.mocap_pos[mocap_id] = (e0 + e1) / 2.0
        data.mocap_quat[mocap_id] = quat


def _add_line_geom(spec: mujoco.MjSpec, name: str, x: float,
                   rgba: tuple[float, float, float, float]) -> None:
    spec.worldbody.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.006, 1.2, 0.001),
        pos=(x, 0.0, 0.001),
        rgba=rgba,
        contype=0,
        conaffinity=0,
    )


def build_tug_spec(n_per_team: int = 5, spacing: float = DUCK_SPACING,
                   gap: float = CENTER_GAP, win_x: float = WIN_X,
                   red_rgba=RED_RGBA, blue_rgba=BLUE_RGBA) -> mujoco.MjSpec:
    """Composite spec: floor + n red ducks (x<0, facing -x) + n blue ducks."""
    red, blue = team_prefixes(n_per_team)

    parent = mujoco.MjSpec.from_file(str(_SCENE_XML))
    _tint_shells(parent, red_rgba)          # unprefixed robot = red duck 0
    _add_rope_sites(parent, "red")

    for prefix, rgba, team in [(p, red_rgba, "red") for p in red[1:]] + [(p, blue_rgba, "blue") for p in blue]:
        child = mujoco.MjSpec.from_file(str(_ROBOT_XML))
        _tint_shells(child, rgba)
        _add_rope_sites(child, team)
        frame = parent.worldbody.add_frame()
        parent.attach(child, prefix=prefix, frame=frame)

    # Rope chain, far red → center → far blue. Two cords per link (left/right
    # waist sites) so a link transmits no yaw torque between teammates.
    chain = list(reversed(red)) + blue
    knot_use: list[tuple[str, str, float]] = []
    for pa, pb in zip(chain[:-1], chain[1:]):
        nominal = gap if pa == red[0] else spacing
        link_len = nominal + ROPE_SLACK
        site_a, site_b = _span_hook_sites(pa, pb, red)
        # The knot coils the rim FACING the other duck. Red ducks are
        # rotated 180° (world +x = their trunk-local -x), blue are not.
        knot_use.append((pa, site_a, (1.0 if pa in blue else -1.0) * +1.0))
        knot_use.append((pb, site_b, (1.0 if pb in blue else -1.0) * -1.0))
        cord = parent.add_tendon(
            name=f"tug_{pa or 'r0_'}{pb}cord",
            stiffness=ROPE_STIFFNESS,
            springlength=(0.0, link_len),
            limited=True,
            range=(0.0, link_len + ROPE_LIMIT_MARGIN),
            width=ROPE_WIDTH,
            rgba=ROPE_RGBA,
            group=4,  # physics only; the swept variants do the visuals
            solref_limit=(0.02, 1.0),
            solimp_limit=(0.90, 0.95, 0.001, 0.5, 2.0),
        )
        cord.wrap_site(f"{pa}{site_a}")
        cord.wrap_site(f"{pb}{site_b}")

    _add_hemp_material(parent)
    _matte_floor(parent)
    _add_tug_lighting(parent)
    _add_rig_materials(parent)
    _load_hook_stls(parent)
    _add_harness_rings(parent, n_per_team)
    _knot_mesh(parent)
    _add_knots(parent, knot_use, red)
    _build_span_variants(parent)
    n_strands = 2 * n_per_team - 1
    for idx in range(n_strands):
        body = parent.worldbody.add_body(name=f"tug_span_{idx}", mocap=True)
        body.add_geom(
            name=f"tug_span_{idx}",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="tug_var_5_1",
            material="tug_twist",
            contype=0,
            conaffinity=0,
        )

    _add_line_geom(parent, "center_line", 0.0, (1.0, 1.0, 1.0, 1.0))
    _add_line_geom(parent, "win_line_red", -win_x, red_rgba)
    _add_line_geom(parent, "win_line_blue", win_x, blue_rgba)
    # Offscreen framebuffer big enough for 1280x720+ renders.
    parent.visual.global_.offwidth = 1920
    parent.visual.global_.offheight = 1080
    return parent


@dataclass
class DuckRig:
    """Runtime indices for one prefixed robot inside the compiled model."""

    prefix: str
    team: str
    trunk_body_id: int
    free_qpos_adr: int
    imu_ang_vel_adr: int
    actuator_ids: np.ndarray          # (14,) actuator ids in JOINT_NAMES order
    joint_qpos_idx: np.ndarray        # (14,)
    joint_qvel_idx: np.ndarray        # (14,)
    last_action: np.ndarray = field(default_factory=lambda: np.zeros(14, dtype=np.float32))


def find_duck_rigs(model: mujoco.MjModel, n_per_team: int) -> list[DuckRig]:
    red, blue = team_prefixes(n_per_team)
    rigs = []
    for prefix, team in [(p, "red") for p in red] + [(p, "blue") for p in blue]:
        trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}trunk_base")
        if trunk_id < 0:
            raise KeyError(f"body {prefix}trunk_base missing")
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}trunk_base_freejoint")
        sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"{prefix}imu_ang_vel")
        actuator_ids = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}{jn}")
            for jn in JOINT_NAMES
        ], dtype=int)
        if (actuator_ids < 0).any():
            missing = [jn for jn, a in zip(JOINT_NAMES, actuator_ids) if a < 0]
            raise KeyError(f"actuators missing for {prefix!r}: {missing}")
        rigs.append(DuckRig(
            prefix=prefix,
            team=team,
            trunk_body_id=trunk_id,
            free_qpos_adr=int(model.jnt_qposadr[joint_id]),
            imu_ang_vel_adr=int(model.sensor_adr[sensor_id]),
            actuator_ids=actuator_ids,
            joint_qpos_idx=np.array([int(model.jnt_qposadr[model.actuator_trnid[a, 0]]) for a in actuator_ids]),
            joint_qvel_idx=np.array([int(model.jnt_dofadr[model.actuator_trnid[a, 0]]) for a in actuator_ids]),
        ))
    return rigs


def _quat_rotate_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    u = np.array([x, y, z])
    return 2.0 * np.dot(u, vec) * u + (w * w - np.dot(u, u)) * vec - 2.0 * w * np.cross(u, vec)


def compute_obs(model: mujoco.MjModel, data: mujoco.MjData, rigs: list[DuckRig],
                pull_speeds: np.ndarray) -> np.ndarray:
    """(n_ducks, 61) obs batch matching the walking/running ONNX contract:
    ang_vel(3) | projected_gravity(3) | joint_pos_rel(14) | joint_vel(14) |
    last_action(14) | twist(3) | head_pose(4 zeros) | body_pose(6 zeros)."""
    obs = np.zeros((len(rigs), 61), dtype=np.float32)
    world_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    for k, rig in enumerate(rigs):
        gyro = data.sensordata[rig.imu_ang_vel_adr:rig.imu_ang_vel_adr + 3]
        quat = data.xquat[rig.trunk_body_id]
        proj_g = _quat_rotate_inverse(quat, world_gravity)
        joint_pos = data.qpos[rig.joint_qpos_idx] - DEFAULT_POSE
        joint_vel = data.qvel[rig.joint_qvel_idx]
        row = np.concatenate([
            gyro, proj_g, joint_pos, joint_vel, rig.last_action,
            [pull_speeds[k], 0.0, 0.0], np.zeros(10, dtype=np.float32),
        ])
        obs[k] = row
    return obs


def apply_actions(data: mujoco.MjData, rigs: list[DuckRig], actions: np.ndarray) -> None:
    for k, rig in enumerate(rigs):
        action = actions[k].astype(np.float32)
        data.ctrl[rig.actuator_ids] = DEFAULT_POSE + action
        rig.last_action = action.copy()


def team_states(model: mujoco.MjModel, data: mujoco.MjData, rigs: list[DuckRig],
                ) -> dict[str, dict[str, object]]:
    world_gravity = np.array([0.0, 0.0, -1.0])
    states: dict[str, dict[str, object]] = {}
    for team in ("red", "blue"):
        team_rigs = [r for r in rigs if r.team == team]
        fallen = 0
        for rig in team_rigs:
            z = data.xpos[rig.trunk_body_id][2]
            pg_z = _quat_rotate_inverse(data.xquat[rig.trunk_body_id], world_gravity)[2]
            if z < FALLEN_TRUNK_Z or pg_z > FALLEN_GRAVITY_Z:
                fallen += 1
        states[team] = {
            "fallen": fallen,
            "center_x": float(data.xpos[team_rigs[0].trunk_body_id][0]),
        }
    return states


def check_winner(model: mujoco.MjModel, data: mujoco.MjData, rigs: list[DuckRig],
                 win_x: float = WIN_X, min_fallen: int = MIN_FALLEN) -> tuple[str | None, str | None]:
    """Return (winner, reason) or (None, None) while the round is live."""
    states = team_states(model, data, rigs)
    for team, other in (("red", "blue"), ("blue", "red")):
        if states[team]["fallen"] >= min_fallen:
            return other, f"{team} team down ({states[team]['fallen']} fallen)"
    midpoint_x = (states["red"]["center_x"] + states["blue"]["center_x"]) / 2.0
    if midpoint_x < -win_x:
        return "red", f"rope pulled past red line (midpoint {midpoint_x:+.2f} m)"
    if midpoint_x > win_x:
        return "blue", f"rope pulled past blue line (midpoint {midpoint_x:+.2f} m)"
    return None, None
