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


def _add_rope_sites(spec: mujoco.MjSpec) -> None:
    trunk = _find_body(spec, "trunk_base")
    for side, y in (("left", ROPE_SITE_Y), ("right", -ROPE_SITE_Y)):
        trunk.add_site(
            name=f"rope_{side}",
            pos=(0.0, y, ROPE_SITE_Z),
            size=(0.004,),
            rgba=ROPE_RGBA,
        )
    trunk.add_site(
        name="rope_chest",
        pos=(ROPE_CHEST_X, 0.0, ROPE_SITE_Z),
        size=(0.004,),
        rgba=ROPE_RGBA,
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


# Hemp rope PBR maps: ambientCG Rope001 (CC0, https://ambientcg.com/view?id=Rope001),
# natural twisted fibre — the same look as the reference tug-of-war video.
_ROPE_COLOR = _ROBOT_DIR / "assets" / "rope_hemp_color.png"
_ROPE_NORMAL = _ROBOT_DIR / "assets" / "rope_hemp_normal.png"
_ROPE_ROUGHNESS = _ROBOT_DIR / "assets" / "rope_hemp_roughness.png"


def _add_hemp_material(spec: mujoco.MjSpec) -> None:
    for name, path, role in (("tug_hemp_tex", _ROPE_COLOR, mujoco.mjtTextureRole.mjTEXROLE_RGB),
                             ("tug_hemp_nrm", _ROPE_NORMAL, mujoco.mjtTextureRole.mjTEXROLE_NORMAL),
                             ("tug_hemp_rgh", _ROPE_ROUGHNESS, mujoco.mjtTextureRole.mjTEXROLE_ROUGHNESS)):
        tex = spec.add_texture(name=name)
        tex.type = mujoco.mjtTexture.mjTEXTURE_2D
        tex.file = str(path)
    mat = spec.add_material(name="tug_hemp")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "tug_hemp_tex"
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_NORMAL] = "tug_hemp_nrm"
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_ROUGHNESS] = "tug_hemp_rgh"
    mat.texrepeat = (12.0, 1.0)
    mat.texuniform = True


# Waist wrap: a 3.5-turn twisted-rope spiral hugging the belly, tilted and
# slightly irregular like a hand-tied coil (reference video: 3 snug turns,
# visibly tilted, working end leaving from mid-coil). One continuous rope:
# the inter-duck strands pick up exactly at the wrap's two ends.
WRAP_ELLIPSE_X = 0.041    # torso half-depth + rope radius — snug on the shell
WRAP_ELLIPSE_Y = 0.055    # torso half-width + rope radius
WRAP_Z0 = 0.004           # bottom turn (center side) — belly, below mid-torso
WRAP_Z1 = 0.030           # top turn (away side)
WRAP_TURNS = 3.5
WRAP_TILT = np.radians(12.0)   # coil plane tips up toward the duck's back
WRAP_JITTER = 0.0015           # per-turn radius/z wobble — kills the CNC look
WRAP_FRONT_LOCAL = np.array([WRAP_ELLIPSE_X, 0.0, WRAP_Z1])
WRAP_BACK_LOCAL = np.array([-WRAP_ELLIPSE_X, 0.0, WRAP_Z0])
ROPE_PLY_CENTER_R = 0.0038   # 3-ply rope ~16 mm overall — chunky like the reference
ROPE_PLY_TUBE_R = 0.0043
ROPE_PLY_TWISTS_PER_TURN = 4  # ply rotations per coil turn


def _wrap_path(rng: np.random.Generator):
    jitter_r = rng.uniform(-WRAP_JITTER, WRAP_JITTER, 16)
    jitter_z = rng.uniform(-WRAP_JITTER, WRAP_JITTER, 16)

    def path(t: float) -> np.ndarray:
        theta = 2.0 * np.pi * WRAP_TURNS * t   # 0 → 7π: back → 3.5 turns → front
        seg = min(int(t * 16), 15)
        r_scale = 1.0 + jitter_r[seg]
        point = np.array([
            WRAP_ELLIPSE_X * r_scale * np.cos(theta),
            WRAP_ELLIPSE_Y * r_scale * np.sin(theta),
            WRAP_Z0 + (WRAP_Z1 - WRAP_Z0) * t + jitter_z[seg],
        ])
        point[2] += np.tan(WRAP_TILT) * point[0]
        return point
    return path


def _straight_path(length: float):
    def path(t: float) -> np.ndarray:
        return np.array([0.0, 0.0, (t - 0.5) * length])
    return path


def _twisted_tube_mesh(spec: mujoco.MjSpec, name: str,
                       path, t_segments: int, alpha_segments: int,
                       twists: float, uv_repeats: float,
                       sub_fibers: int = 5, flyaway: int = 0,
                       seed: int = 42) -> None:
    """Two-level twisted rope following an arbitrary path (inline mesh, UVs).

    Level 1: three plies spiral around the path. Level 2 (sub_fibers > 1):
    each ply is itself a bundle of thinner yarns spiralling inside the ply
    tube — the same construction as real 3-ply hemp. `flyaway` bakes that
    many short fibre spikes into the mesh, sticking a few mm off the yarns:
    the fuzzy halo of real manila rope."""
    rng = np.random.default_rng(seed)
    verts: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    faces: list[tuple[int, int, int]] = []
    eps = 1e-3

    def frame(t: float):
        center = path(t)
        tangent = path(min(1.0, t + eps)) - path(max(0.0, t - eps))
        tangent /= np.linalg.norm(tangent)
        ref = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(tangent, ref)) > 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        u = ref - np.dot(ref, tangent) * tangent
        u /= np.linalg.norm(u)
        return center, tangent, u, np.cross(tangent, u)

    yarns_per_ply = max(1, sub_fibers)
    yarn_center_r = ROPE_PLY_TUBE_R * 0.55 if yarns_per_ply > 1 else 0.0
    yarn_tube_r = ROPE_PLY_TUBE_R * (0.5 if yarns_per_ply > 1 else 1.0)
    sub_twists = twists * 3.0   # yarns counter-twist faster than plies

    for k in range(3):
        for f in range(yarns_per_ply):
            base = len(verts)
            for i in range(t_segments):
                t = i / (t_segments - 1)
                center, tangent, u, w = frame(t)
                phi = 2.0 * np.pi * (k / 3 + twists * t)
                u2 = np.cos(phi) * u + np.sin(phi) * w
                ply_center = center + ROPE_PLY_CENTER_R * u2
                yarn_center = ply_center
                if yarns_per_ply > 1:
                    psi = 2.0 * np.pi * (f / yarns_per_ply + sub_twists * t)
                    yarn_center = ply_center + yarn_center_r * (np.cos(psi) * u2 + np.sin(psi) * np.cross(tangent, u2))
                yarn_out = yarn_center - ply_center
                n = np.linalg.norm(yarn_out)
                yarn_out = yarn_out / n if n > 1e-9 else u2
                for j in range(alpha_segments):
                    alpha = 2.0 * np.pi * j / alpha_segments
                    point = yarn_center + yarn_tube_r * (
                        np.cos(alpha) * yarn_out + np.sin(alpha) * np.cross(tangent, yarn_out))
                    verts.append(tuple(point))
                    uvs.append((t * uv_repeats, (j / alpha_segments + f / yarns_per_ply) % 1.0))
            for i in range(t_segments - 1):
                for j in range(alpha_segments):
                    j2 = (j + 1) % alpha_segments
                    a = base + i * alpha_segments + j
                    b = base + (i + 1) * alpha_segments + j
                    c = base + (i + 1) * alpha_segments + j2
                    d = base + i * alpha_segments + j2
                    faces.append((a, b, c))
                    faces.append((a, c, d))

    for _ in range(flyaway):
        # A tapered 3-segment fibre sticking 2-5 mm off a random yarn.
        t = rng.uniform(0.02, 0.98)
        k = rng.integers(0, 3)
        f = rng.integers(0, yarns_per_ply)
        center, tangent, u, w = frame(t)
        phi = 2.0 * np.pi * (k / 3 + twists * t)
        u2 = np.cos(phi) * u + np.sin(phi) * w
        ply_center = center + ROPE_PLY_CENTER_R * u2
        psi = 2.0 * np.pi * (f / yarns_per_ply + sub_twists * t)
        yarn_center = ply_center + yarn_center_r * (np.cos(psi) * u2 + np.sin(psi) * np.cross(tangent, u2))
        out_dir = u2 * 0.7 + w * rng.uniform(-0.5, 0.5) + tangent * rng.uniform(-0.4, 0.4)
        out_dir /= np.linalg.norm(out_dir)
        length = rng.uniform(0.002, 0.005)
        root = yarn_center + yarn_tube_r * out_dir
        mid = root + out_dir * length * 0.6 + tangent * rng.uniform(-0.001, 0.001)
        tip = root + out_dir * length
        base = len(verts)
        for point, radius in ((root, 0.00035), (mid, 0.00022), (tip, 0.00006)):
            side = np.cross(out_dir, tangent)
            n_side = np.linalg.norm(side)
            side = side / n_side if n_side > 1e-9 else u
            for j in range(4):
                ang = np.pi / 2 * j
                verts.append(tuple(point + radius * (np.cos(ang) * side + np.sin(ang) * np.cross(out_dir, side))))
                uvs.append((t * uv_repeats, 0.5))
        for ring in range(2):
            for j in range(4):
                j2 = (j + 1) % 4
                a = base + ring * 4 + j
                b = base + ring * 4 + j2
                c = base + (ring + 1) * 4 + j2
                d = base + (ring + 1) * 4 + j
                faces.append((a, b, c))
                faces.append((a, c, d))
        for j in range(4):  # tip cap fan
            faces.append((base + 8 + j, base + 8 + (j + 1) % 4, base + 8 + (j + 2) % 4))

    mesh = spec.add_mesh(name=name)
    mesh.uservert = np.array(verts, dtype=np.float32).flatten()
    mesh.userface = np.array(faces, dtype=np.int32).flatten()
    mesh.usertexcoord = np.array(uvs, dtype=np.float32).flatten()


def _add_waist_wraps(spec: mujoco.MjSpec, n_per_team: int) -> None:
    red, blue = team_prefixes(n_per_team)
    for prefix in red + blue:
        trunk = _find_body(spec, f"{prefix}trunk_base")
        trunk.add_geom(
            name=f"{prefix}tug_wrap",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="tug_wrap",
            material="tug_hemp",
            contype=0,
            conaffinity=0,
            density=0.0,   # pure visual — no added mass or inertia
        )


SPAN_PIECE_LENGTH = 0.05    # fixed-length twisted piece, chains overlap
SPAN_PIECES = 6             # pieces per inter-duck strand
SAG_SUBSEGMENTS = SPAN_PIECES


@dataclass
class RopeVisual:
    body_id: int
    geom_id: int
    trunk_a_id: int
    trunk_b_id: int
    local_a: np.ndarray    # endpoint on duck a (its back/center-side wrap end)
    local_b: np.ndarray    # endpoint on duck b (its front/away-side wrap end)
    nominal: float
    fraction: float        # sub-segment start fraction along the strand


def resolve_rope_visuals(model: mujoco.MjModel, n_per_team: int,
                         spacing: float = DUCK_SPACING,
                         gap: float = CENTER_GAP) -> list[RopeVisual]:
    red, blue = team_prefixes(n_per_team)
    chain = list(reversed(red)) + blue
    visuals = []
    for idx, (pa, pb) in enumerate(zip(chain[:-1], chain[1:])):
        nominal = (gap if pa == red[0] else spacing)
        trunk_a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{pa}trunk_base")
        trunk_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{pb}trunk_base")
        for sub in range(SAG_SUBSEGMENTS):
            body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, f"tugvis_{idx}_{sub}")
            geom_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, f"tugvis_{idx}_{sub}")
            visuals.append(RopeVisual(
                body_id=body_id,
                geom_id=geom_id,
                trunk_a_id=trunk_a,
                trunk_b_id=trunk_b,
                local_a=WRAP_BACK_LOCAL.copy(),
                local_b=WRAP_FRONT_LOCAL.copy(),
                nominal=nominal,
                fraction=sub / SAG_SUBSEGMENTS,
            ))
    return visuals


def _site_world(data: mujoco.MjData, body_id: int, local: np.ndarray,
                out: np.ndarray) -> None:
    mujoco.mju_rotVecQuat(out, local, data.xquat[body_id])
    out += data.xpos[body_id]


def update_rope_visuals(model: mujoco.MjModel, data: mujoco.MjData,
                        visuals: list[RopeVisual]) -> None:
    """Lay every fixed-length twisted piece along its sagging strand curve.

    Pieces keep their mesh length (mesh scale is an asset-level property and
    cannot change per frame); consecutive pieces overlap into each other and
    the end pieces sink into the waist wraps, hiding every joint."""
    quat = np.zeros(4)
    p0 = np.zeros(3)
    p1 = np.zeros(3)
    for vis in visuals:
        _site_world(data, vis.trunk_a_id, vis.local_a, p0)
        _site_world(data, vis.trunk_b_id, vis.local_b, p1)
        chord = float(np.linalg.norm(p1 - p0))
        slack = max(0.0, vis.nominal - chord)
        sag = min(SAG_MAX, SAG_PER_SLACK * slack + 0.002)
        t0, t1 = vis.fraction, vis.fraction + 1.0 / SAG_SUBSEGMENTS
        tm = (t0 + t1) / 2.0
        center = p0 + (p1 - p0) * tm
        center[2] -= 4.0 * sag * tm * (1.0 - tm)
        # Tangent of the sag parabola at tm.
        tangent = (p1 - p0).copy()
        tangent[2] -= 4.0 * sag * (1.0 - 2.0 * tm)
        norm = float(np.linalg.norm(tangent))
        if norm < 1e-6:
            continue
        mocap_id = model.body_mocapid[vis.body_id]
        data.mocap_pos[mocap_id] = center
        mujoco.mju_quatZ2Vec(quat, tangent / norm)
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
    _add_rope_sites(parent)

    for prefix, rgba in [(p, red_rgba) for p in red[1:]] + [(p, blue_rgba) for p in blue]:
        child = mujoco.MjSpec.from_file(str(_ROBOT_XML))
        _tint_shells(child, rgba)
        _add_rope_sites(child)
        frame = parent.worldbody.add_frame()
        parent.attach(child, prefix=prefix, frame=frame)

    # Rope chain, far red → center → far blue. Two cords per link (left/right
    # waist sites) so a link transmits no yaw torque between teammates.
    chain = list(reversed(red)) + blue
    for pa, pb in zip(chain[:-1], chain[1:]):
        nominal = gap if pa == red[0] else spacing
        link_len = nominal + ROPE_SLACK
        for side_a, side_b in _link_side_pairs(pa, pb, red):
            cord = parent.add_tendon(
                name=f"tug_{pa or 'r0_'}{pb}cord_{side_a}",
                stiffness=ROPE_STIFFNESS,
                springlength=(0.0, link_len),
                limited=True,
                range=(0.0, link_len + ROPE_LIMIT_MARGIN),
                width=ROPE_WIDTH,
                rgba=ROPE_RGBA,
                group=4,  # physics only; the hemp mocap geoms do the visuals
                solref_limit=(0.02, 1.0),
                solimp_limit=(0.90, 0.95, 0.001, 0.5, 2.0),
            )
            cord.wrap_site(f"{pa}rope_{side_a}")
            cord.wrap_site(f"{pb}rope_{side_b}")

    _add_hemp_material(parent)
    _twisted_tube_mesh(parent, "tug_wrap", _wrap_path(np.random.default_rng(20260914)),
                       t_segments=66, alpha_segments=6,
                       twists=WRAP_TURNS * ROPE_PLY_TWISTS_PER_TURN, uv_repeats=14.0,
                       sub_fibers=5, flyaway=260)
    _twisted_tube_mesh(parent, "tug_piece", _straight_path(SPAN_PIECE_LENGTH),
                       t_segments=30, alpha_segments=6, twists=2.5, uv_repeats=2.0,
                       sub_fibers=5, flyaway=40)
    _add_waist_wraps(parent, n_per_team)
    n_strands = 2 * n_per_team - 1
    for idx in range(n_strands):
        for sub in range(SAG_SUBSEGMENTS):
            body = parent.worldbody.add_body(name=f"tugvis_{idx}_{sub}", mocap=True)
            body.add_geom(
                name=f"tugvis_{idx}_{sub}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname="tug_piece",
                material="tug_hemp",
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
