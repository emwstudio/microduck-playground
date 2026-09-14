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
ROPE_SITE_Z = 0.035
ROPE_CHEST_X = 0.022
ROPE_RADIUS = 0.005

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


def rope_visual_segments(n_per_team: int, spacing: float = DUCK_SPACING,
                         gap: float = CENTER_GAP) -> list[tuple[str, str, float]]:
    """(site_a, site_b, nominal_length) per physics cord — the hemp mocap
    strands span these, with sag computed from slack vs nominal. Order
    matches the tugvis_* body groups in build_tug_spec."""
    red, blue = team_prefixes(n_per_team)
    chain = list(reversed(red)) + blue
    segments: list[tuple[str, str, float]] = []
    for pa, pb in zip(chain[:-1], chain[1:]):
        nominal = gap if pa == red[0] else spacing
        for side_a, side_b in _link_side_pairs(pa, pb, red):
            segments.append((f"{pa}rope_{side_a}", f"{pb}rope_{side_b}", nominal))
    return segments


SAG_SUBSEGMENTS = 4       # mocap cylinders per cord
SAG_PER_SLACK = 0.6       # parabola depth per metre of slack
SAG_MAX = 0.045


def _hemp_textures(size: int = 128) -> tuple[bytes, bytes]:
    """Procedural 3-ply twisted-rope tiles: color with per-strand cylindrical
    shading + groove shadows + fibre noise, and a matching tangent-space
    normal map derived from the strand height profile."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    diag = (xx + yy) / size                      # 45° direction, 0..2
    phase = (diag * 3.0) % 1.0                   # 3 strands per tile
    height = np.cos(2.0 * np.pi * phase)         # strand crown/groove profile
    fibre = np.random.default_rng(7).normal(0.0, 0.06, (size, size))
    fibre += 0.05 * np.sin(xx * 2.4 + yy * 0.3)  # along-strand fibre streaks
    shade = 0.55 + 0.35 * height + fibre
    base = np.array([0.78, 0.63, 0.42])          # hemp beige, deepened
    rgb = np.clip(shade[..., None] * base * 255, 0, 255).astype(np.uint8)

    eps = 1.0 / size
    dhdx = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) / (2 * eps)
    dhdy = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) / (2 * eps)
    strength = 0.012
    nx, ny = -dhdx * strength, -dhdy * strength
    nz = np.ones_like(nx)
    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    nrm = np.stack([(nx / norm + 1) / 2, (ny / norm + 1) / 2, (nz / norm + 1) / 2], axis=-1)
    nrm = (nrm * 255).astype(np.uint8)
    return rgb.tobytes(), nrm.tobytes()


def _add_hemp_material(spec: mujoco.MjSpec) -> None:
    rgb, normal = _hemp_textures()
    tex = spec.add_texture(name="tug_hemp_tex")
    tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    tex.width, tex.height, tex.nchannel = 128, 128, 3
    tex.data = rgb
    ntex = spec.add_texture(name="tug_hemp_nrm")
    ntex.type = mujoco.mjtTexture.mjTEXTURE_2D
    ntex.width, ntex.height, ntex.nchannel = 128, 128, 3
    ntex.data = normal
    mat = spec.add_material(name="tug_hemp")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "tug_hemp_tex"
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_NORMAL] = "tug_hemp_nrm"
    mat.texrepeat = (10.0, 1.0)
    mat.texuniform = True


def _twisted_coil_mesh(spec: mujoco.MjSpec, name: str = "tug_coil",
                       theta_segments: int = 44, alpha_segments: int = 7,
                       twists: int = 11) -> None:
    """3-ply twisted rope bent into a waist coil, as an inline mesh.

    Baked in the xy plane with major radii (x 0.69, y 1.0) and unit scale —
    geoms scale it uniformly to ROPE_COIL_RADIUS, giving an ellipse that
    hugs the torso (half-width y ~0.052 > depth x ~0.035)."""
    n_strands = 3
    strand_center_r = 0.15    # strand centreline distance from coil centreline
    strand_tube_r = 0.095     # each ply's tube radius
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for k in range(n_strands):
        base = len(verts)
        for i in range(theta_segments):
            theta = 2.0 * np.pi * i / theta_segments
            e_r = np.array([0.69 * np.cos(theta), np.sin(theta), 0.0])
            e_r /= np.linalg.norm(e_r)
            e_z = np.array([0.0, 0.0, 1.0])
            phi = 2.0 * np.pi * (k / n_strands + twists * theta / (2.0 * np.pi))
            center = (np.array([0.69 * np.cos(theta), np.sin(theta), 0.0])
                      + strand_center_r * (np.cos(phi) * e_r + np.sin(phi) * e_z))
            # Strand tangent (finite difference) and a stable tube frame.
            d_theta = 1e-3
            theta2 = theta + d_theta
            phi2 = 2.0 * np.pi * (k / n_strands + twists * theta2 / (2.0 * np.pi))
            e_r2 = np.array([0.69 * np.cos(theta2), np.sin(theta2), 0.0])
            e_r2 /= np.linalg.norm(e_r2)
            center2 = (np.array([0.69 * np.cos(theta2), np.sin(theta2), 0.0])
                       + strand_center_r * (np.cos(phi2) * e_r2 + np.sin(phi2) * e_z))
            tangent = (center2 - center) / d_theta
            tangent /= np.linalg.norm(tangent)
            u = center - np.array([0.69 * np.cos(theta), np.sin(theta), 0.0])
            u /= np.linalg.norm(u)
            w = np.cross(tangent, u)
            w /= np.linalg.norm(w)
            for j in range(alpha_segments):
                alpha = 2.0 * np.pi * j / alpha_segments
                point = center + strand_tube_r * (np.cos(alpha) * u + np.sin(alpha) * w)
                verts.append(tuple(point))
        for i in range(theta_segments):
            i2 = (i + 1) % theta_segments
            for j in range(alpha_segments):
                j2 = (j + 1) % alpha_segments
                a = base + i * alpha_segments + j
                b = base + i2 * alpha_segments + j
                c = base + i2 * alpha_segments + j2
                d = base + i * alpha_segments + j2
                faces.append((a, b, c))
                faces.append((a, c, d))
    mesh = spec.add_mesh(name=name)
    mesh.scale = (ROPE_COIL_RADIUS, ROPE_COIL_RADIUS, ROPE_COIL_RADIUS)
    mesh.uservert = np.array(verts, dtype=np.float32).flatten()
    mesh.userface = np.array(faces, dtype=np.int32).flatten()


ROPE_COIL_RADIUS = 0.055    # geom scale for the baked unit coil
ROPE_COIL_ZS = (0.026, 0.044)  # two wraps around the waist


def _add_waist_coils(spec: mujoco.MjSpec, n_per_team: int) -> None:
    red, blue = team_prefixes(n_per_team)
    for prefix in red + blue:
        trunk = _find_body(spec, f"{prefix}trunk_base")
        for zi, z in enumerate(ROPE_COIL_ZS):
            trunk.add_geom(
                name=f"{prefix}tug_coil_{zi}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname="tug_coil",
                material="tug_hemp",
                pos=(0.0, 0.0, z),
                contype=0,
                conaffinity=0,
                density=0.0,   # pure visual — no added mass or inertia
            )


@dataclass
class RopeVisual:
    body_id: int
    geom_id: int
    site_a_id: int
    site_b_id: int
    nominal: float
    fraction: float   # sub-segment start fraction along the cord, 0..1


def resolve_rope_visuals(model: mujoco.MjModel, n_per_team: int) -> list[RopeVisual]:
    segments = rope_visual_segments(n_per_team)
    visuals = []
    for idx, (site_a, site_b, nominal) in enumerate(segments):
        for sub in range(SAG_SUBSEGMENTS):
            body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, f"tugvis_{idx}_{sub}")
            geom_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, f"tugvis_{idx}_{sub}")
            visuals.append(RopeVisual(
                body_id=body_id,
                geom_id=geom_id,
                site_a_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_a),
                site_b_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_b),
                nominal=nominal,
                fraction=sub / SAG_SUBSEGMENTS,
            ))
    return visuals


def update_rope_visuals(model: mujoco.MjModel, data: mujoco.MjData,
                        visuals: list[RopeVisual]) -> None:
    """Move every hemp sub-segment onto its sagging cord curve (per frame)."""
    quat = np.zeros(4)
    for vis in visuals:
        p0 = data.site_xpos[vis.site_a_id]
        p1 = data.site_xpos[vis.site_b_id]
        chord = float(np.linalg.norm(p1 - p0))
        slack = max(0.0, vis.nominal - chord)
        sag = min(SAG_MAX, SAG_PER_SLACK * slack + 0.002)
        t0, t1 = vis.fraction, vis.fraction + 1.0 / SAG_SUBSEGMENTS
        pa = p0 + (p1 - p0) * t0
        pb = p0 + (p1 - p0) * t1
        pa[2] -= 4.0 * sag * t0 * (1.0 - t0)
        pb[2] -= 4.0 * sag * t1 * (1.0 - t1)
        direction = pb - pa
        length = float(np.linalg.norm(direction))
        if length < 1e-6:
            continue
        mocap_id = model.body_mocapid[vis.body_id]
        data.mocap_pos[mocap_id] = (pa + pb) / 2.0
        mujoco.mju_quatZ2Vec(quat, direction / length)
        data.mocap_quat[mocap_id] = quat
        model.geom_size[vis.geom_id][1] = length / 2.0


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
    _twisted_coil_mesh(parent)
    _add_waist_coils(parent, n_per_team)
    for idx in range(len(rope_visual_segments(n_per_team))):
        for sub in range(SAG_SUBSEGMENTS):
            body = parent.worldbody.add_body(name=f"tugvis_{idx}_{sub}", mocap=True)
            body.add_geom(
                name=f"tugvis_{idx}_{sub}",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=(ROPE_RADIUS, 0.1, 0.0),
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
