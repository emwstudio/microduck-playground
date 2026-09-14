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
ROPE_RADIUS = 0.0035

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


def rope_visual_segments(n_per_team: int) -> list[tuple[str, str]]:
    """Site-name pairs the hemp mocap cylinders span: every physics cord plus
    a per-duck belt (left → chest → right) that wraps the rope around each
    torso. Order matches the tugvis_* bodies in build_tug_spec."""
    red, blue = team_prefixes(n_per_team)
    chain = list(reversed(red)) + blue
    segments: list[tuple[str, str]] = []
    for pa, pb in zip(chain[:-1], chain[1:]):
        for side_a, side_b in _link_side_pairs(pa, pb, red):
            segments.append((f"{pa}rope_{side_a}", f"{pb}rope_{side_b}"))
    for prefix in red + blue:
        segments.append((f"{prefix}rope_left", f"{prefix}rope_chest"))
        segments.append((f"{prefix}rope_chest", f"{prefix}rope_right"))
    return segments


def _hemp_rgba_texture(size: int = 64) -> bytes:
    """Procedural twisted-hemp tile: diagonal light/dark strand bands plus
    fibre noise. Tiles seamlessly along the cylinder."""
    yy, xx = np.mgrid[0:size, 0:size]
    strand = ((xx + yy) % 16) < 8
    base = np.where(strand, 0.72, 0.52)[..., None] * np.array([1.0, 0.88, 0.66])
    fibre = np.random.default_rng(7).normal(0.0, 0.05, (size, size, 1))
    rgb = np.clip((base + fibre) * 255, 0, 255).astype(np.uint8)
    return rgb.tobytes()


def _add_hemp_material(spec: mujoco.MjSpec) -> None:
    tex = spec.add_texture(name="tug_hemp_tex")
    tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    tex.width, tex.height, tex.nchannel = 64, 64, 3
    tex.data = _hemp_rgba_texture(64)
    mat = spec.add_material(name="tug_hemp")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "tug_hemp_tex"
    mat.texrepeat = (6.0, 1.0)
    mat.texuniform = True


@dataclass
class RopeVisual:
    body_id: int
    geom_id: int
    site_a_id: int
    site_b_id: int


def resolve_rope_visuals(model: mujoco.MjModel, n_per_team: int) -> list[RopeVisual]:
    segments = rope_visual_segments(n_per_team)
    visuals = []
    for idx, (site_a, site_b) in enumerate(segments):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"tugvis_{idx}")
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"tugvis_{idx}")
        visuals.append(RopeVisual(
            body_id=body_id,
            geom_id=geom_id,
            site_a_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_a),
            site_b_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_b),
        ))
    return visuals


def update_rope_visuals(model: mujoco.MjModel, data: mujoco.MjData,
                        visuals: list[RopeVisual]) -> None:
    """Move every hemp cylinder onto its site-to-site segment (per frame)."""
    quat = np.zeros(4)
    for vis in visuals:
        p0 = data.site_xpos[vis.site_a_id]
        p1 = data.site_xpos[vis.site_b_id]
        direction = p1 - p0
        length = float(np.linalg.norm(direction))
        if length < 1e-6:
            continue
        mocap_id = model.body_mocapid[vis.body_id]
        data.mocap_pos[mocap_id] = (p0 + p1) / 2.0
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
    for idx in range(len(rope_visual_segments(n_per_team))):
        body = parent.worldbody.add_body(name=f"tugvis_{idx}", mocap=True)
        body.add_geom(
            name=f"tugvis_{idx}",
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
