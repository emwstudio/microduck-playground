"""Simple straight staircase ("ordinary stairs") for Microduck.

Full-width 60 mm treads (the whole 54 mm sole fits flat), 30 mm risers at
~27 degrees, eleven mini treads up to a full-depth top platform.
Deliberately plain: no alternating half-width treads, no curved top
section, no clamps or bridge plates — a normal straight flight of stairs.
"""

import os
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg

from ..robot.ladder import StairLadderGeometry
from .microduck_ladder_env_cfg import (
    MicroduckLadderRlCfg,
    make_microduck_ladder_env_cfg,
)

# 12 treads of 30 mm: a 36 cm straight flight ending on the top platform.
SIMPLE_STAIRS_GEOMETRY = StairLadderGeometry(
    num_treads=12,
    alternating=False,
    tread_depth_m=0.060,  # the whole 54 mm sole rests flat (validate cap: 60 mm)
    landing_every=12,  # the 12th tread is the full-depth top platform
)
# Full-width treads space same-side treads one riser apart, so with
# overlapping treads the toe needs riser >= 29 mm
# (StairLadderGeometry.min_riser_m).  The curriculum therefore starts with
# small risers at *open-riser* angles (run >= tread depth: the swing foot
# passes through the gaps, never under a tread) and converges on 30 mm at
# 26.6 deg where the 60 mm treads tile contiguously.  Mirrors LADDER_LEVELS'
# riser bands with angles recomputed to the tiling boundary.
SIMPLE_STAIRS_LEVELS: tuple[dict, ...] = (
    {"riser": (0.015, 0.017), "angle": (13.0, 14.0)},
    {"riser": (0.017, 0.020), "angle": (14.0, 15.8)},
    {"riser": (0.020, 0.023), "angle": (15.8, 18.4)},
    {"riser": (0.023, 0.027), "angle": (18.4, 21.0)},
    {"riser": (0.027, 0.030), "angle": (21.0, 24.2)},
)
SIMPLE_STAIRS_EPISODE_LENGTH_S = 12.0


def _foot_targets_per_side(
    env, asset_cfg, min_rise: float = 0.006, scale: float = 0.05
):
    """ladder_foot_targets with per-side lateral targets on full-width treads.

    Upstream `_stair_foot_target_info` aims BOTH feet at the tread centre for
    full-width treads (tread_side == 0); only landings get per-side targets
    (landing_target_per_side, fixing the converging-step hesitation of the
    2026-09-04 landing study).  On a plain full-width staircase EVERY tread
    is full width, so all mini-tread targets were centreline.  Aim each foot
    at its own side of every full-width tread instead.
    """
    import torch
    from mjlab_microduck.tasks import mdp as _mdp
    from mjlab.managers import SceneEntityCfg as _SEC

    state = _mdp._stair_state(env)
    asset = env.scene[asset_cfg.name]
    out = torch.zeros(env.num_envs, 4, device=env.device)
    if state is None:
        return out
    info = _mdp._stair_foot_target_info(env, asset, min_rise)
    g = state.geometry
    num = g.num_treads
    yaw = _mdp._yaw_from_quat(asset.data.root_link_quat_w)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    lat_mag = 0.5 * g.center_gap_m + 0.5 * g.side_width_m
    landing = torch.tensor([g.is_landing(i) for i in range(num)], device=env.device)
    for slot, side in enumerate((1.0, -1.0)):
        idx = info["index"][:, slot].clamp_max(num - 1)
        full_width = state.tread_side[idx] == 0.0
        needs_offset = full_width & ~landing[idx] & info["valid"][:, slot]
        tyaw = state.tread_yaw.gather(1, idx[:, None]).squeeze(1)
        off = side * lat_mag * torch.stack((-torch.sin(tyaw), torch.cos(tyaw)), dim=-1)
        vec = info["vec"][:, slot].clone()
        vec[:, :2] = vec[:, :2] + torch.where(needs_offset[:, None], off, torch.zeros_like(off))
        fwd = vec[:, 0] * cy + vec[:, 1] * sy
        valid = info["valid"][:, slot]
        out[:, 2 * slot] = torch.where(valid, fwd / scale, 0.0)
        out[:, 2 * slot + 1] = torch.where(valid, vec[:, 2] / scale, 0.0)
    return torch.nan_to_num(out, nan=0.0).clamp(-5.0, 5.0)


def make_microduck_simple_stairs_env_cfg(
    play: bool = False,
    top_spawn_prob: float = 0.0,
    per_side_targets: bool = os.getenv("MICRODUCK_SIMPLE_STAIRS_PER_SIDE", "0") == "1",
) -> ManagerBasedRlEnvCfg:
    """Straight full-width staircase with a top platform (see module docstring)."""
    cfg = make_microduck_ladder_env_cfg(
        play=play,
        geometry=SIMPLE_STAIRS_GEOMETRY,
        level_table=SIMPLE_STAIRS_LEVELS,
        episode_length_s=SIMPLE_STAIRS_EPISODE_LENGTH_S,
        max_start_tread=8,
        top_spawn_prob=top_spawn_prob,
    )
    if per_side_targets:
        from mjlab.managers import SceneEntityCfg

        for group in ("actor", "critic"):
            terms = cfg.observations[group].terms
            terms["head_command"] = deepcopy(terms["head_command"])
            terms["head_command"].func = _foot_targets_per_side
            terms["head_command"].params = {"asset_cfg": SceneEntityCfg("robot")}
    return cfg


MicroduckSimpleStairsRlCfg = deepcopy(MicroduckLadderRlCfg)
MicroduckSimpleStairsRlCfg.experiment_name = "simple_stairs"
MicroduckSimpleStairsRlCfg.run_name = "simple_stairs"
