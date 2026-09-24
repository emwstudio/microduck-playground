"""Simple straight staircase ("ordinary stairs") for Microduck.

Full-width 60 mm treads (the whole 54 mm sole fits flat) and OPEN risers:
with the angle shallow enough that run = riser/tan(angle) exceeds the 60 mm
tread depth, consecutive treads leave an open gap and the swing toe passes
through the gap instead of under the next tread.  That makes risers BELOW
the 29 mm toe-under-tread minimum (ladder.py:min_riser_m) legal — the v12
design converges on 25 mm at ~19-20 deg (run 68.7 mm, gap ~8.7 mm), inside
the official gait family's ~25 mm step envelope (FK: foot apex clearance
median 66.6 / p10 34.7 mm).  v1-v3 failed because the reset clamped risers
to 29 mm, above that envelope.

The reset enforces the gap per env: ``reset_stair_ladder(open_riser=True)``
skips the min_riser clamp and instead clamps the angle so
run >= tread depth + 2 mm (see ladder.clamp_riser_angle).  The level table
below is drawn up so every band already satisfies the gap at its lower
riser, so the clamp is a guard, not a distortion.  12 treads up to a
full-depth top platform.
"""

import os
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg

from ..robot.ladder import StairLadderGeometry
from .microduck_ladder_env_cfg import (
    MicroduckLadderRlCfg,
    make_microduck_ladder_env_cfg,
)

# 12 treads up to a full-depth top platform.
SIMPLE_STAIRS_GEOMETRY = StairLadderGeometry(
    num_treads=12,
    alternating=False,
    tread_depth_m=0.060,  # the whole 54 mm sole rests flat (validate cap: 60 mm)
    landing_every=12,  # the 12th tread is the full-depth top platform
)
# v12 course: 15 -> 25.5 mm risers.  Full-width treads space same-side treads
# one riser apart, so the toe-under-tread rule would need riser >= 29 mm
# (min_riser_m) — above the ~25 mm step envelope.  With OPEN risers the
# binding rule is the gap: run = riser/tan(angle) >= 62 mm (60 mm depth +
# 2 mm margin).  Each band's top angle is atan(riser_lo / 0.062), so the
# band always satisfies the gap at its own lower riser; the reset's
# open_riser clamp guards the draws.  Level 4 is 24-25.5 mm at 19-20 deg —
# the target envelope riser with run ~68.7 mm at 25 mm / 20 deg.
SIMPLE_STAIRS_LEVELS: tuple[dict, ...] = (
    {"riser": (0.015, 0.017), "angle": (13.0, 13.6)},
    {"riser": (0.017, 0.020), "angle": (13.6, 15.3)},
    {"riser": (0.020, 0.022), "angle": (15.3, 17.9)},
    {"riser": (0.022, 0.024), "angle": (17.9, 19.5)},
    {"riser": (0.024, 0.0255), "angle": (19.0, 20.0)},
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
        open_riser=True,  # v12: 25 mm risers through the open gap, not under the tread
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
