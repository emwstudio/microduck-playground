"""Simple straight staircase ("ordinary stairs") for Microduck.

Full-width 60 mm treads (the whole 54 mm sole fits flat), 30 mm risers at
~27 degrees, eleven mini treads up to a full-depth top platform.
Deliberately plain: no alternating half-width treads, no curved top
section, no clamps or bridge plates — a normal straight flight of stairs.
"""

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
# Full-width treads space same-side treads one riser apart, so the toe needs
# riser >= 29 mm (StairLadderGeometry.min_riser_m).  30 mm on 105 mm legs is
# stair-like but climbable; the ~26.6 deg angle follows from the 60 mm run.
SIMPLE_STAIRS_LEVELS: tuple[dict, ...] = (
    {"riser": (0.030, 0.030), "angle": (26.6, 26.6)},
)
SIMPLE_STAIRS_EPISODE_LENGTH_S = 12.0


def make_microduck_simple_stairs_env_cfg(
    play: bool = False, top_spawn_prob: float = 0.0
) -> ManagerBasedRlEnvCfg:
    """Straight full-width staircase with a top platform (see module docstring)."""
    return make_microduck_ladder_env_cfg(
        play=play,
        geometry=SIMPLE_STAIRS_GEOMETRY,
        level_table=SIMPLE_STAIRS_LEVELS,
        episode_length_s=SIMPLE_STAIRS_EPISODE_LENGTH_S,
        max_start_tread=8,
        top_spawn_prob=top_spawn_prob,
    )


MicroduckSimpleStairsRlCfg = deepcopy(MicroduckLadderRlCfg)
MicroduckSimpleStairsRlCfg.experiment_name = "simple_stairs"
MicroduckSimpleStairsRlCfg.run_name = "simple_stairs"
