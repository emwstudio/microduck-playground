"""Probe: which tread is the duck on when a fall STARTS?  (v14c fall-location diagnosis)

Background: v14c-ck3998 climbs a median 9 treads but fall_fraction is 1.00
with final_rise_p50 -0.07 m and ~80 % forward falls at L2-L4.  Hypothesis
under test: falls START on treads 8-10 (2-3 treads short of the top
platform, tread 11) and the duck then tumbles back to the floor — "falls
happen before the top edge, not on it".  This probe measures, per fall
episode: the tread at the fall's start, the distance to the top platform,
and the end-of-episode trunk height in risers.

Measurement conventions (the implementable choices — read before quoting):

* SAME environment construction as evaluate_ladder_checkpoint.py: play cfg,
  ALL TERMINATIONS CLEARED, episodes always run the full duration.  One env =
  one episode, so per-env trackers never need re-arming.  A "fall episode"
  is classified afterwards with the evaluator's exact formula
  (tilt > 65 deg | ladder_fallen) so numbers stay comparable with the
  historical eval JSONs.
* FALL START = the first run of ``fall_grace_steps`` consecutive steps with
  trunk tilt > ``fall_tilt_deg`` (default 50 deg).  The env's own fall
  termination fires at 65 deg (bad_orientation) or on ladder_fallen; a 50 deg
  threshold with a 3-step streak catches the start of the topple ~0.05-0.15 s
  earlier while ignoring transient recovery leans (a stepping duck rarely
  holds >50 deg for 3 straight steps without going over).
* "Tread at fall start" = ``state.foot_last_tread`` max over the two feet AT
  THE MARK STEP (the highest tread either foot last stood on — contacts
  update every step via the reward stack).  Also recorded: the running
  max-tread-so-far (they differ when the duck slipped back a tread before
  tipping), the spawn tread, and body-frame gravity at the mark (fall
  direction, same classification as the evaluator).
* Distance to top = (num_treads - 1) - fall-start tread (tread 11 is the top
  platform on simple_stairs).
* Final position = episode-end trunk z above the env origin, in risers
  (per-env sampled riser).  final_risers ~0 = tumbled all the way back to
  the floor (the -0.07 m final_rise_p50 signature).

usage:
  MUJOCO_GL=egl uv run python scripts/probe_fall_location.py \
      --checkpoint-file logs/rsl_rl/simple_stairs/2026-09-24_23-23-31_simple_stairs/model_3998.pt \
      --output-file logs/probe-fall-location.json

Cross-analysis (added 2026-09-26): ``by_max_tread`` buckets episodes by the
highest tread reached (<8 / 8-9 / 10 / >=11) and reports, per bucket, the
fraction that ENDS ON THE DECK — trunk xy inside the top platform's
footprint (its box centre +/- half the landing depth/width, from the
geometry, not hardcoded) and trunk z in [top - 5 cm, top + 20 cm].  It
answers: "of the episodes that DO reach the top platform (~10%), is the
duck already lying on the deck at the end?" — the input to the beveled-
nose decision.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from rsl_rl.runners import OnPolicyRunner

import mjlab_microduck.tasks  # noqa: F401 - populate registry
from mjlab_microduck.tasks import mdp as microduck_mdp


@dataclass(frozen=True)
class Config:
    checkpoint_file: str
    task_id: str = "Mjlab-SimpleStairs-MicroDuck"
    num_envs: int = 256
    duration_s: float = 12.0
    levels: tuple[int, ...] = (0, 1, 2, 3, 4)
    climb_speed: float = 0.04
    floor_spawn_prob: float = 0.5
    swing_spawn_prob: float = 0.0
    seed: int = 123
    output_file: str | None = None
    fall_tilt_deg: float = 50.0  # fall-start detector: trunk tilt threshold
    fall_grace_steps: int = 3    # consecutive steps above the threshold


def _probe_level(cfg: Config, level: int, device: str) -> dict:
    """One level: rollout and per-fall start/end positions (evaluator-style env)."""
    env_cfg = load_env_cfg(cfg.task_id, play=True)
    env_cfg.seed = cfg.seed + level
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.episode_length_s = cfg.duration_s + 1.0
    env_cfg.terminations.clear()  # falls play out; one env = one episode
    twist = env_cfg.commands["twist"]
    twist.ranges.lin_vel_x = (cfg.climb_speed, cfg.climb_speed)
    twist.rel_standing_envs = 0.0
    twist.resampling_time_range = (cfg.duration_s + 1.0, cfg.duration_s + 1.0)
    spawn = env_cfg.events["reset_stair_ladder"].params
    spawn["fixed_level"] = level
    spawn["floor_spawn_prob"] = cfg.floor_spawn_prob
    spawn["swing_spawn_prob"] = cfg.swing_spawn_prob

    agent_cfg = load_rl_cfg(cfg.task_id)
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(cfg.task_id) or OnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(cfg.checkpoint_file, map_location=device)
    policy = runner.get_inference_policy(device=device)

    obs = env.get_observations()
    robot = raw_env.scene["robot"]
    state = raw_env._stair
    n = env_cfg.scene.num_envs
    num_treads = state.geometry.num_treads
    top_idx = num_treads - 1
    riser = state.riser.clone()
    top_z = state.tread_top[:, -1].clone()
    spawn_tread = state.start_tread.clone()
    on_floor = state.spawn_on_floor.clone()
    spawn_z = robot.data.root_link_pos_w[:, 2].clone()

    # Per-env trackers (no re-arming needed: terminations are cleared).
    max_tread = torch.full((n,), -1, dtype=torch.long, device=device)  # running max over foot_last_tread
    streak = torch.zeros(n, dtype=torch.long, device=device)  # consecutive tilt > threshold steps
    marked = torch.zeros(n, dtype=torch.bool, device=device)  # fall start recorded
    fell = torch.zeros(n, dtype=torch.bool, device=device)  # evaluator's fall-episode formula
    mark_tread = torch.full((n,), -99, dtype=torch.long, device=device)  # support tread at fall start
    mark_maxtread = torch.full((n,), -99, dtype=torch.long, device=device)
    mark_gravity = torch.zeros(n, 3, device=device)  # body-frame gravity at fall start (direction)

    steps = round(cfg.duration_s / raw_env.step_dt)
    tilt_thresh = math.radians(cfg.fall_tilt_deg)
    for _ in range(steps):
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)
        tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0))
        last_tread = state.foot_last_tread.max(dim=1).values
        max_tread = torch.maximum(max_tread, last_tread)
        streak = torch.where(tilt > tilt_thresh, streak + 1, torch.zeros_like(streak))
        mark = (streak >= cfg.fall_grace_steps) & ~marked
        mark_tread = torch.where(mark, last_tread, mark_tread)
        mark_maxtread = torch.where(mark, max_tread, mark_maxtread)
        mark_gravity[mark] = robot.data.projected_gravity_b[mark]
        marked |= mark
        fell |= (tilt > math.radians(65.0)) | microduck_mdp.ladder_fallen(raw_env)

    z_rel = robot.data.root_link_pos_w[:, 2] - raw_env.scene.env_origins[:, 2]
    final_risers = z_rel / riser
    final_rise_m = robot.data.root_link_pos_w[:, 2] - spawn_z
    reached_top = robot.data.root_link_pos_w[:, 2] > top_z + 0.117 + 0.05

    # On-deck at episode end: trunk xy inside the top platform's footprint
    # (box centre +/- half landing depth/width from the geometry) and trunk z
    # in [top - 5 cm, top + 20 cm] — "lying on the platform" included.
    top_c = state.tread_centre[:, -1]  # (N, 3) world xy of the platform box centre
    g = state.geometry
    pos = robot.data.root_link_pos_w
    on_deck = (
        ((pos[:, 0] - top_c[:, 0]).abs() <= 0.5 * g.landing_depth_m)
        & ((pos[:, 1] - top_c[:, 1]).abs() <= 0.5 * g.landing_width_m)
        & (pos[:, 2] >= top_z - 0.05)
        & (pos[:, 2] <= top_z + 0.20)
    )

    def _hist(values: torch.Tensor) -> dict:
        out = {}
        for t in range(-1, num_treads):
            out[str(t)] = int((values == t).sum())
        return out

    def _q(v: torch.Tensor, q: float) -> float | None:
        return float(v.quantile(q)) if v.numel() else None

    falls_i = fell.nonzero(as_tuple=False).squeeze(-1)
    nonfalls_i = (~fell).nonzero(as_tuple=False).squeeze(-1)
    marked_falls = marked & fell  # fall episodes whose start was captured
    dist = (top_idx - mark_tread[marked_falls]).float()

    def _spawn_split(mask: torch.Tensor) -> dict:
        fl = mask & on_floor
        la = mask & ~on_floor
        return {
            "floor_spawn": int(fl.sum()),
            "ladder_spawn": int(la.sum()),
            "fall_start_tread_p50_floor": _q(mark_tread[marked_falls & fl].float(), 0.5),
            "fall_start_tread_p50_ladder": _q(mark_tread[marked_falls & la].float(), 0.5),
        }

    # Cross-analysis: episodes bucketed by the highest tread reached, with
    # the on-deck ending fraction per bucket.
    by_max_tread = []
    for label, mask in (
        ("<8", max_tread < 8),
        ("8-9", (max_tread >= 8) & (max_tread <= 9)),
        ("10", max_tread == 10),
        (">=11", max_tread >= 11),
    ):
        cnt = int(mask.sum())
        by_max_tread.append(
            {
                "bucket": label,
                "episodes": cnt,
                "falls": int((mask & fell).sum()),
                "on_deck": int((mask & on_deck).sum()),
                "on_deck_fraction": round(float((mask & on_deck).float().mean()), 4) if cnt else None,
                "on_deck_of_falls": (
                    round(float((mask & fell & on_deck).float().mean()), 4) if bool((mask & fell).any()) else None
                ),
                "final_risers_p50": _q(final_risers[mask], 0.5),
            }
        )

    result = {
        "level": level,
        "riser_m": (float(riser.min()), float(riser.max())),
        "angle_deg": (float(torch.rad2deg(state.angle).min()), float(torch.rad2deg(state.angle).max())),
        "episodes": n,
        "falls": int(fell.sum()),
        "falls_marked": int(marked_falls.sum()),
        "reached_top": int(reached_top.sum()),
        # Q1: tread at fall start (current support tread, and max-so-far variant)
        "fall_start_tread_hist": _hist(mark_tread[marked_falls]),
        "fall_start_maxtread_hist": _hist(mark_maxtread[marked_falls]),
        "fall_start_tread_p10": _q(mark_tread[marked_falls].float(), 0.1),
        "fall_start_tread_p50": _q(mark_tread[marked_falls].float(), 0.5),
        "fall_start_tread_p90": _q(mark_tread[marked_falls].float(), 0.9),
        "fall_start_maxtread_p50": _q(mark_maxtread[marked_falls].float(), 0.5),
        # Q2: distance to the top platform
        "dist_to_top_p10": _q(dist, 0.1),
        "dist_to_top_p50": _q(dist, 0.5),
        "dist_to_top_p90": _q(dist, 0.9),
        # Q3: end position (risers above the floor spawn line)
        "fall_final_risers_p10": _q(final_risers[falls_i], 0.1),
        "fall_final_risers_p50": _q(final_risers[falls_i], 0.5),
        "fall_final_risers_p90": _q(final_risers[falls_i], 0.9),
        "fall_final_rise_m_p50": _q(final_rise_m[falls_i], 0.5),
        # fall direction at the fall start (evaluator's classification)
        "fall_forward_fraction": float((mark_gravity[marked_falls, 0] > 0.3).float().mean()) if bool(marked_falls.any()) else None,
        "fall_backward_fraction": float((mark_gravity[marked_falls, 0] < -0.3).float().mean()) if bool(marked_falls.any()) else None,
        "fall_sideways_fraction": float((mark_gravity[marked_falls, 1].abs() > 0.5).float().mean()) if bool(marked_falls.any()) else None,
        # Q4: non-fall episodes' end position
        "nonfalls": int(nonfalls_i.sum()),
        "nonfall_maxtread_p50": _q(max_tread[nonfalls_i].float(), 0.5),
        "nonfall_final_risers_p50": _q(final_risers[nonfalls_i], 0.5),
        "spawn_tread_p50": _q(spawn_tread.float(), 0.5),
        **_spawn_split(fell),
        "by_max_tread": by_max_tread,
        "_records": {
            "max_tread": max_tread.cpu(),
            "fell": fell.cpu(),
            "on_deck": on_deck.cpu(),
            "final_risers": final_risers.cpu(),
        },
    }
    raw_env.close()
    return result


def main(cfg: Config) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    results = []
    recs = {"max_tread": [], "fell": [], "on_deck": [], "final_risers": []}
    print(
        f"{'lvl':>3} {'falls':>6} {'start_p50':>9} {'max_p50':>8} {'dist_top_p50':>12} "
        f"{'final_ris_p50':>13} {'nonfall_ris_p50':>15} {'fwd%':>6}"
    )
    for level in cfg.levels:
        r = _probe_level(cfg, level, device)
        results.append(r)
        rec = r.pop("_records")
        for k in recs:
            recs[k].append(rec[k])
        print(
            f"{level:>3} {r['falls']:>6} "
            f"{r['fall_start_tread_p50'] if r['fall_start_tread_p50'] is not None else float('nan'):>9.1f} "
            f"{r['fall_start_maxtread_p50'] if r['fall_start_maxtread_p50'] is not None else float('nan'):>8.1f} "
            f"{r['dist_to_top_p50'] if r['dist_to_top_p50'] is not None else float('nan'):>12.1f} "
            f"{r['fall_final_risers_p50'] if r['fall_final_risers_p50'] is not None else float('nan'):>13.2f} "
            f"{r['nonfall_final_risers_p50'] if r['nonfall_final_risers_p50'] is not None else float('nan'):>15.2f} "
            f"{(r['fall_forward_fraction'] or 0.0) * 100:>5.0f}%"
        )
    # Global cross-analysis across all levels: on-deck endings by max tread.
    gt = {k: torch.cat(v) for k, v in recs.items()}
    global_by_max_tread = []
    print(f"\n{'bucket':>6} {'episodes':>8} {'falls':>6} {'on_deck':>8} {'deck%':>7} {'deck|fall%':>10} {'final_ris_p50':>13}")
    for label, mask in (
        ("<8", gt["max_tread"] < 8),
        ("8-9", (gt["max_tread"] >= 8) & (gt["max_tread"] <= 9)),
        ("10", gt["max_tread"] == 10),
        (">=11", gt["max_tread"] >= 11),
    ):
        cnt = int(mask.sum())
        deck = gt["on_deck"] & mask
        row = {
            "bucket": label,
            "episodes": cnt,
            "falls": int((mask & gt["fell"]).sum()),
            "on_deck": int(deck.sum()),
            "on_deck_fraction": round(float(deck.float().mean()), 4) if cnt else None,
            "on_deck_of_falls": (
                round(float((gt["fell"] & deck).sum() / max(int((mask & gt["fell"]).sum()), 1)), 4)
            ),
            "final_risers_p50": float(gt["final_risers"][mask].quantile(0.5)) if cnt else None,
        }
        global_by_max_tread.append(row)
        print(
            f"{label:>6} {cnt:>8} {row['falls']:>6} {row['on_deck']:>8} "
            f"{(row['on_deck_fraction'] or 0.0) * 100:>6.1f}% "
            f"{(row['on_deck_of_falls'] or 0.0) * 100:>9.1f}% "
            f"{row['final_risers_p50'] if row['final_risers_p50'] is not None else float('nan'):>13.2f}"
        )
    if cfg.output_file:
        Path(cfg.output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.output_file, "w") as f:
            json.dump(
                {"config": asdict(cfg), "conventions": __doc__, "results": results,
                 "global_by_max_tread": global_by_max_tread},
                f, indent=1,
            )
        print(f"PROBE_FALL_LOCATION_JSON {cfg.output_file}")


if __name__ == "__main__":
    main(tyro.cli(Config))
