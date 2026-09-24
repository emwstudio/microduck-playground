"""Evaluate / render the jump_step policy: success rate per platform height.

usage (eval):   uv run python ../training/eval_jump_step.py --source <ckpt> --out <dir> --envs 512 --seconds 6
usage (mixed):  uv run python ../training/eval_jump_step.py --source <ckpt> --out <dir> --envs 512 --airborne-frac 0.5
usage (render): uv run python ../training/eval_jump_step.py --source <ckpt> --out <dir> --render --seconds 8 --azimuth 45

Reports per-height-band success fraction (jump_step_landed termination) and
writes eval.json; with --render records a single-env video instead.
With --airborne-frac F > 0, F of the spawns drop the robot above the platform
(v9 airborne mode) and success is reported per spawn class (airborne vs
floor) with per-episode accounting, instead of the height bands.
"""
import argparse, json, os
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.wrappers import VideoRecorder

import mjlab_microduck.tasks  # noqa: F401
from mjlab_microduck.tasks import MicroduckOnPolicyRunner, mdp
from mjlab_microduck.tasks.microduck_jump_step_env_cfg import (
    MicroduckJumpStepRlCfg,
    make_microduck_jump_step_env_cfg,
    spawn_class_outcomes,
    SPAWN_AIRBORNE,
)
from mjlab_microduck.video_effects import configure_video_cfg, fix_render_shadows

p = argparse.ArgumentParser()
p.add_argument("--source", required=True)
p.add_argument("--out", required=True)
p.add_argument("--envs", type=int, default=512)
p.add_argument("--seconds", type=float, default=6.0)
p.add_argument("--seed", type=int, default=777)
p.add_argument("--render", action="store_true")
p.add_argument("--azimuth", type=float, default=45.0)
p.add_argument("--airborne-frac", type=float, default=0.0,
               help="fraction of airborne (drop-onto-platform) spawns; >0 reports per-spawn-class success")
p.add_argument("--ballistic-frac", type=float, default=0.0,
               help="fraction of ballistic (just-jumped) spawns; >0 reports per-spawn-class success")
a = p.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

cfg = make_microduck_jump_step_env_cfg(play=True)
cfg.scene.num_envs = 1 if a.render else a.envs
cfg.seed = a.seed
if not a.render:
    # Sweep the training height range in eval (play pins one height by default).
    cfg.commands["twist"].ranges.lin_vel_x = (0.025, 0.035)
if a.airborne_frac > 0.0 and not a.render:
    cfg.events["reset_jump_step"].params["airborne_spawn_prob"] = a.airborne_frac
if a.ballistic_frac > 0.0 and not a.render:
    cfg.events["reset_jump_step"].params["ballistic_spawn_prob"] = a.ballistic_frac

if a.render:
    configure_video_cfg(cfg)
    cfg.viewer.distance = 0.7
    cfg.viewer.elevation = -12.0
    cfg.viewer.azimuth = a.azimuth
    raw = ManagerBasedRlEnv(cfg, device="cuda:0", render_mode="rgb_array")
    fix_render_shadows(raw, light_dir=(-0.5, 0.2, -1.0))
    steps = int(round(a.seconds / raw.step_dt))
    env = VideoRecorder(raw, video_folder=out, step_trigger=lambda s: s == 0, video_length=steps, disable_logger=True)
else:
    raw = ManagerBasedRlEnv(cfg, device="cuda:0")
    steps = int(round(a.seconds / raw.step_dt))
    env = raw

agent = asdict(MicroduckJumpStepRlCfg)
agent.update(logger="tensorboard", upload_model=False, seed=a.seed)
env = RslRlVecEnvWrapper(env, clip_actions=agent.get("clip_actions"))
runner = MicroduckOnPolicyRunner(env, agent, str(out), "cuda:0")
runner.load(a.source, map_location="cuda:0")
policy = runner.get_inference_policy(device="cuda:0")

raw.reset(seed=a.seed)
obs = env.get_observations()
assert obs["actor"].shape[-1] == 61

bucketed = (a.airborne_frac > 0.0 or a.ballistic_frac > 0.0) and not a.render
heights = None if (a.render or bucketed) else raw.command_manager.get_command("twist")[:, 0].cpu().clone()
landed_envs = torch.zeros(env.num_envs, dtype=torch.bool, device=raw.device)
if bucketed:
    # Per-episode accounting by spawn class: an episode that ends is booked
    # under its own spawn type (auto-reset has already drawn the next one).
    # Semantic labels, NOT positional bincount indices — a swapped index here
    # reported airborne landings as "floor" in the v9 eval (2026-09-24).
    cur_type = raw._jump_step_state.spawn_type.clone()
    ep = {"airborne": [0, 0], "ballistic": [0, 0], "floor": [0, 0]}  # [episodes, landed]
for step in range(steps):
    with torch.inference_mode():
        act = policy(obs)
    obs, _, done, _ = env.step(act)
    term = raw.termination_manager.get_term("jump_step_landed").bool()
    if not a.render:
        landed_envs |= term
    if bucketed:
        fin = done.nonzero(as_tuple=False).squeeze(-1)
        if len(fin):
            outcomes = spawn_class_outcomes(cur_type[fin], term[fin])
            for key in ("airborne", "ballistic"):
                ep[key][0] += outcomes[key][0]
                ep[key][1] += outcomes[key][1]
            for cls in ("floor", "edge", "platform"):
                ep["floor"][0] += outcomes[cls][0]
                ep["floor"][1] += outcomes[cls][1]
            cur_type[fin] = raw._jump_step_state.spawn_type[fin]
    assert torch.isfinite(act).all()

if a.render:
    print(f"[render] done -> {out}")
elif bucketed:
    # Unfinished episodes count as attempts (a landed episode terminates).
    outcomes = spawn_class_outcomes(cur_type, torch.zeros_like(cur_type, dtype=torch.bool))
    ep["airborne"][0] += outcomes["airborne"][0]
    ep["ballistic"][0] += outcomes["ballistic"][0]
    for cls in ("floor", "edge", "platform"):
        ep["floor"][0] += outcomes[cls][0]
    report = {
        "envs": env.num_envs,
        "seconds": a.seconds,
        "seed": a.seed,
        "airborne_frac": a.airborne_frac,
        "ballistic_frac": a.ballistic_frac,
        "airborne": {"episodes": ep["airborne"][0], "success": ep["airborne"][1],
                     "fraction": round(ep["airborne"][1] / max(ep["airborne"][0], 1), 4)},
        "ballistic": {"episodes": ep["ballistic"][0], "success": ep["ballistic"][1],
                      "fraction": round(ep["ballistic"][1] / max(ep["ballistic"][0], 1), 4)},
        "floor": {"episodes": ep["floor"][0], "success": ep["floor"][1],
                  "fraction": round(ep["floor"][1] / max(ep["floor"][0], 1), 4)},
    }
    (out / "eval.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
else:
    bands = {}
    h = heights.numpy()
    l = landed_envs.cpu().numpy()
    for lo, hi in ((0.0245, 0.030), (0.030, 0.0355)):
        m = (h >= lo) & (h < hi)
        bands[f"{int(lo*1000)}-{int(hi*1000)}mm"] = {
            "count": int(m.sum()),
            "success": int((l & m).sum()),
            "fraction": round(float((l & m).sum() / max(m.sum(), 1)), 4),
        }
    report = {
        "envs": env.num_envs,
        "seconds": a.seconds,
        "seed": a.seed,
        "success_total": int(l.sum()),
        "success_fraction": round(float(l.mean()), 4),
        "bands": bands,
    }
    (out / "eval.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
raw.close()
