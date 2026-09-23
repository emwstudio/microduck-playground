"""Evaluate / render the jump_step policy: success rate per platform height.

usage (eval):   uv run python ../training/eval_jump_step.py --source <ckpt> --out <dir> --envs 512 --seconds 6
usage (render): uv run python ../training/eval_jump_step.py --source <ckpt> --out <dir> --render --seconds 8 --azimuth 45

Reports per-height-band success fraction (jump_step_landed termination) and
writes eval.json; with --render records a single-env video instead.
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
a = p.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

cfg = make_microduck_jump_step_env_cfg(play=True)
cfg.scene.num_envs = 1 if a.render else a.envs
cfg.seed = a.seed
if not a.render:
    # Sweep the training height range in eval (play pins one height by default).
    cfg.commands["twist"].ranges.lin_vel_x = (0.025, 0.035)

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

heights = None if a.render else raw.command_manager.get_command("twist")[:, 0].cpu().clone()
landed_envs = torch.zeros(env.num_envs, dtype=torch.bool, device=raw.device)
for step in range(steps):
    with torch.inference_mode():
        act = policy(obs)
    obs, _, done, _ = env.step(act)
    if not a.render:
        term = raw.termination_manager.get_term("jump_step_landed").bool()
        landed_envs |= term
    assert torch.isfinite(act).all()

if a.render:
    print(f"[render] done -> {out}")
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
