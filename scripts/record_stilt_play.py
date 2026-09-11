#!/usr/bin/env python3
"""Record a stilt policy walking inside the real mjlab training env (headless).

Unlike a CPU-MuJoCo rehearsal (scripts/render_stilt_video.py), this steps the
actual warp env the policy was trained in — same BAM actuators, obs pipeline,
DR and command handling as the curriculum gate. Morphology is compile-time:
set MICRODUCK_STILT_HEIGHT_CM / MICRODUCK_STILT_BLEND before running.

Usage (cwd = third_party/microduck-playground):
    MICRODUCK_STILT_HEIGHT_CM=10 MICRODUCK_STILT_BLEND=0.5 PYTHONPATH=src \
    uv run --no-sync python scripts/record_stilt_play.py \
        --checkpoint-file ../../artifacts/stilts_repro/10cm/checkpoint.pt \
        --duration-s 6 --out ../../artifacts/stilts_repro/videos/stilt_h10cm.mp4
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from rsl_rl.runners import OnPolicyRunner

import mjlab_microduck.tasks  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-file", type=Path, required=True)
    ap.add_argument("--task-id", default="Mjlab-Stilt-Flat-MicroDuck")
    ap.add_argument("--duration-s", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import imageio.v2 as imageio

    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    env_cfg = load_env_cfg(args.task_id, play=True)
    env_cfg.seed = args.seed
    env_cfg.scene.num_envs = 1
    env_cfg.viewer.width = args.width
    env_cfg.viewer.height = args.height

    agent_cfg = load_rl_cfg(args.task_id)
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(args.task_id) or OnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(args.checkpoint_file), map_location=device)
    policy = runner.get_inference_policy(device=device)

    fps = int(round(1.0 / raw_env.step_dt))
    num_steps = round(args.duration_s / raw_env.step_dt)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(args.out), fps=fps, macro_block_size=1)

    obs = env.get_observations()
    for _ in range(num_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        writer.append_data(raw_env.render())

    writer.close()
    env.close()
    print(f"recorded {num_steps} steps @ {fps} fps -> {args.out}")


if __name__ == "__main__":
    main()
