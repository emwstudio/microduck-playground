"""Render a swing360 checkpoint to MP4 from the deployable play contract.

Bottom motionless spawn (arc_spawn is disabled in play), single env, fixed
camera from the task viewer cfg. This is the deployment-side visual audit:
what the policy actually does, not what training metrics claim.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from rsl_rl.runners import OnPolicyRunner

import mjlab_microduck.tasks  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--lookat", type=float, nargs=3, default=None)
    parser.add_argument("--distance", type=float, default=None)
    parser.add_argument("--elevation", type=float, default=None)
    parser.add_argument("--azimuth", type=float, default=None)
    args = parser.parse_args()

    configure_torch_backends()
    task_id = "Mjlab-Swing360-MicroDuck"
    env_cfg = load_env_cfg(task_id, play=True)
    env_cfg.seed = args.seed
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = args.duration + 1.0
    env_cfg.terminations.clear()
    env_cfg.viewer.width = args.width
    env_cfg.viewer.height = args.height
    if args.lookat is not None:
        env_cfg.viewer.lookat = tuple(args.lookat)
    if args.distance is not None:
        env_cfg.viewer.distance = args.distance
    if args.elevation is not None:
        env_cfg.viewer.elevation = args.elevation
    if args.azimuth is not None:
        env_cfg.viewer.azimuth = args.azimuth
    agent_cfg = load_rl_cfg(task_id)

    raw = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task_id) or OnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=args.device)
    runner.load(str(args.checkpoint), map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)
    obs = env.get_observations()

    robot = raw.scene["robot"]
    pivot_ids, _ = robot.find_joints("passive_swing_pivot")
    pivot = pivot_ids[0]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame_skip = max(1, round((1.0 / raw.step_dt) / args.fps))
    steps = int(args.duration / raw.step_dt)
    writer = imageio.get_writer(str(args.out), fps=args.fps, macro_block_size=1)
    max_abs = 0.0
    samples: list[dict[str, float]] = []
    with torch.inference_mode():
        for step in range(steps):
            obs, _, _, _ = env.step(policy(obs))
            angle = float(robot.data.joint_pos[0, pivot])
            max_abs = max(max_abs, abs(angle))
            samples.append({"time_s": step * raw.step_dt, "angle_rad": angle})
            if step % frame_skip == 0:
                writer.append_data(raw.render())
    writer.close()
    if args.metrics is not None:
        import json

        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps({"samples": samples}))
    print(f"wrote {args.out}, peak |pivot| = {max_abs:.1f} deg".replace(
        "deg", f"deg ({max_abs * 57.2958:.1f} deg)"))


if __name__ == "__main__":
    main()
