"""Deterministic full-circle metrics for a learned rigid-arm swing policy.

Runs the Mjlab-Swing360-MicroDuck task headless and reports the actual
unwrapped pivot trajectory: peak arc, over-the-top count, and completed
full turns per environment. This is the deployment-side verdict for the
360-degree objective — training reward curves are not evidence.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--duration", type=float, default=36.0)
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--num-envs", type=int, default=16)
    args = parser.parse_args()

    configure_torch_backends()
    task_id = "Mjlab-Swing360-MicroDuck"
    env_cfg = load_env_cfg(task_id, play=True)
    env_cfg.seed = args.seed
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = args.duration + 1.0
    env_cfg.terminations.clear()
    agent_cfg = load_rl_cfg(task_id)

    raw = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task_id) or OnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=args.device)
    runner.load(str(args.checkpoint), map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)
    obs = env.get_observations()

    robot = raw.scene["robot"]
    pivot_ids, _ = robot.find_joints("passive_swing_pivot")
    pivot = pivot_ids[0]

    max_abs = torch.zeros(args.num_envs, device=args.device)
    peak_rate = torch.zeros(args.num_envs, device=args.device)
    steps = int(args.duration / raw.step_dt)
    with torch.inference_mode():
        for _ in range(steps):
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            angle = robot.data.joint_pos[:, pivot]
            rate = robot.data.joint_vel[:, pivot]
            max_abs = torch.maximum(max_abs, torch.abs(angle))
            peak_rate = torch.maximum(peak_rate, torch.abs(rate))

    final_angle = robot.data.joint_pos[:, pivot].cpu()
    max_abs = max_abs.cpu()
    peak_rate = peak_rate.cpu()

    per_env = []
    for i in range(args.num_envs):
        a = float(max_abs[i])
        per_env.append(
            {
                "env": i,
                "max_abs_angle_deg": math.degrees(a),
                "over_the_top": bool(a > math.pi),
                "completed_full_turns": int(a // (2.0 * math.pi)),
                "final_angle_deg": math.degrees(float(final_angle[i])),
                "peak_pivot_rate_rad_s": float(peak_rate[i]),
            }
        )
    over = sum(1 for e in per_env if e["over_the_top"])
    loops = [e["completed_full_turns"] for e in per_env]
    result = {
        "checkpoint": str(args.checkpoint),
        "duration_s": args.duration,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "envs_over_the_top": over,
        "envs_with_full_turn": sum(1 for n in loops if n >= 1),
        "max_completed_full_turns": max(loops),
        "median_max_abs_angle_deg": sorted(e["max_abs_angle_deg"] for e in per_env)[
            args.num_envs // 2
        ],
        "per_env": per_env,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(
        f"over-the-top: {over}/{args.num_envs}, "
        f"full-turn envs: {result['envs_with_full_turn']}, "
        f"max turns: {result['max_completed_full_turns']}, "
        f"median peak arc: {result['median_max_abs_angle_deg']:.1f} deg"
    )


if __name__ == "__main__":
    main()
