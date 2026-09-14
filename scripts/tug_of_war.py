#!/usr/bin/env python3
"""NvN tug-of-war demo: two teams of Microducks pull a rope chain across a center line.

No training involved — every duck runs the same walking ONNX with a constant
forward velocity command, and each duck faces away from the center line, so
forward walking is pulling. A round ends when the rope midpoint crosses a
team's win line or when MIN_FALLEN ducks of one team are down. Rounds
auto-reset; per-duck pull-speed jitter keeps outcomes varied.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

from mjlab_microduck.robot.tug_of_war import (
    CENTER_GAP,
    DEFAULT_POSE,
    DUCK_SPACING,
    MIN_FALLEN,
    TRUNK_Z0,
    WIN_X,
    apply_actions,
    build_tug_spec,
    check_winner,
    compute_obs,
    duck_spawns,
    find_duck_rigs,
    resolve_rope_visuals,
    update_rope_visuals,
)

# Official alpha walking policy (walking gait keeps a support foot — the
# running policy's flight phase topples instantly under rope load). Fetch with:
#   curl -sL -o policies/alpha_walking.onnx \
#     https://huggingface.co/pollen-robotics/microduck-policies/resolve/main/alpha_walking.onnx
DEFAULT_POLICY = "policies/alpha_walking.onnx"
CONTROL_DT = 0.02  # 50 Hz policy rate
SETTLE_S = 0.5     # hold default pose before the pull starts


def reset_round(model, data, rigs, spawns, rng):
    mujoco.mj_resetData(model, data)
    for rig in rigs:
        x, quat = spawns[rig.prefix]
        adr = rig.free_qpos_adr
        data.qpos[adr + 0] = x + rng.uniform(-0.002, 0.002)
        data.qpos[adr + 1] = rng.uniform(-0.002, 0.002)
        data.qpos[adr + 2] = TRUNK_Z0
        data.qpos[adr + 3:adr + 7] = quat
        data.qpos[rig.joint_qpos_idx] = DEFAULT_POSE
        rig.last_action[:] = 0.0
    for rig in rigs:
        data.ctrl[rig.actuator_ids] = DEFAULT_POSE
    mujoco.mj_forward(model, data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--n-per-team", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--round-duration", type=float, default=15.0)
    parser.add_argument("--pull-speed", type=float, default=0.2)
    parser.add_argument("--pull-jitter", type=float, default=0.15,
                        help="per-duck relative pull-speed jitter, uniform ±this fraction")
    parser.add_argument("--team-boost", type=float, default=1.3,
                        help="each round one random team gets its pull speed x(1+this); "
                             "symmetric teams otherwise deadlock into draws")
    parser.add_argument("--win-x", type=float, default=0.25)
    parser.add_argument("--spacing", type=float, default=DUCK_SPACING)
    parser.add_argument("--gap", type=float, default=CENTER_GAP)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--foot-friction", type=float, default=2.0,
                        help="foot sliding friction mu; sim default ~1.0 lets the rope "
                             "drag ducks instead of gripping (real PU sole ~2.0)")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--lookat", type=float, nargs=3, default=(0.0, 0.0, 0.1))
    parser.add_argument("--distance", type=float, default=1.6)
    parser.add_argument("--elevation", type=float, default=-15.0)
    parser.add_argument("--azimuth", type=float, default=90.0)
    args = parser.parse_args()

    if not args.no_render and args.out is None:
        parser.error("--out is required unless --no-render")

    rng = np.random.default_rng(args.seed)
    spec = build_tug_spec(n_per_team=args.n_per_team, spacing=args.spacing,
                          gap=args.gap, win_x=args.win_x)
    model = spec.compile()
    data = mujoco.MjData(model)
    rigs = find_duck_rigs(model, args.n_per_team)
    spawns = duck_spawns(args.n_per_team, args.spacing, args.gap)
    if args.foot_friction > 0:
        foot_geoms = [i for i in range(model.ngeom)
                      if "foot" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "")]
        model.geom_friction[foot_geoms, 0] = args.foot_friction
        print(f"foot friction {args.foot_friction} on {len(foot_geoms)} geoms")

    decimation = max(1, round(CONTROL_DT / model.opt.timestep))
    control_dt = decimation * model.opt.timestep
    print(f"{len(rigs)} ducks, {model.nu} actuators, {model.ntendon} rope cords; "
          f"timestep {model.opt.timestep}, control {1.0 / control_dt:.0f} Hz")

    session = ort.InferenceSession(args.policy, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    rope_visuals = resolve_rope_visuals(model, args.n_per_team, args.spacing, args.gap)
    renderer = None
    writer = None
    if not args.no_render:
        import imageio.v2 as imageio
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(args.out), fps=args.fps, macro_block_size=1)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat = np.array(args.lookat)
    camera.distance = args.distance
    camera.elevation = args.elevation
    camera.azimuth = args.azimuth
    render_skip = max(1, round((1.0 / args.fps) / control_dt))

    results = []
    score = {"red": 0, "blue": 0, "draw": 0}
    for round_idx in range(args.rounds):
        reset_round(model, data, rigs, spawns, rng)
        pull_speeds = args.pull_speed * rng.uniform(
            1.0 - args.pull_jitter, 1.0 + args.pull_jitter, size=len(rigs))
        boost_team = rng.choice(["red", "blue"])
        for k, rig in enumerate(rigs):
            if rig.team == boost_team:
                pull_speeds[k] *= 1.0 + args.team_boost
        # Settle: hold the default pose briefly so every duck starts upright.
        for _ in range(int(SETTLE_S / model.opt.timestep)):
            mujoco.mj_step(model, data)

        winner, reason = None, None
        steps = int(args.round_duration / control_dt)
        step = 0
        t_start = time.time()
        while step < steps:
            obs = compute_obs(model, data, rigs, pull_speeds)
            if not np.isfinite(obs).all():
                winner, reason = "draw", "NaN in observations"
                break
            # The exported ONNX pins batch=1, so infer per duck.
            actions = np.concatenate([
                session.run([output_name], {input_name: obs[k:k + 1]})[0]
                for k in range(len(rigs))
            ])
            apply_actions(data, rigs, actions)
            for _ in range(decimation):
                mujoco.mj_step(model, data)
            step += 1
            if renderer is not None and step % render_skip == 0:
                update_rope_visuals(model, data, rope_visuals)
                renderer.update_scene(data, camera)
                writer.append_data(renderer.render())
            if step * control_dt > 1.0:  # grace period: spawn transients
                winner, reason = check_winner(model, data, rigs, args.win_x, MIN_FALLEN)
                if winner:
                    break
        if winner is None:
            winner, reason = "draw", f"round cap {args.round_duration:.0f}s reached"
        score[winner] += 1
        elapsed = step * control_dt
        results.append({"round": round_idx + 1, "winner": winner, "reason": reason,
                        "boosted": boost_team, "duration_s": round(elapsed, 2)})
        print(f"round {round_idx + 1}: {winner.upper()} wins — {reason} "
              f"({elapsed:.1f}s, wall {time.time() - t_start:.1f}s)")
        # Victory beat: keep rendering ~1s so the outcome reads on video.
        if renderer is not None:
            for _ in range(int(1.0 / control_dt)):
                for _ in range(decimation):
                    mujoco.mj_step(model, data)
                if step % render_skip == 0:
                    update_rope_visuals(model, data, rope_visuals)
                    renderer.update_scene(data, camera)
                    writer.append_data(renderer.render())
                step += 1

    if writer is not None:
        writer.close()
        print(f"wrote {args.out}")
    summary = {"score": score, "rounds": results,
               "config": {"n_per_team": args.n_per_team, "pull_speed": args.pull_speed,
                          "win_x": args.win_x, "seed": args.seed}}
    print(json.dumps(summary, indent=2))
    if args.metrics is not None:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
