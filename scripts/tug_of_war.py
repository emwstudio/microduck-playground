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
    ROPE_SLACK,
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


def grade_frame(frame):
    """Filmic-ish grade toward the reference video's warm three.js look:
    highlight-lifted S-curve, warm tint, subtle bloom from the highlights."""
    img = frame.astype(np.float32) / 255.0
    img = np.clip(img ** 0.92, 0, 1)                      # gentle gamma lift
    img = np.clip(img + 0.55 * img * (img - 0.5), 0, 1)   # soft S-curve
    img *= np.array([1.04, 1.01, 0.97])                   # warm tint
    lum = img.mean(axis=2)
    bloom_src = np.clip(lum - 0.72, 0, 1) ** 2            # highlight mask
    ky = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16
    bloom = np.apply_along_axis(lambda r: np.convolve(r, ky, mode="same"), 0, bloom_src)
    bloom = np.apply_along_axis(lambda r: np.convolve(r, ky, mode="same"), 1, bloom)
    img = np.clip(img + 0.22 * bloom[..., None], 0, 1)
    return (img * 255).astype(np.uint8)
CONTROL_DT = 0.02  # 50 Hz policy rate
SETTLE_S = 0.5     # hold default pose before the pull starts


def reset_round(model, data, rigs, spawns, rng):
    mujoco.mj_resetData(model, data)
    for rig in rigs:
        x, quat = spawns[rig.prefix]
        adr = rig.free_qpos_adr
        data.qpos[adr + 0] = x + rng.uniform(-0.002, 0.002)
        from mjlab_microduck.robot.tug_of_war import SPAWN_Y
        data.qpos[adr + 1] = SPAWN_Y + rng.uniform(-0.002, 0.002)
        data.qpos[adr + 2] = TRUNK_Z0
        data.qpos[adr + 3:adr + 7] = quat
        data.qpos[rig.joint_qpos_idx] = DEFAULT_POSE
        rig.last_action[:] = 0.0
    for rig in rigs:
        data.ctrl[rig.actuator_ids] = DEFAULT_POSE
    mujoco.mj_forward(model, data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=DEFAULT_POLICY,
                        help="fallback policy for both teams")
    parser.add_argument("--policy-red", default=None,
                        help="red-team policy ONNX (defaults to --policy)")
    parser.add_argument("--policy-blue", default=None,
                        help="blue-team policy ONNX (defaults to --policy)")
    parser.add_argument("--red-kind", choices=["walk", "tug"], default="walk",
                        help="walk: velocity-commanded policy (vx=pull speed); "
                             "tug: self-directed pull policy (vx zeroed — trained "
                             "with a zero-padded twist slot, a real vx is OOD)")
    parser.add_argument("--blue-kind", choices=["walk", "tug"], default="walk",
                        help="same as --red-kind, for the blue team")
    parser.add_argument("--n-per-team", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--round-duration", type=float, default=15.0)
    parser.add_argument("--pull-speed", type=float, default=0.2)
    parser.add_argument("--pull-jitter", type=float, default=0.15,
                        help="per-duck relative pull-speed jitter, uniform ±this fraction")
    parser.add_argument("--team-boost", type=float, default=2.2,
                        help="each round one random team gets its pull speed x(1+this); "
                             "symmetric teams otherwise deadlock into draws")
    parser.add_argument("--win-x", type=float, default=0.18)
    parser.add_argument("--win-x-red", type=float, default=None,
                        help="override red-side win distance (handicap balancing: "
                             "a stronger red team can be given a longer grind)")
    parser.add_argument("--win-x-blue", type=float, default=None,
                        help="override blue-side win distance")
    parser.add_argument("--spacing", type=float, default=DUCK_SPACING)
    parser.add_argument("--gap", type=float, default=CENTER_GAP)
    parser.add_argument("--pretension", type=float, default=0.015,
                        help="spawn each link this far past its taut length so the "
                             "rope carries tension from t=0 (the tug training env's "
                             "sled rope starts taut; a slack match start lets "
                             "lean-back policies topple before tension builds)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rope-damping", type=float, default=2.0,
                        help="tendon damping (N·s/m) on every rope cord; 0 = pure "
                             "spring. The stiff 200 N/m dead-band cord yanks ducks "
                             "over during the first seconds' tension transients "
                             "(outer ducks toppled at 1-4s and lay there for the "
                             "whole round); 2.0 absorbs the yanks — zero early "
                             "falls, rounds stay 10-15s, outcomes stay balanced")
    parser.add_argument("--foot-friction", type=float, default=2.0,
                        help="foot sliding friction mu; sim default ~1.0 lets the rope "
                             "drag ducks instead of gripping (real PU sole ~2.0)")
    parser.add_argument("--get-up", type=float, default=0.0, metavar="SECONDS",
                        help="video mode: a duck down this long is stood back up at "
                             "its spot (real pullers get up; keeps all 10 ducks "
                             "visibly pulling). Rounds are then decided by the rope "
                             "crossing a win line, never by team wipe. 0 = off.")
    parser.add_argument("--surge-start", type=float, default=0.0,
                        help="seconds of pure deadlock before the surge team digs deep "
                             "(ramp to 1+surge-amount over 3s). 0 = off")
    parser.add_argument("--surge-amount", type=float, default=1.5,
                        help="surge multiplier on the surge team's pull speed")
    parser.add_argument("--surge-team", choices=["random", "red", "blue", "alternate"],
                        default="random",
                        help="which team surges; 'alternate' = red on even rounds, blue on odd")
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
    parser.add_argument("--no-grade", action="store_true",
                        help="disable the filmic post-processing grade")
    args = parser.parse_args()

    if not args.no_render and args.out is None:
        parser.error("--out is required unless --no-render")

    rng = np.random.default_rng(args.seed)
    # The red floor line sits at the EFFECTIVE red win distance (handicap).
    spec_win_x_red = args.win_x_red if args.win_x_red is not None else args.win_x
    spec = build_tug_spec(n_per_team=args.n_per_team, spacing=args.spacing,
                          gap=args.gap, win_x=spec_win_x_red,
                          win_x_blue=args.win_x_blue)
    model = spec.compile()
    data = mujoco.MjData(model)
    rigs = find_duck_rigs(model, args.n_per_team)
    # Spawn each link slightly past its taut length (cord = nominal + ROPE_SLACK)
    # so the chain carries tension from t=0 like the training sled rope does.
    pret = ROPE_SLACK + args.pretension
    spawns = duck_spawns(args.n_per_team, args.spacing + pret, args.gap + pret)
    if args.rope_damping > 0:
        model.tendon_damping[:] = args.rope_damping
        print(f"rope damping {args.rope_damping} N·s/m on {model.ntendon} cords")
    if args.foot_friction > 0:
        foot_geoms = [i for i in range(model.ngeom)
                      if "foot" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "")]
        model.geom_friction[foot_geoms, 0] = args.foot_friction
        print(f"foot friction {args.foot_friction} on {len(foot_geoms)} geoms")

    decimation = max(1, round(CONTROL_DT / model.opt.timestep))
    control_dt = decimation * model.opt.timestep
    print(f"{len(rigs)} ducks, {model.nu} actuators, {model.ntendon} rope cords; "
          f"timestep {model.opt.timestep}, control {1.0 / control_dt:.0f} Hz")

    sessions = {}
    for team, policy_path in (("red", args.policy_red or args.policy),
                              ("blue", args.policy_blue or args.policy)):
        s = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
        sessions[team] = (s, s.get_inputs()[0].name, s.get_outputs()[0].name)
        print(f"{team} policy: {policy_path}")

    rope_spans = resolve_rope_visuals(model, args.n_per_team, args.spacing, args.gap)
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
        down_time: dict[str, float] = {}
        kinds = {"red": args.red_kind, "blue": args.blue_kind}
        if args.surge_start > 0:
            if args.surge_team == "alternate":
                surge_team = "red" if round_idx % 2 == 0 else "blue"
            elif args.surge_team == "random":
                surge_team = rng.choice(["red", "blue"])
            else:
                surge_team = args.surge_team
        else:
            surge_team = None
        while step < steps:
            # Tug-style policies are self-directed: their twist slot was
            # zero-padded in training, so feeding a real pull speed is OOD
            # and topples them instantly. Zero the command for those ducks.
            cmd_speeds = np.array([
                0.0 if kinds[rig.team] == "tug" else pull_speeds[k]
                for k, rig in enumerate(rigs)
            ])
            # Surge: after a pure-deadlock stalemate the surge team digs deep
            # (ramps to 1+amount over 3s) and grinds the rope across — rounds
            # end decisively at 12-19s with zero falls.
            if surge_team is not None:
                t_now = step * control_dt
                if t_now > args.surge_start:
                    k = min(1.0, (t_now - args.surge_start) / 3.0)
                    for j, rig in enumerate(rigs):
                        if rig.team == surge_team and kinds[rig.team] == "walk":
                            cmd_speeds[j] *= 1.0 + args.surge_amount * k
            obs = compute_obs(model, data, rigs, cmd_speeds)
            if not np.isfinite(obs).all():
                winner, reason = "draw", "NaN in observations"
                break
            # The exported ONNX pins batch=1, so infer per duck on its team's policy.
            actions = np.concatenate([
                sessions[rig.team][0].run(
                    [sessions[rig.team][2]],
                    {sessions[rig.team][1]: obs[k:k + 1]})[0]
                for k, rig in enumerate(rigs)
            ])
            apply_actions(data, rigs, actions)
            for _ in range(decimation):
                mujoco.mj_step(model, data)
            step += 1
            if renderer is not None and step % render_skip == 0:
                update_rope_visuals(model, data, rope_spans, renderer._mjr_context)
                renderer.update_scene(data, camera)
                frame = renderer.render()
                writer.append_data(frame if args.no_grade else grade_frame(frame))
            if args.get_up > 0:
                # Video mode: downed ducks get back up after args.get_up seconds
                # and keep pulling (real tug-of-war: you stumble, you recover).
                from mjlab_microduck.robot.tug_of_war import FALLEN_TRUNK_Z, team_states
                for rig in rigs:
                    down = data.xpos[rig.trunk_body_id][2] < FALLEN_TRUNK_Z
                    if down:
                        down_time[rig.prefix] = down_time.get(rig.prefix, 0.0) + control_dt
                    else:
                        down_time[rig.prefix] = 0.0
                    if down_time[rig.prefix] >= args.get_up:
                        _, quat = spawns[rig.prefix]
                        adr = rig.free_qpos_adr
                        data.qpos[adr + 2] = TRUNK_Z0          # keep x, y
                        data.qpos[adr + 3:adr + 7] = quat
                        data.qvel[adr:adr + 6] = 0.0
                        data.qpos[rig.joint_qpos_idx] = DEFAULT_POSE
                        data.qvel[rig.joint_qvel_idx] = 0.0
                        data.ctrl[rig.actuator_ids] = DEFAULT_POSE
                        rig.last_action[:] = 0.0
                        down_time[rig.prefix] = 0.0
                states = team_states(model, data, rigs)
                midpoint_x = (states["red"]["center_x"] + states["blue"]["center_x"]) / 2.0
                win_x_red = args.win_x_red if args.win_x_red is not None else args.win_x
                win_x_blue = args.win_x_blue if args.win_x_blue is not None else args.win_x
                if step * control_dt > 1.0:
                    if midpoint_x < -win_x_red:
                        winner, reason = "red", f"rope pulled past red line (midpoint {midpoint_x:+.2f} m)"
                        break
                    if midpoint_x > win_x_blue:
                        winner, reason = "blue", f"rope pulled past blue line (midpoint {midpoint_x:+.2f} m)"
                        break
            elif step * control_dt > 1.0:  # grace period: spawn transients
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
                    update_rope_visuals(model, data, rope_spans, renderer._mjr_context)
                    renderer.update_scene(data, camera)
                    frame = renderer.render()
                writer.append_data(frame if args.no_grade else grade_frame(frame))
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
