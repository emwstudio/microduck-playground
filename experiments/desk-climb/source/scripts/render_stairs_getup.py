"""Render the climber -> getup relay on simple_stairs (the Hannes route).

v14c (rsl_rl climber, 61D mjlab_microduck contract) charges the stairs;
when it falls ON THE DECK REGION control hard-switches to the official
getup.onnx, which should stand the duck back up on the platform.

OBS CONTRACT (verified against training/render_climb_video.py, the official
handoff implementation): the getup ONNX reads the SAME 61D obs["actor"] the
rsl_rl policy consumes — no dimension conversion.  Three contract details,
all copied from the official renderer:
  * [48:61] command slots must be ZERO for the getup (it was trained blind;
    the desk-recovery env asserts obs[:, 48:] == 0).  The climber needs
    those slots (twist phase + foot targets), so the switch builds a
    per-policy obs variant: climber gets the raw obs, getup gets a clone
    with [48:] zeroed.
  * [34:48] last_action slot carries the ONNX's RAW previous output, not
    the alpha-filtered executed action (official line 142/177).
  * After the switch the executed action is alpha-filtered toward the
    previous executed action (0.7 everywhere, 0.5 on the four head
    servos) and every actuator's kp_scale drops to 80% (official lines
    178-181).
The ONNX was exported with its obs normalizer baked in, so it is fed the
raw (unnormalized) obs exactly like the rsl_rl inference policy.

SWITCH CONDITION: fall-on-deck — trunk tilt > 60 deg AND trunk z above
(platform top - 10 cm), held for ``switch_grace_steps`` steps (0.1 s), OR a
fixed-time fallback (``--switch-after-s``).  auto_reset is OFF so a fall
never resets the episode mid-video; _manual_reset_pending is zeroed every
step (the official renderer's pattern).

STAND METRIC (relay ``stood``): tilt < 25 deg and trunk z above
top + 0.09 m, in a CONTINUOUS run for ``--stand-hold-s`` seconds, AND both
feet inside the landing box's xy footprint for >= 80% of the run's samples
(``StandTracker`` — the flush deck's top is level with the last tread, so
trunk height alone cannot separate "standing on the deck" from "standing
on the last tread"; brief getup foot lifts inside a run are tolerated,
a run that is mostly off-deck is discarded).

WALK-ON MODE (``--walk-on``): after BOTH feet stand on the top platform
(foot_tread == num_treads-1 for both), the relay does NOT switch on the
timer.  Instead the climber's foot-target obs slot ([51:55], see
ladder_foot_targets in mdp.py) is fed a VIRTUAL next tread: a point on the
deck advancing along the stairs' forward direction at the climb speed
(0.04 m/s), one riser above the deck, per-side at the landing's lateral
target offset — bit-for-bit in ladder_foot_targets' format (world vector
to (target - foot) in the robot's yaw frame, scaled 1/0.05, clamped +/-5).
The climber reads "one more stair" and keeps walking across the deck until
it trips on the tabletop (the user's goal: "fall on the table if you must,
but keep walking") — then the normal fall switch fires and getup takes
over on the deck.  Safety: the virtual target never advances past the
deck's far edge (4 cm margin); a fall (tilt > 60 deg) stops the feed
immediately and switches.  Walk-on measures: distance walked on the deck,
whether the fall happened inside the deck footprint, and the getup stand.

usage:
  uv run python scripts/render_stairs_getup.py \
      --source logs/rsl_rl/simple_stairs/2026-09-24_23-23-31_simple_stairs/model_3998.pt \
      --out logs/render-stairs-getup --seeds 19927 19928 19929 19930 --walk-on
env vars: SIMPLE_STAIRS_NOSE_BEVEL_M / SIMPLE_STAIRS_LANDING_SETBACK_M /
          SIMPLE_STAIRS_LANDING_FLUSH / SIMPLE_STAIRS_TABLE_LEGS /
          SIMPLE_STAIRS_RAIL_OVERHANG_M (geometry knobs, see the env cfg)
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch


# --- pure helpers (unit-tested) -------------------------------------------------


def walk_on_advance(vt_x: float, deck_far_x: float, speed_m_s: float, dt: float) -> float:
    """Advance the virtual target along the deck at the climb speed, clamped
    at the deck's far edge (caller includes its own safety margin)."""
    return min(vt_x + speed_m_s * dt, deck_far_x)


def walk_on_obs_targets(
    vt_x: float, lat_m: float, vt_z: float, feet_pos: np.ndarray, yaw: float
) -> np.ndarray:
    """Format a virtual "next tread" for the foot-target obs slot, in exactly
    ``ladder_foot_targets``' layout (mdp.py): per foot the world vector to
    (target - foot) projected into the robot's yaw frame, scaled 1/0.05 and
    clamped to +/-5 — [fwd_L, up_L, fwd_R, up_R].  ``lat_m`` is the per-side
    lateral target offset (the landing convention: each foot aims at its own
    side of the platform, not the centreline).  ``feet_pos``: (2, 3) world
    foot-site positions, (left, right) order.
    """
    out = np.zeros(4, dtype=np.float32)
    for slot, side in enumerate((1.0, -1.0)):
        vec = np.array([vt_x, side * lat_m, vt_z], dtype=np.float64) - feet_pos[slot].astype(np.float64)
        fwd = (vec[0] * math.cos(yaw) + vec[1] * math.sin(yaw)) / 0.05
        up = vec[2] / 0.05
        out[2 * slot] = np.clip(fwd, -5.0, 5.0)
        out[2 * slot + 1] = np.clip(up, -5.0, 5.0)
    return out


def on_deck_xy(pos_xy: np.ndarray, top_c_xy: np.ndarray, half_depth: float, half_width: float) -> bool:
    """Trunk xy inside the top platform's footprint (box centre +/- half extents)."""
    return bool(
        abs(float(pos_xy[0]) - float(top_c_xy[0])) <= half_depth
        and abs(float(pos_xy[1]) - float(top_c_xy[1])) <= half_width
    )


class StandTracker:
    """Continuous-hold stand tracker for the relay's ``stood`` metric.

    The flush landing's top is LEVEL with the last mini tread, so "trunk z
    above top + 0.09" alone cannot tell "standing on the deck" from
    "standing on the last tread, never made it up" (a real seed-19927 false
    positive: 14.3 s 'stand' that was actually on the tread).  A run
    accumulates only upright-at-height steps with BOTH feet inside the
    landing box's xy footprint.  Getup re-stands lift the feet briefly, so
    a run is only DISCARDED when off-deck samples exceed ``off_frac_max``
    of the run's steps (default 0.2 — i.e. >= 80% of the hold must be on
    the deck); a topple (not upright-at-height) resets the run instantly.
    """

    def __init__(self, off_frac_max: float = 0.2):
        self.off_frac_max = off_frac_max
        self.held = 0.0
        self.off = 0.0
        self.started = False

    def update(self, upright_at_height: bool, feet_on_deck: bool, dt: float) -> None:
        if not upright_at_height:
            self.held = 0.0
            self.off = 0.0
            self.started = False
            return
        if feet_on_deck:
            self.held += dt
            self.started = True
        else:
            self.off += dt
        total = self.held + self.off
        if total > 0.0 and self.off / total > self.off_frac_max:
            # The run is mostly off the deck — it never happened.
            self.held = 0.0
            self.off = 0.0
            self.started = False

    def stood(self, hold_s: float) -> bool:
        return self.held >= hold_s


def main() -> None:
    import argparse
    import json
    from dataclasses import asdict

    import onnxruntime as ort
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends
    from mjlab.utils.wrappers import VideoRecorder

    import mjlab_microduck.tasks  # noqa: F401 - populate registry
    from mjlab_microduck.tasks import MicroduckOnPolicyRunner, mdp
    from mjlab_microduck.tasks.microduck_simple_stairs_env_cfg import (
        MicroduckSimpleStairsRlCfg,
        make_microduck_simple_stairs_env_cfg,
    )
    from mjlab_microduck.video_effects import configure_video_cfg, fix_render_shadows

    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="climber checkpoint (rsl_rl .pt)")
    p.add_argument("--out", required=True)
    p.add_argument("--getup", default=None, help="getup.onnx path (default: desk-climb training/models/getup.onnx)")
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--seeds", type=int, nargs="+", default=[19923, 19924, 19925, 19926])
    p.add_argument("--level", type=int, default=4, help="fixed curriculum level (4 = 24-25.5 mm risers)")
    p.add_argument("--azimuth", type=float, default=100.0)
    p.add_argument("--distance", type=float, default=0.9)
    p.add_argument("--elevation", type=float, default=-12.0)
    p.add_argument("--switch-after-s", type=float, default=6.0, help="fallback switch time (pre-top only in --walk-on)")
    p.add_argument("--switch-grace-steps", type=int, default=5)
    p.add_argument("--switch-tilt-deg", type=float, default=60.0)
    p.add_argument("--switch-drop-m", type=float, default=0.10, help="deck region: trunk z > top - this")
    p.add_argument("--stand-hold-s", type=float, default=3.0)
    p.add_argument("--stand-tilt-deg", type=float, default=25.0)
    p.add_argument("--stand-above-top-m", type=float, default=0.09)
    p.add_argument("--dump-obs", action="store_true", help="print the 61D layout + ONNX io contract and exit")
    p.add_argument("--walk-on", action="store_true", help="feed a virtual next tread on the deck and keep walking")
    a = p.parse_args()

    torch.set_num_threads(1)
    configure_torch_backends()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    getup_path = a.getup or str(Path(__file__).resolve().parents[2] / "training" / "models" / "getup.onnx")

    # --- env: simple_stairs (geometry knobs default on), single env, no resets
    cfg = make_microduck_simple_stairs_env_cfg(play=True)
    cfg.seed = a.seeds[0]
    cfg.scene.num_envs = 1
    cfg.auto_reset = False  # a fall must never reset the episode mid-relay
    cfg.episode_length_s = a.seconds + 1.0
    spawn = cfg.events["reset_stair_ladder"].params
    spawn.update(fixed_level=a.level, floor_spawn_prob=1.0, swing_spawn_prob=0.0)
    configure_video_cfg(cfg)
    cfg.viewer.distance = a.distance
    cfg.viewer.elevation = a.elevation
    cfg.viewer.azimuth = a.azimuth

    raw = ManagerBasedRlEnv(cfg, device="cuda:0", render_mode="rgb_array")
    fix_render_shadows(raw, light_dir=(-0.5, 0.2, -1.0))
    steps = int(round(a.seconds / raw.step_dt))

    agent = asdict(MicroduckSimpleStairsRlCfg)
    agent.update(logger="tensorboard", upload_model=False, seed=a.seeds[0])
    _env0 = RslRlVecEnvWrapper(raw, clip_actions=agent.get("clip_actions"))
    runner = MicroduckOnPolicyRunner(_env0, agent, str(out), "cuda:0")
    runner.load(a.source, map_location="cuda:0")
    climber = runner.get_inference_policy(device="cuda:0")

    session = ort.InferenceSession(getup_path, providers=["CPUExecutionProvider"])

    robot = raw.scene["robot"]
    servo_ids = mdp._servo_joint_ids(raw, robot)
    head_ids, _ = robot.find_joints("^(neck_pitch|head_pitch|head_yaw|head_roll)$")
    head_actions = [i for i, j in enumerate(servo_ids) if j in head_ids]
    alpha = torch.full((14,), 0.7, device=raw.device)
    alpha[head_actions] = 0.5
    base_gains = [v.kp_scale.clone() for v in robot.actuators]

    if a.dump_obs:
        raw.reset(seed=a.seeds[0])
        probe_obs = raw.observation_manager.compute(update_history=False)["actor"]
        print(f"[dump] obs['actor'] shape: {tuple(probe_obs.shape)}")
        layout = [
            ("base_ang_vel", 0, 3), ("projected_gravity", 3, 6), ("joint_pos", 6, 20),
            ("joint_vel", 20, 34), ("actions(last)", 34, 48), ("twist", 48, 51),
            ("head_command(foot targets)", 51, 55), ("body_command", 55, 61),
        ]
        for name, lo, hi in layout:
            seg = probe_obs[:, lo:hi]
            print(f"[dump] [{lo:2d}:{hi:2d}] {name:<26} nonzero={int(torch.count_nonzero(seg))}/{seg.numel()}")
        i0 = session.get_inputs()[0]
        o0 = session.get_outputs()[0]
        print(f"[dump] getup input: name={i0.name!r} shape={i0.shape} type={i0.type}")
        print(f"[dump] getup output: name={o0.name!r} shape={o0.shape} type={o0.type}")
        g = probe_obs.clone()
        g[:, 48:] = 0.0
        y = session.run(None, {i0.name: g.cpu().numpy().astype(np.float32)})[0]
        print(f"[dump] first getup action: shape={y.shape} finite={bool(np.isfinite(y).all())}")
        raw.close()
        raise SystemExit(0)

    def _run_seed(seed: int) -> dict:
        """One relay attempt: climber until fall-on-deck/timeout, then getup."""
        for v, gr in zip(robot.actuators, base_gains):
            v.kp_scale.copy_(gr)  # restore gains modified by the previous run
        rec = VideoRecorder(
            raw, video_folder=out / f"seed_{seed}", step_trigger=lambda s: s == 0,
            video_length=steps, disable_logger=True,
        )
        env = RslRlVecEnvWrapper(rec, clip_actions=agent.get("clip_actions"))
        raw.reset(seed=seed)
        obs = env.get_observations()
        assert obs["actor"].shape == (1, 61), f"obs contract broken: {obs['actor'].shape}"

        st = mdp._stair_state(raw)
        g = st.geometry
        top_idx = g.num_treads - 1
        top_z = st.tread_top[:, top_idx]
        top_c = st.tread_centre[0, top_idx]
        half_d = 0.5 * g.landing_depth_m
        half_w = 0.5 * g.landing_width_m
        # The landing's per-side lateral foot-target offset (landing_target_per_side).
        lat_m = 0.5 * g.center_gap_m + 0.5 * g.side_width_m
        sites = mdp._stair_foot_sites(raw, robot)

        switched = torch.zeros(1, dtype=torch.bool, device=raw.device)
        previous_raw = torch.zeros((1, 14), device=raw.device)
        previous_executed = previous_raw.clone()
        switch_step = None
        switch_reason = None
        fall_streak = 0
        tracker = StandTracker()
        stand_held = 0.0
        tilt = torch.zeros(1, device=raw.device)
        # walk-on state
        walk_active = False
        walk_start_x = None
        vt_x = None
        fell_on_deck = False

        for step in range(steps):
            # --- walk-on bookkeeping (before the policy acts) ---
            if a.walk_on and not bool(switched[0]):
                if not walk_active:
                    # Flush landings share the last tread's level, so classify
                    # "on the deck" by the landing box's xy footprint, not by
                    # the contact geom (a foot mid-seam can read the mini tread).
                    feet_xyz = robot.data.site_pos_w[:, sites, :]
                    if bool(mdp.feet_on_landing_xy(st, feet_xyz).all(dim=1)[0]):
                        walk_active = True
                        walk_start_x = float(robot.data.root_link_pos_w[0, 0])
                        vt_x = max(
                            float(robot.data.site_pos_w[0, sites, 0].max()), walk_start_x
                        ) + 0.05
                        print(f"[seed {seed}] walk-on starts at {step * raw.step_dt:.2f}s (x={walk_start_x:.3f})")
                if walk_active:
                    vt_x = walk_on_advance(vt_x, float(top_c[0]) + half_d - 0.04, 0.04, raw.step_dt)
                    yaw = float(mdp._yaw_from_quat(robot.data.root_link_quat_w)[0])
                    feet_pos = robot.data.site_pos_w[0, sites].cpu().numpy()
                    obs["actor"] = obs["actor"].clone()
                    obs["actor"][:, 51:55] = torch.tensor(
                        walk_on_obs_targets(vt_x, lat_m, float(top_z[0]) + 0.025, feet_pos, yaw),
                        device=raw.device,
                    )[None, :]

            if not bool(switched[0]):
                with torch.inference_mode():
                    act = climber(obs)
                act = act.clone()
            else:
                getup_obs = obs["actor"].clone()
                getup_obs[:, 48:] = 0.0  # blind contract: zero the command slots
                getup_obs[:, 34:48] = previous_raw  # last_action = raw ONNX output
                y = session.run(None, {session.get_inputs()[0].name: getup_obs.cpu().numpy().astype(np.float32)})[0]
                act = torch.tensor(y, device=raw.device)
                previous_raw = act.clone()

            tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0))
            z = robot.data.root_link_pos_w[:, 2]
            if switch_step is None:
                if walk_active:
                    # already on/above the deck: any topple ends the walk
                    on_deck_fall = bool(tilt[0] > math.radians(a.switch_tilt_deg))
                else:
                    on_deck_fall = bool(
                        (tilt[0] > math.radians(a.switch_tilt_deg))
                        and bool(z[0] > top_z[0] - a.switch_drop_m)
                    )
                fall_streak = fall_streak + 1 if on_deck_fall else 0
                if fall_streak >= a.switch_grace_steps:
                    switch_step, switch_reason = step, "deck_fall"
                    fell_on_deck = on_deck_xy(
                        robot.data.root_link_pos_w[0, :2].cpu().numpy(),
                        top_c[:2].cpu().numpy(), half_d, half_w,
                    )
                    print(f"[seed {seed}] switch at {step * raw.step_dt:.2f}s (deck_fall, on_deck={fell_on_deck})")
                elif not walk_active and step * raw.step_dt >= a.switch_after_s:
                    switch_step, switch_reason = step, "timeout"
                    print(f"[seed {seed}] switch at {step * raw.step_dt:.2f}s (timeout)")
                if switch_step is not None:
                    switched[:] = True

            # Official contract: gains to 80% and alpha-filtered actions after the switch.
            for v, gr in zip(robot.actuators, base_gains):
                v.kp_scale.copy_(torch.where(switched[:, None], gr * 0.8, gr))
            act = torch.where(switched[:, None], alpha * act + (1 - alpha) * previous_executed, act)
            previous_executed = act.clone()
            raw._manual_reset_pending.zero_()
            obs, _, done, _ = env.step(act)
            assert torch.isfinite(act).all()

            if switch_step is not None:
                # stood = upright at deck height for a CONTINUOUS hold with
                # >= 80% of samples both-feet-on-deck (flush top == last
                # tread's top, so trunk height alone cannot tell them apart).
                up_hi = bool(
                    (tilt[0] < math.radians(a.stand_tilt_deg))
                    and bool(z[0] > top_z[0] + a.stand_above_top_m)
                )
                feet_ok = bool(
                    mdp.feet_on_landing_xy(st, robot.data.site_pos_w[:, sites, :]).all(dim=1)[0]
                )
                tracker.update(up_hi, feet_ok, raw.step_dt)
                stand_held = tracker.held

        stood = tracker.stood(a.stand_hold_s)
        final_x = float(robot.data.root_link_pos_w[0, 0])
        walk_distance = (final_x - walk_start_x) if walk_start_x is not None else 0.0
        print(
            f"[seed {seed}] switched={switch_step is not None}({switch_reason}) "
            f"walk_on={walk_active} walk_dist={walk_distance:.3f}m fell_on_deck={fell_on_deck} "
            f"stand_held={stand_held:.2f}s(off={tracker.off:.2f}) stood(>= {a.stand_hold_s}s)={stood}"
        )
        return {
            "seed": seed,
            "switched": switch_step is not None,
            "switch_step_s": None if switch_step is None else round(switch_step * raw.step_dt, 2),
            "switch_reason": switch_reason,
            "walk_on_activated": bool(walk_active),
            "walk_distance_m": round(walk_distance, 4),
            "fell_on_deck": bool(fell_on_deck),
            "stood": bool(stood),
            "stand_held_s": round(stand_held, 2),
            "stand_off_deck_s": round(tracker.off, 2),
            "final_tilt_deg": round(math.degrees(float(tilt[0])), 1),
            "video": f"seed_{seed}/",
        }

    runs = [_run_seed(s) for s in a.seeds]
    n_stood = sum(r["stood"] for r in runs)
    n_walked = sum(r["walk_on_activated"] for r in runs)
    report = {
        "source": a.source,
        "getup": getup_path,
        "level": a.level,
        "walk_on": bool(a.walk_on),
        "geometry": {
            "nose_bevel_m": os.getenv("SIMPLE_STAIRS_NOSE_BEVEL_M", "0.0"),
            "landing_flush": os.getenv("SIMPLE_STAIRS_LANDING_FLUSH", "1"),
            "table_legs": os.getenv("SIMPLE_STAIRS_TABLE_LEGS", "1"),
            "rail_overhang_m": os.getenv("SIMPLE_STAIRS_RAIL_OVERHANG_M", "0.12"),
        },
        "stand_hold_s": a.stand_hold_s,
        "runs": runs,
        "stood_count": n_stood,
        "walked_count": n_walked,
        "success_fraction": round(n_stood / len(runs), 4),
    }
    (out / "relay.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}, indent=2))
    print(f"[relay] done -> {out}")
    raw.close()


if __name__ == "__main__":
    main()
