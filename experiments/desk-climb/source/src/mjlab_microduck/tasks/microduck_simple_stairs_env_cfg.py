"""Simple straight staircase ("ordinary stairs") for Microduck.

Full-width 60 mm treads (the whole 54 mm sole fits flat) and OPEN risers:
with the angle shallow enough that run = riser/tan(angle) exceeds the 60 mm
tread depth, consecutive treads leave an open gap and the swing toe passes
through the gap instead of under the next tread.  That makes risers BELOW
the 29 mm toe-under-tread minimum (ladder.py:min_riser_m) legal — the v12
design converges on 25 mm at ~19-20 deg (run 68.7 mm, gap ~8.7 mm), inside
the official gait family's ~25 mm step envelope (FK: foot apex clearance
median 66.6 / p10 34.7 mm).  v1-v3 failed because the reset clamped risers
to 29 mm, above that envelope.

The reset enforces the gap per env: ``reset_stair_ladder(open_riser=True)``
skips the min_riser clamp and instead clamps the angle so
run >= tread depth + 2 mm (see ladder.clamp_riser_angle).  The level table
below is drawn up so every band already satisfies the gap at its lower
riser, so the clamp is a guard, not a distortion.  12 treads up to a
full-depth top platform.

v13 reward patch (v12 diagnosis: the "2-tread wall"): F1 wires the
staircase family's tread-stall price into this recipe (the stance term pays
~0.54/step for parking on treads under a climb command, and the plain
ladder recipe has no anti-park term); F2 widens the swing-overshoot
clearance 25 -> 50 mm so the official gait family's ~67 mm foot apex is
legal at the shallow 15-17 mm risers (the ceiling is support + riser +
3 mm + clearance on this one-riser-spacing geometry).

v14 dose correction (single variable): F1 cured the retreat farm
(0.44 -> 0.04) but the 2 s / 2.0 dose also condemned the normal between-
treads steady pause (falls 0.25 -> 0.99).  The stall window is now 4.0 s
(the duck steps at ~0.4 s/step and measured inter-tread pauses run 1-3 s,
so only a TRUE no-progress park of 4+ s pays) and the weight 1.0 (park
income ~0.54/step vs the -1.0/step price — still a 2x kill, but a steady
pause is no longer fatal).  F2 unchanged.

v15 (top-platform finish): the v14c policy climbs steadily (median 9
treads, mean level ~3.2) but cannot finish the final 1-2 treads onto the
top platform — it pokes its head past the nose, cannot recover the CoM and
falls BACK down.  A ``top_approach_prob`` (0.35 train / 0 eval) share of
non-floor spawns starts as a static stance on (k, k+1) with k in
{num_treads-4 .. num_treads-2} (8, 9, 10 — including the just-arrived
10/top-nose stance, the staircase family's onto-landing formula), marking
them spawn_on_top so stop-fails don't demote (the s34 lesson); the share
decays to 0.15 at iter 1000 (event_param_curriculum).  Warm start: load a
v14c checkpoint with MICRODUCK_WARM_START=1 (mdp.py Patch 5 — counters and
iteration restart at 0, weights/normalizer/optimizer kept) plus
MICRODUCK_SIMPLE_STAIRS_START_LEVEL=3 (one-shot level seed — the loaded
policy consolidated at mean level ~3.2, so the curriculum resumes near its
level instead of re-proving L0).  Rewards untouched for attribution.

v16 (probe-fall-location.json: 78-81 % of falls start at the tread 10 ->
top transition, then slide 2-3 risers down for free).  Two reward patches,
everything else kept from v14/v15: (1) ``tumble_deficit`` — a tread-
quantized high-water deficit (self-negating, weight 2.0) so every step
spent below the episode's best tread bleeds, while crouches/bobbing stay
free (foot_last_tread only moves on contact); (2) the v14 tread_stall dose
is LIFTED while both feet are within 3 treads of the top platform
(``_tread_stall_top_exempt``), so the TOP gate's required slow-down
(|v| < 0.35, 0.5 s hold) is no longer taxed as a stall.

v17 (single variable: the deficit's schedule).  v16 applied the w=2.0 tax
from iteration 0 and the policy chose conservatism over exploration
(rise_p90 2.25 vs v14's 11.3) — the official rule, empirically: any
attempt-tax active while a hard skill is being explored makes "do nothing"
win.  The tax was gated on the per-env adaptive ladder level (L0-L1 free,
L2+ full).

v18 (single variable: the gate condition).  The v17 level gate deadlocked:
in the tax-free L0-L1 zone "climb 2 and retreat" was the free optimum (the
v12 retreat farm resurrected at 0.42), so no env ever reached L2 and the
tax never engaged — exploration-tax again, this time via the farm eating
the free zone.  The gate now keys on the EPISODE'S OWN high-water mark:
once this episode has stood on the threshold tread the tax is on for its
remainder (finish pressure arrives with the first success); episodes that
never got high stay free to explore.

v19 (single variable: threshold 5 -> 8).  The v18 threshold taxed the
breakthrough itself — v14's reckless 11-tread charges are built out of
mid-charge slips, and the gate made every slip back below 5 bleed, so the
frontier died with the farm still untouchable (two runs, no ignition).
8 confines the finish tax to the probe-proven crash zone (78-81 % of
v14c's falls start at the tread 8-11 top-nose region,
probe-fall-location.json): the 5-7 waist is free to charge through, the
~2-3-tread farm still cannot reach the gate.  Watch:
Episode_Reward/tumble_deficit (weighted effect — zero below 8, negative
above), swing_retreated_fraction (farm signature; must not rise), and
rise_p90 (must recover toward v14's 11 — the 5-7 frontier is free again).

Beveled nose (2026-09-26, the Hannes route — the getup relay needs the
duck to END UP on the deck, not bouncing off it): the probe shows 78-81 %
of falls start at the tread 10 -> top transition, face-first into the top
slab's 20 mm side wall above the 25 mm riser.  ``landing_nose_bevel_m``
(default 0.025 via SIMPLE_STAIRS_NOSE_BEVEL_M, factory-time so train and
play/eval share it) replaces that wall with a triangular ramp at the top
platform's nose; the mini-tread open-riser gaps are untouched.  Set the
env var to 0 to restore the wall for A/B.
"""

import os
from copy import deepcopy
from dataclasses import replace

import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, EventTermCfg, RewardTermCfg

from ..robot.ladder import StairLadderGeometry
from .microduck_ladder_env_cfg import (
    MicroduckLadderRlCfg,
    make_microduck_ladder_env_cfg,
)
from .microduck_velocity_env_cfg import NUM_STEPS_PER_ENV
from mjlab_microduck.tasks import mdp as microduck_mdp

# 12 treads up to a full-depth top platform.
SIMPLE_STAIRS_GEOMETRY = StairLadderGeometry(
    num_treads=12,
    alternating=False,
    tread_depth_m=0.060,  # the whole 54 mm sole rests flat (validate cap: 60 mm)
    landing_every=12,  # the 12th tread is the full-depth top platform
)
# v12 course: 15 -> 25.5 mm risers.  Full-width treads space same-side treads
# one riser apart, so the toe-under-tread rule would need riser >= 29 mm
# (min_riser_m) — above the ~25 mm step envelope.  With OPEN risers the
# binding rule is the gap: run = riser/tan(angle) >= 62 mm (60 mm depth +
# 2 mm margin).  Each band's top angle is atan(riser_lo / 0.062), so the
# band always satisfies the gap at its own lower riser; the reset's
# open_riser clamp guards the draws.  Level 4 is 24-25.5 mm at 19-20 deg —
# the target envelope riser with run ~68.7 mm at 25 mm / 20 deg.
SIMPLE_STAIRS_LEVELS: tuple[dict, ...] = (
    {"riser": (0.015, 0.017), "angle": (13.0, 13.6)},
    {"riser": (0.017, 0.020), "angle": (13.6, 15.3)},
    {"riser": (0.020, 0.022), "angle": (15.3, 17.9)},
    {"riser": (0.022, 0.024), "angle": (17.9, 19.5)},
    {"riser": (0.024, 0.0255), "angle": (19.0, 20.0)},
)
SIMPLE_STAIRS_EPISODE_LENGTH_S = 12.0


def _foot_targets_per_side(
    env, asset_cfg, min_rise: float = 0.006, scale: float = 0.05
):
    """ladder_foot_targets with per-side lateral targets on full-width treads.

    Upstream `_stair_foot_target_info` aims BOTH feet at the tread centre for
    full-width treads (tread_side == 0); only landings get per-side targets
    (landing_target_per_side, fixing the converging-step hesitation of the
    2026-09-04 landing study).  On a plain full-width staircase EVERY tread
    is full width, so all mini-tread targets were centreline.  Aim each foot
    at its own side of every full-width tread instead.
    """
    import torch
    from mjlab_microduck.tasks import mdp as _mdp
    from mjlab.managers import SceneEntityCfg as _SEC

    state = _mdp._stair_state(env)
    asset = env.scene[asset_cfg.name]
    out = torch.zeros(env.num_envs, 4, device=env.device)
    if state is None:
        return out
    info = _mdp._stair_foot_target_info(env, asset, min_rise)
    g = state.geometry
    num = g.num_treads
    yaw = _mdp._yaw_from_quat(asset.data.root_link_quat_w)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    lat_mag = 0.5 * g.center_gap_m + 0.5 * g.side_width_m
    landing = torch.tensor([g.is_landing(i) for i in range(num)], device=env.device)
    for slot, side in enumerate((1.0, -1.0)):
        idx = info["index"][:, slot].clamp_max(num - 1)
        full_width = state.tread_side[idx] == 0.0
        needs_offset = full_width & ~landing[idx] & info["valid"][:, slot]
        tyaw = state.tread_yaw.gather(1, idx[:, None]).squeeze(1)
        off = side * lat_mag * torch.stack((-torch.sin(tyaw), torch.cos(tyaw)), dim=-1)
        vec = info["vec"][:, slot].clone()
        vec[:, :2] = vec[:, :2] + torch.where(needs_offset[:, None], off, torch.zeros_like(off))
        fwd = vec[:, 0] * cy + vec[:, 1] * sy
        valid = info["valid"][:, slot]
        out[:, 2 * slot] = torch.where(valid, fwd / scale, 0.0)
        out[:, 2 * slot + 1] = torch.where(valid, vec[:, 2] / scale, 0.0)
    return torch.nan_to_num(out, nan=0.0).clamp(-5.0, 5.0)


# v18: the deficit tax is gated on the EPISODE'S OWN high-water mark instead
# of the adaptive ladder level.  The v17 level gate deadlocked: in the
# tax-free L0-L1 zone "climb 2, retreat" was the free optimum (the v12
# retreat farm resurrected at 0.42), so no env ever reached L2 and the tax
# never engaged.  Gating on per-episode performance instead: once THIS
# episode has stood on tread >= MIN_TREAD the tax is on for its remainder
# (the finish pressure arrives with the first success); episodes that never
# got high stay free to explore.  A mid-episode latch means the tax engages
# the moment the duck first stands on the threshold tread, not at the next
# reset.
# v19: threshold 5 -> 8 (single variable).  5 taxed the breakthrough itself:
# v14's reckless 11-tread charges were built out of mid-charge slips, and
# the v18 gate made exactly those slip back below 5 bleed (two runs, no
# ignition).  8 confines the finish tax to the probe-proven crash zone —
# probe-fall-location.json puts 78-81 % of v14c's falls at the tread 8-11
# top-nose region — while the 5-7 waist stays free and the ~2-3-tread farm
# still cannot reach the gate.
TUMBLE_DEFICIT_MIN_TREAD = 8


def simple_stairs_tumble_deficit_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Self-negating cost (<= 0), in TREADS per step: the current support
    tread minus the episode's high-water mark, GATED on that same mark
    (zero until the episode has stood on TUMBLE_DEFICIT_MIN_TREAD, v18).

    v16 (probe-fall-location.json): v14c's falls start at the tread 10 ->
    top-platform transition and the duck slides 2-3 risers back down for
    free — upward_progress pays per metre climbed, but a tumble only loses
    the truncated future, so "climb 10, fall off" was a fine episode.  Every
    step spent below the high-water mark now bleeds.  Tread quantization is
    what keeps normal climbing free: ``foot_last_tread`` only moves on tread
    CONTACT, so crouches and CoM bobbing between steps never pay; only
    genuinely landing on a lower tread (or the floor, -1) registers.  A hold
    episode or a stand at the best tread pays exactly zero.  At weight 2.0 a
    2-3-riser slide costs ~-40 to -90 across the tumble, against the
    ~+16-25 a riser earns from upward_progress — losing ground costs several
    times what gaining it paid, while a failed attempt still stays
    net-positive over parking (the attempt is not made fatal, just
    unprofitable vs finishing).
    """
    n, dev = env.num_envs, env.device
    state = microduck_mdp._stair_state(env)
    if state is None or not hasattr(state, "foot_last_tread"):
        return torch.zeros(n, device=dev)
    cur = state.foot_last_tread.max(dim=1).values  # highest tread either foot last stood on (-1 = floor)
    if not hasattr(state, "tumble_best"):
        state.tumble_best = cur.clone()
    # Re-latch at episode start so the previous episode's high-water mark
    # cannot bleed into the new one (same fresh pattern as the stall term).
    fresh = env.episode_length_buf <= 1
    state.tumble_best = torch.where(fresh, cur.clone(), state.tumble_best)
    state.tumble_best = torch.maximum(state.tumble_best, cur)
    deficit = (cur - state.tumble_best).clamp_max(0.0)
    gate = (state.tumble_best >= TUMBLE_DEFICIT_MIN_TREAD).float()
    return torch.nan_to_num(deficit * gate, nan=0.0).float()


def _tread_stall_top_exempt(
    env: ManagerBasedRlEnv,
    stall_s: float = 4.0,
    exempt_below_top: int = 3,
) -> torch.Tensor:
    """ladder_tread_stall_penalty with a top-approach exemption (v16).

    The TOP success gate requires slowing to |v| < 0.35 m/s and holding 0.5 s
    at the platform edge — the 4 s stall window punishes exactly that
    deceleration (probe: falls start at the 10 -> top transition carrying
    speed).  While BOTH feet's last tread is within ``exempt_below_top`` of
    the top platform (treads >= num_treads - 3 = 9 on simple_stairs), the
    stall price is lifted so the finish may be taken carefully; everywhere
    else the v14 dose applies unchanged.  Residual risk to watch: a free
    park on treads 9-10 (if it emerges, tighten the zone to >= 10 rather
    than raising the dose).
    """
    base = microduck_mdp.ladder_tread_stall_penalty(env, stall_s=stall_s)
    state = microduck_mdp._stair_state(env)
    if state is None or not hasattr(state, "foot_last_tread"):
        return base
    exempt_tread = state.geometry.num_treads - exempt_below_top
    near_top = (state.foot_last_tread >= exempt_tread).all(dim=1)
    return base * (~near_top).float()


def _seed_start_level(env, env_ids, geometry, level: int) -> None:
    """One-shot curriculum-level seed for warm starts (v15).

    A fresh env starts every env at level 0 (``_stair_ensure_state``), so a
    warm-started policy that consolidated at a high level would waste its
    competence re-proving the shallow levels — and the v15 target (the top
    transition at the target risers) needs L3+ exposure from iteration 0.
    Registered as a reset-mode event BEFORE reset_stair_ladder: on the first
    reset it pins every env to ``level`` (the first spawn is already at the
    seeded level), then never fires again — the adaptive
    ladder_level_curriculum takes over (demote on early falls, promote on
    climbs).  From-scratch runs leave ``start_level`` at 0 (unchanged).
    """
    state = microduck_mdp._stair_ensure_state(env, geometry)
    if getattr(state, "start_level_seeded", False):
        return
    state.start_level_seeded = True
    state.level[:] = int(level)


def make_microduck_simple_stairs_env_cfg(
    play: bool = False,
    top_spawn_prob: float = 0.0,
    per_side_targets: bool = os.getenv("MICRODUCK_SIMPLE_STAIRS_PER_SIDE", "0") == "1",
    top_approach_prob: float | None = None,
    start_level: int | None = None,
) -> ManagerBasedRlEnvCfg:
    """Straight full-width staircase with a top platform (see module docstring)."""
    if top_approach_prob is None:
        # v15: near-top spawn in training; OFF in play/eval so the success
        # rate measures the natural climb, not the bolt-on (house rule).
        top_approach_prob = 0.0 if play else 0.35
    if start_level is None:
        start_level = int(os.getenv("MICRODUCK_SIMPLE_STAIRS_START_LEVEL", "0"))
    # Beveled top-platform nose (2026-09-26, the Hannes route): the probe-
    # proven crash wall is the top slab's 20 mm face above the 25 mm riser;
    # a 25 mm wedge turns it into a ramp so a forward pitch rides up onto
    # the deck instead of bouncing back downstairs.  Only the top landing's
    # nose — the open-riser mini-tread gaps are untouched.  Read at factory
    # time so train AND play/eval share it (this one is meant to be
    # measured in eval); SIMPLE_STAIRS_NOSE_BEVEL_M=0 restores the wall.
    geometry = replace(
        SIMPLE_STAIRS_GEOMETRY,
        landing_nose_bevel_m=float(os.getenv("SIMPLE_STAIRS_NOSE_BEVEL_M", "0.025")),
    )
    cfg = make_microduck_ladder_env_cfg(
        play=play,
        geometry=geometry,
        level_table=SIMPLE_STAIRS_LEVELS,
        episode_length_s=SIMPLE_STAIRS_EPISODE_LENGTH_S,
        max_start_tread=8,
        top_spawn_prob=top_spawn_prob,
        open_riser=True,  # v12: 25 mm risers through the open gap, not under the tread
        top_approach_prob=top_approach_prob,
    )
    # v13 F1: anti-park.  The stance composite pays ~0.54/step for standing on
    # treads under a climb command (track factor 0.27 x weight 2.0), and the
    # plain ladder recipe carries no stall price — the v12 policy mounts 1-2
    # treads and parks ("低头站桩", retreated rising).  The staircase-landing
    # family fixed this exact basin with ladder_tread_stall_penalty (s15-17);
    # it was never wired into the plain ladder recipe.  Self-negating function
    # -> POSITIVE weight.
    # v14 dose correction (v13 cured the retreat farm 0.44 -> 0.04 but
    # overdosed: stall_s 2.0 / w 2.0 also condemned the normal between-treads
    # steady pause, falls 0.25 -> 0.99).  stall_s 4.0: the duck steps at
    # ~0.4 s/step and measured between-treads pauses run 1-3 s, so a 4 s
    # window only catches a TRUE park (no new tread in 4+ s).  weight 1.0:
    # the park's income stays ~0.54/step vs the -1.0/step price — still a 2x
    # kill — but an unlucky steady pause is no longer fatal.
    cfg.rewards["tread_stall"] = RewardTermCfg(
        func=_tread_stall_top_exempt,  # v16: lifted while both feet are within 3 of the top
        weight=1.0,
        params={"stall_s": 4.0},
    )
    # v16: make rolling back down the stairs expensive.  See the function
    # docstring for the mechanism (tread-quantized high-water deficit) and
    # the weight rationale: 2.0 x (treads below the episode best) per step,
    # several times the ~+16-25 a riser earns, attempts stay non-fatal.
    cfg.rewards["tumble_deficit"] = RewardTermCfg(
        func=simple_stairs_tumble_deficit_penalty,
        weight=2.0,
    )
    # v13 F2: let the natural step fit under the overshoot ceiling.  With
    # same_side_spacing=1 the ceiling is support + riser + 3 mm + clearance =
    # 44 mm at the 15-17 mm level-0 risers, taxing the official gait family's
    # median 66.6 mm foot apex at ~1.15/step (the alternating ladder is free:
    # two risers + 28 mm = 76-88 mm).  clearance 0.025 -> 0.05 raises the L0
    # ceiling to 69 mm (covers the 66.6 median; p90 72.4 keeps a small tax as
    # anti-fling pressure) and the L4 ceiling to 78 mm.
    cfg.rewards["swing_overshoot"].params["clearance"] = 0.05
    # v15: decay the near-top spawn share.  The top transition is a bolt-on,
    # not the main course: dense last-mile practice in the first half, then a
    # retention share so ordinary climbs re-dominate (the v10 lesson — a
    # nearly-done spawn left at full share retrains away the base skill).
    if top_approach_prob > 0.0 and not play:
        cfg.curriculum["top_approach_spawn"] = CurriculumTermCfg(
            func=microduck_mdp.event_param_curriculum,
            params={
                "event_name": "reset_stair_ladder",
                "param_stages": [
                    {"step": 0, "params": {"top_approach_prob": top_approach_prob}},
                    {"step": 1000 * NUM_STEPS_PER_ENV, "params": {"top_approach_prob": 0.15}},
                ],
            },
        )
    # v15 warm-start curriculum seed (see _seed_start_level).  Seeded via
    # MICRODUCK_SIMPLE_STAIRS_START_LEVEL (3 for the v14c warm start, whose
    # policy consolidated at mean level ~3.2); 0/off for from-scratch runs.
    if start_level > 0:
        cfg.events = {
            "seed_start_level": EventTermCfg(
                func=_seed_start_level,
                mode="reset",
                params={"geometry": geometry, "level": start_level},
            ),
            **cfg.events,
        }
    if per_side_targets:
        from mjlab.managers import SceneEntityCfg

        for group in ("actor", "critic"):
            terms = cfg.observations[group].terms
            terms["head_command"] = deepcopy(terms["head_command"])
            terms["head_command"].func = _foot_targets_per_side
            terms["head_command"].params = {"asset_cfg": SceneEntityCfg("robot")}
    return cfg


MicroduckSimpleStairsRlCfg = deepcopy(MicroduckLadderRlCfg)
MicroduckSimpleStairsRlCfg.experiment_name = "simple_stairs"
MicroduckSimpleStairsRlCfg.run_name = "simple_stairs"
