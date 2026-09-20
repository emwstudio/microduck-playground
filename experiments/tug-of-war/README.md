# Tug-of-war — two ducks, one rope, real physics

Two Microducks (each wearing the printable harness collar with butt + chest
tow rings) pull a hemp rope back-to-back. A red ribbon marks the rope's
midpoint; dragging it past your coloured win line wins the bout.

Final shipped bout (`media/preview.mp4`, 4K in DuckEMW artifacts): the v16
fine-tuned steady-lean policy mirror, red wins at 12.3 s, zero falls, near
zero visible rotation.

## Reproduce

```bash
PYTHONPATH=src uv run scripts/tug_of_war.py \
  --n-per-team 1 --gap 0.55 \
  --policy-red  policies/tugchain_steady_v16_2250.onnx --red-kind  tug \
  --policy-blue policies/tugchain_steady_v16_2250.onnx --blue-kind tug \
  --rounds 1 --round-duration 25 --win-x 0.15 \
  --rope-stiffness 60 --rope-damping 2.0 --pretension 0.005 \
  --fatigue-start 6 --fatigue-floor 0.8 --fatigue-team red \
  --yaw-kp 0.8 --cam-track 0.3 --fps 50 \
  --out match.mp4
```

## Training tasks (registered in `tasks/__init__.py`)

| Task id | Layout |
|---|---|
| `Mjlab-Microduck-Tug-Steady/-Shuffle` | duck + harness drags a weighted sled (first recipe family) |
| `Mjlab-Microduck-TugChain-Steady/-Shuffle` | 1v1 duck chain, frozen v6 opponent |
| `Mjlab-Microduck-TugChain3-Steady/-Shuffle` | 3v3 chain, learner = middle duck, 5 frozen ducks |
| `Mjlab-Microduck-TugChain3S-Steady/-Shuffle` | 3v3 chain, all 6 ducks live learners (shared policy) |
| `Mjlab-Microduck-TugChainF-Steady/-Shuffle` | face-to-face variant (chest-ring rope, backward pull) |

## Key mechanisms (all measured, none guessed)

- **Real rope**: tension-only dead-band cord (stiffness/damping/pretension),
  taut length measured ring-to-ring — the trunk-distance taut length once
  made every rope decorative (the mother bug).
- **Deciding without falls**: `--fatigue-*` ramps one team's foot mu down so
  it slides on its feet and bleeds ground; `--get-up` stands stumblers back
  up for video mode.
- **Straight pulling**: tug policies have no heading channel and orbit
  ±150°/bout. Root fix (v16) = fine-tune the proven puller with the
  `track_angular_velocity` command channel (alpha's own mechanism), then feed
  a gentle per-duck yaw-rate PD command at deployment (`--yaw-kp`).
  Cosmetic fallback: `--cam-track` orbit-cancelling camera.
- **Red ribbon** at the rope midpoint (mocap) so the win reads on video.
- **Visual rope**: swept hemp variants chosen per chord length; taut when
  pulled, sagging on stumbles (measured 76% taut / 24% brief slack in the
  shipped bout).

## Lesson log (the short version)

v1/v2 reward hacks (posing not pulling) → v3 burst-and-die → v4 real pulls
but hanging posture → v6 hard 35° lean gate → v7 1v1 chain (the keeper) →
v8 frozen teammates (94% down) → v9 self-play mirror works but orbits →
v11/v12 forward lean (38%-mass head face-plants, dead end) →
v13 heading-hold (destabilized) → v14 face-to-face (no consolidation) →
v15 yaw channel from scratch (inconsistent) → **v16 fine-tune the champion
with the yaw channel (works)**.

Policies used in the final bout: `policies/tugchain_steady_v16_2250.onnx`
(v7 `model_1999` + 500 iterations of yaw-channel fine-tuning, checkpoint
`model_2250`).
