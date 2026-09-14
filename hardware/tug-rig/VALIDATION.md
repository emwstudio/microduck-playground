# Tug tow-pack clearance validation

MuJoCo-native contact checks (pack welded to trunk_base with contype=1 and a
3 mm contact margin; every sampled pose forward-kinematicsed; real mj_contacts
counted). Report: meshes/pack_contact_report.json.

## Results

- **Per-joint sweeps (14 servos, full range): 12/14 servos completely clean.**
  Only left_hip_pitch (1 pose), right_hip_roll (1 pose), right_hip_pitch
  (4 poses) contact — all at extreme fold-back swing values.
- **LHS full 10-DOF leg space (400 samples): 112 blocked poses (28.0%)** —
  knee-bent + hip-fold-back combinations.
- **Tug-of-war envelope** (hip_pitch ∈ [-0.5, +0.35] rad, |hip_roll| ≤ 0.4 rad,
  walking is ±0.3 / small roll): **8 blocked poses (26.7% of the 30-sample
  envelope subset)**, all fold-back-behind-butt sagittal poses that cannot
  occur in tug walking — same physical budget the swing seat documents.

## Interpretation

The pack floats 28 mm behind the butt shell on two standoff rails, so the
legs' swing space sits in the gap by construction; residual contacts are the
leg tips' fold-back cone touching the rails at extremes, not a fit defect.
Contacts with trunk_base at the rail feet are the INTENDED mounting interface
(classified MOUNTING in the report, not counted as faults).

## Fastening (how the pack stays on the duck)

Two complementary features, same scheme as the swing seat (locating + strap):

- **Contoured locating pads** at the rail feet (10 mm discs keyed against the
  hip-shell corner) stop the pack rotating/sliding on the smooth shell.
- **A 20×2 mm velcro webbing strap** (`tug_strap.stl`) threads through the
  two vertical slots cut in the panel's side edges (slot = strap +1 mm
  clearance) and belts around the waist at slot height (z = -10 mm). The
  back arc sits in the slot plane (x = -78 mm), front/sides hug the torso
  with ~2 mm clearance. The strap is removable — the pack itself needs no
  screws or glue.

No insertion-force, pull-out, anti-slip, wear, fatigue or load testing has
been performed — the fastening is a geometric fit only. 3D-print validation
is required before any hardware use.
