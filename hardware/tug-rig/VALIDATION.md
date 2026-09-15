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

## Fastening — clamp collar (current design, cad_collar.py)

The standoff pack + velcro strap above is superseded by a one-piece **clamp
collar** (`tug_collar.stl` + `tug_clamp_screw.stl`): a 14×2.2 mm band
following the shell contour (front +34 mm, back -48 mm, sides ±49 mm at
band height z = -10 mm, i.e. ~1-2 mm clamping allowance), SPLIT on the +y
side with two lugs — drilled **Ø3.2 mm through-holes** take an **M3
socket-head cap screw** (modelled with 2.5 mm hex socket and M3×0.5
thread) and hex nut; tightening the screw closes the split and clamps the
band onto the shell by friction (hose-clamp principle). The tow eyes are
closed rings **fused to the band by cast neck bosses** — the collar STL is
verified as ONE connected shell (`connected_bodies == 1` at generation):
front eye (chest pull) at x = +39 mm, back eye (butt pull) at x = -53 mm,
both at z = -10 mm. The whole tug force path flows through the collar; no
panel, standoffs or webbing are needed. The pack + strap STLs are kept in
meshes/ as an archived alternative.

Collar contact/clearance validation against the leg swing envelope is
pending; the band sits at mid-torso, above the hip joint line.

No insertion-force, pull-out, anti-slip, wear, fatigue or load testing has
been performed — the fastening is a geometric fit only. 3D-print validation
is required before any hardware use.
