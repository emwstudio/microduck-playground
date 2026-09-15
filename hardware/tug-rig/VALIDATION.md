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
band height z = -4 mm — above the hip-shell bulge so the clamp is visible
from the side), SPLIT (~18 mm) on the +y side with two slim PARALLEL
cantilever lugs (5×8 mm, faces exactly perpendicular to the screw axis so
the head and nut seat flat and clamp squarely) reaching 20 mm radially
OUTBOARD — the **M3 socket-head cap screw** (Ø3.2 mm holes, 2.5 mm hex
socket, M3×0.5 thread, black-oxide in sim) bridges the lug tips ~14 mm
OUTSIDE the ring's outer surface, so the hex key tightens it with zero
obstruction from the ring. Tightening pulls the lug arms together and
clamps the band onto the shell by friction. The tow eyes are D-ring style: ring plane
VERTICAL and fore-aft (rope threads from the side, pull stays in the
ring's plane), **fused to the band by cast neck bosses** — the collar STL
is verified as ONE connected shell (`connected_bodies == 1` at
generation). Each eye has a **clear Ø9 mm hole** (Ø4 mm rope passes
doubled for a lark's head; Ø6 mm single for a bowline) — the generator
pushes a Ø8 mm gauge pin sideways through each eye and asserts the
channel is unobstructed:
front eye (chest pull) at x = +46.6 mm, back eye (butt pull) at
x = -60.6 mm, both at z = -4 mm. The whole tug force path flows through
the collar; no panel, standoffs or webbing are needed. The pack + strap
STLs are kept in meshes/ as an archived alternative.

Collar contact/clearance validation against the leg swing envelope is
pending; the band sits at mid-torso, above the hip joint line.

No insertion-force, pull-out, anti-slip, wear, fatigue or load testing has
been performed — the fastening is a geometric fit only. 3D-print validation
is required before any hardware use.
