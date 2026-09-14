# Tug-of-war harness rig

Parametric, printable Microduck harness rig for the rope-tug demos: a
**standoff tow-pack** — a crowned back panel floating 28 mm behind the butt
shell on two standoff rails (jetpack-style), carrying the butt carabiner
(the pulled end), plus a chest carabiner (the pulling end), so the rope's
force path runs through the duck like a human puller's. Clearance is
validated with MuJoCo's own collision engine — see
[VALIDATION.md](VALIDATION.md).

<table>
  <tr>
    <td><img src="renders/frame_front.png" width="300" alt="Frame front view"></td>
    <td><img src="renders/frame_three_quarter.png" width="300" alt="Frame three-quarter view"></td>
    <td><img src="renders/frame_side.png" width="300" alt="Frame side view"></td>
  </tr>
</table>

## Files

- `source/cad_rig.py` — waist-band variant (superseded by the saddle);
- `source/cad_towpack.py` — standoff tow-pack (current design): rounded
  panel + standoff rails + carabiner eye, watertight STL;
- `source/cad_saddle.py` — form-fitted butt saddle (superseded; lessons:
  a snug back cup inevitably overlaps the legs' fold-back swing cone);
- `source/verify_pack_contacts.py` — MuJoCo-native contact validation
  (per-joint sweeps + LHS + tug envelope, JSON report);
- `source/verify_saddle_clearance.py` — earlier trimesh/FCL validator
  (superseded by the MuJoCo-native one);
- `meshes/*_mm.stl` — millimetre-scale printable meshes;
- `../../src/mjlab_microduck/robot/microduck/assets/tug_frame.stl`,
  `tug_hook_carabiner.stl`, `tug_chest_carabiner.stl` — metre-scale MuJoCo
  visual meshes (water-tight, density-0 in the sim).
