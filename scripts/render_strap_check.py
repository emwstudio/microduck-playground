"""Single-duck close-up renders of the tow-pack + waist strap (fix check)."""
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mjlab_microduck.robot.tug_of_war import build_tug_spec

OUT = Path(__file__).resolve().parents[3] / "artifacts" / "tug_of_war_v1"

spec = build_tug_spec(n_per_team=1)
model = spec.compile()
data = mujoco.MjData(model)

# stand the duck up (freejoint occupies qpos[0:7])
jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
adr = model.jnt_qposadr[jid]
data.qpos[adr:adr + 7] = (0, 0, 0.12, 1, 0, 0, 0)
mujoco.mj_forward(model, data)

# hide span ropes + center line so only the duck and its rig show
for i in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
    if name.startswith("tug_span") or name.startswith("center_line"):
        model.geom_rgba[i, 3] = 0.0

renderer = mujoco.Renderer(model, height=900, width=1200)
scene_option = mujoco.MjvOption()

cams = {
    "strap_back34": dict(azimuth=205, elevation=-12, distance=0.40, lookat=(-0.04, 0, 0.10)),
    "strap_side": dict(azimuth=90, elevation=-8, distance=0.36, lookat=(-0.02, 0, 0.10)),
    "strap_front34": dict(azimuth=25, elevation=-12, distance=0.40, lookat=(0.0, 0, 0.10)),
    "strap_top": dict(azimuth=180, elevation=-55, distance=0.45, lookat=(-0.02, 0, 0.10)),
}
cam = mujoco.MjvCamera()
for name, cfg in cams.items():
    cam.azimuth, cam.elevation, cam.distance = cfg["azimuth"], cfg["elevation"], cfg["distance"]
    cam.lookat = cfg["lookat"]
    renderer.update_scene(data, camera=cam, scene_option=scene_option)
    Image.fromarray(renderer.render()).save(OUT / f"{name}.png")
    print("wrote", OUT / f"{name}.png")
