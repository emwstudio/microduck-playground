"""Render the reproduced desk-climb sequence (continued climber + official getup ONNX).

Mirrors evaluate_sequence.py's official-handoff contract, single env, with VideoRecorder.
usage: uv run python ../training/render_climb_video.py --source ../logs/desk-climber/final.pt --out ../logs/render --seconds 45 --seed 19923
env vars (same as run.py): ENDING_ROLE LADDER_SHIFT HANDOFF_ARM TRIGGER_MODE SWITCH_MARGIN SWITCH_MAX_SPIN GETUP_FILE RECOVERY_MIN_ROOT_Z
"""
import argparse, json, os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

import mjlab_microduck.tasks  # noqa: F401
import geometry_patch
from mjlab_microduck.tasks import MicroduckOnPolicyRunner, mdp
from mjlab_microduck.tasks.microduck_desk_recovery_env_cfg import (
    DeskRecoveryRlCfg,
    make_desk_recovery,
)
from mjlab_microduck.video_effects import configure_video_cfg, fix_render_shadows

def style_hf(raw):
    """HF 展示视频观感：梯子木色、桌腿白、机器人橙脚橙喙、暖光。纯视觉，不动物理。"""
    import mujoco

    model = raw.sim.mj_model
    WOOD = (0.60, 0.40, 0.20, 1.0)
    WOOD_DARK = (0.45, 0.28, 0.13, 1.0)
    WOOD_LIGHT = (0.86, 0.70, 0.47, 1.0)
    LEG_WHITE = (0.92, 0.92, 0.92, 1.0)
    ORANGE = (0.96, 0.45, 0.08, 1.0)
    for i in range(model.ngeom):
        name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").split("/")[-1]
        if name.startswith("desktop"):
            model.geom_rgba[i] = WOOD_LIGHT
        elif name.startswith("desk_"):
            model.geom_rgba[i] = LEG_WHITE
        elif name.startswith("clamp"):
            model.geom_rgba[i] = WOOD_DARK
        elif name.startswith(("tread", "root_", "bridge", "spine", "rail", "base_", "landing")):
            model.geom_rgba[i] = WOOD
    ORANGE_PARTS = {
        "foot_left_material", "foot_right_material",
        "ankle_left_material", "ankle_right_material",
        "sole_left_material", "sole_right_material",
        "jaw_material", "jaw_soft_material", "bottom_head_shell_material",
    }
    recolored = set()
    for i in range(model.nmat):
        name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, i) or "").split("/")[-1]
        if name in ORANGE_PARTS:
            model.mat_rgba[i] = ORANGE
            recolored.add(name)
    print(f"[style] recolored {len(recolored)}/{len(ORANGE_PARTS)} robot materials: {sorted(recolored)}")
    for i in range(model.nlight):
        model.light_diffuse[i] = (0.95, 0.90, 0.82)
        model.light_ambient[i] = (0.35, 0.35, 0.38)
        model.light_specular[i] = (0.30, 0.30, 0.30)


p = argparse.ArgumentParser()
p.add_argument("--source", required=True)
p.add_argument("--out", required=True)
p.add_argument("--seconds", type=float, default=45.0)
p.add_argument("--seed", type=int, default=19923)
p.add_argument("--azimuth", type=float, default=100.0)
p.add_argument("--distance", type=float, default=1.0)
p.add_argument("--elevation", type=float, default=-12.0)
p.add_argument("--style", choices=["default", "hf"], default="default")
a = p.parse_args()
torch.set_num_threads(1)
configure_torch_backends()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

cfg = make_desk_recovery(play=True, floor_probability=1.0)
geometry_patch.configure_cfg(cfg)
cfg.seed = a.seed
cfg.scene.num_envs = 1
cfg.events["reset_stair_ladder"].params.update(
    min_start_tread=0, max_start_tread=0, desk_probability=0.0, floor_spawn_prob=1.0
)
cfg.auto_reset = False
cfg.terminations.pop("reached_top", None)
cfg.episode_length_s = a.seconds + 1
configure_video_cfg(cfg)
cfg.viewer.distance = a.distance
cfg.viewer.elevation = a.elevation
cfg.viewer.azimuth = a.azimuth

raw = ManagerBasedRlEnv(cfg, device="cuda:0", render_mode="rgb_array")
fix_render_shadows(raw, light_dir=(-0.5, 0.2, -1.0))
if a.style == "hf":
    style_hf(raw)
steps = int(round(a.seconds / raw.step_dt))
env = VideoRecorder(raw, video_folder=out, step_trigger=lambda s: s == 0, video_length=steps, disable_logger=True)
agent = asdict(DeskRecoveryRlCfg)
agent.update(logger="tensorboard", upload_model=False, seed=a.seed)
env = RslRlVecEnvWrapper(env, clip_actions=agent.get("clip_actions"))
runner = MicroduckOnPolicyRunner(env, agent, str(out), "cuda:0")
runner.load(a.source, map_location="cuda:0")
policy = runner.get_inference_policy(device="cuda:0")

import onnxruntime as ort

session = ort.InferenceSession(
    str(Path(__file__).parent / "models" / os.environ["GETUP_FILE"]),
    providers=["CPUExecutionProvider"],
)
robot = raw.scene["robot"]
servo_ids = mdp._servo_joint_ids(raw, robot)
head_ids, _ = robot.find_joints("^(neck_pitch|head_pitch|head_yaw|head_roll)$")
head_actions = [i for i, j in enumerate(servo_ids) if j in head_ids]
alpha = torch.full((14,), 0.7, device=raw.device)
alpha[head_actions] = 0.5
base_gains = [v.kp_scale.clone() for v in robot.actuators]

raw.reset(seed=a.seed)
obs = env.get_observations()
assert obs["actor"].shape == (1, 61) and torch.count_nonzero(obs["actor"][:, 48:]) == 0

import mujoco

model = raw.sim.mj_model
geom_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(model.ngeom)]
desktop = geom_names.index("tread_31/desktop")
foot_geoms = [geom_names.index("robot/left_foot_collision"), geom_names.index("robot/right_foot_collision")]
desk = next(r for r in __import__("mjlab_microduck.robot.floor_desk", fromlist=["RECORDS"]).RECORDS if r["name"] == "desktop")

switched = torch.zeros(1, dtype=torch.bool, device=raw.device)
previous_raw = torch.zeros((1, 14), device=raw.device)
previous_executed = previous_raw.clone()
switch_step = None

for step in range(steps):
    obs["actor"][:, 34:48] = previous_raw
    with torch.inference_mode():
        act = policy(obs)
    act = act.clone()

    c = raw.sim.data.contact
    n = int(raw.sim.data.nacon[0])
    g = c.geom[:n]
    feet_on_desk = torch.zeros(1, 2, dtype=torch.bool, device=raw.device)
    for side, fg in enumerate(foot_geoms):
        hit = (c.dist[:n] <= 0) & (g == desktop).any(1) & (g == fg).any(1)
        feet_on_desk[c.worldid[:n][hit], side] = True
    root_local = (robot.data.root_link_pos_w - raw.scene.env_origins).clone()
    st = mdp._stair_state(raw)
    root_local[:, 0] -= st.x0
    root_local[:, 1] -= st.y0
    margin = float(os.environ.get("SWITCH_MARGIN", ".04"))
    eligible = (
        feet_on_desk.any(dim=1)
        & (root_local[:, 0] > desk["pos"][0] - desk["size"][0] + margin)
        & (root_local[:, 0] < desk["pos"][0] + desk["size"][0] - 0.05)
        & (root_local[:, 1].abs() < desk["size"][1] - 0.05)
        & (root_local[:, 2] > 0.66)
    )
    spin = robot.data.root_link_ang_vel_w.norm(dim=1)
    eligible &= spin < float(os.environ["SWITCH_MAX_SPIN"])
    arriving = ~switched & eligible
    if bool(arriving[0]) and switch_step is None:
        switch_step = step
        print(f"[render] switch at {step * raw.step_dt:.2f}s (spin {float(spin[0]):.2f})")
    switched |= arriving
    if bool(switched[0]):
        x = obs["actor"].cpu().numpy().astype(np.float32)
        y = session.run(None, {session.get_inputs()[0].name: x})[0]
        act = torch.tensor(y, device=raw.device)
    previous_raw = act.clone()
    for v, gr in zip(robot.actuators, base_gains):
        v.kp_scale.copy_(torch.where(switched[:, None], gr * 0.8, gr))
    act = torch.where(switched[:, None], alpha * act + (1 - alpha) * previous_executed, act)
    previous_executed = act.clone()
    raw._manual_reset_pending.zero_()
    obs, _, done, _ = env.step(act)
    assert torch.isfinite(act).all()

print(f"[render] done. switched={bool(switched[0])} switch_step={switch_step}")
raw.close()
