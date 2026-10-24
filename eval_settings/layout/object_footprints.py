"""Measure each tracked object's top-down footprint at its nominal (unrandomized) pose.

For every tracked object: root xyz (the point the success checks use), the world
bounding box, and the xy corners of its oriented bounding box. Written to
out/footprints_<task>.json for plot_layout.py.

Run inside the isaaclab container, from /workspace/physicalai:

    python eval_settings/layout/object_footprints.py --headless --enable_cameras
"""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--tasks", nargs="+", default=["cutlery_arrangement", "toy_blocks_collection"])
parser.add_argument("--out_dir", default="eval_settings/layout/out")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(vars(args)).app

import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import omni.usd
from pxr import Usd, UsdGeom
from isaaclab_tasks.utils import parse_env_cfg
import simulator.tasks  # noqa: F401
from simulator.tasks.external import resolve_task

out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
for task in args.tasks:
    task_id = resolve_task(f"eval/{task}_eval.py")
    cfg = parse_env_cfg(task_id, device=args.device, num_envs=1)
    cfg.use_joint_position_actions()
    cfg.recorders = None
    cfg.terminations.time_out = None
    for name, term in list(vars(cfg.events).items()):
        if term is not None and getattr(term, "mode", None) == "reset" and name != "reset_all":
            setattr(cfg.events, name, None)  # nominal poses
    env = gym.make(task_id, cfg=cfg).unwrapped
    env.reset()
    env.sim.step(render=True)
    origin = env.scene.env_origins[0].cpu().numpy()
    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    objects = {}
    for n in env.cfg.tracked_object_names:
        obj = env.scene[n]
        root = obj.data.root_pos_w[0].cpu().numpy() - origin
        quat = obj.data.root_quat_w[0].cpu().numpy()
        prim = stage.GetPrimAtPath(f"/World/envs/env_0/Scene/{n}")
        bound = cache.ComputeWorldBound(prim)
        box = bound.GetBox()
        m = np.array(bound.GetMatrix())  # row-vector convention: p_world = p_local @ m
        lo, hi = np.array(box.GetMin()), np.array(box.GetMax())
        corners = np.array([[x, y, z, 1.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        world = (corners @ m)[:, :3] - origin
        aligned = bound.ComputeAlignedRange()
        objects[n] = {
            "root_xyz": root.round(4).tolist(),
            "root_quat_wxyz": quat.round(5).tolist(),
            "aabb_min": (np.array(aligned.GetMin()) - origin).round(4).tolist(),
            "aabb_max": (np.array(aligned.GetMax()) - origin).round(4).tolist(),
            "obb_corners_xyz": world.round(4).tolist(),
        }
        print(f"[fp] {task} {n:<12} root {root.round(3).tolist()} aabb {objects[n]['aabb_min']} .. {objects[n]['aabb_max']}",
              flush=True)
    (out / f"footprints_{task}.json").write_text(json.dumps({"task": task, "objects": objects}, indent=2))
    env.close()

os._exit(0)
