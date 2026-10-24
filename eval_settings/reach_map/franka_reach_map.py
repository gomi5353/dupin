"""Measure where the Franka can reach a top-down grasp on each task's table.

Runs inside the Isaac Lab container, one task per run:

    python eval_settings/reach_map/franka_reach_map.py --task cup_stacking --headless
    python eval_settings/reach_map/franka_reach_map.py --task cutlery_arrangement --headless
    python eval_settings/reach_map/franka_reach_map.py --task toy_blocks_collection --headless

Method (same as obj_cvg_exps/reach_map on test/obj-cvg-influence, generalized)
------------------------------------------------------------------------------
The robot config (USD, base pose, home joints, joint limits) and the object spawn
ranges are taken from the task's public eval file (``eval/<task>_eval.py``); the
table top comes from TABLES below (measured with duck/tools/measure_tables.py).
One robot per target is cloned into a robot-only scene (no table, no objects),
and all targets are solved in parallel:

* targets: a grid over the table top, fingertip point (``panda_hand`` + 0.1034 m,
  same offset as the task's ``ee_frame``) at ``table_z + h`` for each height ``h``;
* orientation: gripper pointing straight down; yaw follows the direction from
  the robot base to the target (yaw is treated as free);
* solver: damped-least-squares differential IK from the task's home joints.
  Joint positions are written directly each iteration (kinematic IK, no PD
  tracking lag) and clamped to the soft joint limits;
* reachable: final position error < ``--pos_tol`` and orientation error < ``--rot_tol``.

Each solved configuration is also checked against the table: a point is flagged
``collides`` when any arm link origin (panda_link1..7, hand) over the table
footprint ends up below the table top. Other scene geometry (cup stacking
backsplash and wall cabinets) is only drawn as lines on the plot.

Outputs (``--out_dir``): ``reach_<task>.npz``, ``reach_<task>.json``, ``reach_<task>.png``.
"""

import argparse
import json
import math
import os
from pathlib import Path

from isaaclab.app import AppLauncher

# Table tops in the env-local frame (env 0 origin = world origin), from duck/tools/measure_tables.py (2026-10-20).
TABLES = {
    "cup_stacking": {"shape": "rect", "x": (8.07, 10.59), "y": (6.17, 6.85), "z_top": 0.875},  # y capped at backsplash
    "cutlery_arrangement": {"shape": "rect", "x": (6.47, 7.57), "y": (2.48, 4.48), "z_top": 0.709},
    "toy_blocks_collection": {"shape": "circle", "center": (3.30, 1.10), "radius": 0.61, "z_top": 0.364},
}
# Extra geometry drawn as lines (cup stacking kitchen, measured from the scene USD).
LINES = {
    "cup_stacking": [
        {"y": 6.849, "label": "backsplash", "lw": 2, "ls": "-"},
        {"y": 6.496, "label": "wall cabinets above z=1.54", "lw": 1, "ls": ":"},
    ],
}
TCP_OFFSET = 0.1034  # panda_hand -> fingertip midpoint, same as ee_frame "end_effector"

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", required=True, choices=sorted(TABLES))
parser.add_argument("--step", type=float, default=0.02, help="Grid spacing on the table (m).")
parser.add_argument(
    "--heights",
    type=float,
    nargs="+",
    default=[0.05, 0.20],
    help="Fingertip heights above the table top (m). First = grasp height, others = carry/place heights.",
)
parser.add_argument("--iters", type=int, default=300, help="IK iterations.")
parser.add_argument("--pos_tol", type=float, default=0.01, help="Position tolerance (m).")
parser.add_argument("--rot_tol", type=float, default=5.0, help="Orientation tolerance (deg).")
parser.add_argument("--out_dir", type=str, default="eval_settings/reach_map/out")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.assets import ArticulationCfg  # noqa: E402
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import simulator.tasks  # noqa: E402,F401
from simulator.tasks.external import resolve_task  # noqa: E402

TASK = args.task
TABLE = TABLES[TASK]


def table_bounds():
    if TABLE["shape"] == "rect":
        return TABLE["x"], TABLE["y"]
    (cx, cy), r = TABLE["center"], TABLE["radius"]
    return (cx - r, cx + r), (cy - r, cy + r)


def on_table(x, y):
    if TABLE["shape"] == "rect":
        return (x >= TABLE["x"][0]) & (x <= TABLE["x"][1]) & (y >= TABLE["y"][0]) & (y <= TABLE["y"][1])
    (cx, cy), r = TABLE["center"], TABLE["radius"]
    return (x - cx) ** 2 + (y - cy) ** 2 <= r**2


def table_area():
    if TABLE["shape"] == "rect":
        return (TABLE["x"][1] - TABLE["x"][0]) * (TABLE["y"][1] - TABLE["y"][0])
    return math.pi * TABLE["radius"] ** 2


def build_robot_cfg():
    """Robot cfg and per-object spawn boxes (init pos + pose range) from the public eval file."""
    task_cfg = parse_env_cfg(resolve_task(f"eval/{TASK}_eval.py"), device="cpu", num_envs=1)
    robot = task_cfg.scene.robot.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.rigid_props.disable_gravity = True
    ranges = {}
    for term in vars(task_cfg.events).values():
        params = getattr(term, "params", None)
        if not isinstance(params, dict):
            continue
        if "pose_ranges" in params:
            ranges.update(params["pose_ranges"])
        elif "pose_range" in params and getattr(params.get("asset_cfg"), "name", None):
            ranges[params["asset_cfg"].name] = params["pose_range"]
    spawn = {}
    for name in task_cfg.tracked_object_names:
        pos = getattr(task_cfg.scene, name).init_state.pos
        pr = ranges.get(name, {})
        spawn[name] = {
            "x": (pos[0] + pr.get("x", (0.0, 0.0))[0], pos[0] + pr.get("x", (0.0, 0.0))[1]),
            "y": (pos[1] + pr.get("y", (0.0, 0.0))[0], pos[1] + pr.get("y", (0.0, 0.0))[1]),
        }
    return robot, spawn


def make_targets(step):
    (x0, x1), (y0, y1) = table_bounds()
    xs = np.arange(x0 + step / 2, x1, step)
    ys = np.arange(y0 + step / 2, y1, step)
    X, Y = np.meshgrid(xs, ys)  # (ny, nx)
    return xs, ys, X, Y


def solve(scene, sim, X, Y, height):
    robot = scene["robot"]
    n = X.size
    device = sim.device
    origins = scene.env_origins
    arm_ids, _ = robot.find_joints("panda_joint.*")
    hand_idx = robot.find_bodies("panda_hand")[0][0]
    link_ids, _ = robot.find_bodies(["panda_link[1-7]", "panda_hand"])

    # reset to home
    q_home = robot.data.default_joint_pos.clone()
    robot.write_joint_state_to_sim(q_home, torch.zeros_like(q_home))
    robot.set_joint_position_target(q_home)
    scene.write_data_to_sim()
    sim.step(render=False)
    scene.update(sim.get_physics_dt())

    base_pos = robot.data.root_pos_w - origins  # (n,3) env-local
    base_yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)[2]
    home_quat = robot.data.body_quat_w[:, hand_idx].clone()

    # targets in env-local frame -> world; the home gripper faces along the base's forward axis
    tcp = torch.tensor(np.stack([X.ravel(), Y.ravel(), np.full(n, TABLE["z_top"] + height)], -1),
                       dtype=torch.float32, device=device)
    yaw = torch.atan2(tcp[:, 1] - base_pos[:, 1], tcp[:, 0] - base_pos[:, 0]) - base_yaw
    q_yaw = math_utils.quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    target_quat_w = math_utils.quat_mul(q_yaw, home_quat)
    tcp_axis = math_utils.quat_apply(target_quat_w, torch.tensor([0.0, 0.0, TCP_OFFSET], device=device).expand(n, 3))
    target_hand_w = tcp + origins - tcp_axis

    ik = DifferentialIKController(
        DifferentialIKControllerCfg(
            command_type="pose", ik_method="dls", use_relative_mode=False, ik_params={"lambda_val": 0.05}
        ),
        num_envs=n,
        device=device,
    )
    lo = robot.data.soft_joint_pos_limits[:, arm_ids, 0]
    hi = robot.data.soft_joint_pos_limits[:, arm_ids, 1]
    q = q_home.clone()

    def hand_pose_b():
        d = robot.data
        return math_utils.subtract_frame_transforms(
            d.root_pos_w, d.root_quat_w, d.body_pos_w[:, hand_idx], d.body_quat_w[:, hand_idx]
        )

    tgt_pos_b, tgt_quat_b = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, target_hand_w, target_quat_w
    )
    ik.set_command(torch.cat([tgt_pos_b, tgt_quat_b], -1))
    base_rot = math_utils.matrix_from_quat(math_utils.quat_inv(robot.data.root_quat_w))

    for _ in range(args.iters):
        hp, hq = hand_pose_b()
        jac_w = robot.root_physx_view.get_jacobians()[:, hand_idx - 1, :, arm_ids]
        jac_b = torch.cat([base_rot @ jac_w[:, :3], base_rot @ jac_w[:, 3:]], 1)
        q_arm = ik.compute(hp, hq, jac_b, q[:, arm_ids])
        q[:, arm_ids] = torch.clamp(q_arm, lo, hi)
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        robot.set_joint_position_target(q)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(sim.get_physics_dt())

    hp, hq = hand_pose_b()
    pos_err = torch.linalg.norm(hp - tgt_pos_b, dim=-1)
    rot_err = torch.rad2deg(math_utils.quat_error_magnitude(hq, tgt_quat_b))
    ok = (pos_err < args.pos_tol) & (rot_err < args.rot_tol)

    links = (robot.data.body_pos_w[:, link_ids] - origins[:, None]).cpu().numpy()  # (n, L, 3)
    over_table = on_table(links[..., 0], links[..., 1])
    collides = (over_table & (links[..., 2] < TABLE["z_top"])).any(-1)

    shape = X.shape
    return {
        "reachable": ok.cpu().numpy().reshape(shape),
        "collides": collides.reshape(shape),
        "pos_err": pos_err.cpu().numpy().reshape(shape),
        "rot_err": rot_err.cpu().numpy().reshape(shape),
        "base_xy": base_pos[0, :2].cpu().numpy(),
    }


def box_area(b):
    return (b["x"][1] - b["x"][0]) * (b["y"][1] - b["y"][0])


def plot(path, xs, ys, results, spawn, base_xy, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Patch, Rectangle

    grasp_h = args.heights[0]
    X, Y = np.meshgrid(xs, ys)
    inside = on_table(X, Y)
    r = results[grasp_h]
    img = np.full(X.shape, np.nan)
    img[inside] = 0
    img[inside & r["reachable"]] = 2
    img[inside & r["reachable"] & r["collides"]] = 1
    step = xs[1] - xs[0]
    extent = [xs[0] - step / 2, xs[-1] + step / 2, ys[0] - step / 2, ys[-1] + step / 2]

    (tx0, tx1), (ty0, ty1) = table_bounds()
    width, height = tx1 - tx0 + 0.3, max(ty1, base_xy[1]) - min(ty0, base_xy[1]) + 0.3
    fig, ax = plt.subplots(figsize=(min(14, max(7, 5 * width / height + 2)), 7))
    ax.imshow(img, origin="lower", extent=extent, cmap=ListedColormap(["#e9e9e9", "#f2b27a", "#6fb37b"]),
              vmin=0, vmax=2, interpolation="nearest")
    for h in args.heights[1:]:
        good = (results[h]["reachable"] & ~results[h]["collides"] & inside).astype(float)
        ax.contour(X, Y, good, levels=[0.5], colors="#2b6cb0", linestyles="--", linewidths=1.5)

    if TABLE["shape"] == "rect":
        ax.add_patch(Rectangle((tx0, ty0), tx1 - tx0, ty1 - ty0, fill=False, ec="black", lw=1.5))
    else:
        ax.add_patch(Circle(TABLE["center"], TABLE["radius"], fill=False, ec="black", lw=1.5))
    for line in LINES.get(TASK, []):
        ax.axhline(line["y"], color="#7a4b2a", lw=line["lw"], ls=line["ls"])
        ax.text(tx1 - 0.02, line["y"] + 0.01, line["label"], ha="right", va="bottom", fontsize=8, color="#7a4b2a")

    palette = ["#1f4fd1", "#d63384", "#e8a20c", "#6f42c1"]
    for k, (name, b) in enumerate(spawn.items()):
        c = palette[k % len(palette)]
        w, hgt = max(b["x"][1] - b["x"][0], 0.01), max(b["y"][1] - b["y"][0], 0.01)
        ax.add_patch(Rectangle((b["x"][0], b["y"][0]), w, hgt, fill=True, alpha=0.55, fc=c, ec=c, lw=1.5))
        ax.annotate(f"{name} spawn\n{summary['spawn'][name]['share_of_reachable_pct']:.2f}% of reachable",
                    (b["x"][1], b["y"][1]), xytext=(40, 60 - 34 * k), textcoords="offset points", fontsize=8,
                    color=c, arrowprops={"arrowstyle": "-", "color": c, "lw": 0.8})

    ax.plot(*base_xy, marker="s", ms=10, color="black")
    ax.annotate("Franka base", base_xy, xytext=(8, -4), textcoords="offset points", fontsize=8)

    g = summary["heights"][f"{grasp_h:.2f}"]
    handles = [
        Patch(fc="#6fb37b", label=f"reachable, top-down @ {grasp_h*100:.0f} cm ({g['area_m2']:.2f} m², "
              f"{g['share_of_table_pct']:.0f}% of table)"),
        Patch(fc="#f2b27a", label="IK solves but arm link below table top"),
        Patch(fc="#e9e9e9", label="not reachable"),
    ] + [
        Line2D([], [], color="#2b6cb0", ls="--", label=f"reachable @ {h*100:.0f} cm "
               f"({summary['heights'][f'{h:.2f}']['area_m2']:.2f} m²)") for h in args.heights[1:]
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8, frameon=False)
    ax.set_xlim(tx0 - 0.15, tx1 + 0.15)
    ax.set_ylim(min(ty0, base_xy[1]) - 0.15, max(ty1, base_xy[1]) + 0.15)
    ax.set_aspect("equal")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_title(f"Franka top-down reach on the {TASK.replace('_', ' ')} table\n(grid {args.step*100:.0f} cm, "
                 f"tol {args.pos_tol*100:.0f} cm / {args.rot_tol:.0f}°)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def main():
    robot_cfg, spawn = build_robot_cfg()
    xs, ys, X, Y = make_targets(args.step)
    n = X.size

    @configclass
    class ReachSceneCfg(InteractiveSceneCfg):
        robot: ArticulationCfg = robot_cfg

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 60, device=args.device))
    scene = InteractiveScene(ReachSceneCfg(num_envs=n, env_spacing=3.0, replicate_physics=True))
    sim.reset()
    print(f"[reach] {TASK}: {n} targets ({len(xs)} x {len(ys)}), heights {args.heights}", flush=True)

    results = {}
    with torch.inference_mode():
        for h in args.heights:
            results[h] = solve(scene, sim, X, Y, h)
            print(f"[reach] h={h:.2f}: {results[h]['reachable'][on_table(X, Y)].mean()*100:.1f}% of table reachable",
                  flush=True)

    cell = args.step**2
    inside = on_table(X, Y)
    summary = {"task": TASK, "grid_step_m": args.step, "pos_tol_m": args.pos_tol, "rot_tol_deg": args.rot_tol,
               "table": TABLE, "table_area_m2": table_area(), "heights": {}, "spawn": {}}
    base_xy = results[args.heights[0]]["base_xy"]
    summary["robot_base_xy"] = base_xy.tolist()
    for h, r in results.items():
        good = r["reachable"] & ~r["collides"] & inside
        summary["heights"][f"{h:.2f}"] = {
            "area_m2": float(good.sum() * cell),
            "share_of_table_pct": float(good.sum() * cell / table_area() * 100),
            "x_range": [float(X[good].min()), float(X[good].max())] if good.any() else None,
            "y_range": [float(Y[good].min()), float(Y[good].max())] if good.any() else None,
            "ik_only_area_m2": float((r["reachable"] & inside).sum() * cell),
        }
    reach_area = summary["heights"][f"{args.heights[0]:.2f}"]["area_m2"]
    for name, b in spawn.items():
        summary["spawn"][name] = {"x": list(b["x"]), "y": list(b["y"]), "area_m2": box_area(b),
                                  "share_of_reachable_pct": box_area(b) / reach_area * 100 if reach_area else None}

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"reach_{TASK}.npz", xs=xs, ys=ys, base_xy=base_xy, inside=inside,
             **{f"{k}_h{h:.2f}": v for h, r in results.items() for k, v in r.items() if k != "base_xy"})
    (out / f"reach_{TASK}.json").write_text(json.dumps(summary, indent=2))
    plot(out / f"reach_{TASK}.png", xs, ys, results, spawn, base_xy, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
    # SimulationApp.close() hangs in this container after a headless physics-only run;
    # all outputs are written by now, so exit hard.
    os._exit(0)
