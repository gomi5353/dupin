"""Measure where the Franka can reach a top-down grasp on the cup-stacking counter.

Runs inside the Isaac Lab container:

    python duck/reach_map/franka_reach_map.py --headless

Method
------
The robot config (USD, base pose, home joints, joint limits) is taken from
``CupStackingEnvCfg``. One robot per target is cloned into a robot-only scene
(no kitchen, no cups), and all targets are solved in parallel:

* targets: a grid over the counter top, fingertip point (``panda_hand`` + 0.1034 m,
  same offset as the task's ``ee_frame``) at ``table_z + h`` for each height ``h``;
* orientation: gripper pointing straight down; yaw follows the direction from
  the robot base to the target (cups are rotationally symmetric, so yaw is free);
* solver: damped-least-squares differential IK from the task's home joints.
  Joint positions are written directly each iteration (kinematic IK, no PD
  tracking lag) and clamped to the soft joint limits;
* reachable: final position error < ``--pos_tol`` and orientation error < ``--rot_tol``.

Each solved configuration is also checked against the counter: a point is
flagged ``collides`` when any arm link origin (panda_link1..7, hand) over the
counter footprint ends up below the counter top. Collision against the backsplash
and wall cabinets is only reported by geometry lines on the plot.

Outputs (``--out_dir``): ``reach_<task>.npz``, ``reach_<task>.json``, ``reach_<task>.png``.
"""

import argparse
import json
import math
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--step", type=float, default=0.02, help="Grid spacing on the counter (m).")
parser.add_argument(
    "--heights",
    type=float,
    nargs="+",
    default=[0.05, 0.20],
    help="Fingertip heights above the counter top (m). First = grasp height, others = carry/place heights.",
)
parser.add_argument("--iters", type=int, default=300, help="IK iterations.")
parser.add_argument("--pos_tol", type=float, default=0.01, help="Position tolerance (m).")
parser.add_argument("--rot_tol", type=float, default=5.0, help="Orientation tolerance (deg).")
parser.add_argument("--out_dir", type=str, default="duck/reach_map/out")
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

from simulator.tasks.cup_stacking.cup_stacking_env_cfg import CupStackingEnvCfg  # noqa: E402

TASK = "cup_stacking"
TCP_OFFSET = 0.1034  # panda_hand -> fingertip midpoint, same as ee_frame "end_effector"

# Kitchen geometry measured from the scene USD (world frame, env 0 origin = world origin).
COUNTER = {"x": (8.070, 10.598), "y": (6.166, 6.894), "z_top": 0.875}
BACKSPLASH_Y = 6.849  # front face of the backsplash, z 0.875..1.555
CABINET = {"y_front": 6.496, "z_bottom": 1.545}  # upper wall cabinets
TABLE = {"x": (8.09, 10.59), "y": (6.17, 6.85)}  # usable table range (user-given)


def build_robot_cfg():
    task_cfg = CupStackingEnvCfg()
    robot = task_cfg.scene.robot.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.rigid_props.disable_gravity = True
    spawn_ranges = {}
    for name in ("blue_cup", "pink_cup"):
        pos = getattr(task_cfg.scene, name).init_state.pos
        for term_name in dir(task_cfg.events):
            term = getattr(task_cfg.events, term_name)
            params = getattr(term, "params", None)
            if isinstance(params, dict) and getattr(params.get("asset_cfg"), "name", None) == name:
                pr = params["pose_range"]
                spawn_ranges[name] = {
                    "x": (pos[0] + pr["x"][0], pos[0] + pr["x"][1]),
                    "y": (pos[1] + pr["y"][0], pos[1] + pr["y"][1]),
                }
    return robot, spawn_ranges


def make_targets(step):
    xs = np.arange(TABLE["x"][0] + step / 2, TABLE["x"][1], step)
    ys = np.arange(TABLE["y"][0] + step / 2, TABLE["y"][1], step)
    X, Y = np.meshgrid(xs, ys)  # (ny, nx)
    return xs, ys, X, Y


def solve(scene, sim, robot_cfg, X, Y, height):
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
    home_quat = robot.data.body_quat_w[:, hand_idx].clone()

    # targets in env-local frame -> world
    tcp = torch.tensor(np.stack([X.ravel(), Y.ravel(), np.full(n, COUNTER["z_top"] + height)], -1),
                       dtype=torch.float32, device=device)
    yaw = torch.atan2(tcp[:, 1] - base_pos[:, 1], tcp[:, 0] - base_pos[:, 0]) - math.pi / 2  # home faces +y
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

    links = robot.data.body_pos_w[:, link_ids] - origins[:, None]  # (n, L, 3)
    over_counter = (
        (links[..., 0] > COUNTER["x"][0]) & (links[..., 0] < COUNTER["x"][1])
        & (links[..., 1] > COUNTER["y"][0]) & (links[..., 1] < COUNTER["y"][1])
    )
    collides = (over_counter & (links[..., 2] < COUNTER["z_top"])).any(-1)

    shape = X.shape
    return {
        "reachable": ok.cpu().numpy().reshape(shape),
        "collides": collides.cpu().numpy().reshape(shape),
        "pos_err": pos_err.cpu().numpy().reshape(shape),
        "rot_err": rot_err.cpu().numpy().reshape(shape),
        "base_xy": base_pos[0, :2].cpu().numpy(),
    }


def box_area(b):
    return (b["x"][1] - b["x"][0]) * (b["y"][1] - b["y"][0])


def plot(path, xs, ys, results, spawn_ranges, base_xy, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle

    step = xs[1] - xs[0]
    extent = [xs[0] - step / 2, xs[-1] + step / 2, ys[0] - step / 2, ys[-1] + step / 2]
    grasp_h = args.heights[0]
    r = results[grasp_h]
    cls = np.where(r["reachable"] & ~r["collides"], 2, np.where(r["reachable"], 1, 0))

    fig, ax = plt.subplots(figsize=(13, 5.2))
    cmap = ListedColormap(["#e9e9e9", "#f2b27a", "#6fb37b"])
    ax.imshow(cls, origin="lower", extent=extent, cmap=cmap, vmin=0, vmax=2, interpolation="nearest", aspect="equal")

    for h in args.heights[1:]:
        m = results[h]["reachable"] & ~results[h]["collides"]
        ax.contour(xs, ys, m.astype(float), levels=[0.5], colors="#2b6cb0", linewidths=1.5, linestyles="--")

    ax.add_patch(Rectangle((TABLE["x"][0], TABLE["y"][0]), TABLE["x"][1] - TABLE["x"][0],
                           TABLE["y"][1] - TABLE["y"][0], fill=False, ec="black", lw=1.2))
    ax.axhline(BACKSPLASH_Y, color="#7a4b2a", lw=2)
    ax.text(TABLE["x"][1] - 0.02, BACKSPLASH_Y + 0.01, "backsplash", ha="right", va="bottom", fontsize=8, color="#7a4b2a")
    ax.axhline(CABINET["y_front"], color="#7a4b2a", lw=1, ls=":")
    ax.text(TABLE["x"][1] - 0.02, CABINET["y_front"] + 0.01, f"wall cabinets above z={CABINET['z_bottom']:.2f}",
            ha="right", va="bottom", fontsize=8, color="#7a4b2a")

    colors = {"blue_cup": "#1f4fd1", "pink_cup": "#d63384"}
    for name, b in spawn_ranges.items():
        ax.add_patch(Rectangle((b["x"][0], b["y"][0]), b["x"][1] - b["x"][0], b["y"][1] - b["y"][0],
                               fill=True, alpha=0.55, fc=colors[name], ec=colors[name], lw=1.5))
        ax.annotate(f"{name} spawn\n{summary['spawn'][name]['share_of_reachable_pct']:.2f}% of reachable",
                    (b["x"][1], b["y"][1]), xytext=(18, 18 if name == "blue_cup" else -34),
                    textcoords="offset points", fontsize=8, color=colors[name],
                    arrowprops={"arrowstyle": "-", "color": colors[name], "lw": 0.8})

    ax.plot(*base_xy, marker="s", ms=10, color="black")
    ax.annotate("Franka base", base_xy, xytext=(8, -4), textcoords="offset points", fontsize=8)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    g = summary["heights"][f"{grasp_h:.2f}"]
    handles = [
        Patch(fc="#6fb37b", label=f"reachable, top-down @ {grasp_h*100:.0f} cm ({g['area_m2']:.2f} m², {g['share_of_table_pct']:.0f}% of table)"),
        Patch(fc="#f2b27a", label="IK solves but arm link below counter top"),
        Patch(fc="#e9e9e9", label="not reachable"),
    ] + [
        Line2D([], [], color="#2b6cb0", ls="--", label=f"reachable @ {h*100:.0f} cm "
               f"({summary['heights'][f'{h:.2f}']['area_m2']:.2f} m²)") for h in args.heights[1:]
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8, frameon=False)
    ax.set_xlim(TABLE["x"][0] - 0.1, TABLE["x"][1] + 0.1)
    ax.set_ylim(base_xy[1] - 0.1, COUNTER["y"][1] + 0.05)
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_title(f"Franka top-down reach on the cup-stacking counter (grid {step*100:.0f} cm, "
                 f"tol {args.pos_tol*100:.0f} cm / {args.rot_tol:.0f}°)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def main():
    robot_cfg, spawn_ranges = build_robot_cfg()
    xs, ys, X, Y = make_targets(args.step)
    n = X.size

    @configclass
    class ReachSceneCfg(InteractiveSceneCfg):
        robot: ArticulationCfg = robot_cfg

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 60, device=args.device))
    scene = InteractiveScene(ReachSceneCfg(num_envs=n, env_spacing=3.0, replicate_physics=True))
    sim.reset()
    print(f"[reach] {n} targets ({len(xs)} x {len(ys)}), heights {args.heights}", flush=True)

    results = {}
    with torch.inference_mode():
        for h in args.heights:
            results[h] = solve(scene, sim, robot_cfg, X, Y, h)
            print(f"[reach] h={h:.2f}: {results[h]['reachable'].mean()*100:.1f}% of grid reachable", flush=True)

    cell = args.step**2
    table_area = box_area(TABLE)
    summary = {"task": TASK, "grid_step_m": args.step, "pos_tol_m": args.pos_tol, "rot_tol_deg": args.rot_tol,
               "counter_top_z": COUNTER["z_top"], "table": TABLE, "table_area_m2": table_area, "heights": {}, "spawn": {}}
    base_xy = results[args.heights[0]]["base_xy"]
    for h, r in results.items():
        good = r["reachable"] & ~r["collides"]
        summary["heights"][f"{h:.2f}"] = {
            "area_m2": float(good.sum() * cell),
            "share_of_table_pct": float(good.sum() * cell / table_area * 100),
            "x_range": [float(X[good].min()), float(X[good].max())] if good.any() else None,
            "y_range": [float(Y[good].min()), float(Y[good].max())] if good.any() else None,
            "ik_only_area_m2": float(r["reachable"].sum() * cell),
        }
    reach_area = summary["heights"][f"{args.heights[0]:.2f}"]["area_m2"]
    for name, b in spawn_ranges.items():
        summary["spawn"][name] = {"x": list(b["x"]), "y": list(b["y"]), "area_m2": box_area(b),
                                  "share_of_reachable_pct": box_area(b) / reach_area * 100}

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"reach_{TASK}.npz", xs=xs, ys=ys, base_xy=base_xy,
             **{f"{k}_h{h:.2f}": v for h, r in results.items() for k, v in r.items() if k != "base_xy"})
    (out / f"reach_{TASK}.json").write_text(json.dumps(summary, indent=2))
    plot(out / f"reach_{TASK}.png", xs, ys, results, spawn_ranges, base_xy, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
    # SimulationApp.close() hangs in this container after a headless physics-only run;
    # all outputs are written by now, so exit hard.
    os._exit(0)
