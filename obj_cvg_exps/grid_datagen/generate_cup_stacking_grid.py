"""Grid-sweep cup-stacking data generation with guaranteed per-cell coverage.

Both cups share one region (centre ``--center``, half-extents ``--half_x`` /
``--half_y``) split into ``--grid_x`` x ``--grid_y`` cells. Cells are visited
row-major from top-left (max y, min x) to bottom-right. For pairing index
``k = (row, col)``:

* ``blue_cup`` goes to cell ``(row, col)``;
* ``pink_cup`` goes to cell ``(row, (col + grid_x // 2) % grid_x)``, i.e. half a
  region away in x, so the cups never spawn on top of each other.

Both maps are bijections, so one success per pairing covers every cell once for
each cup. Inside a cell the pose is drawn uniformly, inset by ``--cell_margin``
so it never crosses into a neighbouring cell. Each pairing gets up to
``--max_attempts`` tries (fresh pose each try); the whole sweep repeats until
every cell has ``--min_episodes_per_cell`` successes or ran out of attempts.

Only successful episodes are written (``EXPORT_SUCCEEDED_ONLY``). The recorder
adds ``object_pose.<name>`` columns (env-local ``[x, y, z, qw, qx, qy, qz]``) so
the dataset can be checked with the object-pose coverage checker. A matching
coverage spec (``coverage_spec.ttl``) and a per-episode log are written to
``--out_dir``.

Run inside the Isaac Lab container:

    python obj_cvg_exps/grid_datagen/generate_cup_stacking_grid.py \
        --headless --enable_cameras --device cuda \
        --repo_id local/cup_stacking_grid8x4
"""

import argparse
import json
import math
import os
import random
import signal
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Grid-sweep cup-stacking data generation.")
parser.add_argument("--task", type=str, default="HCIS-CupStacking-SingleArm-v0")
parser.add_argument("--repo_id", type=str, default="local/cup_stacking_grid8x4", help="LeRobot repo id.")
parser.add_argument("--fps", type=int, default=30, help="LeRobot dataset fps.")
parser.add_argument("--center", type=float, nargs=2, default=[9.0, 6.5], help="Region centre (x, y), env-local.")
parser.add_argument("--half_x", type=float, default=0.2, help="Region half-extent in x (m).")
parser.add_argument("--half_y", type=float, default=0.1, help="Region half-extent in y (m).")
parser.add_argument(
    "--spawn_z", type=float, default=None,
    help="Spawn z (m). Default: drop once from 1.0, then spawn every cup at its measured rest height + 3 mm.",
)
parser.add_argument("--grid_x", type=int, default=8)
parser.add_argument("--grid_y", type=int, default=4)
parser.add_argument("--min_episodes_per_cell", type=int, default=1)
parser.add_argument("--max_attempts", type=int, default=5, help="Tries per pairing before giving up on it.")
parser.add_argument("--cell_margin", type=float, default=0.005, help="Inset from cell edges when sampling (m).")
parser.add_argument("--settle_steps", type=int, default=60, help="Physics steps to let the cups settle (not recorded).")
parser.add_argument("--max_pairings", type=int, default=None, help="Debug: stop after this many pairings.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--step_hz", type=int, default=0,
    help="Cap env steps per wall-clock second, as in scripts/datagen/generate.py (60 = real time). 0 = no cap.",
)
parser.add_argument("--out_dir", type=str, default="obj_cvg_exps/grid_datagen/out")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import simulator.tasks  # noqa: E402, F401
from isaaclab.managers import DatasetExportMode, TerminationTermCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from leisaac.enhance.datasets.lerobot_dataset_handler import LeRobotDatasetCfg  # noqa: E402
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim  # noqa: E402

from simulator.datagen.state_machine.cup_stacking import CupStackingStateMachine  # noqa: E402
from simulator.utils.object_pose_recording import (  # noqa: E402
    OBJECT_POSE_KEY_PREFIX,
    LeRobotRecorderManager,
    read_object_poses,
)

OBJECTS = ("blue_cup", "pink_cup")
# A cup whose z ends below this has fallen off the counter (counter top z = 0.875).
FALL_Z = 0.80
# A cup tilted more than this after settling has tipped over.
MAX_TILT_DEG = 10.0


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------


class Grid:
    def __init__(self, center, half_x, half_y, nx, ny):
        self.x0, self.x1 = center[0] - half_x, center[0] + half_x
        self.y0, self.y1 = center[1] - half_y, center[1] + half_y
        self.nx, self.ny = nx, ny
        self.dx = (self.x1 - self.x0) / nx
        self.dy = (self.y1 - self.y0) / ny

    def pairings(self):
        """Row-major from top-left: row 0 = max y, col 0 = min x. Yields (blue_cell, pink_cell)."""
        for row in range(self.ny):
            for col in range(self.nx):
                yield (row, col), (row, (col + self.nx // 2) % self.nx)

    def cell_bounds(self, cell):
        row, col = cell
        x0 = self.x0 + col * self.dx
        y1 = self.y1 - row * self.dy
        return x0, x0 + self.dx, y1 - self.dy, y1

    def sample(self, cell, margin, rng):
        x0, x1, y0, y1 = self.cell_bounds(cell)
        return rng.uniform(x0 + margin, x1 - margin), rng.uniform(y0 + margin, y1 - margin)

    def cell_of(self, x, y):
        """Checker-style binning: half-open cells, last row/col closed. None if outside."""
        if not (self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1):
            return None
        col = min(int((x - self.x0) / self.dx), self.nx - 1)
        i_from_bottom = min(int((y - self.y0) / self.dy), self.ny - 1)
        return (self.ny - 1 - i_from_bottom, col)


def write_coverage_spec(path: Path, grid: Grid, min_per_cell: int):
    lines = [
        "# Object pose coverage spec generated by generate_cup_stacking_grid.py.",
        "# Frame: env-local (world minus env origin), metres.",
        "",
        "@prefix dv:   <urn:aicapstone:dv#> .",
        "@prefix spec: <urn:aicapstone:spec:cup_stacking#> .",
        "",
        'spec:task dv:taskName "cup_stacking" .',
    ]
    for name in OBJECTS:
        lines += [
            "",
            f'spec:{name} a dv:TaskObject ; dv:objectName "{name}" .',
            "",
            f"spec:{name}_coverage a dv:CoverageRequirement ;",
            "    dv:targetsAxis dv:ObjectPoseAxis ;",
            f"    dv:forObject spec:{name} ;",
            f"    dv:xMin {grid.x0:.3f} ; dv:xMax {grid.x1:.3f} ;",
            f"    dv:yMin {grid.y0:.3f} ; dv:yMax {grid.y1:.3f} ;",
            f"    dv:gridX {grid.nx} ; dv:gridY {grid.ny} ;",
            f"    dv:minEpisodesPerCell {min_per_cell} .",
        ]
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------


def set_success(env, success: bool):
    fill = torch.ones if success else torch.zeros
    env.termination_manager.set_term_cfg(
        "success", TerminationTermCfg(func=lambda env: fill(env.num_envs, dtype=torch.bool, device=env.device))
    )
    env.termination_manager.compute()


def place_cups(env, xy: dict, z: float):
    """Teleport the cups (upright, zero velocity) and let them settle without recording."""
    for name, (x, y) in xy.items():
        obj = env.scene[name]
        origin = env.scene.env_origins[0]
        pose = torch.tensor([[x, y, z, 1.0, 0.0, 0.0, 0.0]], device=env.device)
        pose[:, :3] += origin
        obj.write_root_pose_to_sim(pose)
        obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))
    # env.sim.step bypasses the managers, so nothing is recorded while settling.
    for _ in range(args.settle_steps):
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)


def stacked(env, sm) -> bool:
    """State-machine xy check plus the env's built-in height check (blue > pink + 10 cm)."""
    if not sm.check_success(env):
        return False
    blue = env.scene["blue_cup"].data.root_pos_w[0]
    pink = env.scene["pink_cup"].data.root_pos_w[0]
    return bool(blue[2] > pink[2] + SUCCESS_HEIGHT)


SUCCESS_HEIGHT = 0.10


# Everything runs under inference_mode: env.step makes the sim buffers inference
# tensors, and a later env.reset() outside it would fail on in-place writes.
@torch.inference_mode()
def main():
    global SUCCESS_HEIGHT
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = Grid(args.center, args.half_x, args.half_y, args.grid_x, args.grid_y)
    write_coverage_spec(out_dir / "coverage_spec.ttl", grid, args.min_episodes_per_cell)

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.use_teleop_device("keyboard")  # 7 absolute joint targets + gripper
    env_cfg.seed = args.seed
    SUCCESS_HEIGHT = env_cfg.terminations.success.params.get("height_threshold", SUCCESS_HEIGHT)
    # Cup poses come from the grid, not from the cfg's reset randomization.
    for term_name in list(vars(env_cfg.events)):
        term = getattr(env_cfg.events, term_name)
        params = getattr(term, "params", None)
        if isinstance(params, dict) and getattr(params.get("asset_cfg"), "name", None) in OBJECTS:
            setattr(env_cfg.events, term_name, None)
            print(f"[grid] disabled reset randomization term '{term_name}'")
    env_cfg.terminations.time_out = None
    env_cfg.terminations.success = TerminationTermCfg(
        func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    )
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    env = gym.make(args.task, cfg=env_cfg).unwrapped

    # Same as scripts/datagen/generate.py: robot links ignore gravity.
    import omni.usd
    from pxr import PhysxSchema, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    for prim in stage.Traverse():
        if "Robot" in str(prim.GetPath()) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True)

    del env.recorder_manager
    # Writes object_pose.<name> for env_cfg.tracked_object_names (blue_cup, pink_cup).
    env.recorder_manager = LeRobotRecorderManager(
        env_cfg.recorders, LeRobotDatasetCfg(repo_id=args.repo_id, fps=args.fps), env
    )

    sm = CupStackingStateMachine()
    sm.setup(env)

    counts = {name: [[0] * grid.nx for _ in range(grid.ny)] for name in OBJECTS}
    log = []
    interrupted = False

    def on_sigint(signum, frame):
        nonlocal interrupted
        interrupted = True
        print("\n[grid] Ctrl+C: finishing current attempt and finalizing.")

    signal.signal(signal.SIGINT, on_sigint)
    t_start = time.time()
    spawn_z = 1.0 if args.spawn_z is None else args.spawn_z
    env.reset()
    set_success(env, False)

    next_tick = time.time()

    def rate_limit(env):
        """Same pacing as RateLimiter in scripts/datagen/generate.py: render while waiting."""
        nonlocal next_tick
        next_tick += 1.0 / args.step_hz
        while time.time() < next_tick:
            time.sleep(min(0.0166, 1.0 / args.step_hz))
            env.sim.render()
        next_tick = max(next_tick, time.time())

    def run_attempt(blue_cell, pink_cell):
        """One episode. Returns the log entry."""
        xy = {"blue_cup": grid.sample(blue_cell, args.cell_margin, rng),
              "pink_cup": grid.sample(pink_cell, args.cell_margin, rng)}
        nonlocal spawn_z
        place_cups(env, xy, spawn_z)
        sm.reset()
        if args.spawn_z is None and spawn_z == 1.0:
            rest = min(float(env.scene[n].data.root_pos_w[0, 2] - env.scene.env_origins[0, 2]) for n in OBJECTS)
            spawn_z = rest + 0.003
            print(f"[grid] measured cup rest z = {rest:.4f}; spawning at {spawn_z:.4f} from now on", flush=True)
        settled = read_object_poses(env, list(OBJECTS))
        entry = {
            "blue_cell": list(blue_cell), "pink_cell": list(pink_cell),
            "sampled_xy": {k: [round(v[0], 4), round(v[1], 4)] for k, v in xy.items()},
            "settled_pose": {k[len(OBJECT_POSE_KEY_PREFIX):]: [round(float(t), 4) for t in v] for k, v in settled.items()},
        }
        settled_cells = {
            name: grid.cell_of(*entry["settled_pose"][name][:2]) for name in OBJECTS
        }
        if any(entry["settled_pose"][n][2] < FALL_Z for n in OBJECTS):
            entry["result"] = "fell_on_spawn"
            return entry
        for n in OBJECTS:
            qw, qx, qy, qz = entry["settled_pose"][n][3:]
            tilt = math.degrees(2 * math.asin(min(1.0, math.hypot(qx, qy))))
            if tilt > MAX_TILT_DEG:
                entry["result"] = "tipped_on_spawn"
                return entry
        if settled_cells["blue_cup"] != tuple(blue_cell) or settled_cells["pink_cup"] != tuple(pink_cell):
            entry["result"] = "drifted_out_of_cell"
            return entry

        while not sm.is_episode_done and not interrupted:
            if env.cfg.dynamic_reset_gripper_effort_limit:
                dynamic_reset_gripper_effort_limit_sim(env, "keyboard")
            sm.pre_step(env)
            env.step(sm.get_action(env))
            sm.advance()
            if args.step_hz > 0:
                rate_limit(env)
            if any(env.scene[n].data.root_pos_w[0, 2] - env.scene.env_origins[0, 2] < FALL_Z for n in OBJECTS):
                entry["result"] = "fell_during_episode"
                return entry
        if interrupted:
            entry["result"] = "interrupted"
            return entry
        entry["result"] = "success" if stacked(env, sm) else "failed"
        return entry

    try:
        n_pairings = 0
        for sweep in range(args.min_episodes_per_cell):
            for blue_cell, pink_cell in grid.pairings():
                if interrupted or (args.max_pairings is not None and n_pairings >= args.max_pairings):
                    break
                n_pairings += 1
                br, bc = blue_cell
                pr, pc = pink_cell
                if counts["blue_cup"][br][bc] > sweep and counts["pink_cup"][pr][pc] > sweep:
                    continue
                for attempt in range(1, args.max_attempts + 1):
                    entry = run_attempt(blue_cell, pink_cell)
                    entry.update({"sweep": sweep, "attempt": attempt})
                    ok = entry["result"] == "success"
                    # Export decision happens at the next reset: success -> saved, else discarded.
                    set_success(env, ok)
                    env.reset()
                    set_success(env, False)
                    if ok:
                        counts["blue_cup"][br][bc] += 1
                        counts["pink_cup"][pr][pc] += 1
                        entry["episode_index"] = env.recorder_manager.exported_successful_episode_count - 1
                    log.append(entry)
                    color = "\033[92m" if ok else "\033[91m"
                    print(
                        f"{color}[grid] sweep {sweep} blue{blue_cell} pink{pink_cell} "
                        f"attempt {attempt}/{args.max_attempts}: {entry['result']}\033[0m "
                        f"(saved {env.recorder_manager.exported_successful_episode_count}, "
                        f"{time.time() - t_start:.0f}s)",
                        flush=True,
                    )
                    if ok or interrupted:
                        break
            if interrupted:
                break
    finally:
        env.recorder_manager.finalize()
        summary = {
            "args": {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, list, type(None)))},
            "region": {"x": [grid.x0, grid.x1], "y": [grid.y0, grid.y1]},
            "grid": [grid.nx, grid.ny],
            "saved_episodes": env.recorder_manager.exported_successful_episode_count,
            "attempts": len(log),
            "results": {r: sum(e["result"] == r for e in log) for r in sorted({e["result"] for e in log})},
            "counts_row_major_top_left": counts,
            "uncovered_cells": {
                name: [[r, c] for r in range(grid.ny) for c in range(grid.nx)
                       if counts[name][r][c] < args.min_episodes_per_cell]
                for name in OBJECTS
            },
            "elapsed_s": time.time() - t_start,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        (out_dir / "episodes.json").write_text(json.dumps(log, indent=1))
        print(json.dumps({k: summary[k] for k in ("saved_episodes", "attempts", "results", "uncovered_cells")},
                         indent=2), flush=True)
        env.close()


if __name__ == "__main__":
    main()
    # SimulationApp.close() hangs in this container after headless runs; outputs are written.
    os._exit(0)
