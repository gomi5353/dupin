"""Grid-sweep object spawning for the object-pose coverage experiment.

Shared by ``scripts/datagen/generate.py`` (data collection) and
``scripts/rollout.py`` (evaluation) so both draw start poses the same way.

A rectangular region (``--grid_center`` +- ``--grid_half``) is split into
``gridX x gridY`` cells. Rows count from the top (max y), cols from the left
(min x). Each episode is a ``GridSlot``; ``GridSlot.sample()`` draws a fresh
pose per attempt according to ``--grid_layout``:

* ``grid``: object ``i`` goes to cell ``((row + i * sweep) % gridY,
  (col + i * gridX // n) % gridX)``, uniform inside the cell. Slots run
  row-major from top-left. The column shift keeps the objects apart; the
  per-sweep row shift pairs each cell with a different partner cell in every
  sweep. Each object visits every cell exactly once per sweep.
* ``uniform``: every object uniform over the whole region (no stratification),
  resampled until all objects are ``--grid_min_separation`` apart.
* ``fixed``: every episode uses the same pose (``--grid_fixed_xy``).
* ``outside``: eval only. Every object uniform over a band around the region,
  ``--grid_outside_band`` = (d_min, d_max) metres beyond its edge, so no object
  starts inside the training region. Separation as in ``uniform``.

All layouts share region, slot count and retry budget, so datasets built with
them differ only in start-pose coverage.

Pure Python: importable before the Isaac Sim app is launched.
"""

from __future__ import annotations

import argparse
import math
import os
import random


def add_grid_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--grid_sweep",
        action="store_true",
        help="Spawn task objects from a grid sweep (object-pose coverage experiment) instead of the cfg's "
        "reset randomization. Episode count = gridX*gridY*grid_min_episodes_per_cell slots.",
    )
    parser.add_argument(
        "--grid_layout",
        choices=["grid", "uniform", "fixed", "outside"],
        default="grid",
        help="grid: stratified per cell; uniform: random over the region; fixed: one pose for every episode; "
        "outside: random in a band around the region (eval only, see --grid_outside_band).",
    )
    parser.add_argument("--grid_objects", type=str, nargs="+", default=["blue_cup", "pink_cup"])
    parser.add_argument("--grid_center", type=float, nargs=3, default=[9.0, 6.5, 1.0], help="Region centre (x, y, z).")
    parser.add_argument("--grid_half", type=float, nargs=2, default=[0.2, 0.1], help="Region half-extent (x, y).")
    parser.add_argument("--grid_size", type=int, nargs=2, default=[8, 4], help="gridX gridY.")
    parser.add_argument("--grid_min_episodes_per_cell", type=int, default=1, help="Number of sweeps.")
    parser.add_argument("--grid_max_attempts", type=int, default=5, help="Datagen: tries per slot before moving on.")
    parser.add_argument("--grid_cell_margin", type=float, default=0.005, help="Keep samples this far inside a cell (m).")
    parser.add_argument(
        "--grid_fixed_xy",
        type=float,
        nargs="+",
        default=[8.95, 6.52, 9.07, 6.43],
        help="--grid_layout fixed: x y per grid object, in --grid_objects order (default: cup-stacking init_state).",
    )
    parser.add_argument("--grid_min_separation", type=float, default=0.12, help="--grid_layout uniform: min xy distance.")
    parser.add_argument(
        "--grid_outside_band",
        type=float,
        nargs=2,
        default=[0.0, 0.05],
        help="--grid_layout outside: min/max distance (m) beyond the region edge (Chebyshev, i.e. a rectangular ring).",
    )
    parser.add_argument("--grid_spec_out", type=str, default=None, help="Write a matching coverage spec .ttl here.")


class GridSlot:
    """One episode slot; ``cells`` holds the cell each object was last sampled in."""

    def __init__(self, row: int, col: int, sweep: int, args):
        self.row, self.col, self.sweep = row, col, sweep
        self.args = args
        self.cells: dict[str, tuple[int, int]] = {}

    def _bounds(self):
        (cx, cy, _), (hx, hy), (nx, ny) = self.args.grid_center, self.args.grid_half, self.args.grid_size
        return cx - hx, cy + hy, 2 * hx / nx, 2 * hy / ny

    def _cell_of(self, x: float, y: float) -> tuple[int, int]:
        x0, y1, dx, dy = self._bounds()
        nx, ny = self.args.grid_size
        return (min(max(int((y1 - y) / dy), 0), ny - 1), min(max(int((x - x0) / dx), 0), nx - 1))

    def _sample_xy(self) -> dict[str, tuple[float, float]]:
        a = self.args
        names = a.grid_objects
        x0, y1, dx, dy = self._bounds()
        nx, ny = a.grid_size
        if a.grid_layout == "fixed":
            return {name: tuple(a.grid_fixed_xy[2 * i : 2 * i + 2]) for i, name in enumerate(names)}
        if a.grid_layout == "outside":
            lo, hi = a.grid_outside_band
            xa, xb, ya, yb = x0, x0 + nx * dx, y1 - ny * dy, y1
            while True:
                xy = {}
                for n in names:
                    while True:
                        x, y = random.uniform(xa - hi, xb + hi), random.uniform(ya - hi, yb + hi)
                        d = outside_distance(x, y, (xa, xb, ya, yb))
                        if d > 0 and d >= lo:  # d > 0: never inside, even for lo = 0
                            break
                    xy[n] = (x, y)
                pts = list(xy.values())
                if all(math.dist(p, q) >= a.grid_min_separation for i, p in enumerate(pts) for q in pts[i + 1 :]):
                    return xy
        if a.grid_layout == "uniform":
            while True:
                xy = {n: (random.uniform(x0, x0 + nx * dx), random.uniform(y1 - ny * dy, y1)) for n in names}
                pts = list(xy.values())
                if all(math.dist(p, q) >= a.grid_min_separation for i, p in enumerate(pts) for q in pts[i + 1 :]):
                    return xy
        m = a.grid_cell_margin
        xy = {}
        for i, name in enumerate(names):
            row = (self.row + i * self.sweep) % ny
            col = (self.col + i * nx // len(names)) % nx
            cx0, cy1 = x0 + col * dx, y1 - row * dy
            xy[name] = (random.uniform(cx0 + m, cx0 + dx - m), random.uniform(cy1 - dy + m, cy1 - m))
        return xy

    def sample(self) -> dict:
        """Fresh pose dict ``{name: (pos_xyz, quat_wxyz)}`` (env-local, upright)."""
        z = self.args.grid_center[2]
        xy = self._sample_xy()
        # Out-of-region objects are counted in the nearest edge cell; ``outside`` keeps the true distance.
        self.cells = {name: self._cell_of(x, y) for name, (x, y) in xy.items()}
        x0, y1, dx, dy = self._bounds()
        nx, ny = self.args.grid_size
        rect = (x0, x0 + nx * dx, y1 - ny * dy, y1)
        self.outside = {name: round(outside_distance(x, y, rect), 4) for name, (x, y) in xy.items()}
        return {name: ((x, y, z), (1.0, 0.0, 0.0, 0.0)) for name, (x, y) in xy.items()}

    def __repr__(self):
        return f"{self.args.grid_layout} sweep {self.sweep} slot ({self.row}, {self.col})"


def outside_distance(x: float, y: float, rect: tuple[float, float, float, float]) -> float:
    """Chebyshev distance (m) from (x, y) to rect ``(x0, x1, y0, y1)``; 0 inside."""
    x0, x1, y0, y1 = rect
    return max(x0 - x, x - x1, y0 - y, y - y1, 0.0)


def make_slots(args, seed: int | None) -> list[GridSlot]:
    """All slots, sweep by sweep; seeds the sampler so a seed reproduces the same poses."""
    random.seed(seed)
    nx, ny = args.grid_size
    return [
        GridSlot(row, col, sweep, args)
        for sweep in range(args.grid_min_episodes_per_cell)
        for row in range(ny)
        for col in range(nx)
    ]


def write_coverage_spec(args, path: str) -> None:
    (cx, cy, _), (hx, hy), (nx, ny) = args.grid_center, args.grid_half, args.grid_size
    lines = [
        "# Object pose coverage spec written by --grid_sweep (simulator.utils.grid_sweep).",
        "@prefix dv:   <urn:aicapstone:dv#> .",
        "@prefix spec: <urn:aicapstone:spec:grid_sweep#> .",
        "",
        'spec:task dv:taskName "grid_sweep" .',
    ]
    for name in args.grid_objects:
        lines += [
            "",
            f'spec:{name} a dv:TaskObject ; dv:objectName "{name}" .',
            f"spec:{name}_coverage a dv:CoverageRequirement ;",
            "    dv:targetsAxis dv:ObjectPoseAxis ;",
            f"    dv:forObject spec:{name} ;",
            f"    dv:xMin {cx - hx:.3f} ; dv:xMax {cx + hx:.3f} ;",
            f"    dv:yMin {cy - hy:.3f} ; dv:yMax {cy + hy:.3f} ;",
            f"    dv:gridX {nx} ; dv:gridY {ny} ;",
            f"    dv:minEpisodesPerCell {args.grid_min_episodes_per_cell} .",
        ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[grid] wrote coverage spec to {path}")


def cell_tables(args, results: list[dict]) -> dict:
    """Per object: ``trials`` and ``successes`` per cell (row 0 = top / max y)."""
    nx, ny = args.grid_size
    tables = {}
    for name in args.grid_objects:
        trials = [[0] * nx for _ in range(ny)]
        successes = [[0] * nx for _ in range(ny)]
        for res in results:
            r, c = res["cells"][name]
            trials[r][c] += 1
            successes[r][c] += int(res["success"])
        tables[name] = {"trials": trials, "successes": successes}
    return tables


def print_summary(args, results: list[dict], min_successes: int | None = None) -> None:
    """Print success/trial heatmaps per object; list cells under ``min_successes`` if given."""
    if not results:
        return
    n_ok = sum(int(r["success"]) for r in results)
    print(f"[grid] layout={args.grid_layout}: {n_ok}/{len(results)} episodes succeeded")
    for name, t in cell_tables(args, results).items():
        print(f"[grid] {name}: successes/trials per cell (row 0 = max y, col 0 = min x)")
        for s_row, t_row in zip(t["successes"], t["trials"]):
            print("    " + " ".join(f"{s:2d}/{n:<2d}" for s, n in zip(s_row, t_row)))
        if min_successes is not None:
            nx, ny = args.grid_size
            missing = [(r, c) for r in range(ny) for c in range(nx) if t["successes"][r][c] < min_successes]
            print(f"[grid] {name} cells below {min_successes} successes: {missing or 'none'}")
