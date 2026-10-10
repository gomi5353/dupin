"""Start-pose / trajectory distribution of the obj_cvg datasets.

    uv run --no-project --with pyarrow --with numpy \
        python obj_cvg_exps/grid_datagen/dataset_stats.py [dataset_dir ...]
"""

import sys

import numpy as np
import pyarrow.parquet as pq

DEFAULT = [
    "datasets/local/cup_stacking_grid8x4_m4",
    "datasets/local/cup_stacking_uniform_128",
    "datasets/local/cup_stacking_fixed_128",
]
CUPS = ["blue_cup", "pink_cup"]
X0, X1, Y0, Y1, NX, NY = 8.8, 9.2, 6.4, 6.6, 8, 4


def load(root):
    t = pq.read_table(f"{root}/data/chunk-000/file-000.parquet")
    d = {c: np.stack(t.column(c).to_numpy(zero_copy_only=False)) for c in ["action", "observation.state"]}
    for c in CUPS:
        d[c] = np.stack(t.column(f"object_pose.{c}").to_numpy(zero_copy_only=False))
    d["ep"] = t.column("episode_index").to_numpy()
    d["frame"] = t.column("frame_index").to_numpy()
    return d


def per_episode(d):
    eps = np.unique(d["ep"])
    first = np.array([np.flatnonzero(d["ep"] == e)[np.argmin(d["frame"][d["ep"] == e])] for e in eps])
    last = np.array([np.flatnonzero(d["ep"] == e)[np.argmax(d["frame"][d["ep"] == e])] for e in eps])
    return eps, first, last


def report(root):
    d = load(root)
    eps, first, last = per_episode(d)
    lens = np.bincount(d["ep"])[eps]
    print(f"\n=== {root}\nepisodes {len(eps)}, frames {len(d['ep'])}, length min/mean/max {lens.min()}/{lens.mean():.1f}/{lens.max()}")
    for c in CUPS:
        p = d[c][first]
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        inside = (x >= X0) & (x < X1) & (y >= Y0) & (y < Y1)
        cells = set(zip(((Y1 - y[inside]) / ((Y1 - Y0) / NY)).astype(int), ((x[inside] - X0) / ((X1 - X0) / NX)).astype(int)))
        print(
            f"  {c:9s} start x {x.min():.3f}..{x.max():.3f} (mean {x.mean():.3f}, std {x.std():.3f})"
            f" | y {y.min():.3f}..{y.max():.3f} (mean {y.mean():.3f}, std {y.std():.3f})"
            f" | z {z.min():.3f}..{z.max():.3f} | inside {inside.sum()}/{len(x)} | cells {len(cells)}/32"
            f" | unique xy {len(np.unique(np.round(p[:, :2], 4), axis=0))}"
        )
    b, k = d["blue_cup"][first, :2], d["pink_cup"][first, :2]
    sep = np.linalg.norm(b - k, axis=1)
    print(f"  start separation min/mean/max {sep.min():.3f}/{sep.mean():.3f}/{sep.max():.3f}")
    rel = d["blue_cup"][last, :3] - d["pink_cup"][last, :3]
    print(
        f"  end: blue-pink xy dist max {np.linalg.norm(rel[:, :2], axis=1).max():.3f}, "
        f"dz min/max {rel[:, 2].min():.3f}/{rel[:, 2].max():.3f}"
    )
    a, s = d["action"], d["observation.state"]
    print(f"  action  mean {np.round(a.mean(0), 3)}\n          std  {np.round(a.std(0), 3)}")
    print(f"  state   std  {np.round(s.std(0), 3)}")
    # how different trajectories are: std over episodes of the action at the same frame index
    n = lens.min()
    traj = np.stack([a[d["ep"] == e][np.argsort(d["frame"][d["ep"] == e])][:n] for e in eps])
    print(f"  per-frame action std across episodes (mean over frames, xyz) {np.round(traj.std(0).mean(0)[:3], 4)}")


for root in sys.argv[1:] or DEFAULT:
    report(root)
