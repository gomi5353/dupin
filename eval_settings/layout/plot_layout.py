"""Top-down layout of the cutlery and toy-blocks eval scenes.

Draws, over the reach map from ../reach_map: the table top, every tracked
object's footprint at its nominal pose, each randomized object's spawn box
(range of its root xy), and the success area of the task's success term.
Success terms check object root positions, so spawn boxes and success areas
are drawn in root coordinates; footprints show the object's real extent.

Needs out/footprints_<task>.json (object_footprints.py) and
../reach_map/out/reach_<task>.npz. Runs on the host:

    python3 eval_settings/layout/plot_layout.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Polygon, Rectangle, Wedge

HERE = Path(__file__).resolve().parent
REACH = HERE.parent / "reach_map" / "out"
OUT = HERE / "out"

TABLES = {
    "cutlery_arrangement": {"shape": "rect", "x": (6.47, 7.57), "y": (2.48, 4.48)},
    "toy_blocks_collection": {"shape": "circle", "center": (3.30, 1.10), "radius": 0.61},
}
# Root-xy spawn offsets from eval/<task>_eval.py (fixed objects have none).
SPAWN = {
    "cutlery_arrangement": {"knife": 0.05, "fork": 0.05},
    "toy_blocks_collection": {"green_block": 0.03, "blue_block": 0.03, "red_block": 0.03},
}
COLORS = {
    "plate": "#7f7f7f", "fork": "#1f4fd1", "knife": "#d63384",
    "storage_box": "#8c564b", "green_block": "#2ca02c", "blue_block": "#1f77b4", "red_block": "#d62728",
}
FIXED = {"cutlery_arrangement": "plate", "toy_blocks_collection": "storage_box"}


def hull(points):
    """Convex hull of 2D points (monotone chain)."""
    pts = sorted(map(tuple, points))
    def half(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and np.cross(np.subtract(h[-1], h[-2]), np.subtract(p, h[-2])) <= 0:
                h.pop()
            h.append(p)
        return h[:-1]
    return np.array(half(pts) + half(pts[::-1]))


def draw_success(ax, task, objs):
    if task == "cutlery_arrangement":
        # cutlery_arranged: |fork - plate|_xy <= 0.15 and fork.x < plate.x; knife likewise with knife.x > plate.x.
        px, py = objs["plate"]["root_xyz"][:2]
        r = 0.15
        for name, (t0, t1) in (("fork", (90, 270)), ("knife", (-90, 90))):
            c = COLORS[name]
            ax.add_patch(Wedge((px, py), r, t0, t1, fc=c, alpha=0.15, ec=c, lw=1.5, ls="--", zorder=2))
            ax.text(px + (-1 if name == "fork" else 1) * r * 0.5, py - r * 0.55, f"{name}\nsuccess",
                    ha="center", va="center", fontsize=8, color=c, zorder=6)
        return [Patch(fc="none", ec="black", ls="--", label="success area of object root: within 15 cm of plate root, "
                      "fork at x < plate x, knife at x > plate x")]
    # toys_in_box: each block root within +-0.12 m in x and y (and +-0.08 m in z) of the storage box root.
    bx, by = objs["storage_box"]["root_xyz"][:2]
    h = 0.12
    ax.add_patch(Rectangle((bx - h, by - h), 2 * h, 2 * h, fc="#ffbf00", alpha=0.18, ec="#b8860b", lw=1.5, ls="--",
                           zorder=2))
    ax.text(bx, by + h + 0.01, "success area (all blocks)", ha="center", va="bottom", fontsize=8, color="#b8860b")
    return [Patch(fc="#ffbf00", alpha=0.3, ec="#b8860b", ls="--",
                  label="success area of block roots: |dx|, |dy| < 12 cm (and |dz| < 8 cm) from box root")]


def draw(ax, task, objs, reach, zoom):
    xs, ys = reach["xs"], reach["ys"]
    X, Y = np.meshgrid(xs, ys)
    img = np.full(X.shape, np.nan)
    inside = reach["inside"]
    img[inside] = 0
    img[inside & reach["reachable_h0.05"] & ~reach["collides_h0.05"]] = 1
    step = xs[1] - xs[0]
    ax.imshow(img, origin="lower", extent=[xs[0] - step / 2, xs[-1] + step / 2, ys[0] - step / 2, ys[-1] + step / 2],
              cmap=ListedColormap(["#ececec", "#cfe8d3"]), vmin=0, vmax=1, interpolation="nearest", zorder=0)
    t = TABLES[task]
    if t["shape"] == "rect":
        ax.add_patch(Rectangle((t["x"][0], t["y"][0]), t["x"][1] - t["x"][0], t["y"][1] - t["y"][0], fill=False,
                               ec="black", lw=1.5, zorder=1))
    else:
        ax.add_patch(Circle(t["center"], t["radius"], fill=False, ec="black", lw=1.5, zorder=1))

    handles = draw_success(ax, task, objs)
    for name, o in objs.items():
        c = COLORS[name]
        poly = hull(np.array(o["obb_corners_xyz"])[:, :2])
        fixed = name == FIXED[task]
        ax.add_patch(Polygon(poly, closed=True, fc=c, alpha=0.35 if fixed else 0.6, ec=c, lw=1.2, zorder=3))
        rx, ry = o["root_xyz"][:2]
        ax.plot(rx, ry, marker="+", ms=8, mew=1.5, color="black", zorder=5)
        if name in SPAWN[task]:
            d = SPAWN[task][name]
            ax.add_patch(Rectangle((rx - d, ry - d), 2 * d, 2 * d, fill=False, ec=c, lw=1.5, hatch="///", zorder=4))
        if zoom:
            label = f"{name} (fixed)" if fixed else name
            ax.annotate(label, (poly[:, 0].max(), poly[:, 1].max()), xytext=(4, 2), textcoords="offset points",
                        fontsize=8, color=c, zorder=6)

    base = reach["base_xy"]
    ax.plot(*base, marker="s", ms=9, color="black", zorder=5)
    ax.annotate("Franka base", base, xytext=(8, -4), textcoords="offset points", fontsize=8)
    if zoom:
        pts = [np.array(o["obb_corners_xyz"])[:, :2] for o in objs.values()] + [[base]]
        fixed_xy = np.array(objs[FIXED[task]]["root_xyz"][:2])
        pts.append(fixed_xy + 0.15 * np.array([[-1, -1], [1, 1]]))  # success area
        pts = np.concatenate(pts)
        lo, hi = pts.min(0) - 0.08, pts.max(0) + 0.08
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
    else:
        (x0, y0), (x1, y1) = ((t["x"][0], t["y"][0]), (t["x"][1], t["y"][1])) if t["shape"] == "rect" else (
            np.subtract(t["center"], t["radius"]), np.add(t["center"], t["radius"]))
        ax.set_xlim(x0 - 0.1, x1 + 0.1)
        ax.set_ylim(min(y0, base[1]) - 0.1, y1 + 0.1)
    ax.set_aspect("equal")
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_title("whole table" if not zoom else "work area")
    return handles


def main():
    for task in TABLES:
        objs = json.loads((OUT / f"footprints_{task}.json").read_text())["objects"]
        reach = np.load(REACH / f"reach_{task}.npz")
        fig, axes = plt.subplots(1, 2, figsize=(14, 7.5), gridspec_kw={"width_ratios": [1, 1.2]})
        draw(axes[0], task, objs, reach, zoom=False)
        handles = draw(axes[1], task, objs, reach, zoom=True)
        handles = [
            Patch(fc="#cfe8d3", label="reachable top-down @ 5 cm (reach_map)"),
            Patch(fc="#ececec", label="table top, not reachable"),
            Patch(fc="#999999", alpha=0.6, label="object footprint at nominal pose (oriented bbox)"),
            Patch(fc="none", ec="black", hatch="///", label="spawn box of object root (eval randomization)"),
            Line2D([], [], marker="+", ls="", color="black", ms=8, label="object root (what success checks)"),
        ] + handles
        fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8, frameon=False)
        fig.suptitle(f"{task.replace('_', ' ')} eval layout (nominal poses, public eval file)")
        fig.tight_layout(rect=(0, 0.1, 1, 0.96))
        fig.savefig(OUT / f"layout_{task}.png", dpi=150)
        print(f"wrote {OUT / f'layout_{task}.png'}")


if __name__ == "__main__":
    main()
