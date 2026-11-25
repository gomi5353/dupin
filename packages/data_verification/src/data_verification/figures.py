"""Heatmap figures of the coverage grids, written into ``<store>/figures/`` by ``verify``.

Dataset check: one figure per requirement, episodes per cell (red = fewer than
the minimum). Rollout comparison: the same plus the rollout runs per cell
(colour = success rate, grey = no runs); cells outside the area that runs
started in are drawn too.
"""

from __future__ import annotations

from pathlib import Path

from .axes.image_augmentation.coverage import LightingCheck, LightingComparison
from .axes.object_pose.compare import ObjectPositionComparison
from .axes.object_pose.coverage import ObjectPositionCheck

ANNOTATE_MAX = 12  # print the numbers into the cells up to this many columns and rows
BELOW, COVERED, UNSAMPLED = "#e8a0a0", "#6bb26b", "#dddddd"

# One cell: (x0, x1, y0, y1, ...values)
Rect = tuple[float, float, float, float]


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _draw(ax, rects: list[Rect], colors: list, texts: list[str] | None, inside: Rect, labels: tuple[str, str]) -> None:
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    patches = [Rectangle((x0, y0), x1 - x0, y1 - y0) for x0, x1, y0, y1 in rects]
    ax.add_collection(PatchCollection(patches, facecolors=colors, edgecolors="white" if len(rects) <= 2500 else "none", linewidths=0.4))
    x0, x1, y0, y1 = (min(r[0] for r in rects), max(r[1] for r in rects), min(r[2] for r in rects), max(r[3] for r in rects))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.add_patch(Rectangle((inside[0], inside[2]), inside[1] - inside[0], inside[3] - inside[2], fill=False, edgecolor="black", linewidth=1.2))
    if texts:
        for (a, b, c, d), text in zip(rects, texts):
            ax.text((a + b) / 2, (c + d) / 2, text, ha="center", va="center", fontsize=8)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])


def _count_color(n: int, minimum: int, top: int):
    from matplotlib import colormaps

    if n < minimum:
        return BELOW
    return colormaps["Greens"](0.2 + 0.4 * min(1.0, (n - minimum) / max(1, top - minimum)))


def _rate_color(runs: int, successes: int):
    from matplotlib import colormaps

    return UNSAMPLED if runs == 0 else colormaps["RdYlGn"](successes / runs)


def _legend(ax, minimum: int) -> None:
    from matplotlib.patches import Patch

    ax.legend(
        handles=[Patch(color=BELOW, label=f"< {minimum} episodes"), Patch(color=COVERED, label=f">= {minimum} episodes")],
        loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8, frameon=False,
    )


def _rate_colorbar(fig, ax) -> None:
    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    bar = fig.colorbar(ScalarMappable(Normalize(0, 1), colormaps["RdYlGn"]), ax=ax, fraction=0.046, pad=0.04)
    bar.set_label("rollout success rate (grey = no runs)")


def _counts_panel(ax, cells: list[tuple[Rect, int]], minimum: int, inside: Rect, labels, title: str, annotate: bool) -> None:
    top = max([n for _, n in cells] + [minimum])
    rects = [r for r, _ in cells]
    _draw(ax, rects, [_count_color(n, minimum, top) for _, n in cells], [str(n) for _, n in cells] if annotate else None, inside, labels)
    ax.set_title(title, fontsize=10)
    _legend(ax, minimum)


def _runs_panel(fig, ax, cells: list[tuple[Rect, int, int]], inside: Rect, labels, title: str, annotate: bool) -> None:
    rects = [r for r, _, _ in cells]
    _draw(ax, rects, [_rate_color(n, s) for _, n, s in cells], [f"{s}/{n}" if n else "" for _, n, s in cells] if annotate else None, inside, labels)
    ax.set_title(title, fontsize=10)
    _rate_colorbar(fig, ax)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    _pyplot().close(fig)
    return path


def _grid_rects(x_range, y_range, counts) -> list[tuple[Rect, int]]:
    nh, nw = len(counts), len(counts[0])
    dx, dy = (x_range[1] - x_range[0]) / nw, (y_range[1] - y_range[0]) / nh
    return [((x_range[0] + i * dx, x_range[0] + (i + 1) * dx, y_range[0] + j * dy, y_range[0] + (j + 1) * dy), counts[j][i]) for j in range(nh) for i in range(nw)]


def _dataset_figure(cells, minimum, inside, labels, title, annotate, path: Path) -> Path:
    fig, ax = _pyplot().subplots(figsize=(7, 5.5))
    _counts_panel(ax, cells, minimum, inside, labels, title, annotate)
    return _save(fig, path)


def _comparison_figure(count_cells, run_cells, minimum, inside, labels, title, annotate, path: Path) -> Path:
    fig, (left, right) = _pyplot().subplots(1, 2, figsize=(14, 5.5))
    _counts_panel(left, count_cells, minimum, inside, labels, f"{title}: dataset episodes per cell", annotate)
    _runs_panel(fig, right, run_cells, inside, labels, f"{title}: rollout runs per cell (succeeded/runs)", annotate)
    return _save(fig, path)


def figure_dir(store_root: Path) -> Path:
    return store_root / "figures"


def save_figures(store_root: Path, result) -> list[Path]:
    """Write the figures of a ``VerifyResult`` into ``<store_root>/figures/`` (replacing the earlier ones); returns their paths."""
    out = figure_dir(store_root)
    if out.is_dir():
        for old in out.glob("*.png"):
            old.unlink()
    paths: list[Path] = []
    for check in result.checks:
        for axis in check.axes:
            if isinstance(axis, ObjectPositionCheck):
                for t in axis.coverage:
                    cells = _grid_rects(t.x_range, t.y_range, t.counts)
                    annotate = len(t.counts) <= ANNOTATE_MAX and len(t.counts[0]) <= ANNOTATE_MAX
                    inside = (*t.x_range, *t.y_range)
                    paths.append(_dataset_figure(
                        cells, t.min_episodes_per_cell, inside, ("x (m)", "y (m)"),
                        f"{t.object_name}: episodes per cell ({t.cells_covered}/{t.cells_total} covered)", annotate,
                        out / f"{check.spec}__{t.object_name}.png",
                    ))
            elif isinstance(axis, LightingCheck):
                for t in axis.coverage:
                    r = t.req
                    cells = _grid_rects((float(r.i0), float(r.i1)), (float(r.t0), float(r.t1)), t.counts)
                    annotate = len(t.counts) <= ANNOTATE_MAX and len(t.counts[0]) <= ANNOTATE_MAX
                    paths.append(_dataset_figure(
                        cells, r.min_episodes, (float(r.i0), float(r.i1), float(r.t0), float(r.t1)), ("intensity", "color temperature (K)"),
                        f"{t.light}: episodes per cell ({t.cells_covered}/{t.cells_total} covered)", annotate,
                        out / f"{check.spec}__{t.light}.png",
                    ))
    for comparison in result.comparisons:
        for axis in comparison.axes:
            stem = f"{comparison.spec}__{comparison.rollout}"
            if isinstance(axis, ObjectPositionComparison):
                for o in axis.objects:
                    rows = [c for row in o.cells for c in row] + list(o.outside)
                    rect = lambda c: (*c.x_range, *c.y_range)  # noqa: E731
                    annotate = len(o.cells) <= ANNOTATE_MAX and len(o.cells[0]) <= ANNOTATE_MAX and not o.outside
                    x0, x1, y0, y1 = o.requirement_area
                    paths.append(_comparison_figure(
                        [(rect(c), c.episodes) for c in rows], [(rect(c), c.runs, c.successes) for c in rows],
                        o.min_episodes, (x0, x1, y0, y1), ("x (m)", "y (m)"), o.object_name, annotate,
                        out / f"{stem}__{o.object_name}.png",
                    ))
            elif isinstance(axis, LightingComparison):
                for light in axis.lights:
                    r = light.req
                    cells = {(i, j): light.cells.get((i, j)) for j in range(r.nt) for i in range(r.ni)}
                    cells.update({key: c for key, c in light.cells.items() if not c.inside})
                    rows = []
                    for (i, j), c in cells.items():
                        episodes = light.counts[j * r.ni + i] if r.inside(i, j) else 0
                        rows.append((r.bounds(i, j), episodes, c.runs if c else 0, c.successes if c else 0))
                    annotate = r.ni <= ANNOTATE_MAX and r.nt <= ANNOTATE_MAX and all(c.inside for c in light.cells.values())
                    paths.append(_comparison_figure(
                        [(rect, e) for rect, e, _, _ in rows], [(rect, n, s) for rect, _, n, s in rows],
                        r.min_episodes, (float(r.i0), float(r.i1), float(r.t0), float(r.t1)), ("intensity", "color temperature (K)"), light.light, annotate,
                        out / f"{stem}__{light.light}.png",
                    ))
    return paths
