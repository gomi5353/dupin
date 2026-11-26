"""Object Poses axis, factor object position, rollout side: place each finished run in the lattice of each requirement's grid.

Each run is placed into the lattice of the requirement's grid: column
i = floor((x - areaXMin) / cell width), row j likewise, with the area's upper
edges in the last column and row. A run inside the area is covered when its
cell has at least minEpisodesPerCell dataset episodes; a run outside the area
is not covered. A run is covered on this axis when every requirement covers it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal
from urllib.parse import quote

from rdflib import RDF, XSD, Graph, Literal, URIRef

from ...graph import CP
from ...verifier import AxisComparison, Finding, ReportTable, pct, rate
from .coverage import cell_counts, cell_edge, lattice_index, requirement_rows

# Initial positions of the finished runs of ?rollout (bound by the caller).
RUN_POSITIONS_QUERY = """
SELECT ?run ?index ?status ?name ?x ?y WHERE {
    ?rollout cp:hasRun ?run .
    ?run cp:index ?index ; cp:status ?status ; cp:hasTaskObject ?obj .
    FILTER (?status IN ("succeeded", "failed"))
    ?obj cp:name ?name ; cp:hasPosition ?p .
    ?p cp:positionX ?x ; cp:positionY ?y .
}
"""


@dataclass
class RunCell:
    """One cell of the requirement grid's lattice, inside the area or (for runs that started there) outside it."""

    i: int
    j: int
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    inside: bool
    episodes: int = 0
    min_episodes: int = 0
    runs: int = 0
    successes: int = 0

    @property
    def sampled(self) -> bool:
        return self.runs > 0

    @property
    def covered(self) -> bool:
        return self.inside and self.episodes >= self.min_episodes

    @property
    def needed(self) -> int:
        return max(0, self.min_episodes - self.episodes) if self.inside else 0


@dataclass
class ObjectComparison:
    object_name: str
    cell_size: tuple[float, float]
    requirement_area: tuple[float, float, float, float]
    min_episodes: int
    cells: list[list[RunCell]] = field(default_factory=list)
    """cells[j][i]: the requirement grid, column i (x), row j (y), row-major index k = j * numCellW + i."""
    outside: list[RunCell] = field(default_factory=list)
    """Lattice cells outside the area that runs started in, sorted by (j, i)."""
    runs: int = 0
    runs_covered: int = 0
    successes_covered: int = 0
    successes_not_covered: int = 0

    @property
    def runs_outside(self) -> int:
        return sum(c.runs for c in self.outside)

    @property
    def sampled_cells(self) -> int:
        return sum(c.sampled for row in self.cells for c in row) + len(self.outside)

    @property
    def covered_cells(self) -> int:
        return sum(c.sampled and c.covered for row in self.cells for c in row)


@dataclass
class FillTarget:
    """A requirement cell to record more episodes in."""

    object_name: str
    i: int
    j: int
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    episodes: int
    needed: int
    runs: int
    failed: int


@dataclass
class AreaSuggestion:
    """Grow the requirement area by whole cells so it contains the runs that started outside it."""

    object_name: str
    current: tuple[float, float, float, float]
    suggested: tuple[float, float, float, float]
    num_cell_w: int
    num_cell_h: int


@dataclass
class ObjectPositionComparison(AxisComparison):
    objects: list[ObjectComparison] = field(default_factory=list)
    to_fill: list[FillTarget] = field(default_factory=list)
    widen: list[AreaSuggestion] = field(default_factory=list)

    def lines(self) -> list[str]:
        return [line for obj in self.objects for line in _object_lines(obj)]

    def tables(self) -> list[ReportTable]:
        rows, notes = [], []
        for o in self.objects:
            x0, x1, y0, y1 = o.requirement_area
            if o.outside:
                notes.append(f"{o.object_name}: {o.runs_outside} run(s) started outside your area, in {len(o.outside)} cell(s) of the same size")
            not_covered = o.runs - o.runs_covered
            rows.append(
                [
                    o.object_name, f"{x0:.3f} .. {x1:.3f}", f"{y0:.3f} .. {y1:.3f}", f"{len(o.cells[0])} x {len(o.cells)}", f">= {o.min_episodes}",
                    f"{o.covered_cells}/{o.sampled_cells}", f"{o.runs_covered}/{o.runs} ({pct(o.runs_covered, o.runs)})",
                    rate(o.successes_covered, o.runs_covered), rate(o.successes_not_covered, not_covered),
                ]
            )
        columns = ["Object", "x (m)", "y (m)", "Grid", "Episodes per cell", "Sampled cells covered", "Runs covered", "Success covered", "Success not covered"]
        return [ReportTable("Object Poses: rollout runs", columns, rows, notes)] if rows else []

    def hint_lines(self) -> list[str]:
        lines = []
        if self.to_fill:
            total = sum(t.needed for t in self.to_fill)
            lines.append(f"Record more episodes here ({total} more at least; cells with failed rollout runs first):")
            for t in self.to_fill:
                cx, cy = (t.x_range[0] + t.x_range[1]) / 2, (t.y_range[0] + t.y_range[1]) / 2
                lines.append(
                    f"  {t.object_name} cell (i={t.i}, j={t.j}) x=[{t.x_range[0]:.3f}, {t.x_range[1]:.3f}] "
                    f"y=[{t.y_range[0]:.3f}, {t.y_range[1]:.3f}] centre ({cx:.3f}, {cy:.3f}): "
                    f"has {t.episodes}, needs {t.needed} more; {t.runs} rollout run(s) started here, {t.failed} failed"
                )
            lines.append("")
        if self.widen:
            lines.append("Rollout runs started outside your area. Grow it by whole cells (same cell size), then record there:")
            for w in self.widen:
                lines.append(
                    f"  {w.object_name}: area x=[{w.current[0]:.3f}, {w.current[1]:.3f}] y=[{w.current[2]:.3f}, {w.current[3]:.3f}]"
                    f" -> areaXMin {w.suggested[0]:.6g} ; areaXMax {w.suggested[1]:.6g} ; areaYMin {w.suggested[2]:.6g} ;"
                    f" areaYMax {w.suggested[3]:.6g} ; numCellW {w.num_cell_w} ; numCellH {w.num_cell_h}"
                )
            lines.append("")
        return lines


@dataclass
class _Requirement:
    iri: URIRef
    grid: URIRef
    name: str
    x0: Decimal
    x1: Decimal
    y0: Decimal
    y1: Decimal
    nw: int
    nh: int
    min_episodes: int
    counts: Counter

    @property
    def size(self) -> tuple[Decimal, Decimal]:
        return (self.x1 - self.x0) / self.nw, (self.y1 - self.y0) / self.nh

    def lattice_index(self, x: Decimal, y: Decimal) -> tuple[int, int]:
        """Column and row of the lattice cell (x, y) is in, with the membership rule of the coverage check."""
        return lattice_index(x, self.x0, self.x1, self.nw), lattice_index(y, self.y0, self.y1, self.nh)

    def cell(self, i: int, j: int) -> RunCell:
        inside = 0 <= i < self.nw and 0 <= j < self.nh
        x_range = cell_edge(self.x0, self.x1, self.nw, i), cell_edge(self.x0, self.x1, self.nw, i + 1)
        y_range = cell_edge(self.y0, self.y1, self.nh, j), cell_edge(self.y0, self.y1, self.nh, j + 1)
        cell = RunCell(i, j, _floats(x_range), _floats(y_range), inside)
        if inside:
            cell.episodes, cell.min_episodes = self.counts[j * self.nw + i], self.min_episodes
        return cell


def compare_object_position(graph: Graph, rollout: URIRef, dataset: URIRef, comparison: URIRef, axis: str) -> ObjectPositionComparison:
    result = ObjectPositionComparison(axis, {})
    requirements = _requirements(graph, dataset, result)
    if result.errors:
        return result

    positions: dict[str, dict[int, tuple[Decimal, Decimal]]] = {}
    outcome: dict[int, bool] = {}
    run_nodes: dict[int, URIRef] = {}
    for row in graph.query(RUN_POSITIONS_QUERY, initNs={"cp": CP}, initBindings={"rollout": rollout}):
        index = int(row["index"])
        outcome[index] = str(row.status) == "succeeded"
        run_nodes[index] = row.run
        positions.setdefault(str(row.name), {})[index] = (Decimal(str(row.x)), Decimal(str(row.y)))

    result.run_covered = {index: True for index in outcome}
    for name, req in sorted(requirements.items()):
        dx, dy = req.size
        obj = ObjectComparison(name, _floats((dx, dy)), _floats((req.x0, req.x1, req.y0, req.y1)), req.min_episodes)
        obj.cells = [[req.cell(i, j) for i in range(req.nw)] for j in range(req.nh)]
        outside: dict[tuple[int, int], RunCell] = {}
        for index, (x, y) in sorted(positions.get(name, {}).items()):
            i, j = req.lattice_index(x, y)
            inside = 0 <= i < req.nw and 0 <= j < req.nh
            cell = obj.cells[j][i] if inside else outside.setdefault((i, j), req.cell(i, j))
            ok = outcome[index]
            cell.runs += 1
            cell.successes += ok
            obj.runs += 1
            if cell.covered:
                obj.runs_covered += 1
                obj.successes_covered += ok
            else:
                obj.successes_not_covered += ok
                result.run_covered[index] = False
            _add_placement(result.derived, comparison, run_nodes[index], req, i, j, inside, cell.covered)
        obj.outside = [outside[key] for key in sorted(outside, key=lambda k: (k[1], k[0]))]
        result.objects.append(obj)
        not_covered = [c for row in obj.cells for c in row if c.sampled and not c.covered] + obj.outside
        if not_covered:
            runs = sum(c.runs for c in not_covered)
            failed = runs - sum(c.successes for c in not_covered)
            result.findings.append(
                Finding(
                    "Warning",
                    f"'{name}': {len(not_covered)} of {obj.sampled_cells} sampled cell(s) are not covered by the "
                    f"dataset ({runs} run(s), {failed} failed)",
                    [f"{name}/cell/({c.i}, {c.j})" for c in not_covered],
                    axis,
                )
            )

    for name, runs in sorted(positions.items()):
        if name in requirements:
            continue
        if len(set(runs.values())) > 1:
            result.findings.append(
                Finding(
                    "Warning",
                    f"'{name}' starts at different positions across the rollout's runs, but the spec has no "
                    "requirement for it; it is not compared",
                    [name],
                    axis,
                )
            )
    _guidance(result, requirements)
    return result


def _add_placement(graph: Graph, comparison: URIRef, run: URIRef, req: _Requirement, i: int, j: int, inside: bool, covered: bool) -> None:
    placement = URIRef(f"{comparison}/placement/{run.rsplit('/', 1)[-1]}/{quote(req.name, safe='')}")
    graph.add((comparison, CP.hasPlacement, placement))
    graph.add((placement, RDF.type, CP.RunPlacement))
    graph.add((placement, CP.ofRun, run))
    graph.add((placement, CP.ofRequirement, req.iri))
    graph.add((placement, CP.cellI, Literal(i, datatype=XSD.integer)))
    graph.add((placement, CP.cellJ, Literal(j, datatype=XSD.integer)))
    if inside:
        graph.add((placement, CP.inCell, URIRef(f"{req.grid}/cell/{j * req.nw + i}")))
    graph.add((placement, CP.covered, Literal(covered)))


def _requirements(graph: Graph, dataset: URIRef, result: AxisComparison) -> dict[str, _Requirement]:
    counts = cell_counts(graph, dataset)
    requirements: dict[str, _Requirement] = {}
    for row in requirement_rows(graph, dataset):
        name = str(row.name)
        if name in requirements:
            result.errors.append(f"Spec has more than one requirement for '{name}'; compare needs one per object")
            continue
        requirements[name] = _Requirement(
            row.req,
            row.grid,
            name,
            *(Decimal(str(v.toPython())) for v in (row.x0, row.x1, row.y0, row.y1)),
            int(row.nx),
            int(row.ny),
            int(row.min),
            counts.get(str(row.req), Counter()),
        )
    if not requirements:
        result.errors.append("The spec has no object position requirement; nothing to compare")
    return requirements


def _guidance(result: ObjectPositionComparison, requirements: dict[str, _Requirement]) -> None:
    targets: list[FillTarget] = []
    widen_cells: dict[str, list[RunCell]] = {}
    for obj in result.objects:
        for cell in (c for row in obj.cells for c in row if c.sampled and not c.covered):
            targets.append(
                FillTarget(
                    obj.object_name, cell.i, cell.j, cell.x_range, cell.y_range, cell.episodes, cell.needed,
                    cell.runs, cell.runs - cell.successes,
                )
            )
        if obj.outside:
            widen_cells[obj.object_name] = obj.outside
    result.to_fill = sorted(targets, key=lambda t: (-t.failed, -t.needed, t.object_name, t.j, t.i))

    for name, cells in sorted(widen_cells.items()):
        req = requirements[name]
        dx, dy = req.size
        # Grow by whole cells, so the cell size and the cells already recorded stay the same.
        left = max(0, -min(c.i for c in cells))
        right = max(0, max(c.i for c in cells) - (req.nw - 1))
        down = max(0, -min(c.j for c in cells))
        up = max(0, max(c.j for c in cells) - (req.nh - 1))
        result.widen.append(
            AreaSuggestion(
                name,
                _floats((req.x0, req.x1, req.y0, req.y1)),
                _floats((req.x0 - left * dx, req.x1 + right * dx, req.y0 - down * dy, req.y1 + up * dy)),
                req.nw + left + right,
                req.nh + down + up,
            )
        )


def _floats(values) -> tuple[float, ...]:
    return tuple(float(v) for v in values)


def _object_lines(obj: ObjectComparison) -> list[str]:
    x0, x1, y0, y1 = obj.requirement_area
    nw = len(obj.cells[0])
    lines = [
        f"[{obj.object_name}] your area x={x0:.3f}..{x1:.3f} y={y0:.3f}..{y1:.3f}, "
        f"cell {obj.cell_size[0]:.4g} x {obj.cell_size[1]:.4g}, {nw}x{len(obj.cells)} cells, >= {obj.min_episodes} episodes each",
    ]
    if obj.outside:
        lines.append(f"  {obj.runs_outside} run(s) started outside your area, in {len(obj.outside)} cell(s) of the same size")
    not_covered = obj.runs - obj.runs_covered
    lines.append(
        f"  covered: {obj.covered_cells}/{obj.sampled_cells} sampled cells, {obj.runs_covered}/{obj.runs} runs "
        f"({pct(obj.runs_covered, obj.runs)})\n  success: {rate(obj.successes_covered, obj.runs_covered)} covered vs "
        f"{rate(obj.successes_not_covered, not_covered)} not covered"
    )
    lines.append("")
    return lines
