"""Image Augmentations axis, factor lighting: episodes per (intensity, color temperature) cell, and where the rollout runs' lights fall.

Each requirement splits [intensityMin, intensityMax] into numBinsIntensity bins
(column i) and [colorTemperatureMin, colorTemperatureMax] into
numBinsColorTemperature bins (row j), with the edge rule of ``grid.py``. A
cell is covered when at least minEpisodesPerCell dataset episodes started with
the light inside it. A rollout run is covered on this axis when, for every
requirement, its light falls in a covered cell.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal

from rdflib import RDF, XSD, Graph, Literal, URIRef

from ...graph import CP, index_literal
from ...grid import cell_edge, dec, lattice_index
from ...verifier import AxisCheck, AxisComparison, Finding, ReportTable, pct, rate

REQUIREMENTS_QUERY = """
SELECT ?req ?name ?i0 ?i1 ?t0 ?t1 ?ni ?nt ?min WHERE {
    ?dataset cp:hasGeneralizationAxis ?axis .
    ?axis cp:hasRequirement ?req .
    ?req a cp:LightingRequirement ; cp:lightName ?name ;
         cp:intensityMin ?i0 ; cp:intensityMax ?i1 ; cp:colorTemperatureMin ?t0 ; cp:colorTemperatureMax ?t1 ;
         cp:numBinsIntensity ?ni ; cp:numBinsColorTemperature ?nt ; cp:minEpisodesPerCell ?min .
}
"""

# Light conditions of the finished runs of ?rollout (bound by the caller).
RUN_LIGHTS_QUERY = """
SELECT ?run ?index ?status ?name ?i ?t WHERE {
    ?rollout cp:hasRun ?run .
    ?run cp:index ?index ; cp:status ?status ; cp:hasLightCondition ?light .
    FILTER (?status IN ("succeeded", "failed"))
    ?light cp:name ?name ; cp:lightIntensity ?i ; cp:colorTemperature ?t .
}
"""


@dataclass
class Requirement:
    iri: URIRef
    light: str
    i0: Decimal
    i1: Decimal
    t0: Decimal
    t1: Decimal
    ni: int
    nt: int
    min_episodes: int

    @property
    def valid(self) -> bool:
        return self.i0 < self.i1 and self.t0 < self.t1 and self.ni >= 1 and self.nt >= 1

    def lattice(self, intensity: Decimal, temperature: Decimal) -> tuple[int, int]:
        return lattice_index(intensity, self.i0, self.i1, self.ni), lattice_index(temperature, self.t0, self.t1, self.nt)

    def inside(self, i: int, j: int) -> bool:
        return 0 <= i < self.ni and 0 <= j < self.nt

    def bounds(self, i: int, j: int) -> tuple[float, float, float, float]:
        return (
            float(cell_edge(self.i0, self.i1, self.ni, i)), float(cell_edge(self.i0, self.i1, self.ni, i + 1)),
            float(cell_edge(self.t0, self.t1, self.nt, j)), float(cell_edge(self.t0, self.t1, self.nt, j + 1)),
        )

    def cell_iri(self, k: int) -> URIRef:
        return URIRef(f"{self.iri}/cell/{k}")


def requirements(graph: Graph, dataset: URIRef) -> list[Requirement]:
    rows = graph.query(REQUIREMENTS_QUERY, initNs={"cp": CP}, initBindings={"dataset": dataset})
    result = [
        Requirement(row.req, str(row.name), dec(row.i0), dec(row.i1), dec(row.t0), dec(row.t1), int(row.ni), int(row.nt), int(row.min))
        for row in rows
    ]
    return sorted(result, key=lambda r: r.light)


def required_lights(graph: Graph, dataset: URIRef) -> set[str]:
    return {r.light for r in requirements(graph, dataset)}


def episode_lights(graph: Graph, dataset: URIRef) -> dict[str, dict[int, tuple[Decimal, Decimal]]]:
    """light name -> episode index -> (intensity, color temperature)."""
    lights: dict[str, dict[int, tuple[Decimal, Decimal]]] = {}
    for episode in graph.objects(dataset, CP.hasEpisode):
        index = int(graph.value(episode, CP["index"]))
        for light in graph.objects(episode, CP.hasLightCondition):
            i, t = graph.value(light, CP.lightIntensity), graph.value(light, CP.colorTemperature)
            if i is not None and t is not None:
                lights.setdefault(str(graph.value(light, CP.name)), {})[index] = (dec(i), dec(t))
    return lights


def cell_counts(req: Requirement, values: dict[int, tuple[Decimal, Decimal]]) -> Counter:
    """Cell index (row-major, k = j * ni + i) -> number of episodes in it."""
    counts: Counter = Counter()
    if not req.valid:
        return counts
    for intensity, temperature in values.values():
        i, j = req.lattice(intensity, temperature)
        if req.inside(i, j):
            counts[j * req.ni + i] += 1
    return counts


# ---- Dataset check -----------------------------------------------------------


@dataclass
class LightCoverage:
    light: str
    req: Requirement
    counts: list[list[int]]
    """counts[j][i]: episodes in intensity bin i, color temperature bin j."""

    @property
    def cells_covered(self) -> int:
        return sum(n >= self.req.min_episodes for row in self.counts for n in row)

    @property
    def cells_total(self) -> int:
        return self.req.ni * self.req.nt


@dataclass
class LightingCheck(AxisCheck):
    coverage: list[LightCoverage] = field(default_factory=list)

    def lines(self) -> list[str]:
        lines = []
        for table in self.coverage:
            r = table.req
            lines.append(
                f"[{table.light}] intensity={float(r.i0):g}..{float(r.i1):g}  temperature={float(r.t0):g}..{float(r.t1):g} K  "
                f"covered {table.cells_covered}/{table.cells_total} cells (>= {r.min_episodes} each)"
            )
            lines.append("")
        return lines

    def tables(self) -> list[ReportTable]:
        rows = []
        for t in self.coverage:
            r = t.req
            rows.append(
                [
                    t.light, f"{float(r.i0):g} .. {float(r.i1):g}", f"{float(r.t0):g} .. {float(r.t1):g}", f"{r.ni} x {r.nt}",
                    f">= {r.min_episodes}", f"{t.cells_covered}/{t.cells_total}", "met" if t.cells_covered == t.cells_total else "NOT met",
                ]
            )
        return [ReportTable("Image Augmentations: lighting coverage", ["Light", "Intensity", "Temperature (K)", "Grid", "Episodes per cell", "Cells covered", "Status"], rows)] if rows else []


def check_lighting(graph: Graph, dataset: URIRef, axis: str) -> LightingCheck:
    """Episodes per cell; a Violation per cell below the minimum; the cells as derived nodes."""
    result = LightingCheck(axis)
    lights = episode_lights(graph, dataset)
    for req in requirements(graph, dataset):
        if not req.valid:
            continue  # the SHACL requirement shape reports it
        counts = cell_counts(req, lights.get(req.light, {}))
        table = LightCoverage(req.light, req, [[counts[j * req.ni + i] for i in range(req.ni)] for j in range(req.nt)])
        result.coverage.append(table)
        for k in range(req.ni * req.nt):
            i, j = k % req.ni, k // req.ni
            lo_i, hi_i, lo_t, hi_t = req.bounds(i, j)
            cell = req.cell_iri(k)
            for p, o in (
                (RDF.type, CP.LightingCell), (CP.ofRequirement, req.iri), (CP["index"], index_literal(k)),
                (CP.cellI, Literal(i, datatype=XSD.integer)), (CP.cellJ, Literal(j, datatype=XSD.integer)),
                (CP.cellIntensityMin, _decimal(lo_i)), (CP.cellIntensityMax, _decimal(hi_i)),
                (CP.cellColorTemperatureMin, _decimal(lo_t)), (CP.cellColorTemperatureMax, _decimal(hi_t)),
                (CP.episodeCount, index_literal(counts[k])),
            ):
                result.derived.add((cell, p, o))
            if counts[k] < req.min_episodes:
                result.findings.append(
                    Finding(
                        "Violation",
                        f"'{req.light}' cell (i={i}, j={j}) intensity=[{lo_i:g}, {hi_i:g}] temperature=[{lo_t:g}, {hi_t:g}] K "
                        f"has {counts[k]} episode(s); needs at least {req.min_episodes}",
                        [str(cell)],
                        axis,
                    )
                )
    return result


# ---- Rollout comparison ------------------------------------------------------


@dataclass
class LightRunCell:
    i: int
    j: int
    inside: bool
    episodes: int
    covered: bool
    runs: int = 0
    successes: int = 0


@dataclass
class LightComparison:
    light: str
    req: Requirement
    cells: dict[tuple[int, int], LightRunCell] = field(default_factory=dict)
    """Lattice cells that runs started in."""
    counts: Counter = field(default_factory=Counter)
    """Dataset episodes per cell of the requirement grid (row-major index), for every cell."""

    @property
    def runs(self) -> int:
        return sum(c.runs for c in self.cells.values())

    @property
    def runs_covered(self) -> int:
        return sum(c.runs for c in self.cells.values() if c.covered)

    @property
    def successes_covered(self) -> int:
        return sum(c.successes for c in self.cells.values() if c.covered)

    @property
    def successes_not_covered(self) -> int:
        return sum(c.successes for c in self.cells.values() if not c.covered)


@dataclass
class LightingComparison(AxisComparison):
    lights: list[LightComparison] = field(default_factory=list)

    def lines(self) -> list[str]:
        lines = []
        for light in self.lights:
            r = light.req
            lines.append(
                f"[{light.light}] intensity={float(r.i0):g}..{float(r.i1):g}  temperature={float(r.t0):g}..{float(r.t1):g} K, "
                f"{r.ni}x{r.nt} cells, >= {r.min_episodes} episodes each"
            )
            outside = sum(c.runs for c in light.cells.values() if not c.inside)
            if outside:
                lines.append(f"  {outside} run(s) started with the light outside your ranges")
            not_covered = light.runs - light.runs_covered
            lines.append(
                f"  covered: {light.runs_covered}/{light.runs} runs ({pct(light.runs_covered, light.runs)})\n"
                f"  success: {rate(light.successes_covered, light.runs_covered)} covered vs {rate(light.successes_not_covered, not_covered)} not covered"
            )
            lines.append("")
        return lines

    def tables(self) -> list[ReportTable]:
        rows, notes = [], []
        for light in self.lights:
            r = light.req
            outside = sum(c.runs for c in light.cells.values() if not c.inside)
            if outside:
                notes.append(f"{light.light}: {outside} run(s) started with the light outside your ranges")
            rows.append(
                [
                    light.light, f"{float(r.i0):g} .. {float(r.i1):g}", f"{float(r.t0):g} .. {float(r.t1):g}", f"{r.ni} x {r.nt}", f">= {r.min_episodes}",
                    f"{light.runs_covered}/{light.runs} ({pct(light.runs_covered, light.runs)})",
                    rate(light.successes_covered, light.runs_covered), rate(light.successes_not_covered, light.runs - light.runs_covered),
                ]
            )
        columns = ["Light", "Intensity", "Temperature (K)", "Grid", "Episodes per cell", "Runs covered", "Success covered", "Success not covered"]
        return [ReportTable("Image Augmentations: rollout runs", columns, rows, notes)] if rows else []

    def hint_lines(self) -> list[str]:
        lines = []
        for light in self.lights:
            r = light.req
            for c in sorted(light.cells.values(), key=lambda c: (-(c.runs - c.successes), c.j, c.i)):
                if c.covered:
                    continue
                if c.inside:
                    lo_i, hi_i, lo_t, hi_t = r.bounds(c.i, c.j)
                    lines.append(
                        f"  {light.light}: record more episodes with intensity {lo_i:g}..{hi_i:g} and {lo_t:g}..{hi_t:g} K "
                        f"(has {c.episodes}, needs {r.min_episodes}; {c.runs} run(s), {c.runs - c.successes} failed)"
                    )
                else:
                    lines.append(f"  {light.light}: {c.runs} run(s) started outside your ranges; widen the ranges in the spec and in the task's light randomization")
        return (["Lighting: where to record more episodes:", *lines, ""]) if lines else []


def compare_lighting(graph: Graph, rollout: URIRef, dataset: URIRef, comparison: URIRef, axis: str) -> LightingComparison:
    result = LightingComparison(axis, {})
    reqs = [r for r in requirements(graph, dataset) if r.valid]
    if not reqs:
        result.errors.append("The spec has no valid lighting requirement; nothing to compare")
        return result
    lights = episode_lights(graph, dataset)
    runs: dict[str, dict[int, tuple[Decimal, Decimal]]] = {}
    outcome: dict[int, bool] = {}
    nodes: dict[int, URIRef] = {}
    for row in graph.query(RUN_LIGHTS_QUERY, initNs={"cp": CP}, initBindings={"rollout": rollout}):
        index = int(row["index"])
        outcome[index], nodes[index] = str(row.status) == "succeeded", row.run
        runs.setdefault(str(row.name), {})[index] = (dec(row.i), dec(row.t))
    result.run_covered = {index: True for index in outcome}
    for req in reqs:
        counts = cell_counts(req, lights.get(req.light, {}))
        comparison_light = LightComparison(req.light, req, counts=counts)
        for index, (intensity, temperature) in sorted(runs.get(req.light, {}).items()):
            i, j = req.lattice(intensity, temperature)
            inside = req.inside(i, j)
            episodes = counts[j * req.ni + i] if inside else 0
            cell = comparison_light.cells.setdefault((i, j), LightRunCell(i, j, inside, episodes, inside and episodes >= req.min_episodes))
            cell.runs += 1
            cell.successes += outcome[index]
            if not cell.covered:
                result.run_covered[index] = False
            placement = URIRef(f"{comparison}/lighting/{nodes[index].rsplit('/', 1)[-1]}/{req.light}")
            result.derived.add((comparison, CP.hasPlacement, placement))
            for p, o in (
                (RDF.type, CP.LightingPlacement), (CP.ofRun, nodes[index]), (CP.ofRequirement, req.iri),
                (CP.cellI, Literal(i, datatype=XSD.integer)), (CP.cellJ, Literal(j, datatype=XSD.integer)),
                (CP.covered, Literal(cell.covered)),
            ):
                result.derived.add((placement, p, o))
            if inside:
                result.derived.add((placement, CP.inLightingCell, req.cell_iri(j * req.ni + i)))
        result.lights.append(comparison_light)
        not_covered = [c for c in comparison_light.cells.values() if not c.covered]
        if not_covered:
            n = sum(c.runs for c in not_covered)
            result.findings.append(
                Finding(
                    "Warning",
                    f"'{req.light}': {n} run(s) started with lighting the dataset does not cover "
                    f"({n - sum(c.successes for c in not_covered)} failed)",
                    [f"{req.light}/cell/({c.i}, {c.j})" for c in not_covered],
                    axis,
                )
            )
    return result


def _decimal(value: float) -> Literal:
    return Literal(Decimal(str(round(value, 6))), datatype=XSD.decimal)
