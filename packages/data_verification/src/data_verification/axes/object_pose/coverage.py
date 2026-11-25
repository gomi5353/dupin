"""Object Poses axis, factor object position, dataset side: grid cells, per-cell episode counts and coverage tables."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal

from rdflib import RDF, XSD, Graph, Literal, URIRef

from ...graph import CP, index_literal
from ...grid import cell_edge, dec, lattice_index  # noqa: F401  (re-exported for compare.py)
from ...verifier import AxisCheck

AREA_BOUNDS = (CP.areaXMin, CP.areaXMax, CP.areaYMin, CP.areaYMax)

# The requirements of ?dataset (bound by the caller), through its axes.
REQUIREMENTS_QUERY = """
SELECT ?req ?name ?grid ?x0 ?x1 ?y0 ?y1 ?nx ?ny ?min WHERE {
    ?dataset cp:hasGeneralizationAxis ?axis .
    ?axis cp:hasRequirement ?req .
    ?req a cp:ObjectPositionRequirement ; cp:hasTaskObject ?reqObj ;
         cp:minEpisodesPerCell ?min ; cp:hasAreaConfiguration ?area .
    ?reqObj cp:name ?name .
    ?area cp:areaXMin ?x0 ; cp:areaXMax ?x1 ; cp:areaYMin ?y0 ; cp:areaYMax ?y1 ;
          cp:hasCoverageGrid ?grid .
    ?grid cp:numCellW ?nx ; cp:numCellH ?ny .
}
"""

REQUIRED_NAMES_QUERY = """
SELECT DISTINCT ?name WHERE {
    ?dataset cp:hasGeneralizationAxis ?axis .
    ?axis cp:hasRequirement ?req .
    ?req a cp:ObjectPositionRequirement ; cp:hasTaskObject ?reqObj .
    ?reqObj cp:name ?name .
}
"""


def expand_cells(graph: Graph) -> None:
    """Store area bounds as decimals and add a cp:Cell per grid cell, indexed row-major, with its bounds.

    Grids with a missing or invalid numCellW/numCellH are skipped here; the
    SHACL grid shape reports them.
    """
    # Positions are decimals rounded to 6 places; a bound written as a double
    # (8.8e0) would compare against them as 8.800000000000000710.
    for area in set(graph.subjects(RDF.type, CP.AreaConfiguration)):
        for prop in AREA_BOUNDS:
            values = list(graph.objects(area, prop))
            value = values[0].toPython() if len(values) == 1 else None
            if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
                graph.set((area, prop, Literal(Decimal(str(value)), datatype=XSD.decimal)))

    for grid in set(graph.subjects(RDF.type, CP.CoverageGrid)):
        try:
            nw, nh = int(graph.value(grid, CP.numCellW)), int(graph.value(grid, CP.numCellH))
        except (TypeError, ValueError):
            continue
        if nw < 1 or nh < 1:
            continue
        area = graph.value(None, CP.hasCoverageGrid, grid)
        bounds = [graph.value(area, p) for p in AREA_BOUNDS] if area is not None else [None]
        base = str(grid) if isinstance(grid, URIRef) else f"urn:cp:grid:{grid}"
        for k in range(nw * nh):
            cell = URIRef(f"{base}/cell/{k}")
            graph.add((cell, RDF.type, CP.Cell))
            graph.add((grid, CP.hasCell, cell))
            graph.add((cell, CP["index"], index_literal(k)))
            i, j = k % nw, k // nw
            graph.add((cell, CP.cellI, Literal(i, datatype=XSD.integer)))
            graph.add((cell, CP.cellJ, Literal(j, datatype=XSD.integer)))
            if None not in bounds:
                x0, x1, y0, y1 = (dec(v) for v in bounds)
                if x0 < x1 and y0 < y1:
                    for prop, value in (
                        (CP.cellXMin, cell_edge(x0, x1, nw, i)), (CP.cellXMax, cell_edge(x0, x1, nw, i + 1)),
                        (CP.cellYMin, cell_edge(y0, y1, nh, j)), (CP.cellYMax, cell_edge(y0, y1, nh, j + 1)),
                    ):
                        graph.add((cell, prop, Literal(value, datatype=XSD.decimal)))


def required_names(graph: Graph, dataset: URIRef) -> set[str]:
    return {str(r.name) for r in graph.query(REQUIRED_NAMES_QUERY, initNs={"cp": CP}, initBindings={"dataset": dataset})}


def requirement_rows(graph: Graph, dataset: URIRef):
    """One row per object position requirement of ``dataset``: req, name, grid, area bounds, nx, ny, min."""
    return graph.query(REQUIREMENTS_QUERY, initNs={"cp": CP}, initBindings={"dataset": dataset})


def cell_counts(graph: Graph, dataset: URIRef) -> dict[str, Counter]:
    """Requirement IRI -> Counter of ``dataset``'s episodes per cell index.

    Same cell bounds and membership rule as cp:CellCoverageShape, counted in
    Python: the SPARQL form takes tens of seconds on a 100-episode dataset.
    """
    positions: dict[str, list[tuple[Decimal, Decimal]]] = {}
    for episode in graph.objects(dataset, CP.hasEpisode):
        for obj in graph.objects(episode, CP.hasTaskObject):
            p = graph.value(obj, CP.hasPosition)
            x, y = graph.value(p, CP.positionX), graph.value(p, CP.positionY)
            if x is not None and y is not None:
                positions.setdefault(str(graph.value(obj, CP.name)), []).append((dec(x), dec(y)))
    counts: dict[str, Counter] = {}
    for row in requirement_rows(graph, dataset):
        x0, x1, y0, y1 = (dec(v) for v in (row.x0, row.x1, row.y0, row.y1))
        nw, nh = int(row.nx), int(row.ny)
        counter = counts.setdefault(str(row.req), Counter())
        if not (x0 < x1 and y0 < y1 and nw >= 1 and nh >= 1):
            continue
        for x, y in positions.get(str(row.name), []):
            i, j = lattice_index(x, x0, x1, nw), lattice_index(y, y0, y1, nh)
            if 0 <= i < nw and 0 <= j < nh:
                counter[j * nw + i] += 1
    return counts


@dataclass
class ObjectCoverage:
    object_name: str
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    min_episodes_per_cell: int
    counts: list[list[int]]
    """counts[j][i]: episodes in column i (x), row j (y)."""

    @property
    def cells_covered(self) -> int:
        return sum(n >= self.min_episodes_per_cell for row in self.counts for n in row)

    @property
    def cells_total(self) -> int:
        return sum(len(row) for row in self.counts)


@dataclass
class ObjectPositionCheck(AxisCheck):
    coverage: list[ObjectCoverage] = field(default_factory=list)

    def lines(self) -> list[str]:
        lines = []
        for table in self.coverage:
            lines.append(
                f"[{table.object_name}] x={table.x_range[0]:.3f}..{table.x_range[1]:.3f}  "
                f"y={table.y_range[0]:.3f}..{table.y_range[1]:.3f}  "
                f"covered {table.cells_covered}/{table.cells_total} cells (>= {table.min_episodes_per_cell} each)"
            )
            lines.append("")
        return lines


def check_coverage(graph: Graph, dataset: URIRef, axis: str) -> ObjectPositionCheck:
    """Coverage tables, and each cell's episode count as a derived triple."""
    counts = cell_counts(graph, dataset)
    result = ObjectPositionCheck(axis)
    for row in requirement_rows(graph, dataset):
        nx, ny = int(row.nx), int(row.ny)
        req_counts = counts.get(str(row.req), Counter())
        result.coverage.append(
            ObjectCoverage(
                object_name=str(row.name),
                x_range=(float(row.x0), float(row.x1)),
                y_range=(float(row.y0), float(row.y1)),
                min_episodes_per_cell=int(row.min),
                # Cell index is row-major: k = j * nx + i.
                counts=[[req_counts[j * nx + i] for i in range(nx)] for j in range(ny)],
            )
        )
        for cell in graph.objects(row.grid, CP.hasCell):
            k = graph.value(cell, CP["index"])
            if k is not None:
                result.derived.add((cell, CP.episodeCount, index_literal(req_counts[int(k)])))
    result.coverage.sort(key=lambda t: t.object_name)
    return result
