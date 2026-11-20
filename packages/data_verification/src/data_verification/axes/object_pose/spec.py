"""Object position spec: the checks that stop ``verify`` for a spec."""

from __future__ import annotations

from decimal import Decimal

from ...graph import CP

def todo_fields(graph) -> list[str]:
    """'<object>: <property>' for every value still set to "TODO"."""
    fields = []
    for subject, predicate, value in graph:
        if str(value) == "TODO":
            fields.append(f"{requirement_object_name(graph, subject) or subject}: {predicate.split('#')[-1]}")
    return sorted(fields)


def requirement_object_name(graph, node) -> str | None:
    """Name of the task object whose requirement owns ``node`` (requirement, area or grid)."""
    if (None, CP.hasCoverageGrid, node) in graph:
        node = graph.value(None, CP.hasCoverageGrid, node)
    if (None, CP.hasAreaConfiguration, node) in graph:
        node = graph.value(None, CP.hasAreaConfiguration, node)
    obj = graph.value(node, CP.hasTaskObject)
    name = graph.value(obj, CP.name) if obj is not None else None
    return None if name is None else str(name)


_NUMBER_FIELDS = ("areaXMin", "areaXMax", "areaYMin", "areaYMax")
_WHOLE_NUMBER_FIELDS = ("numCellW", "numCellH", "minEpisodesPerCell")


def value_type_errors(graph) -> list[str]:
    """Requirement, area and grid values that are not numbers (e.g. still quoted)."""
    errors = []
    for field in _NUMBER_FIELDS + _WHOLE_NUMBER_FIELDS:
        for subject, value in graph.subject_objects(CP[field]):
            python_value = value.toPython()
            whole = field in _WHOLE_NUMBER_FIELDS
            ok = isinstance(python_value, int) if whole else isinstance(python_value, (int, float, Decimal))
            if isinstance(python_value, bool) or not ok:
                kind = "a whole number" if whole else "a number"
                shown = f'"{value}"' if isinstance(python_value, str) else str(value)
                name = requirement_object_name(graph, subject) or subject
                errors.append(f"{name}: {field} must be {kind} without quotes, got {shown}")
    return sorted(errors)


def value_range_errors(graph) -> list[str]:
    """Areas, grids and minimums that are numbers but cannot describe a grid (e.g. swapped min/max)."""
    errors = []
    for area in set(graph.subjects(CP.areaXMin, None)) | set(graph.subjects(CP.areaXMax, None)):
        x0, x1, y0, y1 = (graph.value(area, CP[f]) for f in _NUMBER_FIELDS)
        if None in (x0, x1, y0, y1):
            continue  # missing fields are reported by the SHACL shapes
        x0, x1, y0, y1 = (v.toPython() for v in (x0, x1, y0, y1))
        if not (x0 < x1 and y0 < y1):
            name = requirement_object_name(graph, area) or area
            errors.append(
                f"{name}: coverage area is empty: areaXMin must be < areaXMax and areaYMin < areaYMax, "
                f"got x {x0}..{x1}, y {y0}..{y1}"
            )
    for field, low in (("numCellW", 1), ("numCellH", 1), ("minEpisodesPerCell", 1)):
        for subject, value in graph.subject_objects(CP[field]):
            if value.toPython() < low:
                name = requirement_object_name(graph, subject) or subject
                errors.append(f"{name}: {field} must be at least {low}, got {value}")
    return sorted(errors)
