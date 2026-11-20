"""Object color axis (STAR-Gen: Semantic): are the objects shown in the colors the student requires, evenly, and named in the task text?

Files:
  ontology.ttl        ObjectColorAxis, its requirement, color categories, object colors, run placements
  shapes.ttl          SHACL rules on the dataset (every episode has the color, spec well-formed)
  rollout_shapes.ttl  SHACL rules on a rollout (every finished run has the color)
  coverage.py         categories, counts, language check (verify) and run groups (compare)
"""

from __future__ import annotations

import math
from pathlib import Path

from rdflib import Graph, Literal, URIRef

from ...dataset import OBJECT_COLOR_KEY_PREFIX, OBJECT_COLOR_NAMES, read_first_frame_vectors
from ...graph import CP, add_object_color
from ...rollout_log import RolloutLog
from ...spec_template import number, todo_fields, value_type_errors
from ...verifier import AxisCheck, AxisComparison, Verifier
from .coverage import check_object_color, compare_object_color, required_objects

_NUMBER_FIELDS = [CP.maxColorDistance, CP.colorR, CP.colorG, CP.colorB]
_WHOLE_FIELDS = [CP.minEpisodesPerColor]


def _owner(spec: Graph, node) -> str | None:
    """Object name of the requirement that owns ``node`` (requirement or color category)."""
    req = node if (node, None, CP.ObjectColorAxisRequirement) in spec else spec.value(None, CP.hasColorCategory, node)
    obj = spec.value(req, CP.hasTaskObject) if req is not None else None
    name = spec.value(obj, CP.name) if obj is not None else None
    if name is None:
        return None
    category = spec.value(node, CP.name) if req is not node else None
    return f"{name}" + (f" color '{category}'" if category is not None and str(category) != "TODO" else "")


class ObjectColorVerifier(Verifier):
    name = "object_color"
    axis_class = CP.ObjectColorAxis
    package = __name__

    def recorded_names(self, graph: Graph, dataset: URIRef) -> list[str]:
        return sorted(str(n) for n in graph.objects(dataset, CP.recordedColorObject))

    def ingest_dataset(self, root: Path, graph: Graph, dataset: URIRef) -> list[str]:
        values = read_first_frame_vectors(root, OBJECT_COLOR_KEY_PREFIX, len(OBJECT_COLOR_NAMES))
        errors = []
        for obj, episodes in sorted(values.items()):
            graph.add((dataset, CP.recordedColorObject, Literal(obj)))
            for episode_index, rgb in sorted(episodes.items()):
                if not all(math.isfinite(v) for v in rgb):
                    errors.append(f"Episode {episode_index}: non-finite color for '{obj}'")
                    continue
                add_object_color(graph, URIRef(f"{dataset}/episode/{episode_index}"), obj, rgb)
        return errors

    def ingest_rollout(self, log: RolloutLog, graph: Graph, rollout: URIRef) -> list[str]:
        for run in log.runs:
            for obj, rgb in run.object_colors.items():
                add_object_color(graph, URIRef(f"{rollout}/run/{run.index}"), obj, rgb)
        return []

    def spec_errors(self, spec: Graph) -> list[str]:
        if not any(spec.subjects(None, CP.ObjectColorAxis)):
            return []
        todo = todo_fields(spec, lambda n: _owner(spec, n))
        if todo:
            return ["Spec still has TODO values: " + ", ".join(todo)]
        errors = value_type_errors(spec, _NUMBER_FIELDS, _WHOLE_FIELDS, lambda n: _owner(spec, n))
        if errors:
            return errors
        for req in spec.subjects(None, CP.ObjectColorAxisRequirement):
            name = _owner(spec, req) or req
            distance, minimum = (number(spec, req, p) for p in (CP.maxColorDistance, CP.minEpisodesPerColor))
            if distance is not None and distance <= 0:
                errors.append(f"{name}: maxColorDistance must be > 0, got {distance:g}")
            if minimum is not None and minimum < 1:
                errors.append(f"{name}: minEpisodesPerColor must be at least 1, got {minimum:g}")
            if not any(spec.objects(req, CP.hasColorCategory)):
                errors.append(f"{name}: no color category (cp:hasColorCategory)")
        if not errors and not any(spec.subjects(None, CP.ObjectColorAxisRequirement)):
            errors.append("Spec declares no cp:ObjectColorAxisRequirement; nothing to check")
        return sorted(errors)

    def dataset_errors(self, graph: Graph, dataset: URIRef) -> list[str]:
        recorded = set(self.recorded_names(graph, dataset))
        return [
            f"Dataset has no 'object_color.{name}' column, which the spec requires (record with the repository recorder and a task whose object_color_palettes lists it)"
            for name in sorted(required_objects(graph, dataset) - recorded)
        ]

    def check(self, graph: Graph, dataset: URIRef) -> AxisCheck:
        return check_object_color(graph, dataset, self.name)

    def compare(self, graph: Graph, rollout: URIRef, dataset: URIRef, comparison: URIRef) -> AxisComparison:
        return compare_object_color(graph, rollout, dataset, comparison, self.name)
