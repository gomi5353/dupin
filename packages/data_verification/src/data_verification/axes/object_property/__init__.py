"""Object Properties axis (STAR-Gen S-PROP): does the dataset refer to each object in the ways the spec requires?

Only the task text changes; the objects never do. Factor: referencing objects
based on color ("the pink object" instead of "the fork").

Files:
  ontology.ttl        ObjectPropertyAxis, its color reference requirement, references, object colors, run placements
  shapes.ttl          SHACL rules on the dataset (every episode has the color and a task text, spec well-formed)
  rollout_shapes.ttl  SHACL rules on a rollout (every finished run has its instruction)
  coverage.py         references per episode, counts, unchanged-object check (verify) and run groups (compare)
"""

from __future__ import annotations

import math
from pathlib import Path

from rdflib import RDF, Graph, Literal, URIRef

from ...dataset import OBJECT_COLOR_KEY_PREFIX, OBJECT_COLOR_NAMES, read_first_frame_vectors
from ...graph import CP, add_object_color
from ...spec_template import number, todo_fields, value_type_errors
from ...verifier import AxisCheck, AxisComparison, Verifier
from .coverage import check_object_property, compare_object_property, required_objects

_WHOLE_FIELDS = [CP.minEpisodesPerReference]


def _owner(spec: Graph, node) -> str | None:
    """Object name of the requirement that owns ``node`` (requirement or reference)."""
    req = node if (node, None, CP.ColorReferenceRequirement) in spec else spec.value(None, CP.hasReference, node)
    obj = spec.value(req, CP.hasTaskObject) if req is not None else None
    name = spec.value(obj, CP.name) if obj is not None else None
    if name is None:
        return None
    for kind, cls in (("name", CP.NameReference), ("color", CP.ColorReference)):
        if node != req and (node, RDF.type, cls) in spec:
            return f"{name} {kind} reference"
    return str(name)


class ObjectPropertyVerifier(Verifier):
    name = "object_property"
    axis_class = CP.ObjectPropertyAxis
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

    def spec_errors(self, spec: Graph) -> list[str]:
        if not any(spec.subjects(None, CP.ObjectPropertyAxis)):
            return []
        todo = todo_fields(spec, lambda n: _owner(spec, n))
        if todo:
            return ["Spec still has TODO values: " + ", ".join(todo)]
        errors = value_type_errors(spec, [], _WHOLE_FIELDS, lambda n: _owner(spec, n))
        if errors:
            return errors
        for req in spec.subjects(None, CP.ColorReferenceRequirement):
            name = _owner(spec, req) or req
            minimum = number(spec, req, CP.minEpisodesPerReference)
            if minimum is not None and minimum < 1:
                errors.append(f"{name}: minEpisodesPerReference must be at least 1, got {minimum:g}")
            if not any(spec.objects(req, CP.hasReference)):
                errors.append(f"{name}: no reference (cp:hasReference)")
            for ref in spec.objects(req, CP.hasReference):
                own = number(spec, ref, CP.minEpisodesPerReference)
                if own is not None and own < 1:
                    errors.append(f"{_owner(spec, ref)}: minEpisodesPerReference must be at least 1, got {own:g}")
        if not errors and not any(spec.subjects(None, CP.ColorReferenceRequirement)):
            errors.append("Spec declares no cp:ColorReferenceRequirement; nothing to check")
        return sorted(errors)

    def dataset_errors(self, graph: Graph, dataset: URIRef) -> list[str]:
        recorded = set(self.recorded_names(graph, dataset))
        return [
            f"Dataset has no 'object_color.{name}' column, which the spec requires to check that the object never changes "
            "(record with the repository recorder and a task whose object_references lists it)"
            for name in sorted(required_objects(graph, dataset) - recorded)
        ]

    def check(self, graph: Graph, dataset: URIRef) -> AxisCheck:
        return check_object_property(graph, dataset, self.name)

    def compare(self, graph: Graph, rollout: URIRef, dataset: URIRef, comparison: URIRef) -> AxisComparison:
        return compare_object_property(graph, rollout, dataset, comparison, self.name)
