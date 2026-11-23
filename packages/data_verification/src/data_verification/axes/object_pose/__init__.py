"""Object Poses axis (STAR-Gen VB-POSE), factor object position (STAR-Gen: manipulated object pose; position only): are the objects' initial positions spread over the area the spec requires?

Files:
  ontology.ttl        ObjectPoseAxis, its requirement, area, grid, cells, run placements
  shapes.ttl          SHACL rules on the dataset (cell coverage, area, spec well-formed)
  rollout_shapes.ttl  SHACL rules on a rollout (every finished run has the required positions)
  spec.py             checks that stop ``verify`` for a spec (TODO left, values not numbers, empty area)
  coverage.py         cells (spec load), per-cell episode counts and tables (verify)
  compare.py          run placements, coverage of rollout runs, hints (verify)
"""

from __future__ import annotations

from rdflib import Graph, URIRef

from ...graph import CP
from ...verifier import AxisCheck, AxisComparison, Verifier
from .compare import compare_object_position
from .coverage import check_coverage, expand_cells, required_names
from .spec import todo_fields, value_range_errors, value_type_errors


class ObjectPoseVerifier(Verifier):
    name = "object_pose"
    axis_class = CP.ObjectPoseAxis
    package = __name__

    def spec_errors(self, spec: Graph) -> list[str]:
        todo = todo_fields(spec)
        if todo:
            return ["Spec still has TODO values: " + ", ".join(todo)]
        errors = value_type_errors(spec) or value_range_errors(spec)
        if not errors and any(spec.subjects(None, CP.ObjectPoseAxis)) and not any(
            spec.subjects(None, CP.ObjectPositionRequirement)
        ):
            errors = ["Spec declares no cp:ObjectPositionRequirement; nothing to check"]
        return errors

    def derive_spec(self, spec: Graph) -> None:
        expand_cells(spec)

    def dataset_errors(self, graph: Graph, dataset: URIRef) -> list[str]:
        recorded = {str(n) for n in graph.objects(dataset, CP.recordedObject)}
        return [
            f"Dataset has no 'object_position.{name}' column, which the spec requires"
            for name in sorted(required_names(graph, dataset) - recorded)
        ]

    def check(self, graph: Graph, dataset: URIRef) -> AxisCheck:
        return check_coverage(graph, dataset, self.name)

    def compare(self, graph: Graph, rollout: URIRef, dataset: URIRef, comparison: URIRef) -> AxisComparison:
        return compare_object_position(graph, rollout, dataset, comparison, self.name)
