"""Image Augmentations axis (STAR-Gen V-AUG), factor lighting: is the scene light varied over the intensity and color temperature the spec requires?

Files:
  ontology.ttl        ImageAugmentationAxis, its requirement, light conditions, cells, run placements
  shapes.ttl          SHACL rules on the dataset (every episode has the light, spec well-formed)
  rollout_shapes.ttl  SHACL rules on a rollout (every finished run has the light)
  coverage.py         per-cell episode counts (verify) and run placements (compare)
"""

from __future__ import annotations

import math
from pathlib import Path

from rdflib import Graph, Literal, URIRef

from ...dataset import LIGHTING_KEY_PREFIX, LIGHTING_NAMES, read_first_frame_vectors
from ...graph import CP, add_light_condition
from ...rollout_log import RolloutLog
from ...spec_template import number, todo_fields, value_type_errors
from ...verifier import AxisCheck, AxisComparison, Verifier
from .coverage import check_lighting, compare_lighting, required_lights

_NUMBER_FIELDS = [CP.intensityMin, CP.intensityMax, CP.colorTemperatureMin, CP.colorTemperatureMax]
_WHOLE_FIELDS = [CP.numBinsIntensity, CP.numBinsColorTemperature, CP.minEpisodesPerCell]


class ImageAugmentationVerifier(Verifier):
    name = "image_augmentation"
    axis_class = CP.ImageAugmentationAxis
    package = __name__

    def recorded_names(self, graph: Graph, dataset: URIRef) -> list[str]:
        return sorted(str(n) for n in graph.objects(dataset, CP.recordedLight))

    def ingest_dataset(self, root: Path, graph: Graph, dataset: URIRef) -> list[str]:
        values = read_first_frame_vectors(root, LIGHTING_KEY_PREFIX, len(LIGHTING_NAMES))
        errors = []
        for light, episodes in sorted(values.items()):
            graph.add((dataset, CP.recordedLight, Literal(light)))
            for episode_index, (intensity, temperature) in sorted(episodes.items()):
                if not (math.isfinite(intensity) and math.isfinite(temperature)):
                    errors.append(f"Episode {episode_index}: non-finite lighting for '{light}'")
                    continue
                add_light_condition(graph, URIRef(f"{dataset}/episode/{episode_index}"), light, intensity, temperature)
        return errors

    def ingest_rollout(self, log: RolloutLog, graph: Graph, rollout: URIRef) -> list[str]:
        for run in log.runs:
            for light, (intensity, temperature) in run.lighting.items():
                add_light_condition(graph, URIRef(f"{rollout}/run/{run.index}"), light, intensity, temperature)
        return []

    def spec_errors(self, spec: Graph) -> list[str]:
        if not any(spec.subjects(None, CP.ImageAugmentationAxis)):
            return []
        todo = todo_fields(spec, lambda n: spec.value(n, CP.lightName))
        if todo:
            return ["Spec still has TODO values: " + ", ".join(todo)]
        errors = value_type_errors(spec, _NUMBER_FIELDS, _WHOLE_FIELDS, lambda n: spec.value(n, CP.lightName))
        if errors:
            return errors
        for req in spec.subjects(None, CP.LightingRequirement):
            name = spec.value(req, CP.lightName) or req
            i0, i1, t0, t1 = (number(spec, req, p) for p in _NUMBER_FIELDS)
            if None not in (i0, i1) and not i0 < i1:
                errors.append(f"{name}: intensityMin must be < intensityMax, got {i0:g}..{i1:g}")
            if None not in (t0, t1) and not t0 < t1:
                errors.append(f"{name}: colorTemperatureMin must be < colorTemperatureMax, got {t0:g}..{t1:g}")
            for field in _WHOLE_FIELDS:
                value = number(spec, req, field)
                if value is not None and value < 1:
                    errors.append(f"{name}: {field.split('#')[-1]} must be at least 1, got {value:g}")
        if not errors and not any(spec.subjects(None, CP.LightingRequirement)):
            errors.append("Spec declares no cp:LightingRequirement; nothing to check")
        return sorted(errors)

    def dataset_errors(self, graph: Graph, dataset: URIRef) -> list[str]:
        recorded = set(self.recorded_names(graph, dataset))
        return [
            f"Dataset has no 'lighting.{name}' column, which the spec requires (record with the repository recorder and a task that lists it in recorded_light_names)"
            for name in sorted(required_lights(graph, dataset) - recorded)
        ]

    def check(self, graph: Graph, dataset: URIRef) -> AxisCheck:
        return check_lighting(graph, dataset, self.name)

    def compare(self, graph: Graph, rollout: URIRef, dataset: URIRef, comparison: URIRef) -> AxisComparison:
        return compare_lighting(graph, rollout, dataset, comparison, self.name)
