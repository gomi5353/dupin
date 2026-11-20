"""IRIs and triples for the facts that enter a store: a dataset with its episodes, a rollout with its runs."""

from __future__ import annotations

from decimal import Decimal
from urllib.parse import quote

from rdflib import RDF, XSD, Graph, Literal, Namespace, URIRef

from .dataset import DatasetPlacements
from .rollout_log import RolloutLog

CP = Namespace("http://taica.course/cp/ontology#")
DATA = "http://taica.course/cp/data/"


def dataset_iri(dataset_name: str) -> URIRef:
    return URIRef(f"{DATA}{quote(dataset_name, safe='')}")


def rollout_iri(dataset: URIRef, rollout_name: str) -> URIRef:
    return URIRef(f"{dataset}/rollout/{quote(rollout_name, safe='')}")


def spec_iri(dataset: URIRef, spec_name: str) -> URIRef:
    return URIRef(f"{dataset}/spec/{quote(spec_name, safe='')}")


def add_dataset(graph: Graph, dataset: URIRef, data: DatasetPlacements, frame_counts: dict[int, int] | None = None) -> None:
    """Add the dataset node, its episodes and the objects' finite initial positions."""
    graph.add((dataset, RDF.type, CP.Dataset))
    for name in data.object_names:
        graph.add((dataset, CP.recordedObject, Literal(name)))
    for episode_index in data.episode_indices:
        episode = URIRef(f"{dataset}/episode/{episode_index}")
        graph.add((episode, RDF.type, CP.Episode))
        graph.add((episode, CP["index"], index_literal(episode_index)))
        if frame_counts and episode_index in frame_counts:
            graph.add((episode, CP.frameCount, index_literal(frame_counts[episode_index])))
        graph.add((dataset, CP.hasEpisode, episode))
    for placement in data.placements:
        if not placement.is_finite:
            continue
        episode = URIRef(f"{dataset}/episode/{placement.episode_index}")
        add_task_object(graph, episode, placement.object_name, (placement.x, placement.y, placement.z))


def add_rollout(graph: Graph, log: RolloutLog, rollout: URIRef, dataset: URIRef) -> None:
    """Add the rollout of ``log`` with its training ``dataset``, its runs, their status, instruction and initial positions."""
    graph.add((rollout, RDF.type, CP.Rollout))
    graph.add((rollout, CP.name, Literal(log.name)))
    graph.add((rollout, CP.hasDataset, dataset))
    graph.add((rollout, CP.successEpisodes, index_literal(log.successes)))
    graph.add((rollout, CP.totalEpisodes, index_literal(log.finished)))
    header = log.header
    for key, prop in (("eval_file", CP.evalFile), ("eval_sha256", CP.evalSha256), ("task", CP.task), ("checkpoint", CP.checkpoint)):
        if isinstance(header.get(key), str) and header[key]:
            graph.add((rollout, prop, Literal(header[key])))
    if isinstance(header.get("seed"), int):
        graph.add((rollout, CP.seed, Literal(header["seed"], datatype=XSD.integer)))
    for name, pose in sorted((header.get("fixed_objects") or {}).items()):
        pos = pose.get("pos") if isinstance(pose, dict) else None
        if isinstance(pos, list) and len(pos) == 3:
            add_task_object(graph, rollout, name, tuple(float(v) for v in pos), prop=CP.hasFixedObject, path="fixed")
    for r in log.runs:
        run = URIRef(f"{rollout}/run/{r.index}")
        graph.add((run, RDF.type, CP.Run))
        graph.add((run, CP["index"], index_literal(r.index)))
        graph.add((run, CP.status, Literal(r.status)))
        graph.add((rollout, CP.hasRun, run))
        if r.instruction:
            graph.add((run, CP.taskDescription, Literal(r.instruction)))
        for name, xyz in r.object_positions.items():
            add_task_object(graph, run, name, xyz)


def add_task_object(
    graph: Graph, owner: URIRef, name: str, xyz: tuple[float, float, float], prop: URIRef = CP.hasTaskObject, path: str = "object"
) -> URIRef:
    obj = URIRef(f"{owner}/{path}/{quote(name, safe='')}")
    position = URIRef(f"{obj}/position")
    graph.add((obj, RDF.type, CP.TaskObject))
    graph.add((obj, CP.name, Literal(name)))
    graph.add((obj, CP.hasPosition, position))
    graph.add((position, RDF.type, CP.Position))
    for p, value in zip((CP.positionX, CP.positionY, CP.positionZ), xyz):
        graph.add((position, p, decimal_literal(value)))
    graph.add((owner, prop, obj))
    return obj


def add_light_condition(graph: Graph, owner: URIRef, name: str, intensity: float, color_temperature: float) -> URIRef:
    """A light's intensity and color temperature in an episode or run (``<owner>/light/<name>``)."""
    node = URIRef(f"{owner}/light/{quote(name, safe='')}")
    graph.add((node, RDF.type, CP.LightCondition))
    graph.add((node, CP.name, Literal(name)))
    graph.add((node, CP.lightIntensity, decimal_literal(intensity)))
    graph.add((node, CP.colorTemperature, decimal_literal(color_temperature)))
    graph.add((owner, CP.hasLightCondition, node))
    return node


def add_object_color(graph: Graph, owner: URIRef, name: str, rgb: tuple[float, float, float]) -> URIRef:
    """An object's diffuse color in an episode or run (``<owner>/color/<name>``)."""
    node = URIRef(f"{owner}/color/{quote(name, safe='')}")
    graph.add((node, RDF.type, CP.ObjectColor))
    graph.add((node, CP.name, Literal(name)))
    for p, value in zip((CP.colorR, CP.colorG, CP.colorB), rgb):
        graph.add((node, p, decimal_literal(value)))
    graph.add((owner, CP.hasObjectColor, node))
    return node


def index_literal(value: int) -> Literal:
    return Literal(value, datatype=XSD.nonNegativeInteger)


def decimal_literal(value: float | Decimal) -> Literal:
    # Recorded positions are float32; 6 places keep micrometres and drop float noise.
    return Literal(Decimal(str(round(value, 6))), datatype=XSD.decimal)
