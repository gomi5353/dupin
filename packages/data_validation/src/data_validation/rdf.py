"""Build an RDF graph from LeRobot metadata for OWL/SHACL validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, XSD

LR = Namespace("https://hcis.example.org/lerobot#")
BASE = "https://hcis.example.org/dataset/"


def _literal(value: Any) -> Literal:
    if isinstance(value, bool):
        return Literal(value, datatype=XSD.boolean)
    if isinstance(value, int):
        return Literal(value, datatype=XSD.integer)
    if isinstance(value, float):
        return Literal(value, datatype=XSD.double)
    return Literal(str(value))


def _shape(feature: dict[str, Any]) -> str:
    shape = feature.get("shape")
    if isinstance(shape, (list, tuple)):
        return json.dumps(list(shape), separators=(",", ":"))
    return str(shape or "")


def build_graph(dataset_root: Path) -> Graph:
    meta = dataset_root / "meta"
    info_path = meta / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")

    info = json.loads(info_path.read_text())
    graph = Graph()
    dataset = URIRef(BASE + dataset_root.name)
    graph.add((dataset, RDF.type, LR.Dataset))
    version = info.get("codebase_version", info.get("version", ""))
    graph.add((dataset, LR.formatVersion, Literal(str(version))))
    if info.get("fps") is not None:
        graph.add((dataset, LR.datasetFPS, _literal(info["fps"])))

    for name, feature in (info.get("features") or {}).items():
        feature_id = URIRef(f"{dataset}/feature/{name}")
        feature_type = LR.ImageFeature if name.startswith("observation.images") else LR.Feature
        if name == "observation.state":
            feature_type = LR.StateFeature
        elif name == "action":
            feature_type = LR.ActionFeature
        graph.add((feature_id, RDF.type, feature_type))
        graph.add((feature_id, LR.featureName, Literal(name)))
        if feature.get("dtype") is not None:
            graph.add((feature_id, LR.featureDType, Literal(str(feature["dtype"]))))
        graph.add((feature_id, LR.featureShape, Literal(_shape(feature))))
        graph.add((dataset, LR.hasFeature, feature_id))

    tasks_path = meta / "tasks.jsonl"
    if tasks_path.is_file():
        for index, line in enumerate(tasks_path.read_text().splitlines()):
            if not line.strip():
                continue
            task_data = json.loads(line)
            task_name = task_data.get("task", task_data.get("task_index", index))
            _add_task(graph, dataset, index, task_name)
    tasks_parquet = meta / "tasks.parquet"
    if tasks_parquet.is_file():
        try:
            for index, task_data in enumerate(_read_parquet_rows(tasks_parquet)):
                task_name = task_data.get("task", task_data.get("task_index", index))
                _add_task(graph, dataset, index, task_name)
        except ImportError:
            pass

    episodes = _episode_records(meta)
    for index, episode_data in episodes:
        episode_id = URIRef(f"{dataset}/episode/{index}")
        graph.add((episode_id, RDF.type, LR.Episode))
        if isinstance(episode_data, dict) and episode_data.get("success") is not None:
            graph.add((episode_id, LR.episodeOutcome, Literal(str(episode_data["success"]))))
        graph.add((dataset, LR.hasEpisode, episode_id))
        length = episode_data.get("length") if isinstance(episode_data, dict) else None
        for frame_index in range(int(length or 1)):
            frame_id = URIRef(f"{episode_id}/frame/{frame_index}")
            graph.add((frame_id, RDF.type, LR.Frame))
            graph.add((frame_id, LR.frameIndex, Literal(frame_index, datatype=XSD.integer)))
            graph.add((frame_id, LR.belongsToEpisode, episode_id))
            _link_frame_features(graph, frame_id, dataset)
            graph.add((episode_id, LR.hasFrame, frame_id))
    return graph


def _link_frame_features(graph: Graph, frame_id: URIRef, dataset: URIRef) -> None:
    for predicate, feature_name in (
        (LR.hasObservationState, "observation.state"),
        (LR.hasAction, "action"),
        (LR.hasFrontImage, "observation.images.front"),
        (LR.hasWristImage, "observation.images.wrist"),
    ):
        feature_id = URIRef(f"{dataset}/feature/{feature_name}")
        graph.add((frame_id, predicate, feature_id))


def _add_task(graph: Graph, dataset: URIRef, index: int, task_name: Any) -> None:
    task_id = URIRef(f"{dataset}/task/{index}")
    graph.add((task_id, RDF.type, LR.Task))
    graph.add((task_id, LR.featureName, Literal(str(task_name))))
    graph.add((dataset, LR.hasTask, task_id))


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    columns = table.to_pydict()
    return [
        {name: values[index] for name, values in columns.items()}
        for index in range(table.num_rows)
    ]


def _episode_records(meta: Path) -> list[tuple[int, dict[str, Any]]]:
    records: list[tuple[int, dict[str, Any]]] = []
    for path in sorted(meta.glob("episodes*.jsonl")) + sorted((meta / "episodes").glob("**/*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            index = int(value.get("episode_index", len(records))) if isinstance(value, dict) else len(records)
            records.append((index, value if isinstance(value, dict) else {}))
    for path in sorted((meta / "episodes").glob("**/*.parquet")):
        try:
            for value in _read_parquet_rows(path):
                index = int(value.get("episode_index", len(records)))
                records.append((index, value))
        except ImportError:
            # The normal LeRobot installation provides pyarrow. Keep the
            # validator importable without it and let the file check explain
            # that episode metadata could not be inspected.
            continue
    return sorted({index: record for index, record in records}.items())
