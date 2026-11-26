"""The verification store: one directory of Turtle files per dataset, read as one RDF graph.

    stores/<repo_name>/
        dataset.ttl                       ingest --dataset: dataset, episodes, initial positions
        specs/<spec>.ttl                  verify --spec: the spec (renamed), derived cells
        findings/<spec>.ttl               verify: Verification of the dataset against the spec
        findings/<spec>__<rollout>.ttl    compare: the rollout log's runs and their Comparison under the spec
        queries/                          custom SPARQL queries

Each file is rewritten as a whole (written to a temporary file, then renamed),
so an interrupted command never leaves half a file. Raw captures stay where
they are: the LeRobot dataset is the source; the store keeps its path and a
fingerprint, so ``verify`` and ``compare`` can tell when it has changed since
it entered the store. A rollout log is never ingested: ``compare`` reads the log
it is given each time and writes only its result (the runs and the Comparison)
to ``findings/<spec>__<rollout>.ttl``, replacing the same file of an earlier compare.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
import re
from pathlib import Path
from urllib.parse import quote

import pyarrow.parquet as pq
from rdflib import OWL, RDF, RDFS, XSD, BNode, Graph, Literal, Namespace, URIRef

from .dataset import DatasetError, DatasetPlacements, read_episode_tasks, read_initial_placements
from .graph import CP, DATA, add_dataset, add_rollout, dataset_iri, rollout_iri, spec_iri
from . import registry
from .rollout_log import RolloutLogError, read_rollout_log
from .verifier import core_graph

STORES = Path("stores")
SH = Namespace("http://www.w3.org/ns/shacl#")
DCTERMS = Namespace("http://purl.org/dc/terms/")


class StoreError(ValueError):
    """The command cannot be done; ``errors`` says why."""

    def __init__(self, errors: list[str] | str):
        self.errors = [errors] if isinstance(errors, str) else list(errors)
        super().__init__("\n".join(self.errors))


@dataclass(frozen=True)
class Store:
    root: Path

    @property
    def name(self) -> str:
        return self.root.name

    @property
    def dataset(self) -> URIRef:
        return dataset_iri(self.name)

    @property
    def dataset_file(self) -> Path:
        return self.root / "dataset.ttl"

    def spec_file(self, name: str) -> Path:
        return self.root / "specs" / f"{name}.ttl"

    def check_file(self, spec: str) -> Path:
        return self.root / "findings" / f"{spec}.ttl"

    def comparison_file(self, spec: str, rollout: str) -> Path:
        return self.root / "findings" / f"{spec}__{rollout}.ttl"

    def spec_names(self) -> list[str]:
        return sorted(p.stem for p in (self.root / "specs").glob("*.ttl"))

    def load(self, *paths: Path) -> Graph:
        graph = new_graph()
        for path in paths:
            if path.is_file():
                graph.parse(path, format="turtle")
        return graph

    def graph(self, with_ontology: bool = False) -> Graph:
        """Everything in the store as one graph (optionally with the ontology, for labels and classes)."""
        graph = self.load(*sorted(self.root.glob("*.ttl")), *sorted(self.root.glob("*/*.ttl")))
        if with_ontology:
            graph += ontology()
        return graph


def open_store(path: str | Path) -> Store:
    store = Store(Path(path))
    if not store.dataset_file.is_file():
        raise StoreError(f"{path} is not a store (no dataset.ttl); create it with: ingest --dataset <dataset>")
    return store


def new_graph() -> Graph:
    graph = Graph()
    graph.bind("cp", CP)
    graph.bind("xsd", XSD)
    return graph


def ontology() -> Graph:
    graph = core_graph("ontology", "core.ttl")
    for verifier in registry.VERIFIERS:
        graph += verifier.ontology()
    return graph


# ---- Ingest -----------------------------------------------------------------------------


def ingest_dataset(dataset_root: str | Path, stores: str | Path = STORES, name: str | None = None, task: str | None = None) -> tuple[Store, DatasetPlacements]:
    """Write stores/<name>/dataset.ttl from a local LeRobot v3 dataset (meta/ and data/ only; no videos)."""
    root = Path(dataset_root)
    try:
        data = read_initial_placements(root)
    except DatasetError as exc:
        raise StoreError(str(exc)) from None
    errors = []
    if not data.episode_indices:
        errors.append("Dataset contains no episodes")
    bad = [p for p in data.placements if not p.is_finite]
    for p in bad[:10]:
        errors.append(f"Episode {p.episode_index}: non-finite initial position for '{p.object_name}'")
    if len(bad) > 10:
        errors.append(f"... {len(bad) - 10} more non-finite initial positions")
    late = {e: f for e, f in data.first_frame_index.items() if f != 0}
    if late:
        sample = ", ".join(f"{e} (frame {f})" for e, f in list(late.items())[:5])
        errors.append(f"{len(late)} episode(s) do not start at frame_index 0: {sample}")
    if errors:
        raise StoreError(errors)

    store = Store(Path(stores) / (name or root.resolve().name))
    dataset = store.dataset
    graph = new_graph()
    graph.bind("ds", Namespace(f"{dataset}/"))
    add_dataset(graph, dataset, data, data.frame_counts)
    graph.add((dataset, CP.name, Literal(store.name)))
    _provenance(graph, dataset, root.resolve(), dataset_fingerprint(root))
    info = json.loads((root / "meta" / "info.json").read_text())
    if isinstance(info.get("robot_type"), str):
        graph.add((dataset, CP.robotType, Literal(info["robot_type"])))
    if isinstance(info.get("fps"), int):
        graph.add((dataset, CP.fps, Literal(info["fps"], datatype=XSD.nonNegativeInteger)))
    for description in _task_descriptions(root):
        graph.add((dataset, CP.taskDescription, Literal(description)))
    for episode_index, text in read_episode_tasks(root).items():
        graph.add((URIRef(f"{dataset}/episode/{episode_index}"), CP.taskDescription, Literal(text)))
    if task:
        graph.add((dataset, CP.task, Literal(task)))
    for verifier in registry.VERIFIERS:
        errors += verifier.ingest_dataset(root, graph, dataset)
    if errors:
        raise StoreError(errors)

    findings = store.root / "findings"
    if findings.is_dir():
        shutil.rmtree(findings)  # every result was about the previous ingest
    (store.root / "queries").mkdir(parents=True, exist_ok=True)
    write_graph(graph, store.dataset_file)
    return store, data


def read_rollout(store: Store, log_path: str | Path) -> tuple[Graph, str]:
    """Read a rollout log (outputs/rollout/<timestamp>.jsonl) into an in-memory graph and its name; nothing is written."""
    try:
        log = read_rollout_log(log_path)
    except RolloutLogError as exc:
        raise StoreError(str(exc)) from None
    rollout = rollout_iri(store.dataset, log.name)
    graph = new_graph()
    graph.bind("ds", Namespace(f"{store.dataset}/"))
    add_rollout(graph, log, rollout, store.dataset)
    _provenance(graph, rollout, Path(log_path).resolve(), file_sha256(log_path))
    errors = [e for verifier in registry.VERIFIERS for e in verifier.ingest_rollout(log, graph, rollout)]
    if errors:
        raise StoreError(errors)
    repo_id = _trained_on(log.header.get("checkpoint"))
    if repo_id:
        graph.add((rollout, CP.trainedOnRepo, Literal(repo_id)))
    return graph, log.name


def _trained_on(checkpoint) -> str | None:
    """The dataset repo_id in the checkpoint's train_config.json (searched upwards), if any."""
    if not isinstance(checkpoint, str) or not checkpoint:
        return None
    config = next(
        (p / "train_config.json" for p in (Path(checkpoint), *Path(checkpoint).parents) if (p / "train_config.json").is_file()),
        None,
    )
    if config is None:
        return None
    try:
        repo_id = json.loads(config.read_text())["dataset"]["repo_id"]
    except (ValueError, KeyError, TypeError):
        return None
    return repo_id if isinstance(repo_id, str) and repo_id else None


def _task_descriptions(root: Path) -> list[str]:
    path = root / "meta" / "tasks.parquet"
    if not path.is_file():
        return []
    table = pq.read_table(path).to_pydict()
    return sorted({v for column in table.values() for v in column if isinstance(v, str) and v})


def _provenance(graph: Graph, node: URIRef, source: Path, digest: str) -> None:
    graph.add((node, CP.sourcePath, Literal(str(source))))
    graph.add((node, CP.sourceSha256, Literal(digest)))
    graph.add((node, CP.ingestedAt, now()))


# ---- Specs ------------------------------------------------------------------------------

_KEEP_PREFIXES = (str(CP), DATA, str(RDF), str(RDFS), str(OWL), str(XSD), str(SH), str(DCTERMS))


# `cp:phrase fork .` : a property followed by a bare word (no quotes, no prefix, not a number or true/false).
_BARE_VALUE = re.compile(r"^\s*(?:[A-Za-z][\w-]*:)?[A-Za-z][\w-]*\s+(?P<value>[A-Za-z_][\w-]*(?:\s+[A-Za-z_][\w-]*)*)\s*[;.,]?\s*(?:#.*)?$")


def _turtle_hint(path: Path, message: str) -> str:
    """A one-line fix for the most common spec typo: text written without quotes."""
    found = re.search(r"at line (\d+)", message)
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return ""
    if not found:
        return ""
    number = int(found.group(1))
    for n in (number, number - 1, number + 1):  # the parser may report the neighbouring line
        if 1 <= n <= len(lines):
            line = lines[n - 1]
            m = _BARE_VALUE.match(line)
            if m and m.group("value") not in ("a", "true", "false"):
                value = m.group("value")
                return (
                    f"\n          line {n}: {line.strip()}"
                    f"\n          Text values need double quotes: write \"{value}\" instead of {value}. "
                    "Numbers are written without quotes."
                )
    return "\n          Check the line above for a missing quote, a missing '.' or ';' at the end of a statement, or an unknown prefix."


def load_spec(store: Store, spec_path: str | Path, name: str | None = None) -> URIRef:
    """Check a spec file and write it into the store as specs/<name>.ttl (see ``build_spec``).

    ``verify`` calls it for every spec it is given, so the store always holds
    the spec files as they were at the last verify.
    """
    node, graph, name = build_spec(store, spec_path, name)
    _remove_findings(store, name)
    write_graph(graph, store.spec_file(name))
    return node


def build_spec(store: Store, spec_path: str | Path, name: str | None = None) -> tuple[URIRef, Graph, str]:
    """Check a spec file, rename its individuals under the spec's IRI, derive its nodes; nothing is written.

    Renaming lets several specs, all copied from the same shipped spec, live in
    one store. The graph has the spec's axes declared on the dataset
    (cp:hasGeneralizationAxis). Raises ``StoreError`` with every problem found
    (TODO left, wrong value types, ...). Returns the spec node, its graph and its name.
    """
    path = Path(spec_path)
    name = name or path.stem
    if not name or quote(name, safe="-_.") != name or "__" in name:
        raise StoreError(f"Spec name '{name}' must use only letters, digits, '-', '_' and '.', without '__'")
    spec = Graph()
    try:
        spec.parse(path, format="turtle")
    except FileNotFoundError:
        raise StoreError(f"Spec file not found: {path}") from None
    except Exception as exc:  # rdflib raises several parser error types
        raise StoreError(f"Spec is not valid Turtle: {exc}{_turtle_hint(path, str(exc))}") from None

    axes = [(s, o) for s, o in spec.subject_objects(RDF.type) if registry.verifier_for(o) is not None]
    errors = []
    for s in spec.subjects(RDF.type, CP.GeneralizationAxis):
        errors.append(f"{s} is typed cp:GeneralizationAxis; use the class of a registered axis (e.g. cp:ObjectPoseAxis)")
    if not axes and not errors:
        registered = ", ".join(v.axis_class.n3(spec.namespace_manager) for v in registry.VERIFIERS)
        errors.append(f"Spec declares no generalization axis with a registered verifier ({registered})")
    verifiers = []
    for _, axis_class in axes:
        verifier = registry.verifier_for(axis_class)
        if verifier not in verifiers:
            verifiers.append(verifier)
    for verifier in verifiers:
        errors += verifier.spec_errors(spec)
    if errors:
        raise StoreError(errors)

    node = spec_iri(store.dataset, name)
    graph = new_graph()
    graph.bind("spec", Namespace(f"{node}/"))
    rename = _renaming(spec, f"{node}/")
    for s, p, o in spec:
        graph.add((rename.get(s, s), p, rename.get(o, o) if p != RDF.type else o))
    for verifier in verifiers:
        verifier.derive_spec(graph)
    graph.add((node, RDF.type, CP.Spec))
    graph.add((node, CP.name, Literal(name)))
    _provenance(graph, node, path.resolve(), file_sha256(path))
    for axis, _ in axes:
        graph.add((node, CP.declaresAxis, rename.get(axis, axis)))
        graph.add((store.dataset, CP.hasGeneralizationAxis, rename.get(axis, axis)))

    return node, graph, name


def drop_spec(store: Store, name: str) -> None:
    """Remove a spec and its findings from the store (``verify`` drops the specs it was not given)."""
    store.spec_file(name).unlink(missing_ok=True)
    _remove_findings(store, name)


def _remove_findings(store: Store, spec: str) -> None:
    for path in [store.check_file(spec), *(store.root / "findings").glob(f"{spec}__*.ttl")]:
        path.unlink(missing_ok=True)


def _renaming(spec: Graph, base: str) -> dict:
    """Spec individuals (IRIs outside the vocabularies, and blank nodes) -> IRIs under ``base``."""
    vocabulary = {s for s in ontology().subjects()}
    terms = []
    for s, p, o in spec:
        terms.append(s)
        if p != RDF.type:
            terms.append(o)
    mapping: dict = {}
    used: set[str] = set()
    blank = 0
    for term in terms:
        if term in mapping:
            continue
        if isinstance(term, BNode):
            blank += 1
            local = f"_{blank}"
        elif isinstance(term, URIRef) and term not in vocabulary and not str(term).startswith(_KEEP_PREFIXES):
            local = quote(str(term).rstrip("/#").replace("#", "/").rsplit("/", 1)[-1], safe="-_.~") or "node"
        else:
            continue
        unique, n = local, 1
        while unique in used:
            n += 1
            unique = f"{local}_{n}"
        used.add(unique)
        mapping[term] = URIRef(base + unique)
    return mapping


# ---- Sources and staleness -----------------------------------------------------------


def dataset_fingerprint(root: str | Path) -> str:
    """SHA-256 of the path, size and modification time of every file under meta/ and data/ (videos are not read)."""
    root = Path(root)
    digest = hashlib.sha256()
    for sub in ("meta", "data"):
        for path in sorted((root / sub).rglob("*")):
            if path.is_file():
                stat = path.stat()
                digest.update(f"{path.relative_to(root).as_posix()}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_changes(graph: Graph, node: URIRef, kind: str) -> list[str]:
    """Why ``node``'s source no longer matches what entered the store, with the command that refreshes it."""
    source, recorded = graph.value(node, CP.sourcePath), graph.value(node, CP.sourceSha256)
    if source is None or recorded is None:
        return []
    path = Path(str(source))
    if not path.exists():
        return []  # the store keeps what was ingested; nothing to compare with
    current = dataset_fingerprint(path) if kind == "dataset" else file_sha256(path)
    if current == str(recorded):
        return []
    again = {"dataset": f"ingest --dataset {path}", "rollout": f"compare --rollout-log {path}", "spec": f"verify --spec {path}"}[kind]
    return [f"The {kind} {path} changed after it entered the store; run: {again}"]


def now() -> Literal:
    return Literal(datetime.now(timezone.utc).replace(microsecond=0), datatype=XSD.dateTime)


def write_graph(graph: Graph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    graph.serialize(destination=tmp, format="turtle")
    os.replace(tmp, path)
