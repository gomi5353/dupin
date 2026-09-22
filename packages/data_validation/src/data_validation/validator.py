"""LeRobot v3, file-integrity, RDF and SHACL validation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

from pyshacl import validate
from rdflib import Graph, Namespace

from .rdf import build_graph

LR = Namespace("https://hcis.example.org/lerobot#")


@dataclass
class ValidationReport:
    dataset: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["valid"] = self.valid
        result["usable_for_training"] = self.valid
        return result


def validate_dataset(
    dataset: str | Path,
    *,
    contract: str | Path | None = None,
    ontology: str | Path | None = None,
    shapes: str | Path | None = None,
) -> ValidationReport:
    root = Path(dataset).expanduser().resolve()
    report = ValidationReport(str(root))
    meta = root / "meta"
    required = [meta / "info.json", meta / "stats.json"]
    for path in required:
        key = f"exists:{path.relative_to(root)}"
        report.checks[key] = path.is_file()
        if not path.is_file():
            report.errors.append(f"Missing required file: {path.relative_to(root)}")
    if not root.is_dir():
        report.errors.append(f"Dataset directory does not exist: {root}")
        return report
    task_metadata = [meta / "tasks.jsonl", meta / "tasks.parquet"]
    if not any(path.is_file() for path in task_metadata):
        report.errors.append("Missing task metadata: expected meta/tasks.jsonl or meta/tasks.parquet")
    else:
        report.checks["task_metadata"] = True
    if not (root / "data").is_dir():
        report.errors.append("Missing data/ directory")
    if not (root / "videos").is_dir():
        report.warnings.append("Missing videos/ directory; this is an error when image features are declared")

    info: dict[str, Any] = {}
    info_path = meta / "info.json"
    if info_path.is_file():
        try:
            info = json.loads(info_path.read_text())
        except json.JSONDecodeError as exc:
            report.errors.append(f"Invalid JSON in meta/info.json: {exc}")
    _check_info(info, report)
    _check_data_files(root, info, report)
    _check_media_files(root, report)
    if not _episode_metadata_present(meta):
        report.errors.append("No episode metadata found under meta/")

    if not report.errors or info_path.is_file():
        try:
            data_graph = build_graph(root)
            ontology_graph = _load_graph("lerobot.ttl") if ontology is None else _parse_graph(ontology)
            shapes_graph = _load_graph("lerobot-shapes.ttl") if shapes is None else _parse_graph(shapes)
            if contract:
                _check_contract(Path(contract), info, report)
            conforms, _, details = validate(
                data_graph=data_graph,
                shacl_graph=shapes_graph,
                ont_graph=ontology_graph,
                inference="rdfs",
                abort_on_first=False,
            )
            report.checks["shacl"] = bool(conforms)
            if not conforms:
                report.errors.extend(line.strip() for line in str(details).splitlines() if line.strip())
        except Exception as exc:  # validator must report malformed input, not crash
            report.checks["shacl"] = False
            report.errors.append(f"RDF/SHACL validation could not run: {exc}")
    return report


def _load_graph(name: str) -> Graph:
    return _parse_graph(data=files("data_validation").joinpath("ontology", name).read_text())


def _parse_graph(path: str | Path | None = None, *, data: str | None = None) -> Graph:
    graph = Graph()
    if data is not None:
        graph.parse(data=data, format="turtle")
    else:
        graph.parse(str(path), format="turtle")
    return graph


def _check_info(info: dict[str, Any], report: ValidationReport) -> None:
    version = str(info.get("codebase_version", info.get("version", "")))
    report.checks["format_version"] = version.startswith("v3")
    if not version.startswith("v3"):
        report.errors.append(f"Expected LeRobot v3 metadata, found version {version!r}")
    fps = info.get("fps")
    report.checks["fps"] = isinstance(fps, int) and fps > 0
    if not report.checks["fps"]:
        report.errors.append("meta/info.json must contain a positive integer fps")
    features = info.get("features")
    required = {"observation.state", "action", "observation.images.front", "observation.images.wrist"}
    if not isinstance(features, dict):
        report.errors.append("meta/info.json must contain a features object")
        return
    missing = sorted(required - set(features))
    report.checks["required_features"] = not missing
    if missing:
        report.errors.append(f"Missing required features: {', '.join(missing)}")
    if "task" not in features and "task_index" not in features:
        # LeRobot v3 normally stores the task vocabulary in tasks.parquet and
        # references it from data rows through task_index.
        report.warnings.append("No task/task_index feature declared; task metadata must still exist")
    for name, feature in features.items():
        if not isinstance(feature, dict):
            report.errors.append(f"Feature {name!r} is not an object")
            continue
        if "dtype" not in feature:
            report.errors.append(f"Feature {name!r} has no dtype")
        if name != "task" and "shape" not in feature:
            report.errors.append(f"Feature {name!r} has no shape")


def _check_data_files(root: Path, info: dict[str, Any], report: ValidationReport) -> None:
    data_files = list((root / "data").glob("**/*")) if (root / "data").is_dir() else []
    report.checks["data_files"] = any(path.is_file() for path in data_files)
    if not report.checks["data_files"]:
        report.errors.append("data/ contains no files")
    else:
        _check_data_parquet(root, info, report)
    features = info.get("features", {}) if isinstance(info, dict) else {}
    image_features = [name for name in features if str(name).startswith("observation.images")]
    video_files = list((root / "videos").glob("**/*")) if (root / "videos").is_dir() else []
    if image_features and not any(path.is_file() for path in video_files):
        image_files = list((root / "images").glob("**/*")) if (root / "images").is_dir() else []
        if not any(path.is_file() for path in image_files):
            report.errors.append("Image features are declared but videos/ and images/ contain no files")


def _check_data_parquet(root: Path, info: dict[str, Any], report: ValidationReport) -> None:
    parquet_files = sorted((root / "data").glob("**/*.parquet"))
    if not parquet_files:
        report.warnings.append("data/ contains no parquet shards")
        return
    try:
        import pyarrow.parquet as pq
    except ImportError:
        report.warnings.append("PyArrow is not installed; parquet rows were not inspected")
        return
    total_rows = 0
    required_columns = {"action", "observation.state", "frame_index", "episode_index", "task_index"}
    columns_ok = True
    for path in parquet_files:
        schema = set(pq.read_schema(path).names)
        missing = sorted(required_columns - schema)
        if missing:
            columns_ok = False
            report.errors.append(f"{path.relative_to(root)} is missing columns: {', '.join(missing)}")
        total_rows += pq.read_metadata(path).num_rows
    report.checks["data_columns"] = columns_ok
    expected_rows = info.get("total_frames")
    rows_ok = expected_rows is None or int(expected_rows) == total_rows
    report.checks["total_frames"] = rows_ok
    if not rows_ok:
        report.errors.append(f"info.json declares {expected_rows} frames but data shards contain {total_rows}")


def _check_media_files(root: Path, report: ValidationReport) -> None:
    media_files = [
        path
        for directory in (root / "videos", root / "images")
        if directory.is_dir()
        for path in directory.glob("**/*")
        if path.is_file()
    ]
    if not media_files:
        report.checks["media_decode"] = True
        return
    try:
        import av
    except ImportError:
        report.warnings.append("PyAV is not installed; media bytes were not decoded")
        report.checks["media_decode"] = False
        return
    failures: list[str] = []
    for path in media_files:
        try:
            with av.open(str(path)) as container:
                if not any(True for _ in container.decode(video=0)):
                    failures.append(str(path.relative_to(root)))
        except Exception as exc:
            failures.append(f"{path.relative_to(root)} ({exc})")
    report.checks["media_decode"] = not failures
    for failure in failures[:20]:
        report.errors.append(f"Media file could not be decoded: {failure}")
    if len(failures) > 20:
        report.errors.append(f"Additional undecodable media files: {len(failures) - 20}")


def _episode_metadata_present(meta: Path) -> bool:
    return bool(
        list(meta.glob("episodes*.jsonl"))
        or list((meta / "episodes").glob("**/*.jsonl"))
        or list((meta / "episodes").glob("**/*.parquet"))
    )


def _check_contract(contract_path: Path, info: dict[str, Any], report: ValidationReport) -> None:
    contract_graph = Graph()
    contract_graph.parse(str(contract_path), format="turtle")
    expected: dict[str, tuple[str | None, str | None]] = {}
    for feature in contract_graph.objects(None, LR.contractFeature):
        name = contract_graph.value(feature, LR.featureName)
        if name is None:
            continue
        dtype = contract_graph.value(feature, LR.featureDType)
        shape = contract_graph.value(feature, LR.featureShape)
        expected[str(name)] = (str(dtype) if dtype else None, str(shape) if shape else None)
    actual = info.get("features", {}) if isinstance(info, dict) else {}
    contract_ok = True
    for name, (expected_dtype, expected_shape) in expected.items():
        if name not in actual:
            report.errors.append(f"Contract feature is missing: {name}")
            contract_ok = False
            continue
        feature = actual[name]
        if expected_dtype and str(feature.get("dtype")) != expected_dtype:
            report.errors.append(
                f"Feature {name!r} has dtype {feature.get('dtype')!r}; expected {expected_dtype!r}"
            )
            contract_ok = False
        actual_shape = json.dumps(feature.get("shape", []), separators=(",", ":"))
        if expected_shape and actual_shape != expected_shape:
            report.errors.append(
                f"Feature {name!r} has shape {feature.get('shape')!r}; expected {expected_shape!r}"
            )
            contract_ok = False
    report.checks["dataset_contract"] = contract_ok
