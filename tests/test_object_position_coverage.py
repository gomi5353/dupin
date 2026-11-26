"""Object position coverage through the store (ingest, verify --spec) on synthetic LeRobot v3 layouts (no simulator)."""

from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rdflib import RDF, Graph, URIRef

from data_verification import StoreError, format_check, ingest_dataset, load_spec, verify_store
from data_verification.dataset import OBJECT_POSITION_KEY_PREFIX, OBJECT_POSITION_NAMES
from data_verification.graph import CP

ROOT = Path(__file__).resolve().parents[1]

SPEC = """
@prefix cp:   <http://taica.course/cp/ontology#> .
@prefix spec: <http://taica.course/cp/spec/test#> .

spec:axis a cp:ObjectPoseAxis ; cp:hasRequirement spec:cube_requirement .
spec:cube a cp:TaskObject ; cp:name "cube" .
spec:cube_requirement a cp:ObjectPositionRequirement ;
    cp:hasTaskObject spec:cube ;
    cp:minEpisodesPerCell 1 ;
    cp:hasAreaConfiguration spec:cube_area .
spec:cube_area a cp:AreaConfiguration ;
    cp:areaXMin 0.0 ; cp:areaXMax 0.2 ; cp:areaYMin 0.0 ; cp:areaYMax 0.2 ;
    cp:hasCoverageGrid spec:cube_grid .
spec:cube_grid a cp:CoverageGrid ; cp:numCellW 2 ; cp:numCellH 2 .
"""

# One point per cell of the 2x2 grid over [0, 0.2]^2.
CELL_CENTERS = [(0.05, 0.05), (0.15, 0.05), (0.05, 0.15), (0.15, 0.15)]

# The spec "spec" (file spec.ttl) renames spec:<x> to <dataset>/spec/spec/<x>.
SPEC_NODE = "http://taica.course/cp/data/ds/spec/spec/"


def write_dataset(root: Path, episodes: dict[int, dict[str, list[tuple[float, float]]]], frames: int = 3) -> Path:
    """episodes[episode][object] = xy per frame (padded with the last xy up to ``frames``)."""
    object_names = sorted({name for objs in episodes.values() for name in objs})
    features = {
        "observation.state": {"dtype": "float32", "shape": [9], "names": None},
        **{
            f"{OBJECT_POSITION_KEY_PREFIX}{name}": {"dtype": "float32", "shape": [3], "names": OBJECT_POSITION_NAMES}
            for name in object_names
        },
    }
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"codebase_version": "v3.0", "fps": 30, "robot_type": "franka_panda", "features": features}))

    columns: dict[str, list] = {"episode_index": [], "frame_index": []}
    columns.update({f"{OBJECT_POSITION_KEY_PREFIX}{name}": [] for name in object_names})
    for episode, objs in episodes.items():
        for frame in range(frames):
            columns["episode_index"].append(episode)
            columns["frame_index"].append(frame)
            for name in object_names:
                xys = objs[name]
                x, y = xys[min(frame, len(xys) - 1)]
                columns[f"{OBJECT_POSITION_KEY_PREFIX}{name}"].append([x, y, 0.9])
    (root / "data" / "chunk-000").mkdir(parents=True)
    pq.write_table(pa.table(columns), root / "data" / "chunk-000" / "file-000.parquet")
    return root


def dataset_of(tmp_path: Path, points=CELL_CENTERS, name: str = "cube") -> Path:
    return write_dataset(tmp_path / "ds", {i: {name: [xy]} for i, xy in enumerate(points)})


def check(tmp_path: Path, dataset: Path, spec_text: str = SPEC, extra_shapes=None):
    """Ingest ``dataset``, verify the dataset only against ``spec_text`` as spec "spec"."""
    store, _ = ingest_dataset(dataset, tmp_path / "stores")
    spec = tmp_path / "spec.ttl"
    spec.write_text(spec_text)
    (report,) = verify_store(store, [spec], extra_shapes=extra_shapes).checks
    return report


def spec_load_errors(tmp_path: Path, spec_text: str, dataset: Path | None = None) -> list[str]:
    store, _ = ingest_dataset(dataset or write_dataset(tmp_path / "ds", {0: {"cube": [(0.05, 0.05)]}}), tmp_path / "stores")
    spec = tmp_path / "spec.ttl"
    spec.write_text(spec_text)
    with pytest.raises(StoreError) as exc:
        load_spec(store, spec)
    return exc.value.errors


def coverage(report):
    return report.axes[0].coverage


# ---- Coverage rule ----------------------------------------------------------------------


def test_full_coverage_passes(tmp_path: Path) -> None:
    report = check(tmp_path, dataset_of(tmp_path))
    assert report.passed, format_check(report)
    assert report.num_episodes == 4
    assert coverage(report)[0].counts == [[1, 1], [1, 1]]


def test_empty_cell_is_a_violation(tmp_path: Path) -> None:
    report = check(tmp_path, dataset_of(tmp_path, CELL_CENTERS[:3]))
    assert not report.passed
    [finding] = report.violations
    assert finding.message == "'cube' cell (i=1, j=1) x=[0.1, 0.2] y=[0.1, 0.2] has 0 episode(s); needs at least 1"
    assert finding.focus_nodes == [SPEC_NODE + "cube_grid/cell/3"]
    assert finding.verifier == "object_pose"
    assert coverage(report)[0].cells_covered == 3


def test_initial_position_is_first_frame_not_later(tmp_path: Path) -> None:
    # Object starts in cell (1, 1) and is carried to cell (0, 0) afterwards.
    episodes = {i: {"cube": [xy]} for i, xy in enumerate(CELL_CENTERS[:3])}
    episodes[3] = {"cube": [(0.15, 0.15), (0.05, 0.05)]}
    report = check(tmp_path, write_dataset(tmp_path / "ds", episodes))
    assert report.passed, format_check(report)


def test_every_episode_counts_regardless_of_outcome(tmp_path: Path) -> None:
    # A 'success' bookkeeping column must not filter anything.
    dataset = dataset_of(tmp_path)
    shard = dataset / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(shard)
    pq.write_table(table.append_column("success", pa.array([False] * table.num_rows)), shard)
    assert check(tmp_path, dataset).passed


def test_upper_edge_belongs_to_last_cell(tmp_path: Path) -> None:
    report = check(tmp_path, dataset_of(tmp_path, [*CELL_CENTERS[:3], (0.2, 0.2)]))
    assert report.passed, format_check(report)
    assert not report.warnings


def test_outside_area_is_warning_and_not_counted(tmp_path: Path) -> None:
    report = check(tmp_path, dataset_of(tmp_path, [*CELL_CENTERS, (0.5, 0.05)]))
    assert report.passed
    [warning] = report.warnings
    assert "outside the required area" in warning.message
    assert sum(map(sum, coverage(report)[0].counts)) == 4


def test_cells_are_indexed_row_major(tmp_path: Path) -> None:
    # 3 x 2 grid: cell k = j * 3 + i. Leave only cell (i=2, j=0), k=2, empty.
    spec = SPEC.replace("cp:numCellW 2", "cp:numCellW 3").replace("cp:areaXMax 0.2", "cp:areaXMax 0.3")
    points = [(0.05, 0.05), (0.15, 0.05), (0.05, 0.15), (0.15, 0.15), (0.25, 0.15)]
    report = check(tmp_path, dataset_of(tmp_path, points), spec)
    assert coverage(report)[0].counts == [[1, 1, 0], [1, 1, 1]]
    [finding] = report.violations
    assert finding.message.startswith("'cube' cell (i=2, j=0) x=[0.2, 0.3] y=[0, 0.1]")
    assert finding.focus_nodes == [SPEC_NODE + "cube_grid/cell/2"]


def test_area_bound_written_as_double_keeps_upper_edge(tmp_path: Path) -> None:
    spec = SPEC.replace("cp:areaXMax 0.2", "cp:areaXMax 2.0e-1")
    report = check(tmp_path, dataset_of(tmp_path, [*CELL_CENTERS[:3], (0.2, 0.15)]), spec)
    assert report.passed, format_check(report)
    assert not report.warnings


def test_episodes_of_no_dataset_do_not_count(tmp_path: Path) -> None:
    # An episode in the graph that the dataset does not have (here written into the spec) never covers a cell.
    spec = (
        SPEC
        + "spec:stray a cp:Episode ; cp:index 99 ; cp:hasTaskObject spec:stray_cube .\n"
        + 'spec:stray_cube a cp:TaskObject ; cp:name "cube" ; cp:hasPosition spec:stray_pos .\n'
        + "spec:stray_pos a cp:Position ; cp:positionX 0.15 ; cp:positionY 0.15 ; cp:positionZ 0.9 .\n"
    )
    report = check(tmp_path, dataset_of(tmp_path, CELL_CENTERS[:3]), spec)
    assert coverage(report)[0].counts == [[1, 1], [1, 0]]
    assert "'cube' cell (i=1, j=1) x=[0.1, 0.2] y=[0.1, 0.2] has 0 episode(s); needs at least 1" in [v.message for v in report.violations]


# ---- Ingest -----------------------------------------------------------------------------


def test_missing_required_object_column(tmp_path: Path) -> None:
    report = check(tmp_path, write_dataset(tmp_path / "ds", {0: {"other": [(0.05, 0.05)]}}))
    assert not report.passed
    assert any("object_position.cube" in e for e in report.errors)
    assert any("no initial position for required object 'cube'" in v.message for v in report.violations)


def test_dataset_without_position_columns_is_not_ingested(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"features": {"action": {"dtype": "float32", "shape": [8]}}}))
    with pytest.raises(StoreError, match=r"declares no 'object_position.<name>' features"):
        ingest_dataset(root, tmp_path / "stores")
    assert not (tmp_path / "stores").exists()


def test_non_finite_position_is_not_ingested(tmp_path: Path) -> None:
    with pytest.raises(StoreError, match="non-finite"):
        ingest_dataset(dataset_of(tmp_path, [*CELL_CENTERS, (float("nan"), 0.05)]), tmp_path / "stores")


def test_ingest_records_episodes_metadata_and_provenance(tmp_path: Path) -> None:
    store, _ = ingest_dataset(dataset_of(tmp_path), tmp_path / "stores", task="Test-Cube-v0")
    graph = Graph().parse(store.dataset_file)
    ds = store.dataset
    assert len(list(graph.objects(ds, CP.hasEpisode))) == 4
    assert len(list(graph.subjects(RDF.type, CP.Position))) == 4
    assert {str(v) for v in graph.objects(None, CP.frameCount)} == {"3"}
    assert str(graph.value(ds, CP.task)) == "Test-Cube-v0" and str(graph.value(ds, CP.robotType)) == "franka_panda"
    assert [str(v) for v in graph.objects(ds, CP.recordedObject)] == ["cube"]
    assert graph.value(ds, CP.sourcePath) is not None and len(str(graph.value(ds, CP.sourceSha256))) == 64
    assert not list(graph.objects(ds, CP.hasGeneralizationAxis)), "no axis before the first verify"


# ---- Spec: load -----------------------------------------------------------------------


def _fill(spec_text: str, values: dict[str, str]) -> str:
    for field, value in values.items():
        spec_text = spec_text.replace(f'{field} "TODO"', f"{field} {value}")
    return spec_text


FILLED = {
    "cp:areaXMin": "0.0", "cp:areaXMax": "0.2", "cp:areaYMin": "0.0", "cp:areaYMax": "0.2",
    "cp:numCellW": "2", "cp:numCellH": "2", "cp:minEpisodesPerCell": "1",
}


# SPEC with every value "TODO", as in the shipped specs.
TEMPLATE = re.sub(r"(cp:(?:areaXMin|areaXMax|areaYMin|areaYMax|numCellW|numCellH|minEpisodesPerCell)) [0-9.]+", r'\1 "TODO"', SPEC)


def test_unfilled_spec_parses_and_asks_for_todos(tmp_path: Path) -> None:
    errors = spec_load_errors(tmp_path, TEMPLATE)
    assert errors[0].startswith("Spec still has TODO values: cube: areaXMax")


def test_quoted_numbers_are_rejected(tmp_path: Path) -> None:
    errors = spec_load_errors(tmp_path, _fill(TEMPLATE, {**FILLED, "cp:areaXMin": '"0.0"', "cp:numCellW": "2.5"}))
    assert errors == [
        'cube: areaXMin must be a number without quotes, got "0.0"',
        "cube: numCellW must be a whole number without quotes, got 2.5",
    ]


def test_invalid_turtle_is_reported(tmp_path: Path) -> None:
    assert spec_load_errors(tmp_path, "this is not turtle")[0].startswith("Spec is not valid Turtle")


def test_empty_area_is_reported(tmp_path: Path) -> None:
    errors = spec_load_errors(tmp_path, _fill(TEMPLATE, {**FILLED, "cp:areaXMin": "0.3"}))
    assert errors == ["cube: coverage area is empty: areaXMin must be < areaXMax and areaYMin < areaYMax, got x 0.3..0.2, y 0.0..0.2"]


def test_zero_cells_is_reported(tmp_path: Path) -> None:
    errors = spec_load_errors(tmp_path, _fill(TEMPLATE, {**FILLED, "cp:numCellW": "0"}))
    assert errors == ["cube: numCellW must be at least 1, got 0"]


def test_min_episodes_per_cell_must_be_at_least_one(tmp_path: Path) -> None:
    errors = spec_load_errors(tmp_path, _fill(TEMPLATE, {**FILLED, "cp:minEpisodesPerCell": "0"}))
    assert errors == ["cube: minEpisodesPerCell must be at least 1, got 0"]


def test_spec_without_registered_axis_is_rejected(tmp_path: Path) -> None:
    errors = spec_load_errors(tmp_path, SPEC.replace("spec:axis a cp:ObjectPoseAxis", "spec:axis a cp:GeneralizationAxis"))
    assert any("typed cp:GeneralizationAxis" in e for e in errors)
    errors = spec_load_errors(tmp_path / "b", SPEC.replace("spec:axis a cp:ObjectPoseAxis", "spec:axis a <http://example/LightAxis>"))
    assert errors == [
        "Spec declares no generalization axis with a registered verifier (cp:ObjectPoseAxis, cp:ImageAugmentationAxis, cp:ObjectPropertyAxis)"
    ]


def test_loaded_spec_is_renamed_with_cells_and_provenance(tmp_path: Path) -> None:
    store, _ = ingest_dataset(dataset_of(tmp_path), tmp_path / "stores")
    spec = tmp_path / "spec.ttl"
    spec.write_text(SPEC)
    node = load_spec(store, spec)
    graph = Graph().parse(store.spec_file("spec"))
    assert str(node) == SPEC_NODE.rstrip("/")
    assert graph.value(store.dataset, CP.hasGeneralizationAxis) == URIRef(SPEC_NODE + "axis")
    assert graph.value(node, CP.declaresAxis) == URIRef(SPEC_NODE + "axis")
    assert not [t for t in graph if "cp/spec/test#" in " ".join(map(str, t))], "every spec individual is renamed"
    cells = sorted(graph.objects(URIRef(SPEC_NODE + "cube_grid"), CP.hasCell))
    assert len(cells) == 4
    cell3 = URIRef(SPEC_NODE + "cube_grid/cell/3")
    assert [float(graph.value(cell3, p)) for p in (CP.cellXMin, CP.cellXMax, CP.cellYMin, CP.cellYMax)] == [0.1, 0.2, 0.1, 0.2]
    assert len(str(graph.value(node, CP.sourceSha256))) == 64


# ---- Spec: SHACL rules -------------------------------------------------------------------


def _violations(tmp_path: Path, spec_text: str, points=CELL_CENTERS) -> list[str]:
    report = check(tmp_path, dataset_of(tmp_path, points), "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n" + spec_text)
    return [v.message for v in report.violations]


def test_requirement_must_hang_off_an_axis(tmp_path: Path) -> None:
    messages = _violations(tmp_path, SPEC.replace("cp:hasRequirement spec:cube_requirement", "cp:hasRequirement spec:other"))
    assert "Requirement is not linked from an ObjectPoseAxis (cp:hasRequirement)" in messages


def test_grid_cell_count_must_match_width_times_height(tmp_path: Path) -> None:
    # A hand-written extra cell makes 5 cells on a 2 x 2 grid.
    messages = _violations(tmp_path, SPEC + "spec:cube_grid cp:hasCell spec:extra_cell .\nspec:extra_cell a cp:Cell .\n")
    assert "Coverage grid has 5 cell(s); numCellW * numCellH = 4" in messages


def test_requirement_belongs_to_one_axis(tmp_path: Path) -> None:
    messages = _violations(tmp_path, SPEC + "spec:axis2 a cp:ObjectPoseAxis ; cp:hasRequirement spec:cube_requirement .\n")
    assert "Requirement is linked from more than one axis (cp:hasRequirement); give each axis its own requirement" in messages


def test_area_and_grid_belong_to_one_owner(tmp_path: Path) -> None:
    second = """
spec:axis cp:hasRequirement spec:cube2_requirement .
spec:cube2_requirement a cp:ObjectPositionRequirement ;
    cp:hasTaskObject [ a cp:TaskObject ; cp:name "cube" ] ;
    cp:minEpisodesPerCell 1 ;
    cp:hasAreaConfiguration spec:cube_area .
spec:other_area a cp:AreaConfiguration ;
    cp:areaXMin 0.0 ; cp:areaXMax 0.4 ; cp:areaYMin 0.0 ; cp:areaYMax 0.4 ;
    cp:hasCoverageGrid spec:cube_grid .
"""
    messages = _violations(tmp_path, SPEC + second)
    assert "Area must belong to exactly one requirement (cp:hasAreaConfiguration); give each requirement its own area" in messages
    assert "Grid must belong to exactly one area (cp:hasCoverageGrid); give each area its own grid" in messages


def test_cell_index_must_be_unique_and_inside_the_grid(tmp_path: Path) -> None:
    extra = "spec:cube_grid cp:hasCell spec:c_dup, spec:c_out .\nspec:c_dup a cp:Cell ; cp:index \"0\"^^xsd:nonNegativeInteger .\nspec:c_out a cp:Cell ; cp:index \"9\"^^xsd:nonNegativeInteger .\n"
    messages = _violations(tmp_path, SPEC + extra)
    assert "Coverage grid has more than one cell with index 0" in messages
    assert "Cell index 9 is outside the grid (0 to numCellW * numCellH - 1)" in messages


def test_orphan_cell_is_reported(tmp_path: Path) -> None:
    messages = _violations(tmp_path, SPEC + "spec:orphan a cp:Cell ; cp:index \"0\"^^xsd:nonNegativeInteger .\n")
    assert "Cell must belong to exactly one coverage grid (cp:hasCell)" in messages


def test_episode_index_must_be_unique(tmp_path: Path) -> None:
    # A second episode node with the index of a recorded one, linked from the dataset.
    extra = "<http://taica.course/cp/data/ds> cp:hasEpisode spec:twin .\nspec:twin a cp:Episode ; cp:index \"0\"^^xsd:nonNegativeInteger .\n"
    assert "Dataset has more than one episode with index 0" in _violations(tmp_path, SPEC + extra)


def _doc_extra_shape_example() -> str:
    doc = (ROOT / "docs" / "object_pose.md").read_text()
    return doc.split("```turtle\n@prefix cp:", 1)[1].split("```", 1)[0].join(["@prefix cp:", ""])


def test_extra_shapes_example_from_docs(tmp_path: Path) -> None:
    rules = tmp_path / "rules.ttl"
    # The doc uses the cup-stacking robot base (y 6.0); the synthetic dataset lies in [0, 0.2].
    example = _doc_extra_shape_example()
    assert 'sh:minInclusive "6.0"' in example
    rules.write_text(example.replace('"6.0"', '"0.0"'))
    assert check(tmp_path / "a", dataset_of(tmp_path / "a"), extra_shapes=[rules]).passed

    rules.write_text(example.replace('"6.0"', '"0.1"'))
    report = check(tmp_path / "b", dataset_of(tmp_path / "b"), extra_shapes=[rules])
    assert [(v.message, v.count) for v in report.violations] == [("cp:positionY: Object starts behind the robot", 2)]


# ---- Store: results, several specs, staleness ---------------------------------------------


def test_verify_stores_verification_findings_and_cell_counts(tmp_path: Path) -> None:
    store, _ = ingest_dataset(dataset_of(tmp_path, CELL_CENTERS[:3]), tmp_path / "stores")
    spec = tmp_path / "spec.ttl"
    spec.write_text(SPEC)
    verify_store(store, [spec])
    graph = store.graph()
    (verification,) = graph.subjects(RDF.type, CP.Verification)
    assert graph.value(verification, CP.passed).toPython() is False
    assert graph.value(verification, CP.onDataset) == store.dataset
    findings = list(graph.objects(verification, CP.hasFinding))
    assert [str(graph.value(f, CP.severity)) for f in findings] == ["Violation"]
    assert graph.value(findings[0], CP.focusNode) == URIRef(SPEC_NODE + "cube_grid/cell/3")
    counts = {str(c).rsplit("/", 1)[-1]: int(graph.value(c, CP.episodeCount)) for c in graph.subjects(RDF.type, CP.Cell)}
    assert counts == {"0": 1, "1": 1, "2": 1, "3": 0}


def test_several_specs_live_in_one_store_and_are_verified_separately(tmp_path: Path) -> None:
    store, _ = ingest_dataset(dataset_of(tmp_path, CELL_CENTERS[:3]), tmp_path / "stores")
    paths = []
    for name, text in (("coarse", SPEC.replace("cp:numCellW 2 ; cp:numCellH 2", "cp:numCellW 1 ; cp:numCellH 1")), ("fine", SPEC)):
        paths.append(tmp_path / f"{name}.ttl")
        paths[-1].write_text(text)  # same IRIs in both files
    result = verify_store(store, paths)
    assert [(c.spec, c.passed) for c in result.checks] == [("coarse", True), ("fine", False)]
    assert sorted(p.name for p in (store.root / "findings").iterdir()) == ["coarse.ttl", "fine.ttl"]
    graph = store.graph()
    assert len(set(graph.objects(store.dataset, CP.hasGeneralizationAxis))) == 2
    assert len(list(graph.subjects(RDF.type, CP.Cell))) == 1 + 4


def test_verify_reads_the_spec_files_each_time(tmp_path: Path) -> None:
    dataset = dataset_of(tmp_path, CELL_CENTERS[:3])
    store, _ = ingest_dataset(dataset, tmp_path / "stores")
    spec = tmp_path / "spec.ttl"
    spec.write_text(SPEC)
    assert not verify_store(store, [spec]).passed  # cell (1, 1) is empty
    spec.write_text(SPEC.replace("cp:numCellW 2 ; cp:numCellH 2", "cp:numCellW 1 ; cp:numCellH 1"))
    assert verify_store(store, [spec]).passed, "an edited spec is used at the next verify"

    other = tmp_path / "other.ttl"
    other.write_text(SPEC)
    verify_store(store, [spec, other])
    assert store.spec_names() == ["other", "spec"]
    verify_store(store, [spec])
    assert store.spec_names() == ["spec"], "the store keeps exactly the specs of the last verify"
    assert sorted(p.name for p in (store.root / "findings").iterdir()) == ["spec.ttl"]


def test_verify_reports_spec_problems_and_checks_the_other_specs(tmp_path: Path) -> None:
    store, _ = ingest_dataset(dataset_of(tmp_path), tmp_path / "stores")
    good, todo = tmp_path / "good.ttl", tmp_path / "todo.ttl"
    good.write_text(SPEC)
    todo.write_text(TEMPLATE)
    result = verify_store(store, [todo, good])
    assert [(c.spec, c.passed) for c in result.checks] == [("todo", False), ("good", True)]
    assert result.checks[0].errors[0].startswith("Spec still has TODO values")
    assert store.spec_names() == ["good"]
    (report,) = verify_store(store, []).checks
    assert report.errors == ["No spec given; run with --spec <your spec>.ttl"]
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "good.ttl").write_text(SPEC)
    (report,) = verify_store(store, [good, sub / "good.ttl"]).checks
    assert report.errors == ["Two spec files are both named 'good.ttl'; give each spec its own file name"]


def test_changed_dataset_must_be_ingested_again(tmp_path: Path) -> None:
    dataset = dataset_of(tmp_path)
    store, _ = ingest_dataset(dataset, tmp_path / "stores")
    spec = tmp_path / "spec.ttl"
    spec.write_text(SPEC)
    assert verify_store(store, [spec]).passed
    shard = dataset / "data" / "chunk-000" / "file-000.parquet"
    os.utime(shard, ns=(0, 0))
    (report,) = verify_store(store, [spec]).checks
    assert report.errors and "ingest --dataset" in report.errors[0]
    ingest_dataset(dataset, tmp_path / "stores")
    assert not (store.root / "findings").exists(), "re-ingest drops every result of the previous ingest"
    assert verify_store(store, [spec]).passed


def test_query_over_the_store(tmp_path: Path) -> None:
    from data_verification import query

    store, _ = ingest_dataset(dataset_of(tmp_path, CELL_CENTERS[:3]), tmp_path / "stores")
    spec = tmp_path / "spec.ttl"
    spec.write_text(SPEC)
    verify_store(store, [spec])
    text = query(store, (ROOT / "queries" / "uncovered_cells.rq").read_text(), "csv")
    assert text.splitlines() == ["spec,object,i,j,xMin,xMax,yMin,yMax,episodes,needed", "spec,cube,1,1,0.1,0.2,0.1,0.2,0,1"]
    assert query(store, "ASK { ?e a cp:Episode ; cp:index ?i . FILTER (?i = 2) }") == "true"
    assert query(store, "ASK { ?e a cp:Episode ; cp:index 2 }") == "false"  # 2 is xsd:integer; the index is xsd:nonNegativeInteger
    rows = json.loads(query(store, (ROOT / "queries" / "episodes.rq").read_text(), "json"))
    assert [(r["episode"], r["object"]) for r in rows] == [(0, "cube"), (1, "cube"), (2, "cube")]
    with pytest.raises(ValueError, match="invalid or unsupported SPARQL"):
        query(store, "SELEC nothing")


def test_every_example_query_parses(tmp_path: Path) -> None:
    store, _ = ingest_dataset(dataset_of(tmp_path), tmp_path / "stores")
    from data_verification import query

    for path in sorted((ROOT / "queries").glob("*.rq")):
        query(store, path.read_text())


# ---- Recorder format --------------------------------------------------------------------


def _module_constants(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text())
    return {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id.startswith("OBJECT_POSITION_")
    }


def test_recorder_and_verifier_agree_on_column_format() -> None:
    recorder = _module_constants(ROOT / "packages/simulator/src/simulator/utils/lerobot_recorder.py")
    assert recorder == {"OBJECT_POSITION_KEY_PREFIX": OBJECT_POSITION_KEY_PREFIX, "OBJECT_POSITION_NAMES": OBJECT_POSITION_NAMES}


# ---- Command line -----------------------------------------------------------------------


def test_command_line_end_to_end(tmp_path: Path) -> None:
    import subprocess
    import sys

    dataset = dataset_of(tmp_path, CELL_CENTERS[:3])
    log = tmp_path / "outputs" / "rollout" / "20260101-000000.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text(
        json.dumps({"type": "header", "version": 1, "eval_file": "eval/x.py", "seed": 1})
        + "\n"
        + json.dumps({"type": "episode", "episode": 0, "status": "failed", "factors": {"object_position": {"cube": [0.15, 0.15, 0.9]}}})
        + "\n"
    )

    def cli(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(ROOT / "scripts" / "store.py"), *args], cwd=tmp_path, capture_output=True, text=True)

    assert cli("ingest", "--dataset", str(dataset)).returncode == 0
    assert (tmp_path / "stores" / "ds" / "dataset.ttl").is_file()
    mine = tmp_path / "specs" / "mine.ttl"
    mine.parent.mkdir()
    mine.write_text(TEMPLATE)
    todo = cli("verify", "--store", "stores/ds", "--spec", "specs/mine.ttl")
    assert todo.returncode == 1 and "TODO" in todo.stdout, todo.stdout
    assert cli("verify", "--store", "stores/ds").returncode == 2, "--spec is required"
    both = cli("verify", "--store", "stores/ds", "--spec", "specs/mine.ttl", "specs/mine.ttl")
    assert "both named 'mine.ttl'" in both.stdout, "--spec takes several files"
    mine.write_text(_fill(TEMPLATE, FILLED))
    verify = cli("verify", "--store", "stores/ds", "--spec", "specs/mine.ttl", "--json", "out/result.json")
    assert verify.returncode == 1, verify.stdout  # cell (1, 1) is empty
    assert "RESULT: coverage spec NOT met" in verify.stdout and "Rollout" not in verify.stdout
    assert json.loads((tmp_path / "out" / "result.json").read_text())["passed"] is False
    assert cli("ingest", "--store", "stores/ds", "--rollout-log", str(log)).returncode == 2, "rollout logs are no longer ingested"
    assert cli("verify", "--store", "stores/ds", "--spec", "specs/mine.ttl", "--rollout-log", str(log)).returncode == 2
    compare = cli("compare", "--store", "stores/ds", "--spec", "specs/mine.ttl", "--rollout-log", str(log))
    assert compare.returncode == 0, compare.stdout  # comparable; coverage gaps are reported, not an error
    assert "Record more episodes here" in compare.stdout and "does NOT cover" in compare.stdout
    assert "Record more episodes here" not in cli("compare", "--store", "stores/ds", "--spec", "specs/mine.ttl", "--rollout-log", str(log), "--no-hints").stdout
    assert cli("compare", "--store", "stores/ds", "--spec", "specs/mine.ttl").returncode == 2, "--rollout-log is required"
    assert not (tmp_path / "stores" / "ds" / "rollouts").exists()
    assert (tmp_path / "stores" / "ds" / "findings" / "mine__20260101-000000.ttl").is_file()
    explore = cli("explore", "--store", "stores/ds")
    assert explore.returncode == 0 and "NOT passed" in explore.stdout
    assert "20260101-000000: 0/1 succeeded" in explore.stdout and "vs mine: 0 run(s) in distribution" in explore.stdout
    runs = cli("query", "--store", "stores/ds", "--query-file", str(ROOT / "queries" / "run_outcomes_by_cell.rq"), "--format", "csv")
    assert runs.returncode == 0 and "mine,cube,1,1" in runs.stdout.replace("20260101-000000,", "")
    cli("compare", "--store", "stores/ds", "--spec", "specs/mine.ttl", "--rollout-log", str(log), "--no-save")
    q = cli("query", "--store", "stores/ds", "--query-file", str(ROOT / "queries" / "uncovered_cells.rq"), "--format", "csv")
    assert q.returncode == 0 and q.stdout.splitlines()[1] == "mine,cube,1,1,0.1,0.2,0.1,0.2,0,1"
    for removed in ("init", "attach", "detach"):
        assert cli(removed, "--store", "stores/ds").returncode == 2
