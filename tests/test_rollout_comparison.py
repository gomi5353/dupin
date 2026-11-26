"""Rollout-vs-dataset comparison through the store, on synthetic logs, task files and datasets (no simulator)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from rdflib import RDF, Graph, Literal, URIRef

from data_verification import (
    AxisComparison,
    StoreError,
    Verifier,
    format_combined,
    format_comparison,
    ingest_dataset,
    compare_store,
    load_spec,
    read_rollout,
    registry,
    verify_store,
)
from data_verification.store import build_spec
from data_verification.verify import compare_rollout
from data_verification.graph import CP
from data_verification.rollout_log import OBJECT_POSITION_FACTOR, ROLLOUT_LOG_VERSION, RolloutLogError, read_rollout_log
from data_verification.store import ontology
from data_verification.verifier import core_graph, run_shacl
from test_object_position_coverage import CELL_CENTERS, SPEC, write_dataset

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def spec(tmp_path: Path) -> Path:
    path = tmp_path / "spec.ttl"
    path.write_text(SPEC)
    return path


def write_log(path: Path, runs: list[tuple[float, float, str]], extra: dict | None = None, header: dict | None = None) -> Path:
    """runs: (x, y, status) of 'cube' per run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [{"type": "header", "version": ROLLOUT_LOG_VERSION, "eval_file": "eval/cube_eval.py", "eval_sha256": "0" * 64, "seed": 42, **(header or {})}]
    for i, (x, y, status) in enumerate(runs):
        positions = {"cube": [x, y, 0.9], **(extra or {})}
        lines.append({"type": "episode", "episode": i, "status": status, "factors": {OBJECT_POSITION_FACTOR: positions}})
    path.write_text("".join(json.dumps(r) + "\n" for r in lines))
    return path


def full_dataset(tmp_path: Path, points=CELL_CENTERS) -> Path:
    return write_dataset(tmp_path / "ds", {i: {"cube": [xy]} for i, xy in enumerate(points)})


def store_with(tmp_path: Path, spec: Path, dataset: Path | None = None):
    store, _ = ingest_dataset(dataset or full_dataset(tmp_path), tmp_path / "stores")
    load_spec(store, spec)
    return store


def stored_specs(store) -> list[Path]:
    """The spec files the store was last verified with (verify needs them again)."""
    graph = store.graph()
    return sorted(Path(str(graph.value(s, CP.sourcePath))) for s in graph.subjects(RDF.type, CP.Spec))


def compare(store, log: Path):
    (result,) = compare_store(store, stored_specs(store), [log]).comparisons
    return result


def run(tmp_path, spec, runs, dataset=None):
    store = store_with(tmp_path, spec, dataset)
    return compare(store, write_log(tmp_path / "rollout" / "run.jsonl", runs))


def axis(result):
    return result.axes[0]


def cube(result):
    return axis(result).objects[0]


# ---- Comparison ---------------------------------------------------------------


def test_every_run_in_a_covered_cell_is_in_distribution(tmp_path: Path, spec: Path) -> None:
    result = run(tmp_path, spec, [(x, y, "succeeded") for x, y in CELL_CENTERS])
    assert result.passed, format_comparison(result)
    assert result.fully_covered
    assert [[c.runs for c in row] for row in cube(result).cells] == [[1, 1], [1, 1]]
    assert result.runs_in_distribution == 4 and not result.warnings


def test_run_in_a_short_cell_asks_to_record_there(tmp_path: Path, spec: Path) -> None:
    dataset = full_dataset(tmp_path, CELL_CENTERS[:3])  # cell (1, 1) has no episode
    result = run(tmp_path, spec, [(0.15, 0.15, "failed"), (0.05, 0.05, "succeeded")], dataset=dataset)
    cell = cube(result).cells[1][1]
    assert cell.runs == 1 and not cell.covered
    (target,) = axis(result).to_fill
    assert (target.i, target.j, target.needed, target.runs, target.failed) == (1, 1, 1, 1, 1)
    assert result.runs_in_distribution == 1


def test_hints_can_be_left_out_of_the_report(tmp_path: Path, spec: Path) -> None:
    dataset = full_dataset(tmp_path, CELL_CENTERS[:3])
    store = store_with(tmp_path, spec, dataset)
    result = compare(store, write_log(tmp_path / "run.jsonl", [(0.15, 0.15, "failed")]))
    assert "Record more episodes here" in format_comparison(result)
    assert "Record more episodes here" not in format_comparison(result, hints=False)


def test_area_upper_edge_and_inner_edges_follow_the_check(tmp_path: Path, spec: Path) -> None:
    # x = 0.2 is the area maximum (last column); x = 0.1 is an inner edge (second column).
    result = run(tmp_path, spec, [(0.2, 0.2, "succeeded"), (0.1, 0.0, "succeeded")])
    cells = cube(result).cells
    assert cells[1][1].runs == 1 and cells[0][1].runs == 1
    assert result.fully_covered


def test_runs_outside_the_area_suggest_whole_cells(tmp_path: Path, spec: Path) -> None:
    result = run(tmp_path, spec, [(0.05, 0.05, "succeeded"), (0.22, 0.05, "failed"), (-0.01, 0.15, "failed")])
    obj = cube(result)
    assert [(c.i, c.j, c.runs) for c in obj.outside] == [(2, 0, 1), (-1, 1, 1)]
    assert obj.outside[0].x_range == pytest.approx((0.2, 0.3))
    (widen,) = axis(result).widen
    assert widen.suggested == pytest.approx((-0.1, 0.3, 0.0, 0.2)) and (widen.num_cell_w, widen.num_cell_h) == (4, 2)
    assert obj.runs_covered == 1 and result.runs_in_distribution == 1


def test_unsampled_cells_are_not_judged(tmp_path: Path, spec: Path) -> None:
    result = run(tmp_path, spec, [(0.05, 0.05, "succeeded")], dataset=full_dataset(tmp_path, CELL_CENTERS[:1]))
    assert result.fully_covered and not axis(result).to_fill
    assert cube(result).sampled_cells == 1 and cube(result).covered_cells == 1


def test_varying_object_without_requirement_is_warned_not_compared(tmp_path: Path, spec: Path) -> None:
    log = write_log(tmp_path / "run.jsonl", [(0.05, 0.05, "succeeded"), (0.15, 0.15, "succeeded")])
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    for i, line in enumerate(lines[1:]):
        line["factors"][OBJECT_POSITION_FACTOR]["ball"] = [0.1 * i, 0.0, 0.9]
    log.write_text("".join(json.dumps(r) + "\n" for r in lines))
    result = compare(store_with(tmp_path, spec), log)
    assert result.passed, format_comparison(result)
    assert [o.object_name for o in axis(result).objects] == ["cube"] and result.fully_covered
    assert any("'ball'" in w.message and "no requirement" in w.message for w in result.warnings)


def test_aborted_runs_are_kept_but_not_counted(tmp_path: Path, spec: Path) -> None:
    store = store_with(tmp_path, spec)
    result = compare(store, write_log(tmp_path / "run.jsonl", [(0.05, 0.05, "succeeded"), (0.15, 0.05, "aborted"), (0.15, 0.15, "failed")]))
    assert (result.runs, result.successes, result.aborted) == (2, 1, 1)
    graph, name = read_rollout(store, tmp_path / "run.jsonl")
    assert name == "run" and {str(s) for s in graph.objects(None, CP.status)} == {"succeeded", "failed", "aborted"}
    (rollout,) = graph.subjects(RDF.type, CP.Rollout)
    assert (graph.value(rollout, CP.totalEpisodes).toPython(), graph.value(rollout, CP.successEpisodes).toPython()) == (2, 1)
    assert not (store.root / "rollouts").exists(), "the log is not ingested"


def test_rollout_has_its_training_dataset_and_runs_are_not_episodes(tmp_path: Path, spec: Path) -> None:
    store = store_with(tmp_path, spec, full_dataset(tmp_path, CELL_CENTERS[:1]))
    result = compare(store, write_log(tmp_path / "run.jsonl", [(0.15, 0.15, "succeeded")] * 3))
    graph = read_rollout(store, tmp_path / "run.jsonl")[0]
    (rollout,) = graph.subjects(RDF.type, CP.Rollout)
    assert graph.value(rollout, CP.hasDataset) == store.dataset
    assert len(list(graph.subjects(RDF.type, CP.Run))) == 3 and not list(graph.subjects(RDF.type, CP.Episode))
    assert len(list(store.graph().subjects(RDF.type, CP.Episode))) == 1
    # Three runs in cell (1, 1) do not count as dataset coverage there.
    assert cube(result).cells[1][1].episodes == 0 and result.runs_in_distribution == 0


def test_comparison_graph_has_placements(tmp_path: Path, spec: Path) -> None:
    store = store_with(tmp_path, spec, full_dataset(tmp_path, CELL_CENTERS[:3]))
    log = write_log(tmp_path / "run.jsonl", [(0.05, 0.05, "succeeded"), (0.15, 0.15, "failed"), (0.3, 0.05, "failed")])
    rollout_graph, name = read_rollout(store, log)
    _, spec_graph, spec_name = build_spec(store, spec)
    _, graph = compare_rollout(store, spec_name, spec_graph, rollout_graph, name)
    graph += rollout_graph
    (comparison,) = graph.subjects(RDF.type, CP.Comparison)
    assert list(graph.objects(comparison, CP.inDistributionRun)) == [URIRef(f"{store.dataset}/rollout/run/run/0")]
    placements = {
        int(graph.value(graph.value(p, CP.ofRun), CP["index"])): (int(graph.value(p, CP.cellI)), int(graph.value(p, CP.cellJ)), graph.value(p, CP.covered).toPython(), graph.value(p, CP.inCell) is not None)
        for p in graph.objects(comparison, CP.hasPlacement)
    }
    assert placements == {0: (0, 0, True, True), 1: (1, 1, False, True), 2: (3, 0, False, False)}


def test_every_axis_must_cover_a_run(tmp_path: Path, spec: Path, monkeypatch) -> None:
    """A second axis registered with only an ontology, SHACL rules and a compare step (no template, no check)."""
    light = "http://test/light#"

    class LightingVerifier(Verifier):
        name = "lighting"
        axis_class = URIRef(light + "ImageAugmentationAxis")

        def ontology(self) -> Graph:
            return Graph().parse(data=f"""
                @prefix cp: <http://taica.course/cp/ontology#> . @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
                <{light}ImageAugmentationAxis> rdfs:subClassOf cp:GeneralizationAxis .
                <{light}LightingRequirement> rdfs:subClassOf cp:AxisRequirement .""", format="turtle")

        def shapes(self) -> Graph:
            return Graph().parse(data=f"""
                @prefix sh: <http://www.w3.org/ns/shacl#> . @prefix cp: <http://taica.course/cp/ontology#> .
                <{light}AxisShape> a sh:NodeShape ; sh:targetClass <{light}ImageAugmentationAxis> ;
                    sh:property [ sh:path cp:hasRequirement ; sh:minCount 2 ; sh:message "Lighting needs two requirements" ] .""", format="turtle")

        def ingest_dataset(self, root, graph, dataset):
            for e in graph.objects(dataset, CP.hasEpisode):
                graph.add((e, URIRef(light + "lux"), Literal(500.0)))
            return []

        def ingest_rollout(self, log, graph, rollout):
            for r in log.runs:
                graph.add((URIRef(f"{rollout}/run/{r.index}"), URIRef(light + "lux"), Literal(float(r.factors.get("lighting", {}).get("lux", 0)))))
            return []

        def compare(self, graph, rollout, dataset, comparison):
            indices = [int(graph.value(r, CP["index"])) for r in graph.objects(rollout, CP.hasRun)]
            return AxisComparison(self.name, {i: i % 2 == 0 for i in indices})

    monkeypatch.setattr(registry, "VERIFIERS", [*registry.VERIFIERS, LightingVerifier()])
    spec.write_text(SPEC + f"spec:light a <{light}ImageAugmentationAxis> ; cp:hasRequirement spec:dim .\nspec:dim a <{light}LightingRequirement> .\n")
    store = store_with(tmp_path, spec)
    assert len(list(Graph().parse(store.dataset_file).objects(None, URIRef(light + "lux")))) == 4, "ingest hook adds the factor"
    (check,) = verify_store(store, stored_specs(store)).checks
    assert [(v.message, v.verifier) for v in check.violations] == [("cp:hasRequirement: Lighting needs two requirements", "lighting")]
    result = compare(store, write_log(tmp_path / "run.jsonl", [(x, y, "succeeded") for x, y in CELL_CENTERS], extra=None))
    assert len(list(read_rollout(store, tmp_path / "run.jsonl")[0].objects(None, URIRef(light + "lux")))) == 4
    assert [a.axis for a in result.axes] == ["object_pose", "lighting"]
    assert result.runs_in_distribution == 2 and not result.fully_covered


def test_checkpoint_trained_on_another_dataset_is_warned(tmp_path: Path, spec: Path) -> None:
    checkpoint = tmp_path / "train" / "checkpoints" / "last" / "pretrained_model"
    checkpoint.mkdir(parents=True)
    log = write_log(tmp_path / "run.jsonl", [(0.05, 0.05, "succeeded")], header={"checkpoint": str(checkpoint)})
    store = store_with(tmp_path, spec, write_dataset(tmp_path / "me" / "ds", {i: {"cube": [xy]} for i, xy in enumerate(CELL_CENTERS)}))
    (checkpoint / "train_config.json").write_text(json.dumps({"dataset": {"repo_id": "someone/other"}}))
    result = compare(store, log)
    assert result.passed and any("trained on 'someone/other'" in w.message for w in result.warnings)
    (checkpoint / "train_config.json").write_text(json.dumps({"dataset": {"repo_id": "me/ds"}}))
    result = compare(store, log)  # the log is read again: the repo_id is read at compare
    assert not any("trained on" in w.message for w in result.warnings)


def test_all_aborted_is_an_error(tmp_path: Path, spec: Path) -> None:
    result = run(tmp_path, spec, [(0.05, 0.05, "aborted")])
    assert not result.passed and "no finished run" in result.errors[0]


def test_each_compare_reads_the_log_again(tmp_path: Path, spec: Path) -> None:
    store = store_with(tmp_path, spec)
    log = write_log(tmp_path / "run.jsonl", [(0.05, 0.05, "succeeded")])
    assert compare(store, log).runs == 1
    log.write_text(log.read_text() + json.dumps({"type": "episode", "episode": 1, "status": "failed", "factors": {"object_position": {"cube": [0.15, 0.05, 0.9]}}}) + "\n")
    assert compare(store, log).runs == 2, "no stale copy of the log"


def test_compare_only_sees_the_logs_it_is_given(tmp_path: Path, spec: Path) -> None:
    store = store_with(tmp_path, spec)
    old = write_log(tmp_path / "old.jsonl", [(0.05, 0.05, "succeeded")])
    new = write_log(tmp_path / "new.jsonl", [(0.15, 0.05, "failed")])
    compare(store, old)
    (result,) = compare_store(store, stored_specs(store), [new]).comparisons
    assert result.rollout == "new" and (result.runs, result.successes) == (1, 0)


def test_compare_stores_its_result_next_to_the_verify_findings(tmp_path: Path, spec: Path) -> None:
    store = store_with(tmp_path, spec, full_dataset(tmp_path, CELL_CENTERS[:3]))
    verify_store(store, [spec])
    log = write_log(tmp_path / "run.jsonl", [(0.05, 0.05, "succeeded"), (0.15, 0.15, "failed"), (0.3, 0.05, "failed")])
    compare(store, log)
    assert sorted(p.name for p in (store.root / "findings").glob("*.ttl")) == ["spec.ttl", "spec__run.ttl"]
    assert not (store.root / "rollouts").exists(), "the log is not ingested"
    graph = store.graph()
    (comparison,) = graph.subjects(RDF.type, CP.Comparison)
    assert list(graph.objects(comparison, CP.inDistributionRun)) == [URIRef(f"{store.dataset}/rollout/run/run/0")]
    assert len(list(graph.subjects(RDF.type, CP.Run))) == 3


def test_compare_replaces_its_result_and_verify_ignores_it(tmp_path: Path, spec: Path) -> None:
    store = store_with(tmp_path, spec)
    log = write_log(tmp_path / "run.jsonl", [(0.05, 0.05, "succeeded")])
    compare(store, log)
    log.write_text(log.read_text() + json.dumps({"type": "episode", "episode": 1, "status": "failed", "factors": {"object_position": {"cube": [0.15, 0.05, 0.9]}}}) + "\n")
    compare(store, log)
    graph = store.graph()
    assert len(list(graph.subjects(RDF.type, CP.Run))) == 2 and len(list(graph.subjects(RDF.type, CP.Comparison))) == 1
    assert verify_store(store, [spec]).comparisons == [], "verify reads no comparison"
    assert not (store.root / "findings" / "spec__run.ttl").exists(), "a verified spec invalidates its old comparisons"


def test_no_save_writes_nothing(tmp_path: Path, spec: Path) -> None:
    store = store_with(tmp_path, spec)
    log = write_log(tmp_path / "run.jsonl", [(0.05, 0.05, "succeeded")])
    (result,) = compare_store(store, [spec], [log], write=False).comparisons
    assert result.runs == 1 and not (store.root / "findings").exists()


def test_unreadable_log_is_reported(tmp_path: Path, spec: Path) -> None:
    store = store_with(tmp_path, spec)
    result = compare_store(store, [spec], [tmp_path / "missing.jsonl"])
    assert not result.passed and result.comparisons == [] and result.checks[0].errors


def test_rollout_needs_an_existing_store(tmp_path: Path) -> None:
    from data_verification import open_store

    with pytest.raises(StoreError, match="not a store"):
        open_store(tmp_path / "stores" / "nothing")


def test_fixed_poses_in_header_are_stored_and_not_compared(tmp_path: Path, spec: Path) -> None:
    log = write_log(tmp_path / "run.jsonl", [(0.05, 0.05, "succeeded")])
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    pose = {"pos": [7.0, 2.9, 0.74], "rot": [1.0, 0.0, 0.0, 0.0]}
    lines[0].update(robot_base=pose, fixed_objects={"plate": pose})
    log.write_text("".join(json.dumps(r) + "\n" for r in lines))
    store = store_with(tmp_path, spec)
    result = compare(store, log)
    assert result.passed and [o.object_name for o in axis(result).objects] == ["cube"]
    graph = read_rollout(store, log)[0]
    (fixed,) = graph.objects(None, CP.hasFixedObject)
    assert str(graph.value(fixed, CP.name)) == "plate"


def test_inverted_spec_area_is_rejected(tmp_path: Path) -> None:
    spec = tmp_path / "spec.ttl"
    spec.write_text(SPEC.replace("cp:areaXMin 0.0 ; cp:areaXMax 0.2", "cp:areaXMin 0.2 ; cp:areaXMax 0.0"))
    store, _ = ingest_dataset(full_dataset(tmp_path), tmp_path / "stores")
    with pytest.raises(StoreError) as exc:
        load_spec(store, spec)
    assert any(e.startswith("cube: coverage area is empty") for e in exc.value.errors)


# ---- Log format ---------------------------------------------------------------


def test_log_rejects_duplicate_episodes(tmp_path: Path) -> None:
    log = write_log(tmp_path / "episodes.jsonl", [(0.05, 0.05, "succeeded")])
    log.write_text(log.read_text() + log.read_text().splitlines(keepends=True)[1])
    with pytest.raises(RolloutLogError, match="appears twice"):
        read_rollout_log(log)


def test_log_rejects_position_without_z(tmp_path: Path) -> None:
    log = write_log(tmp_path / "episodes.jsonl", [(0.05, 0.05, "succeeded")])
    log.write_text(log.read_text().replace("[0.05, 0.05, 0.9]", "[0.05, 0.05]"))
    with pytest.raises(RolloutLogError, match=r"must be \[x, y, z\]"):
        read_rollout_log(log)


def test_positions_in_store_have_x_y_z(tmp_path: Path, spec: Path) -> None:
    store = store_with(tmp_path, spec)
    compare(store, write_log(tmp_path / "run.jsonl", [(0.05, 0.05, "succeeded")]))
    graph = store.graph()
    positions = list(graph.subjects(RDF.type, CP.Position))
    assert positions, "dataset episodes and runs have positions"
    for p in positions:
        assert all(graph.value(p, prop) is not None for prop in (CP.positionX, CP.positionY))
        assert float(graph.value(p, CP.positionZ)) == 0.9


def _module_constants(path: Path, prefix: str) -> dict[str, object]:
    tree = ast.parse(path.read_text())
    return {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in prefix.split()
    }


def test_rollout_log_writer_and_reader_agree() -> None:
    names = "ROLLOUT_LOG_VERSION OBJECT_POSITION_FACTOR"
    writer = _module_constants(ROOT / "packages/simulator/src/simulator/utils/rollout_log.py", names)
    assert writer == {"ROLLOUT_LOG_VERSION": ROLLOUT_LOG_VERSION, "OBJECT_POSITION_FACTOR": OBJECT_POSITION_FACTOR}


def test_rollout_writes_log_after_each_reset() -> None:
    source = (ROOT / "scripts" / "rollout.py").read_text()
    assert 'os.path.join("outputs", "rollout", f"{time.strftime(' in source, "every rollout writes outputs/rollout/<timestamp>.jsonl"
    reset = source.index("obs_dict, _ = env.reset()\n        needs_reset = True")
    capture = source.index("round_factors = start_factors(env.scene")
    step = source.index("env.step(actions)")
    assert reset < capture < step, "initial positions, lighting and colors must be read after the round's reset and before stepping"


def _list_attr(path: Path, name: str) -> list[str]:
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == name:
            return ast.literal_eval(node.value)
    return []


def _randomized_names(path: Path) -> set[str]:
    """Objects whose start pose a reset event randomizes (randomize_objects_apart / randomize_object_uniform)."""
    names: set[str] = set()
    for call in (n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.Call)):
        func = call.func.attr if isinstance(call.func, ast.Attribute) else getattr(call.func, "id", None)
        kwargs = {k.arg: k.value for k in call.keywords}
        if func == "randomize_objects_apart":
            node = kwargs.get("pose_ranges") or call.args[0]
            names |= {ast.literal_eval(k) for k in node.keys}
        elif func == "randomize_object_uniform":
            names.add(ast.literal_eval(kwargs.get("name") or call.args[0]))
    return names


TASK_FILES = [
    "eval/cup_stacking_eval.py",
    "eval/cutlery_arrangement_eval.py",
    "eval/toy_blocks_collection_eval.py",
    "packages/simulator/src/simulator/tasks/cup_stacking/cup_stacking_env_cfg.py",
    "packages/simulator/src/simulator/tasks/cutlery_arrangement/cutlery_arrangement_env_cfg.py",
    "packages/simulator/src/simulator/tasks/toy_blocks_collection/toy_blocks_collection_env_cfg.py",
]


@pytest.mark.parametrize("task_file", TASK_FILES)
def test_only_randomized_objects_are_tracked(task_file: str) -> None:
    path = ROOT / task_file
    tracked, fixed = _list_attr(path, "tracked_object_names"), _list_attr(path, "fixed_object_names")
    assert set(tracked) == _randomized_names(path), "tracked_object_names must list exactly the randomized objects"
    assert not set(tracked) & set(fixed)


def _dict_attr(path: Path, name: str):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == name:
            return ast.literal_eval(node.value)
    return None


@pytest.mark.parametrize("task_file", TASK_FILES)
def test_task_text_refers_to_objects_by_name_or_color(task_file: str) -> None:
    """Object Properties (STAR-Gen S-PROP): only the words change; the color form names no object kind."""
    import string

    path = ROOT / task_file
    refs, template = _dict_attr(path, "object_references"), _dict_attr(path, "task_description_template")
    weights = _dict_attr(path, "reference_weights")
    assert refs and template, "every task refers to its objects through the template"
    fields = {f for _, f, _, _ in string.Formatter().parse(template) if f}
    assert fields == set(refs), "the template must have one {field} per referenced object"
    kinds = {"cup", "fork", "knife", "bridge", "cylinder", "triangle", "block", "toy"}
    for name, forms in refs.items():
        assert set(forms) == {"name", "color"}, name
        assert forms["color"].endswith(" object") and not kinds & set(forms["color"].split()), (name, forms["color"])
        assert kinds & set(forms["name"].split()), (name, forms["name"])
    expected = {"name": 0.5, "color": 0.5} if task_file.startswith("eval/") else {"name": 1.0, "color": 0.0}
    assert weights == expected
    assert "object_color_palettes" not in path.read_text(), "S-PROP never recolors the objects"
    assert _list_attr(path, "recorded_light_names") == ["rect_light"]
    source = path.read_text()
    assert 'rect_light: AssetBaseCfg = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Scene/RectLight")' in source
    assert "choose_object_reference_forms(self.reference_weights)" in source
    assert 'randomize_light_conditions("light"' not in source, "the RectLight is the randomized light, as in evaluation"


def test_rollout_header_has_robot_base_and_fixed_objects() -> None:
    source = (ROOT / "scripts" / "rollout.py").read_text()
    assert '"robot_base": asset_pose(env.scene, "robot")' in source
    assert "fixed_object_names" in source
    reset = source.index("obs_dict, _ = env.reset()\n        rollout_log = open_rollout_log(")
    assert reset < source.index("env.step(actions)"), "the header poses must be read after the first reset"


# ---- Rollout shapes on hand-written graphs --------------------------------------


def _rollout_errors(runs: list[tuple[int, str, list[str]]]) -> list[str]:
    """runs = (index, status, object names); every object at (0.05, 0.05)."""
    turtle = ["@prefix cp: <http://taica.course/cp/ontology#> .", "@prefix d: <http://test/> .",
              "d:ds a cp:Dataset .", "d:r a cp:Rollout ; cp:hasDataset d:ds ;"
              f" cp:successEpisodes {sum(s == 'succeeded' for _, s, _ in runs)} ;"
              f" cp:totalEpisodes {sum(s != 'aborted' for _, s, _ in runs)} ."]
    for n, (index, status, names) in enumerate(runs):
        turtle.append(f'd:r cp:hasRun d:run{n} . d:run{n} a cp:Run ; cp:index \"{index}\"^^<http://www.w3.org/2001/XMLSchema#nonNegativeInteger> ; cp:status "{status}" .')
        for m, name in enumerate(names):
            turtle.append(
                f'd:run{n} cp:hasTaskObject d:o{n}_{m} . d:o{n}_{m} a cp:TaskObject ; cp:name "{name}" ; '
                f"cp:hasPosition [ a cp:Position ; cp:positionX 0.05 ; cp:positionY 0.05 ; cp:positionZ 0.9 ] ."
            )
    graph = Graph().parse(data="\n".join(turtle), format="turtle")
    findings = run_shacl(graph, {"core": core_graph("shapes", "rollout.ttl")}, ontology(), path_prefix=False)
    return [f.message for f in findings if f.severity == "Violation"]


def test_rollout_shape_needs_a_finished_run() -> None:
    assert "Rollout has no finished run (all aborted)" in _rollout_errors([(0, "aborted", ["cube"])])
    assert "Rollout has no finished run (all aborted)" not in _rollout_errors([(0, "aborted", []), (1, "failed", ["cube"])])


def test_rollout_shape_run_index_and_object_names_are_unique() -> None:
    errors = _rollout_errors([(0, "failed", ["cube", "cube"]), (0, "succeeded", ["cube"])])
    assert "Rollout has more than one run with index 0" in errors
    assert "Run has more than one task object named 'cube'" in errors


def test_several_spec_files_get_a_combined_verdict(tmp_path: Path) -> None:
    loose, strict = tmp_path / "loose.ttl", tmp_path / "strict.ttl"
    loose.write_text(SPEC)
    strict.write_text(SPEC.replace("cp:minEpisodesPerCell 1", "cp:minEpisodesPerCell 2"))
    store, _ = ingest_dataset(full_dataset(tmp_path, CELL_CENTERS + [CELL_CENTERS[0]]), tmp_path / "stores")
    log = write_log(tmp_path / "rollout" / "run.jsonl", [(x, y, "succeeded") for x, y in CELL_CENTERS])
    result = compare_store(store, [loose, strict], [log])
    by_spec = {c.spec: c for c in result.comparisons}
    assert by_spec["loose"].inside_runs == [0, 1, 2, 3]
    assert by_spec["strict"].inside_runs == [0]
    (combined,) = result.combined
    assert combined.specs == ["loose", "strict"] and not combined.skipped
    assert combined.inside_runs == [0] and combined.runs == 4 and not combined.fully_covered
    text = format_combined(combined)
    assert "1/4 runs" in text and "NOT cover" in text
    assert result.to_dict()["combined"][0]["inside_runs"] == [0]


def test_one_spec_file_has_no_combined_verdict(tmp_path: Path, spec: Path) -> None:
    store = store_with(tmp_path, spec)
    log = write_log(tmp_path / "rollout" / "run.jsonl", [(0.05, 0.05, "succeeded")])
    assert compare_store(store, [spec], [log]).combined == []


def test_failed_spec_is_left_out_of_the_combined_verdict(tmp_path: Path, spec: Path) -> None:
    broken = tmp_path / "broken.ttl"
    broken.write_text(SPEC.replace("cp:minEpisodesPerCell 1", 'cp:minEpisodesPerCell "TODO"'))
    store = store_with(tmp_path, spec)
    log = write_log(tmp_path / "rollout" / "run.jsonl", [(0.05, 0.05, "succeeded")])
    result = compare_store(store, [spec, broken], [log])
    assert [c.spec for c in result.comparisons] == ["spec"]
    (combined,) = result.combined
    assert combined.specs == ["spec"] and combined.skipped == ["broken"]
    assert combined.inside_runs == [0] and not combined.fully_covered
    assert "Not combined" in format_combined(combined) and "every axis given" in format_combined(combined)
