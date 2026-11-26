"""Image Augmentations (lighting) and Object Properties (color references) axes through the store, on synthetic LeRobot v3 layouts (no simulator)."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_verification import StoreError, format_check, format_comparison, compare_store, ingest_dataset, load_spec, verify_store
from data_verification.axes.object_property.coverage import contains_phrase
from data_verification.dataset import (
    LIGHTING_KEY_PREFIX,
    LIGHTING_NAMES,
    OBJECT_COLOR_KEY_PREFIX,
    OBJECT_COLOR_NAMES,
    OBJECT_POSITION_KEY_PREFIX,
    OBJECT_POSITION_NAMES,
)
from data_verification.graph import CP
from data_verification.rollout_log import (
    LIGHTING_FACTOR,
    OBJECT_POSITION_FACTOR,
    ROLLOUT_LOG_VERSION,
    RolloutLogError,
    read_rollout_log,
)

PINK = (1.87, 0.4, 1.88)
BY_NAME = "place the fork on the left of the plate."
BY_COLOR = "place the pink object on the left of the plate."

LIGHT_SPEC = """
@prefix cp:   <http://taica.course/cp/ontology#> .
@prefix spec: <http://taica.course/cp/spec/test#> .

spec:axis a cp:ImageAugmentationAxis ; cp:hasRequirement spec:req .
spec:req a cp:LightingRequirement ;
    cp:lightName "rect_light" ;
    cp:intensityMin 2000.0 ; cp:intensityMax 6000.0 ;
    cp:colorTemperatureMin 3000.0 ; cp:colorTemperatureMax 6000.0 ;
    cp:numBinsIntensity 2 ; cp:numBinsColorTemperature 1 ;
    cp:minEpisodesPerCell 1 .
"""

PROPERTY_SPEC = """
@prefix cp:   <http://taica.course/cp/ontology#> .
@prefix spec: <http://taica.course/cp/spec/test#> .

spec:prop_axis a cp:ObjectPropertyAxis ; cp:hasRequirement spec:prop_req .
spec:prop_req a cp:ColorReferenceRequirement ;
    cp:hasTaskObject spec:fork ;
    cp:hasReference spec:by_name , spec:by_color ;
    cp:minEpisodesPerReference 1 .
spec:fork a cp:TaskObject ; cp:name "fork" .
spec:by_name a cp:NameReference ; cp:phrase "fork" .
spec:by_color a cp:ColorReference ; cp:phrase "pink object" .
"""


def write_dataset(root: Path, episodes: list[dict], frames: int = 2) -> Path:
    """episodes[i] = {"light": (intensity, temperature), "color": rgb, "task": text}; one object 'fork' at a fixed position."""
    features = {
        "observation.state": {"dtype": "float32", "shape": [9], "names": None},
        f"{OBJECT_POSITION_KEY_PREFIX}fork": {"dtype": "float32", "shape": [3], "names": OBJECT_POSITION_NAMES},
        f"{LIGHTING_KEY_PREFIX}rect_light": {"dtype": "float32", "shape": [2], "names": LIGHTING_NAMES},
        f"{OBJECT_COLOR_KEY_PREFIX}fork": {"dtype": "float32", "shape": [3], "names": OBJECT_COLOR_NAMES},
    }
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"codebase_version": "v3.0", "fps": 60, "robot_type": "franka_panda", "features": features}))
    tasks = sorted({e["task"] for e in episodes})
    pq.write_table(pa.table({"task_index": list(range(len(tasks))), "__index_level_0__": tasks}), root / "meta" / "tasks.parquet")
    columns: dict[str, list] = {"episode_index": [], "frame_index": [], "task_index": []}
    for key in features:
        if key != "observation.state":
            columns[key] = []
    for i, e in enumerate(episodes):
        for frame in range(frames):
            columns["episode_index"].append(i)
            columns["frame_index"].append(frame)
            columns["task_index"].append(tasks.index(e["task"]))
            columns[f"{OBJECT_POSITION_KEY_PREFIX}fork"].append([0.1, 0.1, 0.9])
            columns[f"{LIGHTING_KEY_PREFIX}rect_light"].append(list(e["light"]))
            columns[f"{OBJECT_COLOR_KEY_PREFIX}fork"].append(list(e["color"]))
    (root / "data" / "chunk-000").mkdir(parents=True)
    pq.write_table(pa.table(columns), root / "data" / "chunk-000" / "file-000.parquet")
    return root


def episode(light=(3000.0, 4500.0), task=BY_NAME, color=PINK) -> dict:
    return {"light": light, "color": color, "task": task}


BALANCED = [episode((3000.0, 4500.0), BY_NAME), episode((5000.0, 4500.0), BY_COLOR)]


def check(tmp_path: Path, episodes: list[dict], spec_text: str):
    store, _ = ingest_dataset(write_dataset(tmp_path / "ds", episodes), tmp_path / "stores")
    spec = tmp_path / "spec.ttl"
    spec.write_text(spec_text)
    (report,) = verify_store(store, [spec]).checks
    return store, report


def write_log(path: Path, runs: list[tuple[tuple[float, float], str | None, str]]) -> Path:
    """runs: (light, instruction, status) per run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [{"type": "header", "version": ROLLOUT_LOG_VERSION, "eval_file": "eval/x_eval.py", "eval_sha256": "0" * 64, "seed": 1}]
    for i, (light, instruction, status) in enumerate(runs):
        record = {
            "type": "episode", "episode": i, "status": status,
            "factors": {OBJECT_POSITION_FACTOR: {"fork": [0.1, 0.1, 0.9]}, LIGHTING_FACTOR: {"rect_light": list(light)}},
        }
        if instruction is not None:
            record["instruction"] = instruction
        lines.append(record)
    path.write_text("".join(json.dumps(r) + "\n" for r in lines))
    return path


def compare(tmp_path: Path, spec_text: str, runs, episodes=BALANCED):
    store, _ = check(tmp_path, episodes, spec_text)
    log = write_log(tmp_path / "rollout" / "run.jsonl", runs)
    (result,) = compare_store(store, [tmp_path / "spec.ttl"], [log]).comparisons
    return result


# ---- Ingest ---------------------------------------------------------------------------


def test_ingest_records_lights_colors_and_task_text(tmp_path: Path) -> None:
    store, _ = ingest_dataset(write_dataset(tmp_path / "ds", BALANCED), tmp_path / "stores")
    graph = store.load(store.dataset_file)
    assert [str(n) for n in graph.objects(store.dataset, CP.recordedLight)] == ["rect_light"]
    assert [str(n) for n in graph.objects(store.dataset, CP.recordedColorObject)] == ["fork"]
    light = next(graph.subjects(CP.lightIntensity, None))
    assert float(graph.value(light, CP.colorTemperature)) == 4500.0
    texts = {str(o) for o in graph.objects(None, CP.taskDescription)}
    assert {BY_NAME, BY_COLOR} <= texts


def test_dataset_without_the_lighting_column_still_ingests(tmp_path: Path) -> None:
    root = write_dataset(tmp_path / "ds", BALANCED)
    info = json.loads((root / "meta" / "info.json").read_text())
    del info["features"][f"{LIGHTING_KEY_PREFIX}rect_light"]
    (root / "meta" / "info.json").write_text(json.dumps(info))
    store, _ = ingest_dataset(root, tmp_path / "stores")
    spec = tmp_path / "spec.ttl"
    spec.write_text(LIGHT_SPEC)
    (report,) = verify_store(store, [spec]).checks
    assert not report.passed
    assert any("no 'lighting.rect_light' column" in e for e in report.errors)


# ---- Image Augmentations: lighting --------------------------------------------------------


def test_lighting_coverage_passes_and_fails(tmp_path: Path) -> None:
    _, report = check(tmp_path / "a", BALANCED, LIGHT_SPEC)
    assert report.passed, format_check(report)
    assert report.axes[0].axis == "image_augmentation"
    assert report.axes[0].coverage[0].counts == [[1, 1]]

    _, report = check(tmp_path / "b", [episode((3000.0, 4500.0)), episode((3500.0, 4500.0))], LIGHT_SPEC)
    assert not report.passed
    assert any("cell (i=1, j=0)" in v.message and "has 0 episode" in v.message for v in report.violations), format_check(report)


def test_lighting_outside_range_is_a_warning(tmp_path: Path) -> None:
    _, report = check(tmp_path, [*BALANCED, episode((9000.0, 4500.0))], LIGHT_SPEC)
    assert report.passed
    assert any("outside the required ranges" in w.message for w in report.warnings)


def test_lighting_spec_todo_and_empty_range_are_rejected(tmp_path: Path) -> None:
    store, _ = ingest_dataset(write_dataset(tmp_path / "ds", BALANCED), tmp_path / "stores")
    spec = tmp_path / "spec.ttl"
    spec.write_text(LIGHT_SPEC.replace("cp:numBinsIntensity 2", 'cp:numBinsIntensity "TODO"'))
    with pytest.raises(StoreError, match="TODO"):
        load_spec(store, spec)
    spec.write_text(LIGHT_SPEC.replace("cp:intensityMax 6000.0", "cp:intensityMax 1000.0"))
    with pytest.raises(StoreError, match="intensityMin must be < intensityMax"):
        load_spec(store, spec)


def test_lighting_comparison(tmp_path: Path) -> None:
    result = compare(tmp_path, LIGHT_SPEC, [((3000.0, 4500.0), BY_NAME, "succeeded"), ((9000.0, 4500.0), BY_NAME, "failed")])
    assert result.passed, format_comparison(result)
    assert result.runs_in_distribution == 1
    assert any("does not cover" in w.message for w in result.warnings)


# ---- Object Properties: color references -----------------------------------------------


def test_references_covered_passes(tmp_path: Path) -> None:
    _, report = check(tmp_path, BALANCED, PROPERTY_SPEC)
    assert report.passed, format_check(report)
    table = report.axes[0].coverage[0]
    assert report.axes[0].axis == "object_property"
    assert table.counts == {"fork": 1, "pink object": 1}


def test_reference_never_used_fails(tmp_path: Path) -> None:
    _, report = check(tmp_path, [episode(task=BY_NAME), episode(task=BY_NAME)], PROPERTY_SPEC)
    messages = [v.message for v in report.violations]
    assert any('"pink object" (color) is used by 0 episode(s)' in m for m in messages), messages


def test_object_must_not_change(tmp_path: Path) -> None:
    episodes = [*BALANCED, episode(task=BY_COLOR, color=(0.9, 0.1, 0.1))]
    _, report = check(tmp_path, episodes, PROPERTY_SPEC)
    assert any("does not look the same in every episode" in v.message for v in report.violations), format_check(report)


def test_unlisted_reference_is_a_warning(tmp_path: Path) -> None:
    _, report = check(tmp_path, [*BALANCED, episode(task="place the utensil on the left of the plate.")], PROPERTY_SPEC)
    assert report.passed, format_check(report)
    assert any("uses none of the spec's phrases" in w.message for w in report.warnings)


def test_reference_comparison_groups(tmp_path: Path) -> None:
    runs = [
        ((3000.0, 4500.0), BY_COLOR, "succeeded"),
        ((3000.0, 4500.0), BY_NAME, "failed"),
        ((3000.0, 4500.0), "place the utensil on the left of the plate.", "failed"),
    ]
    result = compare(tmp_path, PROPERTY_SPEC, runs)
    assert result.passed, format_comparison(result)
    assert result.runs_in_distribution == 2
    groups = result.axes[0].objects[0].groups
    assert (groups["pink object"].runs, groups["pink object"].successes) == (1, 1)
    assert (groups["fork"].runs, groups["fork"].successes) == (1, 0)
    assert groups["(unrecognized)"].runs == 1


def test_run_without_instruction_is_rejected(tmp_path: Path) -> None:
    result = compare(tmp_path, PROPERTY_SPEC, [((3000.0, 4500.0), None, "succeeded")])
    assert not result.passed
    assert any("has no instruction" in e for e in result.errors), format_comparison(result)


def test_property_spec_problems(tmp_path: Path) -> None:
    store, _ = ingest_dataset(write_dataset(tmp_path / "ds", BALANCED), tmp_path / "stores")
    spec = tmp_path / "spec.ttl"
    spec.write_text(PROPERTY_SPEC.replace('cp:phrase "pink object"', 'cp:phrase "TODO"'))
    with pytest.raises(StoreError, match="fork color reference: phrase"):
        load_spec(store, spec)
    spec.write_text(PROPERTY_SPEC.replace("cp:minEpisodesPerReference 1", "cp:minEpisodesPerReference 0"))
    with pytest.raises(StoreError, match="minEpisodesPerReference must be at least 1"):
        load_spec(store, spec)


def test_reference_can_set_its_own_minimum(tmp_path: Path) -> None:
    text = PROPERTY_SPEC.replace('spec:by_color a cp:ColorReference ; cp:phrase "pink object" .',
                                 'spec:by_color a cp:ColorReference ; cp:phrase "pink object" ; cp:minEpisodesPerReference 2 .')
    _, report = check(tmp_path, BALANCED, text)
    messages = [v.message for v in report.violations]
    assert messages == ['\'fork\' reference "pink object" (color) is used by 1 episode(s); needs at least 2'], messages
    assert "1 / 2     unseen" in "\n".join(report.axes[0].lines())


def test_name_only_requirement_is_warned(tmp_path: Path) -> None:
    text = PROPERTY_SPEC.replace("spec:by_name , spec:by_color", "spec:by_name")
    _, report = check(tmp_path, BALANCED, text)
    assert any("no cp:ColorReference" in w.message for w in report.warnings), format_check(report)


def test_phrase_match() -> None:
    assert contains_phrase("Place the Pink  Object on the left.", "pink object")
    assert not contains_phrase("place the pinkish object.", "pink object")
    assert not contains_phrase("place the forklift.", "fork")


# ---- Both axes in one spec, and the rollout log -------------------------------------


def test_one_spec_with_both_axes(tmp_path: Path) -> None:
    combined = LIGHT_SPEC + PROPERTY_SPEC.split("@prefix spec: <http://taica.course/cp/spec/test#> .", 1)[1]
    result = compare(tmp_path, combined, [((3000.0, 4500.0), BY_COLOR, "succeeded"), ((3000.0, 4500.0), "move it", "failed")])
    assert result.passed, format_comparison(result)
    assert {a.axis for a in result.axes} == {"image_augmentation", "object_property"}
    assert result.runs_in_distribution == 1


def test_log_rejects_bad_lighting_and_instruction(tmp_path: Path) -> None:
    log = write_log(tmp_path / "run.jsonl", [((3000.0, 4500.0), BY_NAME, "succeeded")])
    text = log.read_text()
    log.write_text(text.replace("[3000.0, 4500.0]", "[3000.0]"))
    with pytest.raises(RolloutLogError, match=r"must be \[intensity, color_temperature\]"):
        read_rollout_log(log)
    log.write_text(text.replace(json.dumps(BY_NAME), "42"))
    with pytest.raises(RolloutLogError, match="instruction must be a string"):
        read_rollout_log(log)


# ---- Shipped specs ----------------------------------------------------------------------

SHIPPED = sorted((Path(__file__).resolve().parents[1] / "specs").glob("*/*.ttl"))


@pytest.mark.parametrize("path", SHIPPED, ids=[f"{p.parent.name}/{p.name}" for p in SHIPPED])
def test_shipped_specs_parse_and_only_lack_values(path: Path) -> None:
    from rdflib import Graph

    from data_verification import registry

    spec = Graph().parse(path, format="turtle")
    verifier = next(v for v in registry.VERIFIERS if v.name == path.stem)
    assert any(spec.subjects(None, verifier.axis_class)), f"{path.name} declares no {verifier.axis_class}"
    errors = verifier.spec_errors(spec)
    assert len(errors) == 1 and errors[0].startswith("Spec still has TODO values"), errors


def test_every_task_ships_a_spec_per_axis() -> None:
    from data_verification import registry

    root = Path(__file__).resolve().parents[1] / "specs"
    for task in ("cup_stacking", "cutlery_arrangement", "toy_blocks_collection"):
        assert sorted(p.stem for p in (root / task).glob("*.ttl")) == sorted(v.name for v in registry.VERIFIERS)


# ---- Reference forms (simulator/utils/scene_factors.py, loaded without the simulator) ----------


def _scene_factors():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "packages/simulator/src/simulator/utils/scene_factors.py"
    spec = importlib.util.spec_from_file_location("scene_factors_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pick_reference_form_follows_weights() -> None:
    import random

    pick = _scene_factors().pick_reference_form
    rng = random.Random(0)
    assert {pick(rng, {"name": 1.0, "color": 0.0}) for _ in range(200)} == {"name"}, "weight 0 is never drawn"
    draws = [pick(rng, {"name": 0.5, "color": 0.5}) for _ in range(2000)]
    assert 0.45 < draws.count("color") / len(draws) < 0.55


def test_reference_form_errors() -> None:
    errors = _scene_factors().reference_form_errors
    refs = {"fork": {"name": "fork", "color": "pink object"}, "knife": {"name": "knife"}}
    assert errors(refs, {"name": 1.0, "color": 0.0}) == []
    assert errors(refs, {"name": 0.5, "color": 0.5}) == ["reference form 'color' has weight 0.5 but no phrase for knife in object_references"]
    assert errors(refs, {"name": 0.0}) == ["reference_weights {'name': 0.0} must be >= 0 with a positive sum"]
