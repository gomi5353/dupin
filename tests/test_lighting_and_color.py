"""Lighting and object color axes through the store (ingest, attach, verify) on synthetic LeRobot v3 layouts (no simulator)."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_verification import StoreError, attach_spec, format_check, format_comparison, ingest_dataset, ingest_rollout, verify_store
from data_verification.axes.lighting import LightingVerifier
from data_verification.axes.object_color import ObjectColorVerifier
from data_verification.axes.object_color.coverage import names_color, normalized_entropy
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
    OBJECT_COLOR_FACTOR,
    OBJECT_POSITION_FACTOR,
    ROLLOUT_LOG_VERSION,
    RolloutLogError,
    read_rollout_log,
)

RED, BLUE = (0.9, 0.1, 0.1), (0.1, 0.1, 0.9)

LIGHT_SPEC = """
@prefix cp:   <http://taica.course/cp/ontology#> .
@prefix spec: <http://taica.course/cp/spec/test#> .

spec:axis a cp:LightingAxis ; cp:hasRequirement spec:req .
spec:req a cp:LightingAxisRequirement ;
    cp:lightName "rect_light" ;
    cp:intensityMin 2000.0 ; cp:intensityMax 6000.0 ;
    cp:colorTemperatureMin 3000.0 ; cp:colorTemperatureMax 6000.0 ;
    cp:numBinsIntensity 2 ; cp:numBinsColorTemperature 1 ;
    cp:minEpisodesPerCell 1 .
"""

COLOR_SPEC = """
@prefix cp:   <http://taica.course/cp/ontology#> .
@prefix spec: <http://taica.course/cp/spec/test#> .

spec:axis a cp:ObjectColorAxis ; cp:hasRequirement spec:req .
spec:req a cp:ObjectColorAxisRequirement ;
    cp:hasTaskObject spec:cup ;
    cp:hasColorCategory spec:red , spec:blue ;
    cp:maxColorDistance 0.2 ;
    cp:minEpisodesPerColor 1 ;
    cp:minNormalizedEntropy 0.9 .
spec:cup a cp:TaskObject ; cp:name "cup" .
spec:red a cp:ColorCategory ; cp:name "red" ; cp:colorR 0.9 ; cp:colorG 0.1 ; cp:colorB 0.1 .
spec:blue a cp:ColorCategory ; cp:name "blue" ; cp:colorR 0.1 ; cp:colorG 0.1 ; cp:colorB 0.9 .
"""


def write_dataset(root: Path, episodes: list[dict], frames: int = 2) -> Path:
    """episodes[i] = {"light": (intensity, temperature), "color": rgb, "task": text}; one object 'cup' at a fixed position."""
    features = {
        "observation.state": {"dtype": "float32", "shape": [9], "names": None},
        f"{OBJECT_POSITION_KEY_PREFIX}cup": {"dtype": "float32", "shape": [3], "names": OBJECT_POSITION_NAMES},
        f"{LIGHTING_KEY_PREFIX}rect_light": {"dtype": "float32", "shape": [2], "names": LIGHTING_NAMES},
        f"{OBJECT_COLOR_KEY_PREFIX}cup": {"dtype": "float32", "shape": [3], "names": OBJECT_COLOR_NAMES},
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
            columns[f"{OBJECT_POSITION_KEY_PREFIX}cup"].append([0.1, 0.1, 0.9])
            columns[f"{LIGHTING_KEY_PREFIX}rect_light"].append(list(e["light"]))
            columns[f"{OBJECT_COLOR_KEY_PREFIX}cup"].append(list(e["color"]))
    (root / "data" / "chunk-000").mkdir(parents=True)
    pq.write_table(pa.table(columns), root / "data" / "chunk-000" / "file-000.parquet")
    return root


def episode(light=(3000.0, 4500.0), color=RED, task=None) -> dict:
    name = "red" if color == RED else "blue" if color == BLUE else "grey"
    return {"light": light, "color": color, "task": task or f"pick up the {name} cup."}


BALANCED = [episode((3000.0, 4500.0), RED), episode((5000.0, 4500.0), BLUE)]


def check(tmp_path: Path, episodes: list[dict], spec_text: str):
    store, _ = ingest_dataset(write_dataset(tmp_path / "ds", episodes), tmp_path / "stores")
    spec = tmp_path / "spec.ttl"
    spec.write_text(spec_text)
    attach_spec(store, spec)
    (report,) = verify_store(store, rollouts=[]).checks
    return store, report


def write_log(path: Path, runs: list[tuple[tuple[float, float], tuple[float, float, float], str]]) -> Path:
    """runs: (light, color, status) per run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [{"type": "header", "version": ROLLOUT_LOG_VERSION, "eval_file": "eval/cup_eval.py", "eval_sha256": "0" * 64, "seed": 1}]
    for i, (light, color, status) in enumerate(runs):
        factors = {
            OBJECT_POSITION_FACTOR: {"cup": [0.1, 0.1, 0.9]},
            LIGHTING_FACTOR: {"rect_light": list(light)},
            OBJECT_COLOR_FACTOR: {"cup": list(color)},
        }
        lines.append({"type": "episode", "episode": i, "status": status, "factors": factors})
    path.write_text("".join(json.dumps(r) + "\n" for r in lines))
    return path


def compare(tmp_path: Path, spec_text: str, runs, episodes=BALANCED):
    store, _ = check(tmp_path, episodes, spec_text)
    log = write_log(tmp_path / "rollout" / "run.jsonl", runs)
    ingest_rollout(store, log)
    (result,) = verify_store(store, rollouts=[log.stem]).comparisons
    return result


# ---- Ingest ---------------------------------------------------------------------------


def test_ingest_records_lights_colors_and_task_text(tmp_path: Path) -> None:
    store, _ = ingest_dataset(write_dataset(tmp_path / "ds", BALANCED), tmp_path / "stores")
    graph = store.load(store.dataset_file)
    assert [str(n) for n in graph.objects(store.dataset, CP.recordedLight)] == ["rect_light"]
    assert [str(n) for n in graph.objects(store.dataset, CP.recordedColorObject)] == ["cup"]
    light = next(graph.subjects(CP.lightIntensity, None))
    assert float(graph.value(light, CP.colorTemperature)) == 4500.0
    texts = {str(o) for o in graph.objects(None, CP.taskDescription)}
    assert "pick up the red cup." in texts and "pick up the blue cup." in texts


def test_dataset_without_the_new_columns_still_ingests(tmp_path: Path) -> None:
    root = write_dataset(tmp_path / "ds", BALANCED)
    info = json.loads((root / "meta" / "info.json").read_text())
    del info["features"][f"{LIGHTING_KEY_PREFIX}rect_light"]
    (root / "meta" / "info.json").write_text(json.dumps(info))
    store, _ = ingest_dataset(root, tmp_path / "stores")
    spec = tmp_path / "spec.ttl"
    spec.write_text(LIGHT_SPEC)
    attach_spec(store, spec)
    (report,) = verify_store(store, rollouts=[]).checks
    assert not report.passed
    assert any("no 'lighting.rect_light' column" in e for e in report.errors)


# ---- Lighting -------------------------------------------------------------------------


def test_lighting_coverage_passes_and_fails(tmp_path: Path) -> None:
    _, report = check(tmp_path / "a", BALANCED, LIGHT_SPEC)
    assert report.passed, format_check(report)
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
        attach_spec(store, spec)
    spec.write_text(LIGHT_SPEC.replace("cp:intensityMax 6000.0", "cp:intensityMax 1000.0"))
    with pytest.raises(StoreError, match="intensityMin must be < intensityMax"):
        attach_spec(store, spec)


def test_lighting_comparison(tmp_path: Path) -> None:
    result = compare(tmp_path, LIGHT_SPEC, [((3000.0, 4500.0), RED, "succeeded"), ((9000.0, 4500.0), RED, "failed")])
    assert result.passed, format_comparison(result)
    assert result.runs_in_distribution == 1
    assert any("does not cover" in w.message for w in result.warnings)


# ---- Object color ---------------------------------------------------------------------


def test_color_balanced_named_passes(tmp_path: Path) -> None:
    _, report = check(tmp_path, BALANCED, COLOR_SPEC)
    assert report.passed, format_check(report)
    table = report.axes[0].coverage[0]
    assert table.counts == {"red": 1, "blue": 1}
    assert table.entropy == pytest.approx(1.0)


def test_color_low_entropy_and_missing_color_fail(tmp_path: Path) -> None:
    _, report = check(tmp_path, [episode(color=RED), episode(color=RED), episode(color=RED)], COLOR_SPEC)
    messages = [v.message for v in report.violations]
    assert any("'blue' has 0 episode" in m for m in messages), messages
    assert any("normalized entropy 0.000" in m for m in messages), messages


def test_color_language_check(tmp_path: Path) -> None:
    episodes = [episode(color=RED), episode(color=BLUE, task="pick up the red cup.")]
    _, report = check(tmp_path, episodes, COLOR_SPEC)
    assert any("does not say 'blue'" in v.message for v in report.violations), format_check(report)


def test_color_without_category_is_a_warning(tmp_path: Path) -> None:
    _, report = check(tmp_path, [*BALANCED, episode(color=(0.5, 0.5, 0.5))], COLOR_SPEC)
    assert report.passed
    assert any("within 0.2 of no category" in w.message for w in report.warnings)


def test_color_comparison_groups(tmp_path: Path) -> None:
    runs = [((3000.0, 4500.0), RED, "succeeded"), ((3000.0, 4500.0), (0.1, 0.9, 0.1), "failed")]
    result = compare(tmp_path, COLOR_SPEC, runs)
    assert result.passed, format_comparison(result)
    assert result.runs_in_distribution == 1
    groups = result.axes[0].objects[0].groups
    assert (groups["red"].runs, groups["red"].successes) == (1, 1)
    assert (groups["(unseen)"].runs, groups["(unseen)"].successes) == (1, 0)


def test_color_spec_needs_numbers(tmp_path: Path) -> None:
    store, _ = ingest_dataset(write_dataset(tmp_path / "ds", BALANCED), tmp_path / "stores")
    spec = tmp_path / "spec.ttl"
    spec.write_text(COLOR_SPEC.replace("cp:minNormalizedEntropy 0.9", "cp:minNormalizedEntropy 1.5"))
    with pytest.raises(StoreError, match="between 0 and 1"):
        attach_spec(store, spec)


def test_entropy_and_word_match() -> None:
    assert normalized_entropy([5, 5]) == pytest.approx(1.0)
    assert normalized_entropy([10, 0]) == 0.0
    assert normalized_entropy([3]) == 0.0
    assert names_color("Pick up the Red cup.", "red")
    assert not names_color("pick up the reddish cup.", "red")


# ---- Both axes in one spec, and the rollout log -------------------------------------


def test_one_spec_with_both_axes(tmp_path: Path) -> None:
    combined = LIGHT_SPEC + COLOR_SPEC.split("spec:axis a cp:ObjectColorAxis ; cp:hasRequirement spec:req .", 1)[1].replace(
        "spec:req", "spec:color_req"
    ) + "spec:color_axis a cp:ObjectColorAxis ; cp:hasRequirement spec:color_req .\n"
    result = compare(tmp_path, combined, [((3000.0, 4500.0), RED, "succeeded"), ((5000.0, 4500.0), (0.1, 0.9, 0.1), "failed")])
    assert result.passed, format_comparison(result)
    assert {a.axis for a in result.axes} == {"lighting", "object_color"}
    assert result.runs_in_distribution == 1


def test_log_rejects_bad_lighting(tmp_path: Path) -> None:
    log = write_log(tmp_path / "run.jsonl", [((3000.0, 4500.0), RED, "succeeded")])
    log.write_text(log.read_text().replace("[3000.0, 4500.0]", "[3000.0]"))
    with pytest.raises(RolloutLogError, match=r"must be \[intensity, color_temperature\]"):
        read_rollout_log(log)


def test_init_templates_render(tmp_path: Path) -> None:
    store, _ = ingest_dataset(write_dataset(tmp_path / "ds", BALANCED), tmp_path / "stores")
    graph = store.load(store.dataset_file)
    light = LightingVerifier()
    text = light.render_spec(graph, store.dataset, light.recorded_names(graph, store.dataset))
    assert 'cp:lightName "rect_light"' in text and "Already in the dataset: 2 episode(s)" in text
    color = ObjectColorVerifier()
    text = color.render_spec(graph, store.dataset, color.recorded_names(graph, store.dataset))
    assert 'cp:name "cup"' in text and "2 distinct color(s)" in text


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
