"""Terminal tables of verify and compare (rich); the plain-text reports stay the reference."""

from __future__ import annotations

import io
from pathlib import Path

from rich.console import Console

from data_verification import compare_store, verify_store
from data_verification.render import make_console, render_check, render_combined, render_comparison, use_tables
from test_image_augmentation_and_object_property import BY_COLOR, BY_NAME, LIGHT_SPEC, PROPERTY_SPEC, check, compare, write_log
from test_object_position_coverage import CELL_CENTERS
from test_rollout_comparison import full_dataset, spec, store_with, write_log as write_pose_log  # noqa: F401

BOTH = LIGHT_SPEC + PROPERTY_SPEC.split("@prefix spec: <http://taica.course/cp/spec/test#> .", 1)[1]


def draw(render, *args, **kwargs) -> str:
    out = io.StringIO()
    render(*args, Console(file=out, width=140, highlight=False), **kwargs)
    return out.getvalue()


def test_check_report_has_a_table_per_axis(tmp_path: Path) -> None:
    _, report = check(tmp_path, [{"light": (3000.0, 4500.0), "color": (1.87, 0.4, 1.88), "task": BY_NAME}], BOTH)
    text = draw(render_check, report)
    assert "Image Augmentations: lighting coverage" in text and "Object Properties: references of 'fork'" in text
    assert "NOT met" in text and "unseen" in text and "RESULT: coverage spec NOT met" in text
    assert "Findings" in text and "VIOLATION" in text


def test_pose_comparison_table_has_the_numbers_of_the_text_report(tmp_path: Path, spec: Path) -> None:
    store = store_with(tmp_path, spec, full_dataset(tmp_path, CELL_CENTERS[:3]))
    log = write_pose_log(tmp_path / "run.jsonl", [(0.15, 0.15, "failed"), (0.05, 0.05, "succeeded"), (50.0, 50.0, "failed")])
    (result,) = compare_store(store, [spec], [log]).comparisons
    text = draw(render_comparison, result)
    assert "Object Poses: rollout runs" in text and "In distribution (covered on every axis: object_pose)" in text
    assert f"{result.runs_in_distribution}/{result.runs}" in text and "started outside your area" in text
    assert "Suggested next steps" in text and "Record more episodes here" in text
    assert "Suggested next steps" not in draw(render_comparison, result, hints=False)
    assert "RESULT: the dataset does NOT cover every rollout run" in text


def test_both_axes_compare_and_combined_render(tmp_path: Path) -> None:
    result = compare(tmp_path, BOTH, [((3000.0, 4500.0), BY_COLOR, "succeeded"), ((3000.0, 4500.0), "move it", "failed")])
    text = draw(render_comparison, result)
    assert "Image Augmentations: rollout runs" in text and "Object Properties: rollout runs by reference of 'fork'" in text
    assert "seen" in text and "unseen" in text


def test_combined_block(tmp_path: Path) -> None:
    a, b = tmp_path / "a.ttl", tmp_path / "b.ttl"
    a.write_text(LIGHT_SPEC)
    b.write_text(PROPERTY_SPEC)
    store, _ = check(tmp_path, [{"light": (3000.0, 4500.0), "color": (1.87, 0.4, 1.88), "task": BY_NAME}], LIGHT_SPEC)
    log = write_log(tmp_path / "rollout" / "run.jsonl", [((3000.0, 4500.0), BY_NAME, "succeeded")])
    (combined,) = compare_store(store, [a, b], [log]).combined
    text = draw(render_combined, combined)
    assert "Combined (all spec files together)" in text and "a, b" in text and "RESULT:" in text


def test_brackets_in_messages_are_not_markup(tmp_path: Path) -> None:
    _, report = check(tmp_path, [{"light": (3000.0, 4500.0), "color": (1.87, 0.4, 1.88), "task": BY_NAME}], LIGHT_SPEC)
    report.errors.append("bad [bold]value[/bold] and [spec/x]")
    assert "[bold]value[/bold]" in draw(render_check, report)


def test_layout_modes() -> None:
    assert use_tables("table") and not use_tables("text")
    assert not use_tables("auto", io.StringIO()), "redirected output stays plain text"
    assert make_console(io.StringIO()).width == 140
