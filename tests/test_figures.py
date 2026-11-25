"""Coverage heatmaps written by verify."""

from __future__ import annotations

from pathlib import Path

from data_verification import verify_store
from data_verification.figures import save_figures
from data_verification.verify import format_check
from test_object_position_coverage import CELL_CENTERS
from test_rollout_comparison import compare, spec, store_with, stored_specs, write_log  # noqa: F401


def test_dataset_check_writes_one_png_per_object(tmp_path: Path, spec: Path) -> None:
    store = store_with(tmp_path, spec)
    result = verify_store(store, stored_specs(store))
    paths = save_figures(store.root, result)
    assert [p.name for p in paths] == ["spec__cube.png"]
    assert paths[0].parent == store.root / "figures" and paths[0].stat().st_size > 1000


def test_comparison_adds_one_png_per_rollout_and_object(tmp_path: Path, spec: Path) -> None:
    store = store_with(tmp_path, spec)
    log = write_log(tmp_path / "rollout" / "run.jsonl", [(x, y, "failed") for x, y in CELL_CENTERS] + [(50.0, 50.0, "succeeded")])
    compare(store, log)
    paths = save_figures(store.root, verify_store(store, stored_specs(store), rollouts=[log.stem]))
    assert sorted(p.name for p in paths) == ["spec__cube.png", "spec__run__cube.png"]


def test_figures_of_an_earlier_verify_are_replaced(tmp_path: Path, spec: Path) -> None:
    store = store_with(tmp_path, spec)
    stale = store.root / "figures" / "old.png"
    stale.parent.mkdir()
    stale.write_bytes(b"x")
    save_figures(store.root, verify_store(store, stored_specs(store)))
    assert not stale.exists()


def test_report_text_no_longer_holds_grid_tables(tmp_path: Path, spec: Path) -> None:
    store = store_with(tmp_path, spec)
    (check,) = verify_store(store, stored_specs(store)).checks
    assert "x0" not in format_check(check)


def test_unquoted_text_in_a_spec_is_explained(tmp_path: Path, spec: Path) -> None:
    import pytest
    from data_verification import StoreError, load_spec

    store = store_with(tmp_path, spec)
    bad = tmp_path / "bad.ttl"
    bad.write_text('@prefix cp: <http://taica.course/cp/ontology#> .\n@prefix spec: <http://x#> .\nspec:a a cp:TaskObject ;\n    cp:name cube .\n')
    with pytest.raises(StoreError) as error:
        load_spec(store, bad)
    assert 'write "cube" instead of cube' in str(error.value)
