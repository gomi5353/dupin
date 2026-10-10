"""Static regressions for datagen wiring that can run without Isaac Sim."""

from __future__ import annotations

import ast
from pathlib import Path


GENERATE = Path(__file__).resolve().parents[1] / "scripts" / "datagen" / "generate.py"


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def test_datagen_uses_object_pose_recorder() -> None:
    tree = ast.parse(GENERATE.read_text())
    function = _function(tree, "_replace_recorder_manager")
    imports = {
        (node.module, alias.name)
        for node in ast.walk(function)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }

    assert ("simulator.utils.object_pose_recording", "LeRobotRecorderManager") in imports
    assert ("leisaac.enhance.managers.lerobot_recorder_manager", "LeRobotRecorderManager") not in imports


def test_final_episode_is_reset_before_return() -> None:
    tree = ast.parse(GENERATE.read_text())
    function = _function(tree, "_on_episode_done")
    reset_line = next(
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "env"
        and node.func.attr == "reset"
    )
    final_episode_check_line = next(
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == "next_episode_idx"
        and any(isinstance(op, ast.GtE) for op in node.ops)
    )

    assert reset_line < final_episode_check_line
