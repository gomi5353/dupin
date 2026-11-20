"""Read initial object placements from a local LeRobot v3 dataset.

Only ``meta/info.json`` and the ``data/**/*.parquet`` shards are read; videos
are never decoded. The initial placement of an object in an episode is its
``object_position.<name>`` value in that episode's first recorded frame (lowest
``frame_index``), whatever the collection pipeline was and whether the
episode succeeded or failed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Keep in sync with simulator.utils.lerobot_recorder and simulator.utils.scene_factors.
OBJECT_POSITION_KEY_PREFIX = "object_position."
OBJECT_POSITION_NAMES = ["x", "y", "z"]
LIGHTING_KEY_PREFIX = "lighting."
LIGHTING_NAMES = ["intensity", "color_temperature"]
OBJECT_COLOR_KEY_PREFIX = "object_color."
OBJECT_COLOR_NAMES = ["r", "g", "b"]


class DatasetError(ValueError):
    """The dataset cannot be read for object position verification."""


@dataclass(frozen=True)
class InitialPlacement:
    episode_index: int
    object_name: str
    x: float
    y: float
    z: float

    @property
    def is_finite(self) -> bool:
        return all(math.isfinite(v) for v in (self.x, self.y, self.z))


@dataclass(frozen=True)
class DatasetPlacements:
    episode_indices: list[int]
    object_names: list[str]
    placements: list[InitialPlacement]
    first_frame_index: dict[int, int]
    frame_counts: dict[int, int] = field(default_factory=dict)


def read_initial_placements(dataset_root: str | Path) -> DatasetPlacements:
    root = Path(dataset_root)
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise DatasetError(f"Not a LeRobot dataset: {info_path} does not exist")
    features = json.loads(info_path.read_text()).get("features", {})

    position_keys = sorted(k for k in features if k.startswith(OBJECT_POSITION_KEY_PREFIX))
    if not position_keys:
        raise DatasetError(
            f"meta/info.json declares no '{OBJECT_POSITION_KEY_PREFIX}<name>' features; "
            "record with simulator.utils.lerobot_recorder.LeRobotRecorderManager, not leisaac's recorder"
        )
    for key in position_keys:
        shape = list(features[key].get("shape", []))
        if shape != [len(OBJECT_POSITION_NAMES)]:
            raise DatasetError(f"Feature '{key}' has shape {shape}; expected [{len(OBJECT_POSITION_NAMES)}]")

    shards = sorted((root / "data").glob("**/*.parquet"))
    if not shards:
        raise DatasetError(f"No parquet shards under {root / 'data'}")
    columns = ["episode_index", "frame_index", *position_keys]
    tables = []
    for shard in shards:
        missing = sorted(set(columns) - set(pq.read_schema(shard).names))
        if missing:
            raise DatasetError(f"{shard.relative_to(root)} is missing columns: {', '.join(missing)}")
        tables.append(pq.read_table(shard, columns=columns))
    table = pa.concat_tables(tables)

    episodes = table.column("episode_index").to_numpy()
    frames = table.column("frame_index").to_numpy()
    order = np.lexsort((frames, episodes))
    episode_ids, first, counts = np.unique(episodes[order], return_index=True, return_counts=True)
    first_rows = table.take(pa.array(order[first])).to_pydict()

    object_names = [key[len(OBJECT_POSITION_KEY_PREFIX) :] for key in position_keys]
    placements = []
    for row, episode_index in enumerate(first_rows["episode_index"]):
        for key, name in zip(position_keys, object_names):
            position = first_rows[key][row]
            placements.append(
                InitialPlacement(
                    episode_index=int(episode_index),
                    object_name=name,
                    x=float("nan") if position is None else float(position[0]),
                    y=float("nan") if position is None else float(position[1]),
                    z=float("nan") if position is None else float(position[2]),
                )
            )
    return DatasetPlacements(
        episode_indices=[int(e) for e in episode_ids],
        object_names=object_names,
        placements=placements,
        first_frame_index={int(e): int(f) for e, f in zip(first_rows["episode_index"], first_rows["frame_index"])},
        frame_counts={int(e): int(n) for e, n in zip(episode_ids, counts)},
    )


def _shards(root: Path) -> list[Path]:
    shards = sorted((root / "data").glob("**/*.parquet"))
    if not shards:
        raise DatasetError(f"No parquet shards under {root / 'data'}")
    return shards


def _first_rows(root: Path, columns: list[str]) -> dict[str, list]:
    """``columns`` of each episode's first frame (lowest frame_index), one list entry per episode."""
    wanted = ["episode_index", "frame_index", *[c for c in columns if c not in ("episode_index", "frame_index")]]
    tables = []
    for shard in _shards(root):
        missing = sorted(set(wanted) - set(pq.read_schema(shard).names))
        if missing:
            raise DatasetError(f"{shard.relative_to(root)} is missing columns: {', '.join(missing)}")
        tables.append(pq.read_table(shard, columns=wanted))
    table = pa.concat_tables(tables)
    episodes = table.column("episode_index").to_numpy()
    frames = table.column("frame_index").to_numpy()
    order = np.lexsort((frames, episodes))
    _, first = np.unique(episodes[order], return_index=True)
    return table.take(pa.array(order[first])).to_pydict()


def read_first_frame_vectors(dataset_root: str | Path, prefix: str, width: int) -> dict[str, dict[int, tuple[float, ...]]]:
    """``<prefix><name>`` columns of ``width`` floats at each episode's first frame: name -> episode -> values.

    Returns an empty dict when the dataset declares no such column (an older
    recording, or a factor the task does not record). Missing values are NaN.
    """
    root = Path(dataset_root)
    features = json.loads((root / "meta" / "info.json").read_text()).get("features", {})
    keys = sorted(k for k in features if k.startswith(prefix))
    if not keys:
        return {}
    for key in keys:
        shape = list(features[key].get("shape", []))
        if shape != [width]:
            raise DatasetError(f"Feature '{key}' has shape {shape}; expected [{width}]")
    rows = _first_rows(root, keys)
    result: dict[str, dict[int, tuple[float, ...]]] = {}
    for key in keys:
        values = result.setdefault(key[len(prefix) :], {})
        for episode_index, value in zip(rows["episode_index"], rows[key]):
            values[int(episode_index)] = tuple(float("nan") for _ in range(width)) if value is None else tuple(float(v) for v in value)
    return result


def read_episode_tasks(dataset_root: str | Path) -> dict[int, str]:
    """Task text of each episode's first frame (``task_index`` -> ``meta/tasks.parquet``); empty without task data."""
    root = Path(dataset_root)
    tasks_path = root / "meta" / "tasks.parquet"
    if not tasks_path.is_file():
        return {}
    tasks = pq.read_table(tasks_path).to_pydict()
    # LeRobot v3 writes the task text as the pandas index of tasks.parquet; older writers use a "task" column.
    texts = tasks.get("task") or tasks.get("__index_level_0__")
    if "task_index" not in tasks or texts is None:
        return {}
    by_index = {int(i): str(t) for i, t in zip(tasks["task_index"], texts)}
    if not all("task_index" in pq.read_schema(shard).names for shard in _shards(root)):
        return {}
    rows = _first_rows(root, ["task_index"])
    return {int(e): by_index[int(t)] for e, t in zip(rows["episode_index"], rows["task_index"]) if t is not None and int(t) in by_index}
