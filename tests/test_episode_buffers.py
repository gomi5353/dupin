"""Per-environment LeRobot episode buffers (multi-env recording) on a real LeRobotDataset."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "episode_buffers", ROOT / "packages/simulator/src/simulator/utils/episode_buffers.py"
)
episode_buffers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(episode_buffers)
EpisodeBuffers, TEMP_EPISODE_INDEX = episode_buffers.EpisodeBuffers, episode_buffers.TEMP_EPISODE_INDEX

lerobot_dataset = pytest.importorskip("lerobot.datasets.lerobot_dataset")

FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (2,), "names": None},
    "observation.images.front": {"dtype": "video", "shape": (64, 64, 3), "names": ["height", "width", "channels"]},
}


def make_dataset(root: Path):
    return lerobot_dataset.LeRobotDataset.create(repo_id="test/multi_env", fps=10, features=FEATURES, root=root, use_videos=True)


def frame(env_id: int, step: int, task: str) -> dict:
    image = np.full((64, 64, 3), 40 * env_id + step, dtype=np.uint8)
    return {"observation.state": np.array([env_id, step], dtype=np.float32), "observation.images.front": image, "task": task}


def test_interleaved_envs_save_in_finish_order_and_failed_ones_leave_nothing(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    dataset = make_dataset(root)
    buffers = EpisodeBuffers(dataset)
    # env 0: 3 frames, env 1: 5 frames, env 2: 4 frames, all stepping together
    lengths = {0: 3, 1: 5, 2: 4}
    for step in range(5):
        for env_id, n in lengths.items():
            if step < n:
                buffers.add_frame(env_id, frame(env_id, step, f"task of env {env_id}"))
    assert [buffers.frames(e) for e in range(3)] == [3, 5, 4]
    assert (root / f"images/observation.images.front/episode-{TEMP_EPISODE_INDEX + 1:06d}").is_dir()

    assert buffers.save(2)  # env 2 finishes first and becomes episode 0
    buffers.discard(0)  # env 0 failed
    assert buffers.save(1)  # env 1 becomes episode 1
    assert not buffers.save(0)  # nothing left to save
    buffers.add_frame(0, frame(0, 0, "task of env 0"))  # env 0's next episode
    buffers.add_frame(0, frame(0, 1, "task of env 0"))
    assert buffers.save(0)
    dataset.finalize()

    assert not list(root.rglob("*.png"))  # temporary frames removed after encoding
    loaded = lerobot_dataset.LeRobotDataset("test/multi_env", root=root)
    assert loaded.meta.total_episodes == 3 and loaded.meta.total_frames == 4 + 5 + 2
    episodes = np.asarray(loaded.hf_dataset["episode_index"])
    states = np.stack(loaded.hf_dataset["observation.state"])
    assert sorted(set(states[episodes == 0, 0])) == [2]
    assert sorted(set(states[episodes == 1, 0])) == [1]
    assert sorted(set(states[episodes == 2, 0])) == [0]
    assert list(states[episodes == 1, 1]) == [0, 1, 2, 3, 4]  # frame order kept
    tasks = [loaded[i]["task"] for i in range(len(loaded))]
    assert tasks[0] == "task of env 2" and tasks[4] == "task of env 1" and tasks[-1] == "task of env 0"
    image = loaded[4]["observation.images.front"]  # first frame of env 1: value 40
    assert abs(float(image.mean()) * 255 - 40) < 4


def test_discard_all_removes_unsaved_images(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    buffers = EpisodeBuffers(make_dataset(root))
    buffers.add_frame(0, frame(0, 0, "t"))
    buffers.add_frame(3, frame(3, 0, "t"))
    buffers.discard_all()
    assert not list((root / "images").rglob("*.png"))
    assert buffers.frames(0) == buffers.frames(3) == 0
