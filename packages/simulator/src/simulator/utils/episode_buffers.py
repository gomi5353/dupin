"""One LeRobot episode buffer per parallel environment.

``LeRobotDataset`` holds a single ``episode_buffer``, and its camera frames
are written to ``images/<camera>/episode-<index>/`` as they arrive, where
``<index>`` is the episode's final index. With several environments recording
at once, every environment needs its own buffer and its own image directory,
and only the episodes that succeed may take the next index.

``EpisodeBuffers`` keeps one buffer per environment under a temporary episode
index (``TEMP_EPISODE_INDEX + env_id``). ``save(env_id)`` gives the episode the
dataset's next index, moves its images there, and saves it; ``discard(env_id)``
drops it with its images. Environments therefore finish and save in any
order, and a failed episode in one environment leaves the others untouched.

Only needs ``lerobot``; IsaacLab is not imported, so this is testable on the host.
"""

from __future__ import annotations

import shutil
from pathlib import Path

TEMP_EPISODE_INDEX = 1_000_000
"""Episode index of environment 0's unsaved buffer; environment ``i`` uses this + ``i``."""


class EpisodeBuffers:
    def __init__(self, dataset) -> None:
        if getattr(dataset, "_streaming_encoder", None) is not None:
            raise ValueError("EpisodeBuffers needs a LeRobotDataset without streaming encoding (one encoder per dataset)")
        self.dataset = dataset
        self._buffers: dict[int, dict] = {}

    def _camera_keys(self) -> list[str]:
        return [key for key, ft in self.dataset.features.items() if ft["dtype"] in ("image", "video")]

    def _image_dir(self, episode_index: int, key: str) -> Path:
        return self.dataset._get_image_file_path(episode_index, key, frame_index=0).parent

    def _remove_images(self, episode_index: int) -> None:
        for key in self._camera_keys():
            shutil.rmtree(self._image_dir(episode_index, key), ignore_errors=True)

    def frames(self, env_id: int) -> int:
        buffer = self._buffers.get(env_id)
        return 0 if buffer is None else buffer["size"]

    def add_frame(self, env_id: int, frame: dict) -> None:
        if env_id not in self._buffers:
            temp = TEMP_EPISODE_INDEX + env_id
            # Frames of an earlier unsaved episode (e.g. after a crash) must not end up in this one.
            self._remove_images(temp)
            self._buffers[env_id] = self.dataset.create_episode_buffer(episode_index=temp)
        dataset = self.dataset
        dataset.episode_buffer = self._buffers[env_id]
        try:
            dataset.add_frame(frame)
        finally:
            self._buffers[env_id] = dataset.episode_buffer
            dataset.episode_buffer = None

    def save(self, env_id: int) -> bool:
        """Save ``env_id``'s episode as the dataset's next episode; False if it has no frame."""
        buffer = self._buffers.pop(env_id, None)
        if buffer is None or buffer["size"] == 0:
            return False
        dataset = self.dataset
        temp, index = buffer["episode_index"], dataset.meta.total_episodes
        if dataset.image_writer is not None:
            dataset._wait_image_writer()
        for key in self._camera_keys():
            source, target = self._image_dir(temp, key), self._image_dir(index, key)
            shutil.rmtree(target, ignore_errors=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
            buffer[key] = [str(target / Path(p).name) for p in buffer[key]]
        buffer["episode_index"] = index
        dataset.episode_buffer = buffer
        dataset.save_episode(parallel_encoding=False)
        return True

    def discard(self, env_id: int) -> None:
        buffer = self._buffers.pop(env_id, None)
        if buffer is not None:
            self._remove_images(buffer["episode_index"])

    def discard_all(self) -> None:
        for env_id in list(self._buffers):
            self.discard(env_id)
