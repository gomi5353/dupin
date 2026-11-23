"""LeRobot recorder with extra columns for offline data verification.

This is the repository's base extension of leisaac's ``LeRobotRecorderManager``.
It adds the columns of three verification factors: the object position, the
lighting and the object color; a further factor is appended here.

Every tracked object gets its own dataset column ``object_position.<name>``:
float32 ``(3,)`` = ``[x, y, z]``, position relative to the env origin, metres.
Orientation is not recorded: the tasks randomize only object positions and keep
each object's start rotation fixed (a random rotation can tip an object off the
table, and the reset has to be retried), so the orientation carries no
information about coverage.

The ``object_position.`` prefix is deliberate. LeRobot turns every ``observation.*``
key and every ``action*`` key into a policy input/output
(``lerobot.datasets.utils.dataset_to_policy_features``); any other column is
ignored by policies. These columns are therefore safe for training and rollout,
and are read back by ``packages/data_verification``. New factors use their own
non-``observation.`` prefix, such as ``factor.<name>``, for the same reason.

A factor needs two additions: its feature spec in ``build_with_factors`` below,
and its per-frame value, either from the episode buffer in
``SingleArmFrankaTaskEnvCfg.build_lerobot_frame``
(``tasks/template/single_arm_franka_cfg.py``), like the object positions, or
read from the scene in ``scene_frame`` below, like the lighting
(``lighting.<light>``) and the object colors (``object_color.<name>``,
``scene_factors.py``). The recorder also writes each episode's own task text
(``task_text``), which names the object colors.

Record with ``LeRobotRecorderManager`` from this module, not leisaac's original
class, which writes no verification columns and records env 0 only. This one
records every env (``--num_envs N``): each env has its own episode buffer
(``episode_buffers.py``), and each env's episode is saved when it succeeded.

Keep ``OBJECT_POSITION_KEY_PREFIX`` / ``OBJECT_POSITION_NAMES`` in sync with
``data_verification.dataset``.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import numpy as np

from .scene_factors import lighting_features, object_color_features

if TYPE_CHECKING:
    from isaaclab.utils.datasets.episode_data import EpisodeData

OBJECT_POSITION_KEY_PREFIX = "object_position."
OBJECT_POSITION_NAMES = ["x", "y", "z"]


def object_position_features(object_names: list[str]) -> dict[str, dict]:
    """LeRobot feature specs for the ``object_position.<name>`` columns."""
    return {
        f"{OBJECT_POSITION_KEY_PREFIX}{name}": {
            "dtype": "float32",
            "shape": (len(OBJECT_POSITION_NAMES),),
            "names": list(OBJECT_POSITION_NAMES),
        }
        for name in object_names
    }


def object_positions_from_episode(episode_data: EpisodeData, dataset_features: dict) -> dict[str, np.ndarray]:
    """Latest object positions from an IsaacLab ``EpisodeData`` buffer.

    Reads the ``states`` entry written by ``PostStepStatesRecorder`` (part of
    ``ActionStateRecorderManagerCfg``), which stores env-relative root poses
    ``[x, y, z, qw, qx, qy, qz]``; only the position is kept. Only columns declared in ``dataset_features`` are returned.
    """
    rigid_states = episode_data._data.get("states", {}).get("rigid_object", {})
    frame = {}
    for key in dataset_features:
        if not key.startswith(OBJECT_POSITION_KEY_PREFIX):
            continue
        name = key[len(OBJECT_POSITION_KEY_PREFIX) :]
        frame[key] = rigid_states[name]["root_pose"][-1][:3].cpu().numpy().astype(np.float32)
    return frame


def scene_frame(env, light_names: list[str], color_names: list[str], env_id: int = 0) -> dict[str, np.ndarray]:
    """``lighting.<name>`` and ``object_color.<name>`` values of ``env_id``, read from the scene now."""
    from .scene_factors import LIGHTING_KEY_PREFIX, OBJECT_COLOR_KEY_PREFIX, light_values, object_colors

    frame = {}
    for name in light_names:
        frame[f"{LIGHTING_KEY_PREFIX}{name}"] = np.asarray(light_values(env.scene, name, [env_id])[0], dtype=np.float32)
    for name, values in object_colors(env.scene, color_names, [env_id]).items():
        frame[f"{OBJECT_COLOR_KEY_PREFIX}{name}"] = np.asarray(values[0], dtype=np.float32)
    return frame


@cache
def _make_recorder_manager_class():
    import leisaac.enhance.managers.lerobot_recorder_manager as leisaac_recorder
    import torch
    from isaaclab.managers import RecorderManager
    from isaaclab.utils.datasets.episode_data import EpisodeData

    from .episode_buffers import EpisodeBuffers

    class LeRobotRecorderManager(leisaac_recorder.LeRobotRecorderManager):
        """leisaac's LeRobot recorder plus the verification columns, recording every env.

        leisaac builds the dataset features inside ``__init__`` with no
        extension hook, so the feature builder is wrapped for the duration of
        the base constructor only. Tracked objects come from
        ``env.cfg.tracked_object_names``, which must not be empty.

        leisaac records env 0 only and, with ``num_envs > 1``, saves env 0's
        frames once per finished env. Here every env has its own episode
        buffer (``EpisodeBuffers``): each env's frames go to its own buffer,
        and at reset each env's episode is saved when it succeeded and dropped
        otherwise, so N envs record N episodes per round.
        """

        def __init__(self, cfg, dataset_cfg, env):
            object_names = list(getattr(env.cfg, "tracked_object_names", []))
            if not object_names:
                raise ValueError(
                    f"{type(env.cfg).__name__}.tracked_object_names is empty: no object positions would be recorded "
                    "and the dataset could not be verified. List the task's rigid objects there."
                )
            self._light_names = list(getattr(env.cfg, "recorded_light_names", []))
            self._color_names = list(getattr(env.cfg, "object_references", {}))
            base_build = leisaac_recorder.build_feature_from_env
            light_names, color_names = self._light_names, self._color_names

            def build_with_factors(env, dataset_cfg):
                features = base_build(env, dataset_cfg)
                features.update(object_position_features(object_names))
                features.update(lighting_features(light_names))
                features.update(object_color_features(color_names))
                # Append the feature specs of further verification factors here.
                return features

            leisaac_recorder.build_feature_from_env = build_with_factors
            try:
                super().__init__(cfg, dataset_cfg, env)
            finally:
                leisaac_recorder.build_feature_from_env = base_build

            dataset = self._dataset_file_handler._lerobot_dataset
            # leisaac's handler raises NotImplementedError here; the collection scripts read it on resume.
            self._dataset_file_handler.get_num_episodes = lambda: dataset.meta.total_episodes
            self._buffers = EpisodeBuffers(dataset)

        def record_post_step(self) -> None:
            """Add one frame per env to that env's episode buffer (after the first ``_skip_frames`` steps)."""
            RecorderManager.record_post_step(self)
            env = self._env
            for env_id in range(env.num_envs):
                self._env_steps_record[env_id] += 1
                if self._env_steps_record[env_id] <= self._skip_frames:
                    continue
                frame = env.cfg.build_lerobot_frame(self._episodes[env_id], self._dataset_cfg)
                # Values read from the scene and the episode's own task text, so the frame builder needs no env.
                frame.update(scene_frame(env, self._light_names, self._color_names, env_id))
                if hasattr(env.cfg, "task_text"):
                    frame["task"] = env.cfg.task_text(env, env_id)
                self._buffers.add_frame(env_id, frame)
                self._episodes[env_id]._data.clear()

        def export_episodes(self, env_ids=None) -> None:
            """Save each given env's episode if it succeeded, drop it otherwise."""
            if len(self.active_terms) == 0:
                return
            if env_ids is None:
                env_ids = list(range(self._env.num_envs))
            if isinstance(env_ids, torch.Tensor):
                env_ids = env_ids.tolist()
            for env_id in env_ids:
                if env_id in self._episodes:
                    if self._episodes[env_id].success and self._buffers.save(env_id):
                        self._exported_successful_episode_count[env_id] = (
                            self._exported_successful_episode_count.get(env_id, 0) + 1
                        )
                    else:
                        self._buffers.discard(env_id)
                        self._exported_failed_episode_count[env_id] = (
                            self._exported_failed_episode_count.get(env_id, -1) + 1
                        )  # default -1: the first reset has no episode
                self._episodes[env_id] = EpisodeData()

        def finalize(self):
            self._buffers.discard_all()  # episodes that never reached a reset
            super().finalize()

    return LeRobotRecorderManager


def __getattr__(name: str):
    # Lazy so this module stays importable without leisaac/IsaacLab.
    if name == "LeRobotRecorderManager":
        return _make_recorder_manager_class()
    raise AttributeError(name)
