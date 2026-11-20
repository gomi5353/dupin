"""Read the per-episode rollout log written by ``scripts/rollout.py`` (``outputs/rollout/<timestamp>.jsonl``).

Format: ``packages/simulator/src/simulator/utils/rollout_log.py``. Keep
``ROLLOUT_LOG_VERSION`` and the ``*_FACTOR`` names in sync with it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

ROLLOUT_LOG_VERSION = 1
OBJECT_POSITION_FACTOR = "object_position"
LIGHTING_FACTOR = "lighting"
# Factor -> the values of each entry.
_FACTOR_VALUES = {OBJECT_POSITION_FACTOR: "[x, y, z]", LIGHTING_FACTOR: "[intensity, color_temperature]"}
STATUSES = ("succeeded", "failed", "aborted")


class RolloutLogError(ValueError):
    """The rollout log is missing or malformed."""


@dataclass
class RolloutRun:
    index: int
    status: str
    factors: dict[str, dict]
    instruction: str | None = None
    """The task text the policy was given (Object Properties axis)."""

    @property
    def succeeded(self) -> bool:
        return self.status == "succeeded"

    @property
    def finished(self) -> bool:
        return self.status != "aborted"

    @property
    def object_positions(self) -> dict[str, tuple[float, float, float]]:
        """name -> (x, y, z) of the object's initial position."""
        return {name: (float(p[0]), float(p[1]), float(p[2])) for name, p in self.factors.get(OBJECT_POSITION_FACTOR, {}).items()}

    @property
    def lighting(self) -> dict[str, tuple[float, float]]:
        """light name -> (intensity, color temperature) at the start of the run."""
        return {name: (float(v[0]), float(v[1])) for name, v in self.factors.get(LIGHTING_FACTOR, {}).items()}



@dataclass
class RolloutLog:
    path: Path
    header: dict
    runs: list[RolloutRun] = field(default_factory=list)
    """Every run in the log, aborted ones included."""

    @property
    def finished_runs(self) -> list[RolloutRun]:
        """Succeeded and failed runs: the ones that are counted and compared."""
        return [r for r in self.runs if r.finished]

    @property
    def finished(self) -> int:
        return len(self.finished_runs)

    @property
    def successes(self) -> int:
        return sum(r.succeeded for r in self.runs)

    @property
    def aborted(self) -> int:
        return len(self.runs) - self.finished

    @property
    def name(self) -> str:
        """Run name: the log's file name without extension (outputs/rollout/<timestamp>.jsonl)."""
        return self.path.stem


def read_rollout_log(path: str | Path) -> RolloutLog:
    path = Path(path)
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        raise RolloutLogError(f"Rollout log not found: {path}") from None
    records = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RolloutLogError(f"{path}:{number}: not valid JSON: {exc}") from None
    if not records or records[0].get("type") != "header":
        raise RolloutLogError(f"{path}: the first line must be the rollout header")
    header = records[0]
    if header.get("version") != ROLLOUT_LOG_VERSION:
        raise RolloutLogError(f"{path}: log version {header.get('version')!r}, expected {ROLLOUT_LOG_VERSION}")
    log = RolloutLog(path, header)
    seen = set()
    for number, record in enumerate(records[1:], 2):
        status, index = record.get("status"), record.get("episode")
        if record.get("type") != "episode" or status not in STATUSES or not isinstance(index, int):
            raise RolloutLogError(f"{path}: record {number} is not an episode with a status of {', '.join(STATUSES)}")
        if index in seen:
            raise RolloutLogError(f"{path}: episode {index} appears twice")
        seen.add(index)
        instruction = record.get("instruction")
        if instruction is not None and not isinstance(instruction, str):
            raise RolloutLogError(f"{path}: episode {index}: instruction must be a string, got {instruction!r}")
        run = RolloutRun(index, status, record.get("factors", {}), instruction)
        for factor, shape in _FACTOR_VALUES.items():
            width = shape.count(",") + 1
            entries = run.factors.get(factor, {})
            if not isinstance(entries, dict):
                raise RolloutLogError(f"{path}: episode {index}: factor '{factor}' must map names to values, got {entries!r}")
            for name, v in entries.items():
                if not (isinstance(v, list) and len(v) == width and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)):
                    raise RolloutLogError(f"{path}: episode {index}: {factor} of '{name}' must be {shape}, got {v!r}")
                if not all(math.isfinite(x) for x in v):
                    raise RolloutLogError(f"{path}: episode {index}: non-finite {factor} for '{name}'")
        log.runs.append(run)
    return log
