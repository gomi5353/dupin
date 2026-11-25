"""Limits that keep the terminal report readable when a spec has very many cells."""

from __future__ import annotations

MAX_FINDING_LINES = 10
MAX_HINT_LINES = 15


def cap(lines: list[str], limit: int, what: str) -> list[str]:
    """The first ``limit`` lines, then one line saying how many more there are."""
    if len(lines) <= limit:
        return lines
    return lines[:limit] + [f"          ... and {len(lines) - limit} more {what} (all of them are in findings/*.ttl in the store)"]
