"""Splitting a range into equal cells, shared by the axes that check coverage on a grid.

A range [lo, hi] split into n cells has edges lo + i * (hi - lo) / n; edge n is
exactly hi. A value v is in cell i when edge i <= v < edge i+1, and hi belongs
to the last cell. Values outside [lo, hi] get indices below 0 or from n up, so
a run that started outside the range can still be placed on the same lattice.
"""

from __future__ import annotations

import math
from decimal import Decimal


def cell_edge(lo: Decimal, hi: Decimal, n: int, i: int) -> Decimal:
    """The i-th cell edge of [lo, hi] split into n; edge n is exactly hi."""
    return hi if i == n else lo + i * (hi - lo) / n


def lattice_index(v: Decimal, lo: Decimal, hi: Decimal, n: int) -> int:
    """Index i of the interval [edge i, edge i+1) holding v, extended beyond [lo, hi]; hi belongs to cell n - 1."""
    if v == hi:
        return n - 1
    i = math.floor((v - lo) * n / (hi - lo))
    while v < cell_edge(lo, hi, n, i):  # guard against rounding at an edge
        i -= 1
    while v >= cell_edge(lo, hi, n, i + 1):
        i += 1
    return i


def dec(literal) -> Decimal:
    """An rdflib literal (or plain number) as a Decimal, without float noise."""
    value = literal.toPython() if hasattr(literal, "toPython") else literal
    return Decimal(str(value))
