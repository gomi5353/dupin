"""Helpers for an axis's spec template and the checks that stop ``attach``.

A template (``axes/<axis>/template.ttl``) has a header and one block between
``# --- BEGIN <KIND> ---`` and ``# --- END <KIND> ---`` that ``init`` repeats
once per name (object or light), with the placeholder replaced by the name.
"""

from __future__ import annotations

from decimal import Decimal

from rdflib import Graph, URIRef


def render_blocks(template: str, kind: str, placeholder: str, names: list[str], notes: dict[str, str] | None = None) -> str:
    """``template`` with its block repeated per name; ``notes[name]`` (comment lines) goes on top of that block."""
    begin, end = f"# --- BEGIN {kind} ---\n", f"# --- END {kind} ---\n"
    head, rest = template.split(begin, 1)
    block, tail = rest.split(end, 1)
    blocks = [begin + (notes or {}).get(name, "") + block.replace(placeholder, name) + end for name in names]
    return head + "\n".join(blocks) + tail


def todo_fields(graph: Graph, label) -> list[str]:
    """'<owner>: <property>' for every value still set to "TODO"; ``label(node)`` names the owner."""
    return sorted(f"{label(s) or s}: {p.split('#')[-1]}" for s, p, o in graph if str(o) == "TODO")


def value_type_errors(graph: Graph, number_fields: list[URIRef], whole_fields: list[URIRef], label) -> list[str]:
    """Values of ``number_fields`` that are not numbers and of ``whole_fields`` that are not whole numbers."""
    errors = []
    for field in [*number_fields, *whole_fields]:
        whole = field in whole_fields
        for subject, value in graph.subject_objects(field):
            python_value = value.toPython()
            ok = isinstance(python_value, int) if whole else isinstance(python_value, (int, float, Decimal))
            if isinstance(python_value, bool) or not ok:
                kind = "a whole number" if whole else "a number"
                shown = f'"{value}"' if isinstance(python_value, str) else str(value)
                errors.append(f"{label(subject) or subject}: {field.split('#')[-1]} must be {kind} without quotes, got {shown}")
    return sorted(errors)


def number(graph: Graph, node, prop) -> float | None:
    """The single numeric value of ``prop`` on ``node``, or None when missing or not a number."""
    values = list(graph.objects(node, prop))
    if len(values) != 1:
        return None
    value = values[0].toPython()
    return float(value) if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool) else None
