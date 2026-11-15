"""The registered verifiers: one per generalization axis, run by ``verify`` in this order.

To add an axis, create data_verification/axes/<axis>/ with a Verifier subclass
(see axes/object_position/) and append an instance here. A spec may only use
axis classes that a registered verifier checks. Guide: docs/adding_coverage_factor.md.
"""

from __future__ import annotations

from rdflib import URIRef

from .axes.object_position import ObjectPositionVerifier
from .verifier import Verifier

VERIFIERS: list[Verifier] = [
    ObjectPositionVerifier(),
]


def verifier_for(axis_class: URIRef) -> Verifier | None:
    return next((v for v in VERIFIERS if v.axis_class == axis_class), None)


def verifier_named(name: str) -> Verifier | None:
    return next((v for v in VERIFIERS if v.name == name), None)
