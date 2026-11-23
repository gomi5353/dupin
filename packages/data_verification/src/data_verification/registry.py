"""The registered verifiers: one per generalization axis, run by ``verify`` in this order.

To add an axis, create data_verification/axes/<axis>/ with a Verifier subclass
(see axes/object_pose/) and append an instance here. A spec may only use
axis classes that a registered verifier checks.
"""

from __future__ import annotations

from rdflib import URIRef

from .axes.image_augmentation import ImageAugmentationVerifier
from .axes.object_property import ObjectPropertyVerifier
from .axes.object_pose import ObjectPoseVerifier
from .verifier import Verifier

VERIFIERS: list[Verifier] = [
    ObjectPoseVerifier(),
    ImageAugmentationVerifier(),
    ObjectPropertyVerifier(),
]


def verifier_for(axis_class: URIRef) -> Verifier | None:
    return next((v for v in VERIFIERS if v.axis_class == axis_class), None)


def verifier_named(name: str) -> Verifier | None:
    return next((v for v in VERIFIERS if v.name == name), None)
