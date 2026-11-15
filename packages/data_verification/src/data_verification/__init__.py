"""Verification store for LeRobot datasets: RDF/OWL facts, SHACL + Python verifiers, SPARQL queries.

    ingest_dataset / ingest_rollout   raw capture -> store (stores/<dataset>/)
    attach_spec / detach_spec         the student's coverage specs
    verify_store                      registered verifiers -> findings in the store
    query                             SPARQL over the store
"""

from . import registry
from .query import query
from .registry import VERIFIERS
from .store import Store, StoreError, attach_spec, detach_spec, ingest_dataset, ingest_rollout, open_store
from .verifier import AxisCheck, AxisComparison, Finding, Verifier
from .verify import CheckReport, RolloutComparison, VerifyResult, format_check, format_comparison, verify_store

__all__ = [
    "VERIFIERS",
    "registry",
    "AxisCheck",
    "AxisComparison",
    "CheckReport",
    "Finding",
    "RolloutComparison",
    "Store",
    "StoreError",
    "Verifier",
    "VerifyResult",
    "attach_spec",
    "detach_spec",
    "format_check",
    "format_comparison",
    "ingest_dataset",
    "ingest_rollout",
    "open_store",
    "query",
    "verify_store",
]
