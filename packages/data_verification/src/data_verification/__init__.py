"""Verification store for LeRobot datasets: RDF/OWL facts, SHACL + Python verifiers, SPARQL queries.

    ingest_dataset / ingest_rollout   raw capture -> store (stores/<repo_name>/)
    verify_store                      the spec files + registered verifiers -> findings in the store
    load_spec                         check one spec file and put it into the store (verify does this)
    query                             SPARQL over the store
"""

from . import registry
from .query import query
from .registry import VERIFIERS
from .store import Store, StoreError, ingest_dataset, ingest_rollout, load_spec, open_store
from .verifier import AxisCheck, AxisComparison, Finding, Verifier
from .verify import CheckReport, CombinedComparison, RolloutComparison, VerifyResult, format_check, format_combined, format_comparison, verify_store

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
    "format_check",
    "format_combined",
    "format_comparison",
    "CombinedComparison",
    "ingest_dataset",
    "ingest_rollout",
    "load_spec",
    "open_store",
    "query",
    "verify_store",
]
