"""Verification store for LeRobot datasets: RDF/OWL facts, SHACL + Python verifiers, SPARQL queries.

    ingest_dataset                    recorded dataset -> store (stores/<repo_name>/)
    verify_store                      the spec files + registered verifiers -> findings about the dataset in the store
    compare_store                     rollout logs vs the dataset's coverage under the spec files -> findings in the store
    load_spec                         check one spec file and put it into the store (verify does this)
    query                             SPARQL over the store
"""

from . import registry
from .query import query
from .registry import VERIFIERS
from .store import Store, StoreError, ingest_dataset, load_spec, open_store, read_rollout
from .verifier import AxisCheck, AxisComparison, Finding, Verifier
from .verify import CheckReport, CombinedComparison, RolloutComparison, VerifyResult, format_check, format_combined, format_comparison, compare_store, verify_store

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
    "load_spec",
    "open_store",
    "query",
    "read_rollout",
    "compare_store",
    "verify_store",
]
