"""The verifier interface: one registered verifier per generalization axis.

A verifier owns one axis: its ontology, its SHACL shapes (dataset side and
rollout side), the checks of a spec, and optional Python steps. ``verify`` runs
the SHACL shapes of every registered verifier; the Python steps add what SHACL
cannot say well (counts, tables, run placements, hints). A verifier that only
needs rules can leave every Python step at its default.

Register a verifier in ``data_verification/registry.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.abc import Traversable
from importlib.resources import files
from pathlib import Path

from pyshacl import validate
from rdflib import RDF, Graph, Namespace, URIRef

from .graph import CP
from .rollout_log import RolloutLog

SH = Namespace("http://www.w3.org/ns/shacl#")
SEVERITIES = ("Error", "Violation", "Warning", "Info")


@dataclass
class Finding:
    severity: str
    """Error, Violation, Warning or Info."""
    message: str
    focus_nodes: list[str] = field(default_factory=list)
    verifier: str = "core"

    @property
    def count(self) -> int:
        return len(self.focus_nodes)


@dataclass
class AxisCheck:
    """What one verifier found about the dataset, beyond its SHACL findings."""

    axis: str
    findings: list[Finding] = field(default_factory=list)
    derived: Graph = field(default_factory=Graph)
    """Triples to keep in the store with the verification (e.g. per-cell episode counts)."""

    def lines(self) -> list[str]:
        """Report lines printed before the findings (e.g. coverage tables)."""
        return []


@dataclass
class AxisComparison:
    """How one verifier judged a rollout's finished runs against the dataset."""

    axis: str
    run_covered: dict[int, bool]
    """Finished run index -> whether this axis covers it."""
    errors: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    derived: Graph = field(default_factory=Graph)

    def lines(self) -> list[str]:
        return []

    def hint_lines(self) -> list[str]:
        """Suggested next steps, printed unless --no-hints and stored as Info findings."""
        return []


class Verifier:
    """Base class. Subclass it for each axis and register an instance in registry.py."""

    name: str = ""
    """Short name, used in reports and findings (e.g. "object_pose")."""
    axis_class: URIRef
    """The cp:GeneralizationAxis subclass this verifier checks."""
    package: str = ""
    """Package holding ontology.ttl, shapes.ttl and rollout_shapes.ttl (each optional)."""

    # ---- Resources -------------------------------------------------------------

    def resource(self, name: str) -> Traversable | None:
        if not self.package:
            return None
        path = files(self.package).joinpath(name)
        return path if path.is_file() else None

    def _graph(self, name: str) -> Graph:
        graph = Graph()
        path = self.resource(name)
        if path is not None:
            graph.parse(data=path.read_text(), format="turtle")
        return graph

    def ontology(self) -> Graph:
        return self._graph("ontology.ttl")

    def shapes(self) -> Graph:
        return self._graph("shapes.ttl")

    def rollout_shapes(self) -> Graph:
        return self._graph("rollout_shapes.ttl")

    # ---- Steps (all optional) -------------------------------------------------

    def ingest_dataset(self, root: Path, graph: Graph, dataset: URIRef) -> list[str]:
        """Add this axis's per-episode facts read from the LeRobot dataset at ``root`` (episode IRIs: ``<dataset>/episode/<i>``).

        Return problems that make the dataset unusable; they stop ``ingest``.
        Object positions are added by the store itself.
        """
        return []

    def ingest_rollout(self, log: RolloutLog, graph: Graph, rollout: URIRef) -> list[str]:
        """Add this axis's per-run facts from the rollout log (``log.runs[k].factors[<factor>]``; run IRIs: ``<rollout>/run/<i>``)."""
        return []

    def recorded_names(self, graph: Graph, dataset: URIRef) -> list[str]:
        """The objects (or lights) the dataset recorded this axis for."""
        return sorted(str(n) for n in graph.objects(dataset, CP.recordedObject))

    def spec_errors(self, spec: Graph) -> list[str]:
        """Problems that stop ``verify`` for this spec (e.g. TODO left, values not numbers). ``spec`` is the spec file."""
        return []

    def derive_spec(self, spec: Graph) -> None:
        """Add derived nodes to the spec in the store (e.g. grid cells)."""

    def dataset_errors(self, graph: Graph, dataset: URIRef) -> list[str]:
        """Problems that stop the check before SHACL runs (e.g. a required column is missing)."""
        return []

    def check(self, graph: Graph, dataset: URIRef) -> AxisCheck | None:
        """Python part of the dataset check. ``graph`` holds the dataset and one spec."""
        return None

    def compare(self, graph: Graph, rollout: URIRef, dataset: URIRef, comparison: URIRef) -> AxisComparison | None:
        """Python part of the rollout comparison. ``graph`` also holds the rollout; derived nodes hang off ``comparison``."""
        return None


def core_graph(*parts: str) -> Graph:
    graph = Graph()
    graph.parse(data=files("data_verification").joinpath(*parts).read_text(), format="turtle")
    return graph


def run_shacl(data: Graph, shapes: dict[str, Graph], ontology: Graph, path_prefix: bool = True) -> list[Finding]:
    """Validate ``data`` with the union of ``shapes`` (verifier name -> shapes); findings grouped by message."""
    owner: dict = {}
    union = Graph()
    for name, graph in shapes.items():
        union += graph
        for subject in graph.subjects():
            owner.setdefault(subject, name)
    _, results, _ = validate(data_graph=data, shacl_graph=union, ont_graph=ontology, inference="none", abort_on_first=False)
    grouped: dict[tuple[str, str, str], list[str]] = {}
    for result in results.subjects(RDF.type, SH.ValidationResult):
        severity = str(results.value(result, SH.resultSeverity)).rsplit("#", 1)[-1]
        message = str(results.value(result, SH.resultMessage) or "")
        path = results.value(result, SH.resultPath)
        if path_prefix and isinstance(path, URIRef) and not message.startswith("'"):  # not for inverse paths
            message = f"{path.n3(data.namespace_manager)}: {message}"
        verifier = owner.get(results.value(result, SH.sourceShape), "core")
        grouped.setdefault((severity, message, verifier), []).append(str(results.value(result, SH.focusNode)))
    return [Finding(sev, msg, sorted(nodes), verifier) for (sev, msg, verifier), nodes in sorted(grouped.items())]


def axes_of(graph: Graph, dataset: URIRef) -> set[URIRef]:
    return set(graph.objects(dataset, CP.hasGeneralizationAxis))


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.0f}%" if d else "-"


def rate(n: int, d: int) -> str:
    return f"{n}/{d}" if d else "0/0"
