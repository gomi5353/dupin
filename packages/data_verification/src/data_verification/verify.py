"""verify: check the dataset against every attached spec and compare every ingested rollout, then store the results.

For each attached spec, a Verification: the dataset and the spec are checked
with the core SHACL rules and those of every registered verifier, then each
verifier's Python step runs (e.g. per-cell counts). For each spec and each
ingested rollout, a Comparison: each verifier places the rollout's finished
runs against the dataset's coverage; a run is in distribution when every axis
of the spec covers it. Results, with their findings and derived nodes, are
written to findings/ in the store, replacing the previous ones.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from rdflib import RDF, XSD, Graph, Literal, URIRef

from .graph import CP, rollout_iri, spec_iri
from . import registry
from .store import Store, now, ontology, source_changes, write_graph
from .verifier import AxisCheck, AxisComparison, Finding, Verifier, core_graph, pct, rate, run_shacl


@dataclass
class CheckReport:
    store: str
    spec: str
    spec_path: str = ""
    num_episodes: int = 0
    errors: list[str] = field(default_factory=list)
    violations: list[Finding] = field(default_factory=list)
    warnings: list[Finding] = field(default_factory=list)
    axes: list[AxisCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors and not self.violations

    def to_dict(self) -> dict:
        result = asdict(self)
        result["axes"] = [{k: v for k, v in asdict(a).items() if k != "derived"} for a in self.axes]
        result["passed"] = self.passed
        return result


@dataclass
class RolloutComparison:
    store: str
    spec: str
    rollout: str
    eval_file: str = ""
    seed: int | None = None
    runs: int = 0
    successes: int = 0
    aborted: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[Finding] = field(default_factory=list)
    axes: list[AxisComparison] = field(default_factory=list)
    runs_in_distribution: int = 0
    successes_in_distribution: int = 0

    @property
    def passed(self) -> bool:
        return not self.errors

    @property
    def fully_covered(self) -> bool:
        return self.passed and self.runs > 0 and self.runs_in_distribution == self.runs

    def hint_lines(self) -> list[str]:
        return [line for axis in self.axes for line in axis.hint_lines()]

    def hints(self) -> list[Finding]:
        return [Finding("Info", line.strip(), [], axis.axis) for axis in self.axes for line in axis.hint_lines() if line.strip()]

    def to_dict(self) -> dict:
        result = asdict(self)
        result["axes"] = [{k: v for k, v in asdict(a).items() if k != "derived"} for a in self.axes]
        result.update(passed=self.passed, fully_covered=self.fully_covered, hints=self.hint_lines())
        return result


@dataclass
class VerifyResult:
    checks: list[CheckReport] = field(default_factory=list)
    comparisons: list[RolloutComparison] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks) and all(c.passed for c in self.comparisons)

    def warnings(self) -> bool:
        return any(c.warnings for c in self.checks) or any(c.warnings for c in self.comparisons)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "checks": [c.to_dict() for c in self.checks],
            "comparisons": [c.to_dict() for c in self.comparisons],
        }


def verify_store(
    store: Store,
    specs: list[str] | None = None,
    rollouts: list[str] | None = None,
    extra_shapes: list[str | Path] | None = None,
    write: bool = True,
) -> VerifyResult:
    """Verify ``specs`` (default: every attached spec) and compare ``rollouts`` (default: every ingested rollout)."""
    result = VerifyResult()
    attached = store.spec_names()
    for name in specs if specs is not None else attached:
        if name not in attached:
            result.checks.append(CheckReport(str(store.root), name, errors=[f"No spec '{name}' is attached (attached: {', '.join(attached) or 'none'})"]))
            continue
        report, graph = check_spec(store, name, extra_shapes)
        result.checks.append(report)
        if write:
            write_graph(graph, store.check_file(name))
        for rollout in rollouts if rollouts is not None else store.rollout_names():
            comparison, graph = compare_rollout(store, name, rollout)
            result.comparisons.append(comparison)
            if write and store.rollout_file(rollout).is_file():
                write_graph(graph, store.comparison_file(name, rollout))
    if not attached and specs is None:
        result.checks.append(CheckReport(str(store.root), "", errors=["No spec is attached; run: attach --spec <your spec>"]))
    return result


def _spec_verifiers(graph: Graph, spec: URIRef) -> list[Verifier]:
    verifiers: list[Verifier] = []
    for axis in graph.objects(spec, CP.declaresAxis):
        for axis_class in graph.objects(axis, RDF.type):
            verifier = registry.verifier_for(axis_class)
            if verifier is not None and verifier not in verifiers:
                verifiers.append(verifier)
    return verifiers


# ---- Dataset vs spec --------------------------------------------------------------------


def check_spec(store: Store, name: str, extra_shapes: list[str | Path] | None = None) -> tuple[CheckReport, Graph]:
    dataset, spec = store.dataset, spec_iri(store.dataset, name)
    graph = store.load(store.dataset_file, store.spec_file(name))
    report = CheckReport(str(store.root), name, str(graph.value(spec, CP.sourcePath) or ""))
    report.num_episodes = len(set(graph.objects(dataset, CP.hasEpisode)))
    verifiers = _spec_verifiers(graph, spec)
    report.errors += source_changes(graph, dataset, "dataset") + source_changes(graph, spec, "spec")

    if not report.errors:
        for verifier in verifiers:
            report.errors += verifier.dataset_errors(graph, dataset)
        shapes = {"core": core_graph("shapes", "dataset.ttl")}
        shapes.update({v.name: v.shapes() for v in verifiers})
        for i, path in enumerate(extra_shapes or []):
            try:
                shapes[f"extra:{Path(path).name}" if i == 0 else f"extra:{Path(path).name}:{i}"] = Graph().parse(Path(path), format="turtle")
            except Exception as exc:
                report.errors.append(f"Extra shapes file {path} could not be loaded: {exc}")
        if not any(e.startswith("Extra shapes file") for e in report.errors):
            for finding in run_shacl(graph, shapes, ontology()):
                (report.violations if finding.severity == "Violation" else report.warnings).append(finding)
            for verifier in verifiers:
                axis = verifier.check(graph, dataset)
                if axis is not None:
                    report.axes.append(axis)
                    for finding in axis.findings:
                        (report.violations if finding.severity == "Violation" else report.warnings).append(finding)

    out = Graph()
    out.bind("cp", CP)
    node = URIRef(f"{spec}/verification")
    out.add((node, RDF.type, CP.Verification))
    out.add((node, CP.ofSpec, spec))
    out.add((node, CP.onDataset, dataset))
    out.add((node, CP.verifiedAt, now()))
    out.add((node, CP.passed, Literal(report.passed)))
    _add_findings(out, node, [Finding("Error", e) for e in report.errors] + report.violations + report.warnings)
    for axis in report.axes:
        out += axis.derived
    return report, out


# ---- Rollout vs dataset coverage --------------------------------------------------------


def compare_rollout(store: Store, name: str, rollout_name: str) -> tuple[RolloutComparison, Graph]:
    dataset, spec = store.dataset, spec_iri(store.dataset, name)
    rollout = rollout_iri(dataset, rollout_name)
    result = RolloutComparison(str(store.root), name, rollout_name)
    node = URIRef(f"{rollout}/comparison/{name}")
    out = Graph()
    out.bind("cp", CP)
    out.add((node, RDF.type, CP.Comparison))
    out.add((node, CP.ofSpec, spec))
    out.add((node, CP.ofRollout, rollout))
    out.add((node, CP.verifiedAt, now()))

    if not store.rollout_file(rollout_name).is_file():
        result.errors.append(f"No rollout '{rollout_name}' in the store (ingested: {', '.join(store.rollout_names()) or 'none'})")
        return result, out
    graph = store.load(store.dataset_file, store.spec_file(name), store.rollout_file(rollout_name))
    statuses = {int(graph.value(r, CP["index"])): str(graph.value(r, CP.status)) for r in graph.objects(rollout, CP.hasRun)}
    result.eval_file = str(graph.value(rollout, CP.evalFile) or "")
    seed = graph.value(rollout, CP.seed)
    result.seed = None if seed is None else int(seed)
    result.runs = sum(s != "aborted" for s in statuses.values())
    result.successes = sum(s == "succeeded" for s in statuses.values())
    result.aborted = len(statuses) - result.runs
    if not result.runs:
        result.errors.append(f"Rollout '{rollout_name}' has no finished run (all aborted or none run)")
    result.errors += (
        source_changes(graph, dataset, "dataset") + source_changes(graph, spec, "spec") + source_changes(graph, rollout, "rollout")
    )
    _check_training_dataset(graph, rollout, dataset, result)

    verifiers = _spec_verifiers(graph, spec)
    if not result.errors:
        shapes = {"core": core_graph("shapes", "rollout.ttl")}
        shapes.update({v.name: v.rollout_shapes() for v in verifiers})
        for finding in run_shacl(graph, shapes, ontology(), path_prefix=False):
            if finding.severity == "Violation":
                result.errors.append(finding.message)
            else:
                result.warnings.append(finding)
    if not result.errors:
        for verifier in verifiers:
            axis = verifier.compare(graph, rollout, dataset, node)
            if axis is None:
                result.warnings.append(
                    Finding("Info", f"Axis '{verifier.name}' has no rollout comparison; it does not decide which runs are in distribution", [], verifier.name)
                )
                continue
            result.errors += axis.errors
            result.warnings += axis.findings
            result.axes.append(axis)
            out += axis.derived
    if not result.errors:
        finished = [i for i, s in statuses.items() if s != "aborted"]
        inside = [i for i in finished if all(axis.run_covered.get(i, False) for axis in result.axes)]
        result.runs_in_distribution = len(inside)
        result.successes_in_distribution = sum(statuses[i] == "succeeded" for i in inside)
        for i in sorted(inside):
            out.add((node, CP.inDistributionRun, URIRef(f"{rollout}/run/{i}")))
        out.add((node, CP.runsInDistribution, Literal(len(inside), datatype=XSD.nonNegativeInteger)))
    _add_findings(out, node, [Finding("Error", e) for e in result.errors] + result.warnings + result.hints())
    return result, out


def _check_training_dataset(graph: Graph, rollout: URIRef, dataset: URIRef, result: RolloutComparison) -> None:
    """Warn when the checkpoint was trained on a different dataset than the store's."""
    repo_id, source = graph.value(rollout, CP.trainedOnRepo), graph.value(dataset, CP.sourcePath)
    if repo_id is None or source is None:
        return
    parts = Path(str(repo_id)).parts
    if Path(str(source)).parts[-len(parts):] != parts:
        result.warnings.append(
            Finding(
                "Warning",
                f"The rollout's checkpoint was trained on '{repo_id}', not on {source}; "
                "the comparison describes a dataset the policy did not learn from",
                [str(graph.value(rollout, CP.checkpoint) or rollout)],
            )
        )


def _add_findings(graph: Graph, owner: URIRef, findings: list[Finding]) -> None:
    for n, finding in enumerate(findings):
        node = URIRef(f"{owner}/finding/{n}")
        graph.add((owner, CP.hasFinding, node))
        graph.add((node, RDF.type, CP.Finding))
        graph.add((node, CP.severity, Literal(finding.severity)))
        graph.add((node, CP.message, Literal(finding.message)))
        graph.add((node, CP.verifier, Literal(finding.verifier)))
        for focus in finding.focus_nodes:
            if focus.startswith(("http://", "https://", "urn:")):
                graph.add((node, CP.focusNode, URIRef(focus)))
            else:
                graph.add((node, CP.focus, Literal(focus)))


# ---- Reports ----------------------------------------------------------------------------


def format_check(report: CheckReport, max_nodes: int = 3) -> str:
    lines = [
        f"Store   : {report.store}",
        f"Spec    : {report.spec}" + (f"  ({report.spec_path})" if report.spec_path else ""),
        f"Episodes: {report.num_episodes}",
        "",
    ]
    for axis in report.axes:
        lines += axis.lines()
    for error in report.errors:
        lines.append(f"ERROR     {error}")
    for kind, findings in (("VIOLATION", report.violations), ("WARNING", report.warnings)):
        for f in findings:
            nodes = ", ".join(_short_node(n) for n in f.focus_nodes[:max_nodes])
            more = f" (+{f.count - max_nodes} more)" if f.count > max_nodes else ""
            lines.append(f"{kind:<9} {f.message}  [{nodes}{more}]")
    lines.append("")
    lines.append("RESULT: coverage spec met" if report.passed else "RESULT: coverage spec NOT met")
    return "\n".join(lines)


def format_comparison(result: RolloutComparison, hints: bool = True, max_nodes: int = 3) -> str:
    lines = [
        f"Store   : {result.store}",
        f"Spec    : {result.spec}",
        f"Rollout : {result.rollout}" + (f"  ({result.eval_file}, seed {result.seed})" if result.eval_file else ""),
    ]
    aborted = f", {result.aborted} aborted (not counted)" if result.aborted else ""
    lines += [f"Runs    : {result.runs} finished, {result.successes} succeeded{aborted}", ""]
    for axis in result.axes:
        lines += axis.lines()
    if result.axes and result.runs and result.passed:
        out = result.runs - result.runs_in_distribution
        lines.append(
            f"In distribution (covered on every axis: {', '.join(a.axis for a in result.axes)}): "
            f"{result.runs_in_distribution}/{result.runs} runs ({pct(result.runs_in_distribution, result.runs)})"
        )
        lines.append(
            f"  success {rate(result.successes_in_distribution, result.runs_in_distribution)} in distribution vs "
            f"{rate(result.successes - result.successes_in_distribution, out)} outside"
        )
        lines.append("")
    if hints:
        lines += result.hint_lines()
    for error in result.errors:
        lines.append(f"ERROR     {error}")
    for f in result.warnings:
        nodes = ", ".join(_short_node(n) for n in f.focus_nodes[:max_nodes])
        more = f" (+{f.count - max_nodes} more)" if f.count > max_nodes else ""
        lines.append(f"{f.severity.upper():<9} {f.message}" + (f"  [{nodes}{more}]" if f.focus_nodes else ""))
    lines.append("")
    if not result.passed:
        lines.append("RESULT: comparison could not be made")
    elif result.fully_covered:
        lines.append("RESULT: the dataset covers every rollout run")
    else:
        lines.append("RESULT: the dataset does NOT cover every rollout run; see above")
    return "\n".join(lines)


def _short_node(uri: str) -> str:
    """'http://taica.course/cp/data/ds/episode/3/object/cup' -> 'episode/3/object/cup'; spec nodes -> 'spec/<name>/...'."""
    prefix = "http://taica.course/cp/data/"
    if uri.startswith(prefix):
        return uri[len(prefix):].split("/", 1)[-1]
    return uri.rsplit("#", 1)[-1]
