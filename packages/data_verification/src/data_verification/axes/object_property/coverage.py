"""Object Properties axis (STAR-Gen S-PROP): episodes per reference, the unchanged-object check, and the rollout runs' references.

An episode (or run) uses the reference of a requirement whose phrase its task
text (or instruction) contains, as whole words, case-insensitive; when several
phrases appear, the longest one. With none, the text refers to the object in a
way the spec does not list (unrecognized). Every reference needs at least
minEpisodesPerReference dataset episodes: the requirement's value, or the
reference's own value when it sets one. A reference with that many episodes is
seen; with fewer (none included) it is unseen. A rollout run is covered on this
axis when, for every requirement, its instruction uses a seen reference.

S-PROP changes only the words: the object must look the same in every episode.
The object's recorded color (object_color.<name>) is therefore compared across
episodes; values that differ by more than COLOR_TOLERANCE in a channel fail.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from rdflib import RDF, Graph, Literal, URIRef

from ...graph import CP, index_literal
from ...verifier import AxisCheck, AxisComparison, Finding, pct, rate

COLOR_TOLERANCE = 1e-3
"""Largest per-channel difference still counted as the same color (the recorder stores float32)."""

UNRECOGNIZED = "(unrecognized)"

REQUIREMENTS_QUERY = """
SELECT ?req ?name ?min WHERE {
    ?dataset cp:hasGeneralizationAxis ?axis .
    ?axis cp:hasRequirement ?req .
    ?req a cp:ColorReferenceRequirement ; cp:hasTaskObject ?obj ; cp:minEpisodesPerReference ?min .
    ?obj cp:name ?name .
}
"""

REFERENCES_QUERY = """
SELECT ?ref ?phrase ?kind ?min WHERE {
    ?req cp:hasReference ?ref .
    ?ref cp:phrase ?phrase ; a ?kind .
    FILTER (?kind IN (cp:NameReference, cp:ColorReference))
    OPTIONAL { ?ref cp:minEpisodesPerReference ?min }
}
"""

RUN_TEXTS_QUERY = """
SELECT ?run ?index ?status ?text WHERE {
    ?rollout cp:hasRun ?run .
    ?run cp:index ?index ; cp:status ?status .
    FILTER (?status IN ("succeeded", "failed"))
    OPTIONAL { ?run cp:taskDescription ?text }
}
"""

RGB = tuple[float, float, float]


@dataclass
class Reference:
    iri: URIRef
    phrase: str
    kind: str
    """"name" or "color"."""
    min_episodes: int
    """Episodes this reference needs to be seen (its own value, else the requirement's)."""

    @property
    def label(self) -> str:
        return f'"{self.phrase}" ({self.kind})'


@dataclass
class Requirement:
    iri: URIRef
    obj: str
    min_episodes: int
    references: list[Reference]

    def classify(self, text: str | None) -> Reference | None:
        """The reference whose phrase ``text`` contains (the longest if several), or None."""
        if not text:
            return None
        found = [r for r in self.references if contains_phrase(text, r.phrase)]
        return max(found, key=lambda r: len(r.phrase), default=None)


def contains_phrase(text: str, phrase: str) -> bool:
    """Whether ``text`` contains ``phrase`` as whole words, case-insensitive, any whitespace between words."""
    words = r"\s+".join(re.escape(w) for w in phrase.split())
    return bool(words) and re.search(rf"(?<![A-Za-z0-9]){words}(?![A-Za-z0-9])", text, flags=re.IGNORECASE) is not None


def requirements(graph: Graph, dataset: URIRef) -> list[Requirement]:
    result = []
    for row in graph.query(REQUIREMENTS_QUERY, initNs={"cp": CP}, initBindings={"dataset": dataset}):
        references = [
            Reference(r.ref, str(r.phrase), "name" if r.kind == CP.NameReference else "color", int(r["min"] if r["min"] is not None else row["min"]))
            for r in graph.query(REFERENCES_QUERY, initNs={"cp": CP}, initBindings={"req": row.req})
        ]
        references.sort(key=lambda r: (r.kind != "name", r.phrase))
        result.append(Requirement(row.req, str(row.name), int(row.min), references))
    return sorted(result, key=lambda r: r.obj)


def required_objects(graph: Graph, dataset: URIRef) -> set[str]:
    return {r.obj for r in requirements(graph, dataset)}


def episode_colors(graph: Graph, dataset: URIRef) -> dict[str, dict[int, RGB]]:
    """object name -> episode index -> (r, g, b)."""
    colors: dict[str, dict[int, RGB]] = {}
    for episode in graph.objects(dataset, CP.hasEpisode):
        index = int(graph.value(episode, CP["index"]))
        for color in graph.objects(episode, CP.hasObjectColor):
            rgb = tuple(graph.value(color, p) for p in (CP.colorR, CP.colorG, CP.colorB))
            if None not in rgb:
                colors.setdefault(str(graph.value(color, CP.name)), {})[index] = tuple(float(v) for v in rgb)
    return colors


def episode_texts(graph: Graph, dataset: URIRef) -> dict[int, str]:
    texts = {}
    for episode in graph.objects(dataset, CP.hasEpisode):
        text = graph.value(episode, CP.taskDescription)
        if text is not None:
            texts[int(graph.value(episode, CP["index"]))] = str(text)
    return texts


def changed_color_episodes(colors: dict[int, RGB]) -> tuple[RGB | None, list[int]]:
    """The most common color and the episodes whose color differs from it by more than COLOR_TOLERANCE."""
    if not colors:
        return None, []
    usual = Counter(tuple(round(v, 3) for v in rgb) for rgb in colors.values()).most_common(1)[0][0]
    return usual, sorted(i for i, rgb in colors.items() if max(abs(a - b) for a, b in zip(rgb, usual)) > COLOR_TOLERANCE)


def _fmt_rgb(rgb: RGB) -> str:
    return "(" + ", ".join(f"{v:g}" for v in rgb) + ")"


# ---- Dataset check -----------------------------------------------------------


@dataclass
class ReferenceCoverage:
    req: Requirement
    counts: dict[str, int]
    """reference phrase -> episodes."""
    unrecognized: int
    color: RGB | None


@dataclass
class ObjectPropertyCheck(AxisCheck):
    coverage: list[ReferenceCoverage] = field(default_factory=list)

    def lines(self) -> list[str]:
        lines = []
        for table in self.coverage:
            r = table.req
            ok = sum(table.counts.get(ref.phrase, 0) >= ref.min_episodes for ref in r.references)
            color = f"; color {_fmt_rgb(table.color)}" if table.color else ""
            lines.append(f"[{r.obj}] {ok}/{len(r.references)} references seen (episodes >= needed){color}")
            for ref in r.references:
                n = table.counts.get(ref.phrase, 0)
                lines.append(f"  {ref.label:<28} {n:>5} / {ref.min_episodes:<4}" + ("" if n >= ref.min_episodes else "  unseen"))
            if table.unrecognized:
                lines.append(f"  {UNRECOGNIZED:<28} {table.unrecognized:>5}  not counted")
            lines.append("")
        return lines


def check_object_property(graph: Graph, dataset: URIRef, axis: str) -> ObjectPropertyCheck:
    result = ObjectPropertyCheck(axis)
    colors = episode_colors(graph, dataset)
    texts = episode_texts(graph, dataset)
    episodes = sorted(int(graph.value(e, CP["index"])) for e in graph.objects(dataset, CP.hasEpisode))
    for req in requirements(graph, dataset):
        if not req.references:
            continue  # the SHACL requirement shape reports it
        counts: Counter = Counter()
        unrecognized: list[int] = []
        for index in episodes:
            if index not in texts:
                continue  # the SHACL task text shape reports it
            ref = req.classify(texts[index])
            if ref is None:
                unrecognized.append(index)
            else:
                counts[ref.phrase] += 1
        usual, changed = changed_color_episodes(colors.get(req.obj, {}))
        result.coverage.append(ReferenceCoverage(req, dict(counts), len(unrecognized), usual))

        for ref in req.references:
            result.derived.add((ref.iri, CP.episodeCount, index_literal(counts[ref.phrase])))
            if counts[ref.phrase] < ref.min_episodes:
                result.findings.append(
                    Finding(
                        "Violation",
                        f"'{req.obj}' reference {ref.label} is used by {counts[ref.phrase]} episode(s); needs at least {ref.min_episodes}",
                        [str(ref.iri)],
                        axis,
                    )
                )
        if changed:
            result.findings.append(
                Finding(
                    "Violation",
                    f"'{req.obj}' does not look the same in every episode (color differs from {_fmt_rgb(usual)}); "
                    "Object Properties (S-PROP) changes only the task text, never the object",
                    [f"{dataset}/episode/{i}" for i in changed],
                    axis,
                )
            )
        if unrecognized:
            result.findings.append(
                Finding(
                    "Warning",
                    f"'{req.obj}': the task text uses none of the spec's phrases ({', '.join(r.phrase for r in req.references)}); these episodes do not count",
                    [f"{dataset}/episode/{i}" for i in unrecognized],
                    axis,
                )
            )
    return result


# ---- Rollout comparison ------------------------------------------------------


@dataclass
class ReferenceRunGroup:
    label: str
    episodes: int
    needed: int
    covered: bool
    """Seen: the reference has at least ``needed`` dataset episodes."""
    runs: int = 0
    successes: int = 0


@dataclass
class ReferenceRuns:
    req: Requirement
    groups: dict[str, ReferenceRunGroup] = field(default_factory=dict)
    """reference phrase (or UNRECOGNIZED) -> runs."""


@dataclass
class ObjectPropertyComparison(AxisComparison):
    objects: list[ReferenceRuns] = field(default_factory=list)

    def lines(self) -> list[str]:
        lines = []
        for obj in self.objects:
            lines.append(f"[{obj.req.obj}] per reference: dataset episodes / needed, seen or unseen, rollout successes / runs")
            for group in obj.groups.values():
                if group.label == UNRECOGNIZED:
                    state, counts = "-", f"{'':>5}   {'':<4}"
                else:
                    state, counts = "seen" if group.covered else "unseen", f"{group.episodes:>5} / {group.needed:<4}"
                lines.append(f"  {group.label:<28} {counts} {state:<7} {group.successes}/{group.runs}")
            covered = [g for g in obj.groups.values() if g.covered]
            other = [g for g in obj.groups.values() if not g.covered]
            n_cov, n_other = sum(g.runs for g in covered), sum(g.runs for g in other)
            lines.append(
                f"  covered: {n_cov}/{n_cov + n_other} runs ({pct(n_cov, n_cov + n_other)})\n"
                f"  success: {rate(sum(g.successes for g in covered), n_cov)} seen vs "
                f"{rate(sum(g.successes for g in other), n_other)} unseen or unrecognized references"
            )
            lines.append("")
        return lines

    def hint_lines(self) -> list[str]:
        lines = []
        for obj in self.objects:
            for group in obj.groups.values():
                if group.covered or not group.runs:
                    continue
                if group.label == UNRECOGNIZED:
                    lines.append(f"  {obj.req.obj}: {group.runs} run(s) referred to it in a way your spec does not list; add that phrase to the spec")
                else:
                    lines.append(
                        f"  {obj.req.obj}: record more episodes whose task text says {group.label} "
                        f"(has {group.episodes}, needs {group.needed}; raise its form in reference_weights)"
                    )
        return (["Object Properties: what to record next:", *lines, ""]) if lines else []


def compare_object_property(graph: Graph, rollout: URIRef, dataset: URIRef, comparison: URIRef, axis: str) -> ObjectPropertyComparison:
    result = ObjectPropertyComparison(axis, {})
    reqs = [r for r in requirements(graph, dataset) if r.references]
    if not reqs:
        result.errors.append("The spec has no color reference requirement with references; nothing to compare")
        return result
    texts = episode_texts(graph, dataset)
    runs = [
        (int(row["index"]), row.run, str(row.status) == "succeeded", None if row.text is None else str(row.text))
        for row in graph.query(RUN_TEXTS_QUERY, initNs={"cp": CP}, initBindings={"rollout": rollout})
    ]
    result.run_covered = {index: True for index, _, _, _ in runs}
    for req in reqs:
        counts = Counter(ref.phrase for ref in (req.classify(t) for t in texts.values()) if ref is not None)
        obj = ReferenceRuns(req)
        for ref in req.references:
            obj.groups[ref.phrase] = ReferenceRunGroup(ref.label, counts[ref.phrase], ref.min_episodes, counts[ref.phrase] >= ref.min_episodes)
        obj.groups[UNRECOGNIZED] = ReferenceRunGroup(UNRECOGNIZED, 0, 0, False)
        for index, node, succeeded, text in sorted(runs, key=lambda r: r[0]):
            ref = req.classify(text)
            group = obj.groups[ref.phrase if ref else UNRECOGNIZED]
            group.runs += 1
            group.successes += succeeded
            if not group.covered:
                result.run_covered[index] = False
            placement = URIRef(f"{comparison}/reference/{node.rsplit('/', 1)[-1]}/{req.obj}")
            result.derived.add((comparison, CP.hasPlacement, placement))
            for p, o in (
                (RDF.type, CP.ReferencePlacement), (CP.ofRun, node), (CP.ofRequirement, req.iri), (CP.covered, Literal(group.covered)),
            ):
                result.derived.add((placement, p, o))
            if ref is not None:
                result.derived.add((placement, CP.inReference, ref.iri))
        result.objects.append(obj)
        unrecognized = obj.groups[UNRECOGNIZED]
        if unrecognized.runs:
            result.findings.append(
                Finding(
                    "Warning",
                    f"'{req.obj}': {unrecognized.runs} run(s) referred to it in a way the spec does not list "
                    f"({unrecognized.runs - unrecognized.successes} failed)",
                    [req.obj],
                    axis,
                )
            )
        unseen = [g for key, g in obj.groups.items() if key != UNRECOGNIZED and g.runs and not g.covered]
        if unseen:
            n = sum(g.runs for g in unseen)
            result.findings.append(
                Finding(
                    "Warning",
                    f"'{req.obj}': {n} run(s) used an unseen reference, with fewer dataset episodes than needed "
                    f"({', '.join(f'{g.label}: {g.episodes}/{g.needed}' for g in unseen)})",
                    [req.obj],
                    axis,
                )
            )
    return result
