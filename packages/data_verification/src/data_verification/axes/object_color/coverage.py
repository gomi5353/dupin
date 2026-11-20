"""Object color axis: episodes per color category, the language check, and the rollout runs' colors.

An episode's (or run's) color of an object belongs to the requirement's
nearest category by Euclidean RGB distance, if that distance is at most
maxColorDistance; otherwise to no category (an unseen color). A rollout run is
covered on this axis when, for every requirement, its color belongs to a
category with at least minEpisodesPerColor dataset episodes.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from rdflib import RDF, Graph, Literal, URIRef

from ...graph import CP, index_literal
from ...verifier import AxisCheck, AxisComparison, Finding, pct, rate

REQUIREMENTS_QUERY = """
SELECT ?req ?name ?dist ?min WHERE {
    ?dataset cp:hasGeneralizationAxis ?axis .
    ?axis cp:hasRequirement ?req .
    ?req a cp:ObjectColorAxisRequirement ; cp:hasTaskObject ?obj ;
         cp:maxColorDistance ?dist ; cp:minEpisodesPerColor ?min .
    ?obj cp:name ?name .
}
"""

CATEGORIES_QUERY = """
SELECT ?category ?name ?r ?g ?b WHERE {
    ?req cp:hasColorCategory ?category .
    ?category cp:name ?name ; cp:colorR ?r ; cp:colorG ?g ; cp:colorB ?b .
}
"""

# Object colors of the finished runs of ?rollout (bound by the caller).
RUN_COLORS_QUERY = """
SELECT ?run ?index ?status ?name ?r ?g ?b WHERE {
    ?rollout cp:hasRun ?run .
    ?run cp:index ?index ; cp:status ?status ; cp:hasObjectColor ?color .
    FILTER (?status IN ("succeeded", "failed"))
    ?color cp:name ?name ; cp:colorR ?r ; cp:colorG ?g ; cp:colorB ?b .
}
"""

RGB = tuple[float, float, float]


@dataclass
class Category:
    iri: URIRef
    name: str
    rgb: RGB


@dataclass
class Requirement:
    iri: URIRef
    obj: str
    max_distance: float
    min_episodes: int
    categories: list[Category]

    def classify(self, rgb: RGB) -> Category | None:
        """The nearest category within max_distance, or None."""
        best = min(self.categories, key=lambda c: math.dist(c.rgb, rgb), default=None)
        return best if best is not None and math.dist(best.rgb, rgb) <= self.max_distance else None


def requirements(graph: Graph, dataset: URIRef) -> list[Requirement]:
    result = []
    for row in graph.query(REQUIREMENTS_QUERY, initNs={"cp": CP}, initBindings={"dataset": dataset}):
        categories = [
            Category(c.category, str(c.name), (float(c.r), float(c.g), float(c.b)))
            for c in graph.query(CATEGORIES_QUERY, initNs={"cp": CP}, initBindings={"req": row.req})
        ]
        result.append(
            Requirement(row.req, str(row.name), float(row.dist), int(row.min), sorted(categories, key=lambda c: c.name))
        )
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


def names_color(text: str, word: str) -> bool:
    """Whether ``text`` contains ``word`` as a whole word (case-insensitive)."""
    return re.search(rf"(?<![A-Za-z]){re.escape(word)}(?![A-Za-z])", text, flags=re.IGNORECASE) is not None


# ---- Dataset check -----------------------------------------------------------


@dataclass
class ObjectColorCoverage:
    obj: str
    req: Requirement
    counts: dict[str, int]
    """category name -> episodes."""
    uncategorized: int


@dataclass
class ObjectColorCheck(AxisCheck):
    coverage: list[ObjectColorCoverage] = field(default_factory=list)

    def lines(self) -> list[str]:
        lines = []
        for table in self.coverage:
            r = table.req
            ok = sum(n >= r.min_episodes for n in table.counts.values())
            lines.append(
                f"[{table.obj}] {ok}/{len(r.categories)} colors with >= {r.min_episodes} episodes"
            )
            for c in r.categories:
                n = table.counts.get(c.name, 0)
                lines.append(f"  {c.name:<12} {n:>5}" + ("" if n >= r.min_episodes else "!"))
            if table.uncategorized:
                lines.append(f"  {'(no category)':<12} {table.uncategorized:>5}  not counted")
            lines.append("")
        return lines


def check_object_color(graph: Graph, dataset: URIRef, axis: str) -> ObjectColorCheck:
    result = ObjectColorCheck(axis)
    colors = episode_colors(graph, dataset)
    texts = episode_texts(graph, dataset)
    for req in requirements(graph, dataset):
        if not req.categories:
            continue  # the SHACL requirement shape reports it
        counts: Counter = Counter()
        uncategorized: list[int] = []
        unnamed: dict[str, list[int]] = {}
        for index, rgb in sorted(colors.get(req.obj, {}).items()):
            category = req.classify(rgb)
            if category is None:
                uncategorized.append(index)
                continue
            counts[category.name] += 1
            if index in texts and not names_color(texts[index], category.name):
                unnamed.setdefault(category.name, []).append(index)
        result.coverage.append(ObjectColorCoverage(req.obj, req, dict(counts), len(uncategorized)))

        for c in req.categories:
            result.derived.add((c.iri, CP.episodeCount, index_literal(counts[c.name])))
            if counts[c.name] < req.min_episodes:
                result.findings.append(
                    Finding("Violation", f"'{req.obj}' color '{c.name}' has {counts[c.name]} episode(s); needs at least {req.min_episodes}", [str(c.iri)], axis)
                )
        for name, episodes in sorted(unnamed.items()):
            result.findings.append(
                Finding(
                    "Violation",
                    f"'{req.obj}' is '{name}', but the task text does not say '{name}'",
                    [f"{dataset}/episode/{i}" for i in episodes],
                    axis,
                )
            )
        if uncategorized:
            result.findings.append(
                Finding(
                    "Warning",
                    f"'{req.obj}' has a color within {req.max_distance:g} of no category; these episodes do not count",
                    [f"{dataset}/episode/{i}" for i in uncategorized],
                    axis,
                )
            )
    return result


# ---- Rollout comparison ------------------------------------------------------


@dataclass
class ColorRunGroup:
    label: str
    episodes: int
    covered: bool
    runs: int = 0
    successes: int = 0


@dataclass
class ObjectColorRuns:
    obj: str
    req: Requirement
    groups: dict[str, ColorRunGroup] = field(default_factory=dict)
    """category name (or "(unseen)") -> runs."""


@dataclass
class ObjectColorComparison(AxisComparison):
    objects: list[ObjectColorRuns] = field(default_factory=list)

    def lines(self) -> list[str]:
        lines = []
        for obj in self.objects:
            lines.append(f"[{obj.obj}] dataset episodes, rollout successes / runs per color (! = fewer than {obj.req.min_episodes} episodes)")
            for group in obj.groups.values():
                mark = "" if group.covered else "!"
                lines.append(f"  {group.label:<12} {group.episodes:>5}{mark:<2} {group.successes}/{group.runs}")
            covered = [g for g in obj.groups.values() if g.covered]
            other = [g for g in obj.groups.values() if not g.covered]
            n_cov, n_other = sum(g.runs for g in covered), sum(g.runs for g in other)
            lines.append(
                f"  covered: {n_cov}/{n_cov + n_other} runs ({pct(n_cov, n_cov + n_other)})\n"
                f"  success: {rate(sum(g.successes for g in covered), n_cov)} seen colors vs "
                f"{rate(sum(g.successes for g in other), n_other)} unseen or rare colors"
            )
            lines.append("")
        return lines

    def hint_lines(self) -> list[str]:
        lines = []
        for obj in self.objects:
            for group in obj.groups.values():
                if group.covered or not group.runs:
                    continue
                if group.label == UNSEEN:
                    lines.append(f"  {obj.obj}: {group.runs} run(s) used a color your spec does not name; add colors to the task's palette and to the spec")
                else:
                    lines.append(f"  {obj.obj}: record more '{group.label}' episodes (has {group.episodes}, needs {obj.req.min_episodes})")
        return (["Object color: what to record next:", *lines, ""]) if lines else []


UNSEEN = "(unseen)"


def compare_object_color(graph: Graph, rollout: URIRef, dataset: URIRef, comparison: URIRef, axis: str) -> ObjectColorComparison:
    result = ObjectColorComparison(axis, {})
    reqs = [r for r in requirements(graph, dataset) if r.categories]
    if not reqs:
        result.errors.append("The spec has no object color requirement with categories; nothing to compare")
        return result
    colors = episode_colors(graph, dataset)
    runs: dict[str, dict[int, RGB]] = {}
    outcome: dict[int, bool] = {}
    nodes: dict[int, URIRef] = {}
    for row in graph.query(RUN_COLORS_QUERY, initNs={"cp": CP}, initBindings={"rollout": rollout}):
        index = int(row["index"])
        outcome[index], nodes[index] = str(row.status) == "succeeded", row.run
        runs.setdefault(str(row.name), {})[index] = (float(row.r), float(row.g), float(row.b))
    result.run_covered = {index: True for index in outcome}
    for req in reqs:
        counts = Counter(c.name for c in (req.classify(rgb) for rgb in colors.get(req.obj, {}).values()) if c is not None)
        obj = ObjectColorRuns(req.obj, req)
        for c in req.categories:
            obj.groups[c.name] = ColorRunGroup(c.name, counts[c.name], counts[c.name] >= req.min_episodes)
        obj.groups[UNSEEN] = ColorRunGroup(UNSEEN, 0, False)
        for index, rgb in sorted(runs.get(req.obj, {}).items()):
            category = req.classify(rgb)
            group = obj.groups[category.name if category else UNSEEN]
            group.runs += 1
            group.successes += outcome[index]
            if not group.covered:
                result.run_covered[index] = False
            placement = URIRef(f"{comparison}/color/{nodes[index].rsplit('/', 1)[-1]}/{req.obj}")
            result.derived.add((comparison, CP.hasPlacement, placement))
            for p, o in (
                (RDF.type, CP.ColorPlacement), (CP.ofRun, nodes[index]), (CP.ofRequirement, req.iri), (CP.covered, Literal(group.covered)),
            ):
                result.derived.add((placement, p, o))
            if category is not None:
                result.derived.add((placement, CP.inCategory, category.iri))
        result.objects.append(obj)
        unseen = obj.groups[UNSEEN]
        if unseen.runs:
            result.findings.append(
                Finding("Warning", f"'{req.obj}': {unseen.runs} run(s) used a color the spec does not name ({unseen.runs - unseen.successes} failed)", [req.obj], axis)
            )
        rare = [g for g in obj.groups.values() if g.label != UNSEEN and g.runs and not g.covered]
        if rare:
            n = sum(g.runs for g in rare)
            result.findings.append(
                Finding(
                    "Warning",
                    f"'{req.obj}': {n} run(s) used a color with fewer than {req.min_episodes} dataset episodes",
                    [f"{req.obj}/color/{g.label}" for g in rare],
                    axis,
                )
            )
    return result
