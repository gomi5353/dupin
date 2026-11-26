#!/usr/bin/env python3
"""Verification store: record a dataset as RDF, check it against your specs, compare rollouts with it, query it.

One store per dataset, under stores/<repo_name>/. Reads only meta/ and data/ of
the dataset (no simulator, no videos); rollout logs are read by `compare`, which stores only its result.
Guide: docs/data_verification.md

  # 1. Put the dataset into a new store (again after recording more episodes)
  uv run python scripts/store.py ingest --dataset datasets/${HF_USER}/<repo_name>

  # 2. Copy the spec of an axis (specs/<task>/), replace every "TODO"
  cp specs/<task>/object_pose.ttl my_specs/

  # 3. Check the dataset against your specs; specs and results go into the store
  #    (append the specs of other axes after the first to check them in the same run)
  uv run python scripts/store.py verify --store stores/<repo_name> --spec my_specs/object_pose.ttl

  # 4. After a rollout (scripts/rollout.py writes outputs/rollout/<timestamp>.jsonl), compare its runs with
  #    the dataset's coverage under your specs (the log is read each time; only the result is stored)
  uv run python scripts/store.py compare --store stores/<repo_name> --spec my_specs/object_pose.ttl \
      --rollout-log outputs/rollout/<timestamp>.jsonl

  # 5. Ask the store
  uv run python scripts/store.py explore --store stores/<repo_name>
  uv run python scripts/store.py query --store stores/<repo_name> --query-file queries/uncovered_cells.rq
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rdflib import RDF, URIRef

from data_verification import (
    StoreError,
    format_check,
    format_combined,
    compare_store,
    format_comparison,
    ingest_dataset,
    open_store,
    query,
    verify_store,
)
from data_verification.graph import CP, spec_iri
from data_verification.render import make_console, render_check, render_combined, render_comparison, use_tables
from data_verification.store import STORES, source_changes


def cmd_ingest(args: argparse.Namespace) -> int:
    store, data = ingest_dataset(args.dataset, args.stores, name=args.name, task=args.task)
    print(f"Wrote {store.dataset_file}: {len(data.episode_indices)} episode(s), objects {', '.join(data.object_names)}")
    print(f"Next: verify --store {store.root} --spec <your spec>.ttl (one --spec per axis; copies of specs/<task>/*.ttl, filled in)")
    return 0


def _write_figures(args: argparse.Namespace, store, result, **kinds) -> None:
    if args.no_figures:
        return
    from data_verification.figures import save_figures

    figures = save_figures(store.root, result, **kinds)
    if figures:
        print(f"Figures written to {figures[0].parent}/ ({len(figures)}):")
        for path in figures:
            print(f"  {path}")


def _write_json(args: argparse.Namespace, result) -> None:
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result.to_dict(), indent=2) + "\n")


def _print_reports(args: argparse.Namespace, reports: list[tuple[str, object]]) -> None:
    """Print each report as rich tables on a terminal (or with --format table), as plain text otherwise."""
    hints = not getattr(args, "no_hints", False)
    if use_tables(args.format):
        console = make_console()
        draw = {"check": render_check, "comparison": lambda r, c: render_comparison(r, c, hints=hints), "combined": render_combined}
        for n, (kind, report) in enumerate(reports):
            if n:
                console.rule()
            draw[kind](report, console)
        return
    text = {"check": format_check, "comparison": lambda r: format_comparison(r, hints=hints), "combined": format_combined}
    print(("\n\n" + "=" * 80 + "\n\n").join(text[kind](report) for kind, report in reports))


def cmd_verify(args: argparse.Namespace) -> int:
    store = open_store(args.store)
    result = verify_store(store, args.spec, extra_shapes=args.extra_shapes)
    _print_reports(args, [("check", c) for c in result.checks])
    print(f"\nFindings written to {store.root / 'findings'}/")
    _write_figures(args, store, result, comparisons=False)
    _write_json(args, result)
    return 0 if result.passed and not (args.strict and result.warnings()) else 1


def cmd_compare(args: argparse.Namespace) -> int:
    store = open_store(args.store)
    result = compare_store(store, args.spec, args.rollout_log, write=not args.no_save)
    _print_reports(args, [("check", c) for c in result.checks] + [("comparison", c) for c in result.comparisons] + [("combined", c) for c in result.combined])
    if result.comparisons and not args.no_save:
        print(f"\nFindings written to {store.root / 'findings'}/")
    _write_figures(args, store, result, checks=False)
    _write_json(args, result)
    return 0 if result.passed and not (args.strict and result.warnings()) else 1


def cmd_query(args: argparse.Namespace) -> int:
    store = open_store(args.store)
    text = args.query_file.read_text() if args.query_file else args.query_text
    print(query(store, text, args.format, with_ontology=args.with_ontology))
    return 0


def cmd_explore(args: argparse.Namespace) -> int:
    store = open_store(args.store)
    graph = store.graph()
    ds = store.dataset
    episodes = list(graph.objects(ds, CP.hasEpisode))
    print(f"Store   : {store.root}")
    print(f"Dataset : {graph.value(ds, CP.sourcePath)}  (ingested {graph.value(ds, CP.ingestedAt)})")
    print(f"          {len(episodes)} episode(s); objects {', '.join(sorted(str(n) for n in graph.objects(ds, CP.recordedObject)))}")
    lights, colored = sorted(str(n) for n in graph.objects(ds, CP.recordedLight)), sorted(str(n) for n in graph.objects(ds, CP.recordedColorObject))
    if lights or colored:
        print(f"          lights {', '.join(lights) or '-'}; object colors {', '.join(colored) or '-'}")
    for label, prop in (("task", CP.task), ("task text", CP.taskDescription), ("robot", CP.robotType), ("fps", CP.fps)):
        for value in graph.objects(ds, prop):
            print(f"          {label}: {value}")
    for change in source_changes(graph, ds, "dataset"):
        print(f"          STALE: {change}")

    print("\nSpecs")
    for name in store.spec_names() or ["(none; run verify --spec ...)"]:
        spec = spec_iri(ds, name)
        if not store.spec_file(name).is_file():
            print(f"  {name}")
            continue
        axes = sorted(f"{a.rsplit('/', 1)[-1]} ({t.n3(graph.namespace_manager)})" for a in graph.objects(spec, CP.declaresAxis) for t in graph.objects(a, RDF.type))
        requirements = sum(1 for a in graph.objects(spec, CP.declaresAxis) for _ in graph.objects(a, CP.hasRequirement))
        print(f"  {name}: {graph.value(spec, CP.sourcePath)}; axes {', '.join(axes)}; {requirements} requirement(s)")
        for change in source_changes(graph, spec, "spec"):
            print(f"    STALE: {change}")
        print(f"    {_result_line(graph, _node(spec, 'verification'))}")

    print("\nComparisons")
    names = sorted(graph.subjects(RDF.type, CP.Rollout))
    for rollout in names:
        name = str(rollout).rsplit("/", 1)[-1]
        runs = list(graph.objects(rollout, CP.hasRun))
        aborted = sum(str(graph.value(r, CP.status)) == "aborted" for r in runs)
        print(
            f"  {name}: {graph.value(rollout, CP.successEpisodes)}/{graph.value(rollout, CP.totalEpisodes)} succeeded"
            + (f", {aborted} aborted" if aborted else "")
            + f"; eval {graph.value(rollout, CP.evalFile)}, seed {graph.value(rollout, CP.seed)}"
        )
        for change in source_changes(graph, rollout, "rollout"):
            print(f"    STALE: {change}")
        for comparison in sorted(graph.subjects(CP.ofRollout, rollout)):
            n = graph.value(comparison, CP.runsInDistribution)
            spec_name = str(graph.value(comparison, CP.ofSpec)).rsplit("/", 1)[-1]
            print(f"    vs {spec_name}: " + (f"{n} run(s) in distribution; " if n is not None else "") + _findings_summary(graph, comparison))
    if not names:
        print("  (none; run compare --rollout-log ...)")
    print(f"\nQuery it: query --store {store.root} --query-text 'SELECT ...'  (examples in queries/)")
    return 0


def _node(base, path: str) -> URIRef:
    return URIRef(f"{base}/{path}")


def _findings_summary(graph, node) -> str:
    counts: dict[str, int] = {}
    for f in graph.objects(node, CP.hasFinding):
        severity = str(graph.value(f, CP.severity))
        counts[severity] = counts.get(severity, 0) + 1
    return ", ".join(f"{n} {s}" for s, n in sorted(counts.items())) or "no findings"


def _result_line(graph, verification) -> str:
    passed = graph.value(verification, CP.passed)
    if passed is None:
        return "not verified yet (run verify)"
    return f"verified {graph.value(verification, CP.verifiedAt)}: {'passed' if passed.toPython() else 'NOT passed'}; {_findings_summary(graph, verification)}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Put a recorded dataset into a new store.")
    ingest.add_argument("--dataset", required=True, type=Path, help="Local LeRobot v3 dataset directory; writes stores/<name>/dataset.ttl.")
    ingest.add_argument("--name", help=f"Store name (default: the dataset directory name); the store is {STORES}/<name>.")
    ingest.add_argument("--task", help="The gym task id the dataset was recorded with (stored as cp:task).")
    ingest.add_argument("--stores", type=Path, default=STORES, help=f"Directory of stores (default: {STORES}).")
    ingest.set_defaults(func=cmd_ingest)

    verify = sub.add_parser("verify", help="Check the recorded dataset against your spec files; store specs and findings.")
    verify.add_argument("--store", required=True, type=Path)
    verify.add_argument(
        "--spec", action="extend", nargs="+", type=Path, required=True,
        help="Your spec files (.ttl), one per axis: --spec a.ttl b.ttl (or --spec a.ttl --spec b.ttl). "
        "The store keeps exactly the specs of the last verify.",
    )
    verify.add_argument("--format", choices=("auto", "table", "text"), default="auto", help="Report layout: tables on a terminal, plain text when redirected (auto, default), or force one.")
    verify.add_argument("--no-figures", action="store_true", help="Do not write the coverage heatmaps to <store>/figures/.")
    verify.add_argument("--strict", action="store_true", help="Also fail on warnings.")
    verify.add_argument("--json", type=Path, help="Also write the full result as JSON.")
    verify.add_argument("--extra-shapes", type=Path, action="append", default=[], help="Additional SHACL rules (.ttl) for the dataset check; repeatable.")
    verify.set_defaults(func=cmd_verify)

    compare = sub.add_parser("compare", help="Compare the runs of rollout logs with the dataset's coverage under your spec files; stores the result, not the log.")
    compare.add_argument("--store", required=True, type=Path, help="The store of the dataset the policy was trained on.")
    compare.add_argument("--spec", action="extend", nargs="+", type=Path, required=True, help="Your spec files (.ttl), one per axis (as for verify).")
    compare.add_argument(
        "--rollout-log", action="extend", nargs="+", type=Path, required=True,
        help="outputs/rollout/<timestamp>.jsonl written by scripts/rollout.py: --rollout-log a.jsonl b.jsonl (or repeat the option).",
    )
    compare.add_argument("--no-figures", action="store_true", help="Do not write the coverage heatmaps to <store>/figures/compare/.")
    compare.add_argument("--format", choices=("auto", "table", "text"), default="auto", help="Report layout: tables on a terminal, plain text when redirected (auto, default), or force one.")
    compare.add_argument("--no-save", action="store_true", help="Do not write the comparison to <store>/findings/.")
    compare.add_argument("--no-hints", action="store_true", help="Do not print suggested next steps.")
    compare.add_argument("--strict", action="store_true", help="Also fail on warnings.")
    compare.add_argument("--json", type=Path, help="Also write the full result as JSON.")
    compare.set_defaults(func=cmd_compare)

    q = sub.add_parser("query", help="Run a SPARQL query over the store.")
    q.add_argument("--store", required=True, type=Path)
    text = q.add_mutually_exclusive_group(required=True)
    text.add_argument("--query-file", type=Path)
    text.add_argument("--query-text")
    q.add_argument("--format", choices=("table", "csv", "json"), default="table")
    q.add_argument("--with-ontology", action="store_true", help="Also load the ontology (classes, labels, subclass links).")
    q.set_defaults(func=cmd_query)

    explore = sub.add_parser("explore", help="Summarize the store: dataset, specs, latest verify and compare results.")
    explore.add_argument("--store", required=True, type=Path)
    explore.set_defaults(func=cmd_explore)

    args = parser.parse_args()
    try:
        return args.func(args)
    except (StoreError, ValueError) as exc:
        for line in getattr(exc, "errors", [str(exc)]):
            print(f"ERROR  {line}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
