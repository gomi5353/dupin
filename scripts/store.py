#!/usr/bin/env python3
"""Verification store: record a dataset and its rollouts as RDF, check them against your specs, query them.

One store per dataset, under stores/<repo_name>/. Reads only meta/ and data/ of
the dataset (no simulator, no videos) and the rollout logs.
Guide: docs/data_verification.md

  # 1. Put the dataset into a new store (again after recording more episodes)
  uv run python scripts/store.py ingest --dataset datasets/${HF_USER}/<repo_name>

  # 2. Copy the specs of the axes you want checked (specs/<task>/), replace every "TODO"
  cp specs/<task>/object_pose.ttl specs/<task>/image_augmentation.ttl my_specs/

  # 3. After a rollout (scripts/rollout.py writes outputs/rollout/<timestamp>.jsonl), put it into the store
  uv run python scripts/store.py ingest --store stores/<repo_name> --rollout-log outputs/rollout/<timestamp>.jsonl

  # 4. Check the dataset against your specs and compare every rollout; specs and results go into the store
  uv run python scripts/store.py verify --store stores/<repo_name> \\
      --spec my_specs/object_pose.ttl my_specs/image_augmentation.ttl

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
    format_comparison,
    ingest_dataset,
    ingest_rollout,
    open_store,
    query,
    verify_store,
)
from data_verification.graph import CP, spec_iri
from data_verification.store import STORES, source_changes


def cmd_ingest(args: argparse.Namespace) -> int:
    if args.dataset is not None:
        store, data = ingest_dataset(args.dataset, args.stores, name=args.name, task=args.task)
        print(f"Wrote {store.dataset_file}: {len(data.episode_indices)} episode(s), objects {', '.join(data.object_names)}")
        print(f"Next: verify --store {store.root} --spec <your spec>.ttl (one --spec per axis; copies of specs/<task>/*.ttl, filled in)")
        return 0
    if args.store is None:
        raise StoreError("ingest --rollout-log needs --store stores/<repo_name>: the store of the dataset the policy was trained on")
    store = open_store(args.store)
    rollout = ingest_rollout(store, args.rollout_log)
    print(f"Wrote {store.rollout_file(args.rollout_log.stem)} ({rollout})")
    print(f"Next: verify --store {store.root} --spec <your spec>.ttl ...")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    store = open_store(args.store)
    rollouts = [] if args.no_rollouts else (args.rollout or None)
    result = verify_store(store, args.spec, rollouts=rollouts, extra_shapes=args.extra_shapes)
    blocks = [format_check(c) for c in result.checks] + [format_comparison(c, hints=not args.no_hints) for c in result.comparisons]
    blocks += [format_combined(c) for c in result.combined]
    print(("\n\n" + "=" * 80 + "\n\n").join(blocks))
    print(f"\nFindings written to {store.root / 'findings'}/")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result.to_dict(), indent=2) + "\n")
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

    print("\nRollouts")
    for name in store.rollout_names() or ["(none ingested)"]:
        rollout = _node(ds, f"rollout/{name}")
        if (rollout, RDF.type, CP.Rollout) not in graph:
            print(f"  {name}")
            continue
        runs = list(graph.objects(rollout, CP.hasRun))
        aborted = sum(str(graph.value(r, CP.status)) == "aborted" for r in runs)
        print(
            f"  {name}: {graph.value(rollout, CP.successEpisodes)}/{graph.value(rollout, CP.totalEpisodes)} succeeded"
            + (f", {aborted} aborted" if aborted else "")
            + f"; eval {graph.value(rollout, CP.evalFile)}, seed {graph.value(rollout, CP.seed)}"
        )
        for change in source_changes(graph, rollout, "rollout"):
            print(f"    STALE: {change}")
        for spec_name in store.spec_names():
            comparison = _node(rollout, f"comparison/{spec_name}")
            n = graph.value(comparison, CP.runsInDistribution)
            if (comparison, RDF.type, CP.Comparison) in graph:
                print(f"    vs {spec_name}: " + (f"{n} run(s) in distribution; " if n is not None else "") + _findings_summary(graph, comparison))
            else:
                print(f"    vs {spec_name}: not compared yet (run verify)")
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

    ingest = sub.add_parser("ingest", help="Put a dataset (new store) or a rollout log (existing store) into a store.")
    source = ingest.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", type=Path, help="Local LeRobot v3 dataset directory; writes stores/<name>/dataset.ttl.")
    source.add_argument("--rollout-log", type=Path, help="outputs/rollout/<timestamp>.jsonl written by scripts/rollout.py.")
    ingest.add_argument("--store", type=Path, help="With --rollout-log: the store of the dataset the policy was trained on.")
    ingest.add_argument("--name", help=f"With --dataset: store name (default: the dataset directory name); the store is {STORES}/<name>.")
    ingest.add_argument("--task", help="With --dataset: the gym task id the dataset was recorded with (stored as cp:task).")
    ingest.add_argument("--stores", type=Path, default=STORES, help=f"With --dataset: directory of stores (default: {STORES}).")
    ingest.set_defaults(func=cmd_ingest)

    verify = sub.add_parser("verify", help="Check the dataset against your spec files and compare the rollouts; store specs and findings.")
    verify.add_argument("--store", required=True, type=Path)
    verify.add_argument(
        "--spec", action="extend", nargs="+", type=Path, required=True,
        help="Your spec files (.ttl), one per axis: --spec a.ttl b.ttl (or --spec a.ttl --spec b.ttl). "
        "The store keeps exactly the specs of the last verify.",
    )
    verify.add_argument("--rollout", action="append", help="Only this ingested rollout; repeatable (default: all).")
    verify.add_argument("--no-rollouts", action="store_true", help="Check the dataset only.")
    verify.add_argument("--no-hints", action="store_true", help="Do not print suggested next steps (they are still stored as Info findings).")
    verify.add_argument("--strict", action="store_true", help="Also fail on warnings.")
    verify.add_argument("--json", type=Path, help="Also write the full result as JSON.")
    verify.add_argument("--extra-shapes", type=Path, action="append", default=[], help="Additional SHACL rules (.ttl) for the dataset check; repeatable.")
    verify.set_defaults(func=cmd_verify)

    q = sub.add_parser("query", help="Run a SPARQL query over the store.")
    q.add_argument("--store", required=True, type=Path)
    text = q.add_mutually_exclusive_group(required=True)
    text.add_argument("--query-file", type=Path)
    text.add_argument("--query-text")
    q.add_argument("--format", choices=("table", "csv", "json"), default="table")
    q.add_argument("--with-ontology", action="store_true", help="Also load the ontology (classes, labels, subclass links).")
    q.set_defaults(func=cmd_query)

    explore = sub.add_parser("explore", help="Summarize the store: dataset, specs, rollouts, latest results.")
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
