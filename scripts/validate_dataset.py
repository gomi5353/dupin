#!/usr/bin/env python3
"""Validate a LeRobot v3 dataset before training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from data_validation import validate_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="Local LeRobot dataset directory")
    parser.add_argument("--contract", type=Path, help="Optional RDF/Turtle contract graph")
    parser.add_argument("--ontology", type=Path, help="Optional custom OWL ontology graph")
    parser.add_argument("--shapes", type=Path, help="Optional custom SHACL shapes graph")
    parser.add_argument("--report", type=Path, help="Write a JSON validation report")
    parser.add_argument("--strict", action="store_true", help="Return failure for warnings as well as errors")
    args = parser.parse_args()

    report = validate_dataset(
        args.dataset,
        contract=args.contract,
        ontology=args.ontology,
        shapes=args.shapes,
    )
    for key, passed in report.checks.items():
        print(f"{'PASS' if passed else 'FAIL'}  {key}")
    for warning in report.warnings:
        print(f"WARNING  {warning}")
    for error in report.errors:
        print(f"ERROR  {error}")
    print("Dataset is ready for lerobot-train." if report.valid else "Dataset is not ready for training.")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    return 1 if report.errors or (args.strict and report.warnings) else 0


if __name__ == "__main__":
    sys.exit(main())
