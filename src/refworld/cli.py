"""Small command-line entry points for RefWorldBench."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .datasets.blendedmvs import load_manifest, prepare_bootstrap
from .reporting import load_json, validate_report, write_report


def validate_report_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a RefWorldBench JSON report")
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("schemas/report.schema.json"),
        help="report schema path (default: schemas/report.schema.json)",
    )
    args = parser.parse_args(argv)

    try:
        report = load_json(args.report)
        schema = load_json(args.schema)
        validate_report(report, schema)
    except Exception as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1

    print(f"VALID: {args.report}")
    return 0


def prepare_blendedmvs_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare metadata-only RefWorldBench records from a local BlendedMVS root"
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("datasets/blendedmvs-bootstrap-v0.json"),
        help="frozen scene manifest",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/blendedmvs-bootstrap-v0.prepared.json"),
        help="metadata-only output JSON",
    )
    args = parser.parse_args(argv)

    try:
        frozen = load_manifest(args.manifest)
        prepared = prepare_bootstrap(args.dataset_root, frozen)
        write_report(args.output, prepared)
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"PREPARED: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(validate_report_main())
