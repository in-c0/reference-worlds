"""Small command-line entry points for RefWorldBench."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .datasets.blendedmvs import load_manifest, prepare_bootstrap
from .experiments.exp001 import run_marble_stage1
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


def marble_stage1_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="EXP-001 stage 1: generate a Marble world and materialize safe local exports"
    )
    parser.add_argument("reference_image", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--model", default="marble-1.1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--spz-tier",
        action="append",
        dest="spz_tiers",
        choices=["100k", "500k", "full_res"],
        help="SPZ tier to materialize; repeat for multiple tiers (default: 500k)",
    )
    parser.add_argument("--no-collider", action="store_true")
    parser.add_argument(
        "--allow-recaption",
        action="store_true",
        help="use Marble recaptioning instead of the EXP-001 default disable_recaption=true",
    )
    args = parser.parse_args(argv)

    try:
        manifest = run_marble_stage1(
            args.reference_image,
            args.output_dir,
            display_name=args.display_name,
            model=args.model,
            seed=args.seed,
            disable_recaption=not args.allow_recaption,
            spz_tiers=tuple(args.spz_tiers or ["500k"]),
            include_collider=not args.no_collider,
        )
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"GENERATED: {manifest['world'].get('world_id')} -> {args.output_dir / 'stage1.safe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(validate_report_main())
