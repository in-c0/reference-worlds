"""Small command-line entry points for RefWorldBench."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .reporting import load_json, validate_report


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


if __name__ == "__main__":
    raise SystemExit(validate_report_main())
