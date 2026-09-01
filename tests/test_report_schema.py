import json
from pathlib import Path

from refworld.reporting import validate_report


ROOT = Path(__file__).resolve().parents[1]


def test_public_synthetic_report_matches_repository_schema():
    schema = json.loads((ROOT / "schemas" / "report.schema.json").read_text())
    report = json.loads((ROOT / "examples" / "synthetic-report.json").read_text())
    validate_report(report, schema)
