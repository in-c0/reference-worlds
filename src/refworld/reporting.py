"""Strict JSON report serialization and schema validation."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def json_safe(value: Any) -> Any:
    """Convert benchmark values to strict-JSON-safe Python values.

    Non-finite floats are represented as ``None``. This matters for legitimate
    diagnostics such as infinite PSNR on identical images or an unobserved
    failure radius. NumPy scalars/arrays and dataclasses are normalized too.
    """

    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported report value type: {type(value).__name__}")


def validate_report(report: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    """Validate a report against a JSON Schema draft 2020-12 mapping."""

    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise RuntimeError(
            "report validation requires the optional validation dependency: "
            "pip install 'refworld-bench[validation]'"
        ) from exc

    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(report), key=lambda error: list(error.absolute_path))
    if not errors:
        return

    first = errors[0]
    path = ".".join(str(part) for part in first.absolute_path) or "<root>"
    raise ValueError(f"invalid RefWorldBench report at {path}: {first.message}")


def write_report(
    path: str | Path,
    report: Mapping[str, Any],
    *,
    schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize, optionally validate, and write a strict JSON benchmark report."""

    safe = json_safe(report)
    if not isinstance(safe, dict):
        raise TypeError("report root must be an object")
    if schema is not None:
        validate_report(safe, schema)

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(safe, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return safe


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value
