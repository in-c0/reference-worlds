import json
import math

import numpy as np
import pytest

from refworld.reporting import json_safe, validate_report, write_report


SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["version", "system", "sample", "anchor", "runtime"],
    "properties": {
        "version": {"type": "string"},
        "system": {
            "type": "object",
            "required": ["name"],
            "properties": {"name": {"type": "string", "minLength": 1}},
        },
        "sample": {
            "type": "object",
            "required": ["id", "input_type"],
            "properties": {
                "id": {"type": "string"},
                "input_type": {"enum": ["single-image"]},
            },
        },
        "anchor": {
            "type": "object",
            "properties": {"psnr": {"type": ["number", "null"]}},
        },
        "runtime": {"type": "object"},
    },
}


def test_json_safe_converts_nonfinite_and_numpy_values():
    safe = json_safe({
        "psnr": math.inf,
        "bad": float("nan"),
        "array": np.asarray([1.0, np.inf]),
        "scalar": np.float64(2.5),
    })
    assert safe == {
        "psnr": None,
        "bad": None,
        "array": [1.0, None],
        "scalar": 2.5,
    }


def test_write_report_emits_strict_valid_json(tmp_path):
    report = {
        "version": "0.1",
        "system": {"name": "synthetic"},
        "sample": {"id": "s1", "input_type": "single-image"},
        "anchor": {"psnr": math.inf},
        "runtime": {},
    }
    target = tmp_path / "report.json"
    safe = write_report(target, report, schema=SCHEMA)
    assert safe["anchor"]["psnr"] is None
    parsed = json.loads(target.read_text())
    assert parsed == safe
    assert "Infinity" not in target.read_text()


def test_validate_report_rejects_missing_required_field():
    invalid = {
        "version": "0.1",
        "system": {},
        "sample": {"id": "s1", "input_type": "single-image"},
        "anchor": {},
        "runtime": {},
    }
    with pytest.raises(ValueError, match="system"):
        validate_report(invalid, SCHEMA)


def test_json_safe_rejects_opaque_object():
    with pytest.raises(TypeError):
        json_safe(object())
