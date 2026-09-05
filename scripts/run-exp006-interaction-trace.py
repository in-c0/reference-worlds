#!/usr/bin/env python3
"""Run the frozen EXP-006 deterministic interaction/revisit trace."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from refworld.exp006_interaction_trace import build_interaction_trace


EXPECTED_REFERENCE_SHA256 = "4ee7a137e577378a02600ac8a32dc89a7c8409120273622227ad972cc5aff61a"
EXPECTED_WIDTH = 1672
EXPECTED_HEIGHT = 941


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EXP-006 interaction/revisit trace")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--binding",
        type=Path,
        default=Path("examples/exp006_collaborative_futures_binding.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/exp006/collaborative-futures-interaction-trace-v0"),
    )
    return parser.parse_args()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    reference_path = args.reference.resolve()
    binding_path = args.binding.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)
    if not binding_path.is_file():
        raise FileNotFoundError(binding_path)

    actual_sha = sha256_file(reference_path)
    if actual_sha != EXPECTED_REFERENCE_SHA256:
        raise RuntimeError(f"reference SHA mismatch: {actual_sha}")

    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if binding["reference_binding"]["sha256"] != EXPECTED_REFERENCE_SHA256:
        raise RuntimeError("binding/reference SHA mismatch")

    reference = np.asarray(Image.open(reference_path).convert("RGB"), dtype=np.uint8)
    height, width = reference.shape[:2]
    if (width, height) != (EXPECTED_WIDTH, EXPECTED_HEIGHT):
        raise RuntimeError(f"reference dimensions changed: {(width, height)}")

    result = build_interaction_trace(reference, binding)

    for name, image in result.images.items():
        Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(output / f"{name}.png")

    for name, state in result.states.items():
        _write_json(output / f"{name}.json", state)

    trace_jsonl = output / "interaction-trace.jsonl"
    with trace_jsonl.open("w", encoding="utf-8") as handle:
        for step in result.report["trace_steps"]:
            handle.write(json.dumps(step, sort_keys=True, allow_nan=False) + "\n")

    report = dict(result.report)
    report["reference"] = {
        "sha256": actual_sha,
        "width": width,
        "height": height,
        "binary_committed": False,
    }
    report["artifacts"] = {
        "trace_jsonl": trace_jsonl.name,
        "images": {name: f"{name}.png" for name in result.images},
        "states": {name: f"{name}.json" for name in result.states},
    }
    report_path = output / "EXP006-INTERACTION-TRACE-V0.json"
    _write_json(report_path, report)

    print("EXP-006 INTERACTION TRACE V0 COMPLETE")
    print(f"12-step trace recorded: {report['gate_checks']['twelve_steps_recorded']}")
    print(
        "Neighbors observed: "
        f"left={report['neighbor_observed_fraction']['left']:.4f} "
        f"right={report['neighbor_observed_fraction']['right']:.4f}"
    )
    target = report["trace_steps"][5]
    print(
        "Out-of-view target: "
        f"{target['target_entity_id']} fully_out={target['visibility']['fully_out_of_view']}"
    )
    print(f"Edit survives navigation: {report['persistence']['edit_survives_navigation']}")
    print(f"Stable IDs after reload: {report['persistence']['stable_ids_after_reload']}")
    print(f"Edit survives reload: {report['persistence']['edit_survives_reload']}")
    print(f"Collateral semantic drift: {report['semantic_drift']['collateral_semantic_drift_count']}")
    print(
        "Hero after exact reference match: "
        f"{report['hero_fidelity']['after']['exact_match']}"
    )
    print(f"Automated interaction gate: {'PASS' if report['automated_gate_passed'] else 'FAIL'}")
    print("Semantic edit is intentionally not visually mapped in R1; this is reported, not hidden.")
    print(f"Report: {report_path}")
    print("Rank-4 remains sealed and untouched.")
    return 0 if report["automated_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
