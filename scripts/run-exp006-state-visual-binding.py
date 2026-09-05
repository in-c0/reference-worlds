#!/usr/bin/env python3
"""Run the bounded semantic-state-driven visual binding follow-up to EXP-006."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from refworld.exp006_state_visual import build_state_visual_binding


EXPECTED_REFERENCE_SHA256 = "4ee7a137e577378a02600ac8a32dc89a7c8409120273622227ad972cc5aff61a"
EXPECTED_WIDTH = 1672
EXPECTED_HEIGHT = 941


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(path)


def _save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray((np.asarray(mask, dtype=bool).astype(np.uint8) * 255), mode="L").save(path)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EXP-006 semantic state visual binding v0")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--binding",
        type=Path,
        default=Path("examples/exp006_collaborative_futures_binding.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/exp006/collaborative-futures-state-visual-binding-v0"),
    )
    return parser.parse_args()


def main() -> int:
    args = _args()
    reference_path = args.reference.resolve()
    binding_path = args.binding.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)
    if not binding_path.is_file():
        raise FileNotFoundError(binding_path)
    actual_sha = _sha256(reference_path)
    if actual_sha != EXPECTED_REFERENCE_SHA256:
        raise RuntimeError(f"reference SHA mismatch: {actual_sha}")

    reference = np.asarray(Image.open(reference_path).convert("RGB"), dtype=np.uint8)
    height, width = reference.shape[:2]
    if (width, height) != (EXPECTED_WIDTH, EXPECTED_HEIGHT):
        raise RuntimeError(f"reference dimensions changed: {(width, height)}")

    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if binding["reference_binding"]["sha256"] != EXPECTED_REFERENCE_SHA256:
        raise RuntimeError("binding/reference SHA mismatch")
    if bool(binding["renderer_binding"]["owns_semantic_truth"]):
        raise RuntimeError("renderer cannot own semantic truth")

    result = build_state_visual_binding(reference, binding)
    report = result["report"]

    before = result["views"]["before"]
    edited = result["views"]["edited"]
    reloaded = result["views"]["reloaded_hero"]
    reverted = result["views"]["reverted_hero"]

    _save_rgb(output / "hero-before.png", before["hero"].rgb)
    _save_rgb(output / "hero-edited.png", edited["hero"].rgb)
    _save_rgb(output / "hero-reloaded.png", reloaded.rgb)
    _save_rgb(output / "hero-reverted.png", reverted.rgb)
    _save_rgb(output / "neighbor-left-edited.png", edited["neighbor-left"].rgb)
    _save_rgb(output / "neighbor-right-edited.png", edited["neighbor-right"].rgb)

    for name in ("hero", "neighbor-left", "neighbor-right"):
        view = edited[name]
        _save_mask(output / f"{name}-target-support-mask.png", view.target_support)
        _save_mask(output / f"{name}-changed-mask.png", view.changed)
        _save_mask(output / f"{name}-state-generated-mask.png", view.state_generated)
        _save_mask(output / f"{name}-observed-mask.png", view.observed)
        _save_mask(output / f"{name}-hypothesized-mask.png", view.hypothesized)

    for name, state in result["world_states"].items():
        (output / f"world-{name}.json").write_text(
            json.dumps(state, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    report["reference"] = {
        "sha256": actual_sha,
        "width": width,
        "height": height,
        "binary_committed": False,
    }
    report["artifacts"] = {
        "hero_before": "hero-before.png",
        "hero_edited": "hero-edited.png",
        "hero_reloaded": "hero-reloaded.png",
        "hero_reverted": "hero-reverted.png",
        "neighbor_left_edited": "neighbor-left-edited.png",
        "neighbor_right_edited": "neighbor-right-edited.png",
    }
    report_path = output / "EXP006-STATE-VISUAL-BINDING-V0.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    hero = report["views"]["hero"]
    left = report["views"]["neighbor-left"]
    right = report["views"]["neighbor-right"]
    print("EXP-006 STATE VISUAL BINDING V0 COMPLETE")
    print(f"Pre-edit hero exact reference: {report['gate_checks']['pre_edit_hero_exact_reference']}")
    print(f"Hero changed pixels: {hero['changed_pixels']}")
    print(f"Changed outside target: {hero['changed_outside_target_support']}")
    print(f"Changed pixels still observed: {hero['changed_pixels_still_observed']}")
    print(f"Outside-target hero exact: {hero['outside_target_exact']}")
    print(f"Reload re-derives identical hero: {report['persistence']['reload_rederives_identical_hero']}")
    print(f"Revert restores exact reference: {report['persistence']['revert_restores_exact_reference_hero']}")
    print(f"Neighbor changed pixels: left={left['changed_pixels']} right={right['changed_pixels']}")
    print(f"Automated gate: {'PASS' if report['automated_gate_passed'] else 'FAIL'}")
    print("Human review of hero-edited and both edited neighbors is required before merge.")
    print(f"Report: {report_path}")
    print("Rank-4 remains sealed and untouched.")
    return 0 if report["automated_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
