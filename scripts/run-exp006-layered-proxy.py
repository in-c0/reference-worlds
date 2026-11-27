#!/usr/bin/env python3
"""Run EXP-006 Collaborative Futures authored layered proxy R1.

This consumes only the owner reference image and the bound semantic manifest. It
never touches BlendedMVS rank-4 evidence, never reads target depth, and makes no
metric-reconstruction claim.

R1 keeps the R0 camera/layer geometry frozen and changes only display/provenance
separation: hypothesized fallback pixels may be shown for continuity but remain
non-observed in the provenance masks.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from refworld.exp006_layered_proxy import (
    AUTHORED_HFOV_DEGREES,
    NEIGHBOR_TRANSLATION,
    R1_ALPHA_AFFECTED_THRESHOLD,
    R1_ALPHA_OBSERVED_THRESHOLD,
    R1_FEATHER_RADIUS_PX,
    authored_camera,
    render_triplet,
)
from refworld.semantic_handoff import (
    apply_entity_state_patch,
    semantic_drift_report,
    snapshot_roundtrip,
)


EXPECTED_REFERENCE_SHA256 = "4ee7a137e577378a02600ac8a32dc89a7c8409120273622227ad972cc5aff61a"
EXPECTED_WIDTH = 1672
EXPECTED_HEIGHT = 941
MIN_NEIGHBOR_OBSERVED_FRACTION = 0.90
RENDERER_ID = "refworld.exp006.layered-proxy-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EXP-006 authored layered proxy R1")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--binding",
        type=Path,
        default=Path("examples/exp006_collaborative_futures_binding.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/exp006/collaborative-futures-layered-proxy-v1"),
    )
    return parser.parse_args()


def _save_rgb(path: Path, array: np.ndarray) -> None:
    Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB").save(path)


def _save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray((np.asarray(mask, dtype=bool).astype(np.uint8) * 255), mode="L").save(path)


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
    reference_binding = binding["reference_binding"]
    if reference_binding["status"] != "bound":
        raise RuntimeError("EXP-006 reference binding is not frozen as bound")
    if reference_binding["sha256"] != EXPECTED_REFERENCE_SHA256:
        raise RuntimeError("binding/reference SHA mismatch")
    if bool(binding["renderer_binding"]["owns_semantic_truth"]):
        raise RuntimeError("renderer is forbidden from owning semantic truth")
    if bool(binding["renderer_binding"].get("metric_reconstruction_claim")):
        raise RuntimeError("layered proxy may not claim metric reconstruction")

    reference = np.asarray(Image.open(reference_path).convert("RGB"), dtype=np.uint8)
    height, width = reference.shape[:2]
    if (width, height) != (EXPECTED_WIDTH, EXPECTED_HEIGHT):
        raise RuntimeError(f"reference dimensions changed: {(width, height)}")

    camera = authored_camera(width, height)
    triplet = render_triplet(reference)
    view_reports: dict[str, dict] = {}
    for name, (rgb, observed, shifts) in triplet.items():
        rgb_path = output / f"{name}.png"
        observed_mask_path = output / f"{name}-observed-mask.png"
        hypothesized_mask_path = output / f"{name}-hypothesized-mask.png"
        _save_rgb(rgb_path, rgb)
        _save_mask(observed_mask_path, observed)
        _save_mask(hypothesized_mask_path, ~observed)
        exact = bool(np.array_equal(rgb, reference)) if name == "hero" else False
        observed_fraction = float(np.mean(observed))
        view_reports[name] = {
            "camera_tx": 0.0
            if name == "hero"
            else (-NEIGHBOR_TRANSLATION if name == "neighbor-left" else NEIGHBOR_TRANSLATION),
            "observed_fraction": observed_fraction,
            "hypothesized_fraction": float(1.0 - observed_fraction),
            "exact_reference_match": exact,
            "rgb": rgb_path.relative_to(output).as_posix(),
            "observed_mask": observed_mask_path.relative_to(output).as_posix(),
            "hypothesized_mask": hypothesized_mask_path.relative_to(output).as_posix(),
            "layer_pixel_shifts": shifts,
        }

    before_world = binding["world_state"]
    edit = binding["trace_plan"]["bounded_edit"]
    edited_world = apply_entity_state_patch(
        before_world,
        entity_id=edit["entity_id"],
        patch=edit["state_patch"],
        actor="owner-exp006-trace",
    )
    reloaded_world = snapshot_roundtrip(edited_world)
    drift = semantic_drift_report(
        before_world,
        reloaded_world,
        target_entity_id=edit["entity_id"],
    )

    runtime_world = copy.deepcopy(reloaded_world)
    anchor = runtime_world["anchors"][0]
    anchor["camera_status"] = "authored-proxy-frozen"
    anchor["camera"] = {
        "intrinsics": camera["intrinsics"],
        "extrinsics": camera["extrinsics"],
        "convention": camera["convention"],
    }
    runtime_world_path = output / "runtime-world-state.json"
    runtime_world_path.write_text(
        json.dumps(runtime_world, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    automated_gate_passed = bool(
        view_reports["hero"]["exact_reference_match"]
        and view_reports["neighbor-left"]["observed_fraction"] >= MIN_NEIGHBOR_OBSERVED_FRACTION
        and view_reports["neighbor-right"]["observed_fraction"] >= MIN_NEIGHBOR_OBSERVED_FRACTION
        and drift["stable_id_set"]
        and drift["target_changed"]
        and drift["collateral_semantic_drift_count"] == 0
    )

    report = {
        "version": "0.2",
        "stage": "refworld-exp006-layered-proxy-v1",
        "renderer_id": RENDERER_ID,
        "reference": {
            "sha256": actual_sha,
            "width": width,
            "height": height,
            "binary_committed": False,
        },
        "authored_camera": camera,
        "frozen_method": {
            "kind": "fronto-parallel-authored-layered-proxy",
            "r1_change_scope": "display/provenance separation only; R0 camera, translation, layer polygons and depths unchanged",
            "neighbor_translation": NEIGHBOR_TRANSLATION,
            "minimum_neighbor_observed_fraction": MIN_NEIGHBOR_OBSERVED_FRACTION,
            "hypothesized_display_fill_used": True,
            "hypothesized_display_fill_source": "edge-padded full-reference background proxy",
            "hypothesized_display_fill_is_observed": False,
            "unknown_support_inpainted": False,
            "feather_radius_px": R1_FEATHER_RADIUS_PX,
            "alpha_affected_threshold": R1_ALPHA_AFFECTED_THRESHOLD,
            "alpha_observed_threshold": R1_ALPHA_OBSERVED_THRESHOLD,
            "metric_reconstruction_claim": False,
        },
        "views": view_reports,
        "semantic_trace": {
            "bounded_edit": edit,
            "snapshot_reload_completed": True,
            "drift": drift,
            "runtime_world_state": runtime_world_path.relative_to(output).as_posix(),
        },
        "evidence_boundary": {
            "fresh_benchmark_evidence_consumed": False,
            "rank4_touched": False,
            "target_depth_read": False,
            "renderer_owns_semantic_truth": False,
        },
        "automated_gate_passed": automated_gate_passed,
        "human_visual_review_required": True,
        "r0_visual_failure_addressed": "internal black disocclusion seams from carved background ownership",
    }
    report_path = output / "EXP006-LAYERED-PROXY-V1.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("EXP-006 LAYERED PROXY V1 COMPLETE")
    print(f"Hero exact reference match: {view_reports['hero']['exact_reference_match']}")
    print(
        "Neighbor observed fractions: "
        f"left={view_reports['neighbor-left']['observed_fraction']:.4f} "
        f"right={view_reports['neighbor-right']['observed_fraction']:.4f}"
    )
    print(
        "Neighbor hypothesized fractions: "
        f"left={view_reports['neighbor-left']['hypothesized_fraction']:.4f} "
        f"right={view_reports['neighbor-right']['hypothesized_fraction']:.4f}"
    )
    print(f"Stable IDs after edit/reload: {drift['stable_id_set']}")
    print(f"Collateral semantic drift: {drift['collateral_semantic_drift_count']}")
    print(f"Automated gate: {'PASS' if automated_gate_passed else 'FAIL'}")
    print("Human visual review of the R1 neighboring views is still required.")
    print(f"Report: {report_path}")
    print("Rank-4 remains sealed and untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
