#!/usr/bin/env python3
"""Run the predeclared cross-scene oracle evidence-preservation confirmation.

Primary replication set: frozen BlendedMVS scenes 2-10, one first-published
held-out target per scene. Scene 1 was already examined and is excluded from
primary statistics.

Leakage protocol:
- generation materialization excludes held-out target RGB entirely;
- all nine oracle warps, SD2 candidates and B/C compositions must exist first;
- a hash-pinned generation-complete marker is written;
- only then are the nine target RGBs materialized;
- target depth is never materialized/read;
- scoring occurs only after target materialization.

This remains an oracle source-depth/camera diagnostic, not full single-image
RefWorld-0.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import subprocess
from pathlib import Path
from typing import Any

from refworld.datasets.blendedmvs import load_manifest
from refworld.datasets.mvsnet import parse_pair_text

SCENE_ORDERS = tuple(range(2, 11))
FROZEN_SD2_REVISION = "5f74973cbb64c8568780732c17f43eb269d63a0d"
FROZEN_SEED = 42
FROZEN_STEPS = 30
FROZEN_GUIDANCE = 4.0
FROZEN_CONTEXT_RADIUS = 16
FROZEN_MAX_SIDE = 512


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_checked(label: str, command: list[str], *, cwd: Path) -> None:
    print(f"\n== {label} ==", flush=True)
    print(" ".join(command), flush=True)
    completed = subprocess.run(command, cwd=cwd)
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {completed.returncode}")


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize empty values")
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "minimum_worst_case": float(min(values)),
        "maximum_best_case": float(max(values)),
    }


def selected_scene_entries(frozen: dict[str, Any]) -> list[dict[str, Any]]:
    scenes = frozen.get("scenes")
    if not isinstance(scenes, list):
        raise RuntimeError("frozen manifest scenes are missing")
    by_order = {int(item["order"]): item for item in scenes}
    if len(by_order) != len(scenes):
        raise RuntimeError("frozen manifest contains duplicate scene order")
    missing = [order for order in SCENE_ORDERS if order not in by_order]
    if missing:
        raise RuntimeError(f"frozen manifest missing scene orders: {missing}")
    return [by_order[order] for order in SCENE_ORDERS]


def load_generation_record(scene_root: Path, expected_order: int) -> dict[str, Any]:
    manifest_path = scene_root / "cross-scene-generation-inputs.safe.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(value["frozen_scene_order"]) != expected_order:
        raise RuntimeError(f"scene order {expected_order}: generation manifest order mismatch")
    if int(value["target_source_order"]) != 1:
        raise RuntimeError(f"scene order {expected_order}: target source order is not 1")
    if bool(value["target_rgb_materialized"]):
        raise RuntimeError(f"scene order {expected_order}: generation manifest says target RGB was materialized")
    if bool(value["target_depth_materialized"]):
        raise RuntimeError(f"scene order {expected_order}: target depth unexpectedly materialized")
    return value


def validate_generation_marker(marker_path: Path, repo_root: Path) -> list[dict[str, Any]]:
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("stage") != "refworld-cross-scene-all-candidates-before-targets":
        raise RuntimeError("generation-complete marker stage mismatch")
    rows = marker.get("scenes")
    if not isinstance(rows, list) or len(rows) != len(SCENE_ORDERS):
        raise RuntimeError("generation-complete marker scene count mismatch")
    if [int(row["frozen_scene_order"]) for row in rows] != list(SCENE_ORDERS):
        raise RuntimeError("generation-complete marker scene order mismatch")
    for row in rows:
        for key in ("oracle_manifest", "candidate_manifest", "composition_manifest"):
            path = repo_root / str(row[key]["path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            actual = sha256_file(path)
            if actual != str(row[key]["sha256"]):
                raise RuntimeError(f"generation-complete marker hash mismatch: {path}")
    return rows


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    python = repo_root / ".venv-refworld" / "Scripts" / "python.exe"
    if not python.is_file():
        raise RuntimeError(f"RefWorld venv not found: {python}")

    frozen_manifest_path = repo_root / "datasets" / "blendedmvs-bootstrap-v0.json"
    frozen = load_manifest(frozen_manifest_path)
    selected = selected_scene_entries(frozen)
    data_root = repo_root / "private-data" / "blendedmvs-bootstrap"
    experiment_root = repo_root / "outputs" / "calibrated" / "blendedmvs" / "oracle-cross-scene-2-10"
    experiment_root.mkdir(parents=True, exist_ok=True)
    generation_marker = experiment_root / "ALL-CANDIDATES-BEFORE-TARGETS.safe.json"
    aggregate_path = experiment_root / "CROSS-SCENE-REPLICATION.json"

    print("RefWorld cross-scene calibrated oracle confirmation", flush=True)
    print(f"Primary scenes: {[int(item['order']) for item in selected]} (scene 1 excluded as discovery)", flush=True)
    print("Per-scene target rule: first source from first pair.txt record", flush=True)
    print(f"SD2 revision: {FROZEN_SD2_REVISION}", flush=True)
    print("SD2 config: seed=42 steps=30 guidance=4.0 context=16 max-side=512", flush=True)
    print("Geometry: ORACLE anchor depth/camera + published target camera", flush=True)
    print("Seal: target RGB absent until ALL nine candidate compositions are hash-pinned", flush=True)

    run_checked(
        "Installing/verifying frozen cross-scene dependencies",
        [str(python), "-m", "pip", "install", "-e", ".[dataset,repaint-sd2,dev]"],
        cwd=repo_root,
    )
    run_checked(
        "Compiling cross-scene scripts before network/model work",
        [
            str(python),
            "-m",
            "py_compile",
            "scripts/materialize-blendedmvs-cross-scene.py",
            "scripts/run-windows-blendedmvs-oracle-cross-scene.py",
        ],
        cwd=repo_root,
    )
    run_checked(
        "Running cross-scene preflight contract tests",
        [
            str(python),
            "-m",
            "pytest",
            "-q",
            "tests/test_remote_zip.py",
            "tests/test_pfm.py",
            "tests/test_mvsnet_dataset.py",
            "tests/test_blendedmvs_oracle_selection.py",
        ],
        cwd=repo_root,
    )

    generated_rows: list[dict[str, Any]]
    if generation_marker.is_file():
        print("\nExisting generation-complete marker found; validating hashes for safe resume...", flush=True)
        generated_rows = validate_generation_marker(generation_marker, repo_root)
        print("Generation marker valid. No candidate regeneration required.", flush=True)
    else:
        run_checked(
            "PHASE 1A: materializing generation-only inputs for scenes 2-10 (NO target RGB)",
            [
                str(python),
                "scripts/materialize-blendedmvs-cross-scene.py",
                "--phase",
                "generation",
                "--scene-orders",
                ",".join(str(value) for value in SCENE_ORDERS),
            ],
            cwd=repo_root,
        )

        generated_rows = []
        print("\n========== PHASE 1B: GENERATE ALL NINE SCENE CANDIDATES; TARGET RGB ABSENT ==========", flush=True)
        for entry in selected:
            scene_order = int(entry["order"])
            scene_id = str(entry["id"])
            scene_root = data_root / scene_id
            generation = load_generation_record(scene_root, scene_order)
            anchor_id = int(generation["anchor_view_id"])
            target_id = int(generation["target_view_id"])
            target_rgb = scene_root / "blended_images" / f"{target_id:08d}.jpg"
            if target_rgb.exists():
                raise RuntimeError(f"scene {scene_order}: target RGB exists before candidate generation: {target_rgb}")

            scene_output = experiment_root / f"scene-{scene_order:02d}-{scene_id}-target-{target_id:08d}"
            oracle_output = scene_output / "warp"
            candidate_output = scene_output / "sd2"
            composition = candidate_output / "composition"

            run_checked(
                f"Scene {scene_order}/10: oracle warp anchor {anchor_id} -> first target {target_id}",
                [
                    str(python),
                    "-m",
                    "refworld.runners.blendedmvs_oracle_pair",
                    "--scene-root",
                    str(scene_root),
                    "--output",
                    str(oracle_output),
                    "--held-out-rank",
                    "1",
                ],
                cwd=repo_root,
            )
            oracle_manifest = oracle_output / "oracle-pair.safe.json"
            oracle = json.loads(oracle_manifest.read_text(encoding="utf-8"))
            if int(oracle["selection"]["anchor_view_id"]) != anchor_id:
                raise RuntimeError(f"scene {scene_order}: oracle anchor mismatch")
            if int(oracle["selection"]["target_view_id"]) != target_id:
                raise RuntimeError(f"scene {scene_order}: oracle target mismatch")
            if int(oracle["selection"]["held_out_source_order"]) != 1:
                raise RuntimeError(f"scene {scene_order}: oracle source order mismatch")
            if bool(oracle["method_inputs"]["target_rgb_read"]):
                raise RuntimeError(f"scene {scene_order}: oracle stage unexpectedly read target RGB")
            if target_rgb.exists():
                raise RuntimeError(f"scene {scene_order}: target RGB appeared during oracle generation")
            warp_view = oracle_output / str(oracle["result"]["view_directory"])

            run_checked(
                f"Scene {scene_order}/10: frozen SD2 candidate",
                [
                    str(python),
                    "-m",
                    "refworld.runners.sd2_inpaint_candidate",
                    "--warp-view",
                    str(warp_view),
                    "--output",
                    str(candidate_output),
                    "--seed",
                    str(FROZEN_SEED),
                    "--steps",
                    str(FROZEN_STEPS),
                    "--guidance-scale",
                    str(FROZEN_GUIDANCE),
                    "--context-radius",
                    str(FROZEN_CONTEXT_RADIUS),
                    "--max-side",
                    str(FROZEN_MAX_SIDE),
                    "--revision",
                    FROZEN_SD2_REVISION,
                ],
                cwd=repo_root,
            )
            candidate_manifest = candidate_output / "sd2-inpaint.safe.json"
            candidate = json.loads(candidate_manifest.read_text(encoding="utf-8"))
            if str(candidate["backend"]["resolved_revision"]) != FROZEN_SD2_REVISION:
                raise RuntimeError(f"scene {scene_order}: SD2 revision mismatch")
            if not bool(candidate["backend"].get("revision_was_explicitly_pinned")):
                raise RuntimeError(f"scene {scene_order}: SD2 revision not explicitly pinned")
            if bool(candidate["input"]["held_out_evaluation_image_used"]):
                raise RuntimeError(f"scene {scene_order}: SD2 candidate used held-out evaluation image")
            if target_rgb.exists():
                raise RuntimeError(f"scene {scene_order}: target RGB appeared during candidate generation")

            run_checked(
                f"Scene {scene_order}/10: evidence-preserving B/C composition",
                [
                    str(python),
                    "-m",
                    "refworld.runners.compose_candidate",
                    "--warp-view",
                    str(warp_view),
                    "--candidate",
                    str(candidate_output / "candidate.png"),
                    "--valid-mask-npy",
                    str(candidate_output / "repaint-valid-mask.npy"),
                    "--output",
                    str(composition),
                    "--backend",
                    "sd2-community-stable-diffusion-2-inpainting-openrailpp",
                    "--seed",
                    str(FROZEN_SEED),
                    "--backend-run-id",
                    FROZEN_SD2_REVISION,
                ],
                cwd=repo_root,
            )
            composition_manifest = composition / "compose.safe.json"
            if not composition_manifest.is_file():
                raise FileNotFoundError(composition_manifest)
            if target_rgb.exists():
                raise RuntimeError(f"scene {scene_order}: target RGB appeared before all candidates were complete")

            generated_rows.append(
                {
                    "frozen_scene_order": scene_order,
                    "scene_id": scene_id,
                    "anchor_view_id": anchor_id,
                    "target_view_id": target_id,
                    "observed_fraction": float(oracle["result"]["observed_fraction"]),
                    "unresolved_fraction": float(oracle["result"]["unresolved_fraction"]),
                    "oracle_manifest": {
                        "path": oracle_manifest.relative_to(repo_root).as_posix(),
                        "sha256": sha256_file(oracle_manifest),
                    },
                    "candidate_manifest": {
                        "path": candidate_manifest.relative_to(repo_root).as_posix(),
                        "sha256": sha256_file(candidate_manifest),
                    },
                    "composition_manifest": {
                        "path": composition_manifest.relative_to(repo_root).as_posix(),
                        "sha256": sha256_file(composition_manifest),
                    },
                    "oracle_output": oracle_output.relative_to(repo_root).as_posix(),
                    "composition_output": composition.relative_to(repo_root).as_posix(),
                }
            )

        for row in generated_rows:
            scene_root = data_root / str(row["scene_id"])
            target_rgb = scene_root / "blended_images" / f"{int(row['target_view_id']):08d}.jpg"
            if target_rgb.exists():
                raise RuntimeError(f"target RGB exists before generation-complete marker: {target_rgb}")

        generation_marker.write_text(
            json.dumps(
                {
                    "version": "0.1",
                    "stage": "refworld-cross-scene-all-candidates-before-targets",
                    "scope": "frozen BlendedMVS scenes 2-10; oracle source geometry diagnostic",
                    "scene_orders": list(SCENE_ORDERS),
                    "scene_count": len(generated_rows),
                    "sd2_revision": FROZEN_SD2_REVISION,
                    "seed": FROZEN_SEED,
                    "steps": FROZEN_STEPS,
                    "guidance_scale": FROZEN_GUIDANCE,
                    "context_radius_original_pixels": FROZEN_CONTEXT_RADIUS,
                    "max_model_side": FROZEN_MAX_SIDE,
                    "target_rgb_present_when_marker_written": False,
                    "scenes": generated_rows,
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        generated_rows = validate_generation_marker(generation_marker, repo_root)
        print(f"\nSEALED GENERATION COMPLETE MARKER: {generation_marker}", flush=True)

    target_batch_manifest = data_root / "cross-scene-targets-materialization.safe.json"
    if target_batch_manifest.is_file():
        print("\nTarget-materialization batch manifest already exists; validating target RGBs for resume...", flush=True)
        target_batch = json.loads(target_batch_manifest.read_text(encoding="utf-8"))
        if [int(value) for value in target_batch["scene_orders"]] != list(SCENE_ORDERS):
            raise RuntimeError("target materialization scene-order mismatch")
        for row in generated_rows:
            target_rgb = data_root / str(row["scene_id"]) / "blended_images" / f"{int(row['target_view_id']):08d}.jpg"
            if not target_rgb.is_file():
                raise FileNotFoundError(target_rgb)
    else:
        run_checked(
            "PHASE 2A: all candidates are sealed; NOW materialize held-out target RGBs",
            [
                str(python),
                "scripts/materialize-blendedmvs-cross-scene.py",
                "--phase",
                "targets",
                "--scene-orders",
                ",".join(str(value) for value in SCENE_ORDERS),
            ],
            cwd=repo_root,
        )

    print("\n========== PHASE 2B: SCORE THE NINE PREVIOUSLY UNSEEN SCENES ==========", flush=True)
    rows: list[dict[str, Any]] = []
    for generated in generated_rows:
        scene_order = int(generated["frozen_scene_order"])
        scene_id = str(generated["scene_id"])
        target_id = int(generated["target_view_id"])
        scene_root = data_root / scene_id
        oracle_output = repo_root / str(generated["oracle_output"])
        composition = repo_root / str(generated["composition_output"])
        scene_output = oracle_output.parent
        score_path = scene_output / "calibrated-score.json"

        target_manifest_path = scene_root / "cross-scene-target-rgb.safe.json"
        if not target_manifest_path.is_file():
            raise FileNotFoundError(target_manifest_path)
        target_manifest = json.loads(target_manifest_path.read_text(encoding="utf-8"))
        if int(target_manifest["target_view_id"]) != target_id:
            raise RuntimeError(f"scene {scene_order}: target materialization selection mismatch")
        if bool(target_manifest["target_depth_materialized"]):
            raise RuntimeError(f"scene {scene_order}: target depth unexpectedly materialized")

        run_checked(
            f"Scene {scene_order}/10: held-out scoring target {target_id}",
            [
                str(python),
                "-m",
                "refworld.runners.score_blendedmvs_pair",
                "--scene-root",
                str(scene_root),
                "--oracle-output",
                str(oracle_output),
                "--composition",
                str(composition),
                "--output",
                str(score_path),
            ],
            cwd=repo_root,
        )
        score = json.loads(score_path.read_text(encoding="utf-8"))
        if int(score["selection"]["target_view_id"]) != target_id:
            raise RuntimeError(f"scene {scene_order}: score target mismatch")
        if bool(score["evaluation_inputs"]["target_depth_read"]):
            raise RuntimeError(f"scene {scene_order}: scorer unexpectedly read target depth")

        full = score["metrics"]["full_frame"]
        observed = score["metrics"]["observed_support"]
        contrasts = score["contrasts"]
        rows.append(
            {
                "frozen_scene_order": scene_order,
                "scene_id": scene_id,
                "anchor_view_id": int(generated["anchor_view_id"]),
                "target_view_id": target_id,
                "observed_fraction": float(generated["observed_fraction"]),
                "unresolved_fraction": float(generated["unresolved_fraction"]),
                "B_psnr_full_db": float(full["B_unrestricted"]["psnr"]),
                "C_psnr_full_db": float(full["C_evidence_preserved"]["psnr"]),
                "C_minus_B_psnr_full_db": float(contrasts["C_minus_B_psnr_full_db"]),
                "B_psnr_observed_db": float(observed["B_unrestricted"]["psnr"]),
                "C_psnr_observed_db": float(observed["C_evidence_preserved"]["psnr"]),
                "C_minus_B_psnr_observed_db": float(contrasts["C_minus_B_psnr_observed_db"]),
                "score_path": score_path.relative_to(repo_root).as_posix(),
            }
        )

    if [int(row["frozen_scene_order"]) for row in rows] != list(SCENE_ORDERS):
        raise RuntimeError("final scored scene order does not match frozen primary set")

    full_deltas = [float(row["C_minus_B_psnr_full_db"]) for row in rows]
    observed_deltas = [float(row["C_minus_B_psnr_observed_db"]) for row in rows]
    full_wins = sum(value > 0.0 for value in full_deltas)
    observed_wins = sum(value > 0.0 for value in observed_deltas)

    aggregate = {
        "version": "0.1",
        "stage": "refworld-blendedmvs-oracle-cross-scene-confirmation",
        "scope": "oracle-source-depth/camera diagnostic; not full single-image RefWorld-0",
        "primary_replication_unit": "scene",
        "discovery_scene_excluded": 1,
        "protocol": {
            "frozen_scene_orders": list(SCENE_ORDERS),
            "scene_count": len(rows),
            "per_scene_view_rule": "first source of first pair.txt record",
            "sd2_revision": FROZEN_SD2_REVISION,
            "seed": FROZEN_SEED,
            "steps": FROZEN_STEPS,
            "guidance_scale": FROZEN_GUIDANCE,
            "context_radius_original_pixels": FROZEN_CONTEXT_RADIUS,
            "max_model_side": FROZEN_MAX_SIDE,
            "all_candidates_hash_pinned_before_target_rgb_materialization": True,
            "target_depth_materialized_or_read": False,
            "per_scene_hyperparameter_tuning": False,
        },
        "per_scene": rows,
        "aggregate": {
            "full_frame_C_minus_B_psnr_db": {
                **summarize(full_deltas),
                "positive_win_count": int(full_wins),
                "nonpositive_count": int(len(full_deltas) - full_wins),
            },
            "observed_support_C_minus_B_psnr_db": {
                **summarize(observed_deltas),
                "positive_win_count": int(observed_wins),
                "nonpositive_count": int(len(observed_deltas) - observed_wins),
            },
        },
        "generation_complete_marker": generation_marker.relative_to(repo_root).as_posix(),
    }
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("\nCROSS-SCENE CALIBRATED ORACLE CONFIRMATION COMPLETE", flush=True)
    print("scene  target   C-B full dB   C-B observed dB", flush=True)
    for row in rows:
        print(
            f"{int(row['frozen_scene_order']):>5} {int(row['target_view_id']):>7}"
            f" {float(row['C_minus_B_psnr_full_db']):>+13.4f}"
            f" {float(row['C_minus_B_psnr_observed_db']):>+17.4f}",
            flush=True,
        )

    full_summary = aggregate["aggregate"]["full_frame_C_minus_B_psnr_db"]
    observed_summary = aggregate["aggregate"]["observed_support_C_minus_B_psnr_db"]
    print("\nPRIMARY CROSS-SCENE RESULT — frozen scenes 2-10", flush=True)
    print("Full-frame C-B PSNR:", flush=True)
    print(f"  wins:   {full_wins}/{len(rows)}", flush=True)
    print(f"  mean:   {full_summary['mean']:+.4f} dB", flush=True)
    print(f"  median: {full_summary['median']:+.4f} dB", flush=True)
    print(f"  worst:  {full_summary['minimum_worst_case']:+.4f} dB", flush=True)
    print("Observed-support C-B PSNR:", flush=True)
    print(f"  wins:   {observed_wins}/{len(rows)}", flush=True)
    print(f"  mean:   {observed_summary['mean']:+.4f} dB", flush=True)
    print(f"  median: {observed_summary['median']:+.4f} dB", flush=True)
    print(f"  worst:  {observed_summary['minimum_worst_case']:+.4f} dB", flush=True)
    print(f"Aggregate: {aggregate_path}", flush=True)
    print("Scope: oracle source-depth/camera diagnostic; not full single-image RefWorld-0.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
