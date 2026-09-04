#!/usr/bin/env python3
"""Run the predeclared LaMa backend-independence replication on Windows.

Primary set: frozen BlendedMVS scenes 2-10, second published source from each
first pair.txt record. All rank-2 target RGBs remain absent until every LaMa B/C
composition is generated and hash-pinned. Target depth is never fetched/read.
This is an oracle source-depth/camera diagnostic, not full single-image RefWorld.
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
from refworld.runners.lama_inpaint_candidate import MODEL_SHA256, MODEL_SIZE_BYTES, MODEL_URL

SCENE_ORDERS = tuple(range(2, 11))
TARGET_SOURCE_ORDER = 2
CONTEXT_RADIUS = 16


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_checked(label: str, command: list[str], *, cwd: Path) -> None:
    print(f"\n== {label} ==", flush=True)
    print(" ".join(command), flush=True)
    result = subprocess.run(command, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "minimum_worst_case": float(min(values)),
        "maximum_best_case": float(max(values)),
    }


def selected_scenes(frozen: dict[str, Any]) -> list[dict[str, Any]]:
    scenes = frozen.get("scenes")
    if not isinstance(scenes, list):
        raise RuntimeError("frozen manifest scenes missing")
    by_order = {int(item["order"]): item for item in scenes}
    return [by_order[order] for order in SCENE_ORDERS]


def validate_marker(marker_path: Path, repo_root: Path) -> list[dict[str, Any]]:
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("stage") != "refworld-lama-rank2-all-candidates-before-targets":
        raise RuntimeError("LaMa generation marker stage mismatch")
    if marker.get("model_sha256") != MODEL_SHA256:
        raise RuntimeError("LaMa generation marker model SHA mismatch")
    rows = marker.get("scenes")
    if not isinstance(rows, list) or [int(row["frozen_scene_order"]) for row in rows] != list(SCENE_ORDERS):
        raise RuntimeError("LaMa generation marker scene order mismatch")
    for row in rows:
        for key in ("oracle_manifest", "candidate_manifest", "composition_manifest"):
            artifact = repo_root / str(row[key]["path"])
            if not artifact.is_file() or sha256_file(artifact) != str(row[key]["sha256"]):
                raise RuntimeError(f"LaMa generation marker artifact mismatch: {artifact}")
    return rows


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    python = repo_root / ".venv-refworld" / "Scripts" / "python.exe"
    if not python.is_file():
        raise RuntimeError(f"RefWorld venv not found: {python}")

    frozen = load_manifest(repo_root / "datasets" / "blendedmvs-bootstrap-v0.json")
    selected = selected_scenes(frozen)
    data_root = repo_root / "private-data" / "blendedmvs-bootstrap"
    experiment_root = repo_root / "outputs" / "calibrated" / "blendedmvs" / "oracle-lama-backend-independence-rank2"
    experiment_root.mkdir(parents=True, exist_ok=True)
    marker_path = experiment_root / "ALL-LAMA-CANDIDATES-BEFORE-TARGETS.safe.json"
    aggregate_path = experiment_root / "LAMA-BACKEND-INDEPENDENCE.json"

    print("RefWorld LaMa backend-independence confirmation", flush=True)
    print("Primary scenes: frozen 2-10; discovery scene 1 excluded", flush=True)
    print("Target rule: source rank 2 from first pair.txt record", flush=True)
    print(f"Backend: Big-LaMa TorchScript | SHA256 {MODEL_SHA256}", flush=True)
    print(f"Model bytes: {MODEL_SIZE_BYTES} | source {MODEL_URL}", flush=True)
    print(f"Context: {CONTEXT_RADIUS}px elliptical dilation", flush=True)
    print("Seal: rank-2 target RGB absent until ALL nine B/C compositions are hash-pinned", flush=True)

    run_checked(
        "Installing/verifying dependencies",
        [str(python), "-m", "pip", "install", "-e", ".[dataset,dev]"],
        cwd=repo_root,
    )
    run_checked(
        "Compiling LaMa backend-independence scripts",
        [str(python), "-m", "py_compile", "src/refworld/runners/lama_inpaint_candidate.py", "scripts/materialize-blendedmvs-lama-rank2.py", "scripts/run-windows-blendedmvs-lama-backend-independence.py"],
        cwd=repo_root,
    )
    run_checked(
        "Running existing camera/archive/protocol contract tests",
        [str(python), "-m", "pytest", "-q", "tests/test_remote_zip.py", "tests/test_pfm.py", "tests/test_mvsnet_dataset.py", "tests/test_blendedmvs_oracle_selection.py"],
        cwd=repo_root,
    )

    generated_rows: list[dict[str, Any]]
    if marker_path.is_file():
        print("\nExisting sealed LaMa generation marker found; validating for resume...", flush=True)
        generated_rows = validate_marker(marker_path, repo_root)
        print("LaMa generation marker valid.", flush=True)
    else:
        run_checked(
            "PHASE 1A: materialize rank-2 generation inputs (NO target RGB)",
            [str(python), "scripts/materialize-blendedmvs-lama-rank2.py", "--phase", "generation"],
            cwd=repo_root,
        )
        generated_rows = []
        print("\n========== PHASE 1B: GENERATE ALL NINE LAMA CANDIDATES; RANK-2 TARGET RGB ABSENT ==========", flush=True)

        for entry in selected:
            scene_order = int(entry["order"])
            scene_id = str(entry["id"])
            scene_root = data_root / scene_id
            generation_manifest = scene_root / "lama-rank2-generation-inputs.safe.json"
            generation = json.loads(generation_manifest.read_text(encoding="utf-8"))
            if int(generation["target_source_order"]) != TARGET_SOURCE_ORDER:
                raise RuntimeError(f"scene {scene_order}: rank-2 generation selection mismatch")
            anchor_id = int(generation["anchor_view_id"])
            target_id = int(generation["target_view_id"])
            target_rgb = scene_root / "blended_images" / f"{target_id:08d}.jpg"
            if target_rgb.exists():
                raise RuntimeError(f"scene {scene_order}: rank-2 target RGB exists before generation: {target_rgb}")

            scene_output = experiment_root / f"scene-{scene_order:02d}-{scene_id}-target-{target_id:08d}"
            oracle_output = scene_output / "warp"
            candidate_output = scene_output / "lama"
            composition = candidate_output / "composition"

            run_checked(
                f"Scene {scene_order}/10: oracle warp to rank-2 target {target_id}",
                [str(python), "-m", "refworld.runners.blendedmvs_oracle_pair", "--scene-root", str(scene_root), "--output", str(oracle_output), "--held-out-rank", str(TARGET_SOURCE_ORDER)],
                cwd=repo_root,
            )
            oracle_manifest = oracle_output / "oracle-pair.safe.json"
            oracle = json.loads(oracle_manifest.read_text(encoding="utf-8"))
            if int(oracle["selection"]["held_out_source_order"]) != TARGET_SOURCE_ORDER or int(oracle["selection"]["target_view_id"]) != target_id:
                raise RuntimeError(f"scene {scene_order}: oracle rank-2 selection mismatch")
            if bool(oracle["method_inputs"]["target_rgb_read"]) or target_rgb.exists():
                raise RuntimeError(f"scene {scene_order}: target RGB leakage during oracle stage")
            warp_view = oracle_output / str(oracle["result"]["view_directory"])

            run_checked(
                f"Scene {scene_order}/10: pinned Big-LaMa candidate",
                [str(python), "-m", "refworld.runners.lama_inpaint_candidate", "--warp-view", str(warp_view), "--output", str(candidate_output), "--context-radius", str(CONTEXT_RADIUS)],
                cwd=repo_root,
            )
            candidate_manifest = candidate_output / "lama-inpaint.safe.json"
            candidate = json.loads(candidate_manifest.read_text(encoding="utf-8"))
            if candidate["backend"]["model_sha256"] != MODEL_SHA256 or candidate["backend"]["model_size_bytes"] != MODEL_SIZE_BYTES:
                raise RuntimeError(f"scene {scene_order}: LaMa model pin mismatch")
            if bool(candidate["input"]["held_out_evaluation_image_used"]) or target_rgb.exists():
                raise RuntimeError(f"scene {scene_order}: target RGB leakage during LaMa generation")

            run_checked(
                f"Scene {scene_order}/10: evidence-preserving B/C composition",
                [str(python), "-m", "refworld.runners.compose_candidate", "--warp-view", str(warp_view), "--candidate", str(candidate_output / "candidate.png"), "--valid-mask-npy", str(candidate_output / "repaint-valid-mask.npy"), "--output", str(composition), "--backend", "big-lama-fourier-convolution-apache2", "--backend-run-id", MODEL_SHA256],
                cwd=repo_root,
            )
            composition_manifest = composition / "compose.safe.json"
            if target_rgb.exists():
                raise RuntimeError(f"scene {scene_order}: target RGB appeared before all LaMa candidates completed")

            generated_rows.append({
                "frozen_scene_order": scene_order,
                "scene_id": scene_id,
                "anchor_view_id": anchor_id,
                "target_view_id": target_id,
                "observed_fraction": float(oracle["result"]["observed_fraction"]),
                "unresolved_fraction": float(oracle["result"]["unresolved_fraction"]),
                "oracle_output": oracle_output.relative_to(repo_root).as_posix(),
                "composition_output": composition.relative_to(repo_root).as_posix(),
                "oracle_manifest": {"path": oracle_manifest.relative_to(repo_root).as_posix(), "sha256": sha256_file(oracle_manifest)},
                "candidate_manifest": {"path": candidate_manifest.relative_to(repo_root).as_posix(), "sha256": sha256_file(candidate_manifest)},
                "composition_manifest": {"path": composition_manifest.relative_to(repo_root).as_posix(), "sha256": sha256_file(composition_manifest)},
            })

        for row in generated_rows:
            target = data_root / str(row["scene_id"]) / "blended_images" / f"{int(row['target_view_id']):08d}.jpg"
            if target.exists():
                raise RuntimeError(f"rank-2 target RGB exists before sealed marker: {target}")
        marker_path.write_text(json.dumps({
            "version": "0.1",
            "stage": "refworld-lama-rank2-all-candidates-before-targets",
            "scene_orders": list(SCENE_ORDERS),
            "target_source_order": TARGET_SOURCE_ORDER,
            "model_url": MODEL_URL,
            "model_size_bytes": MODEL_SIZE_BYTES,
            "model_sha256": MODEL_SHA256,
            "context_radius_original_pixels": CONTEXT_RADIUS,
            "target_rgb_present_when_marker_written": False,
            "scenes": generated_rows,
        }, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        generated_rows = validate_marker(marker_path, repo_root)
        print(f"\nSEALED LAMA GENERATION MARKER: {marker_path}", flush=True)

    targets_batch = data_root / "lama-rank2-targets-materialization.safe.json"
    if targets_batch.is_file():
        print("\nRank-2 targets already materialized after seal; validating resume inputs...", flush=True)
        for row in generated_rows:
            target = data_root / str(row["scene_id"]) / "blended_images" / f"{int(row['target_view_id']):08d}.jpg"
            if not target.is_file():
                raise FileNotFoundError(target)
    else:
        run_checked(
            "PHASE 2A: all LaMa candidates sealed; NOW materialize rank-2 target RGBs",
            [str(python), "scripts/materialize-blendedmvs-lama-rank2.py", "--phase", "targets"],
            cwd=repo_root,
        )

    print("\n========== PHASE 2B: SCORE NINE PREVIOUSLY UNSEEN RANK-2 TARGETS ==========", flush=True)
    rows: list[dict[str, Any]] = []
    for generated in generated_rows:
        scene_order = int(generated["frozen_scene_order"])
        scene_id = str(generated["scene_id"])
        target_id = int(generated["target_view_id"])
        scene_root = data_root / scene_id
        target_manifest = scene_root / "lama-rank2-target-rgb.safe.json"
        target_meta = json.loads(target_manifest.read_text(encoding="utf-8"))
        if int(target_meta["target_view_id"]) != target_id or bool(target_meta["target_depth_materialized"]):
            raise RuntimeError(f"scene {scene_order}: target materialization contract mismatch")
        oracle_output = repo_root / str(generated["oracle_output"])
        composition = repo_root / str(generated["composition_output"])
        score_path = oracle_output.parent / "calibrated-score.json"
        run_checked(
            f"Scene {scene_order}/10: held-out scoring rank-2 target {target_id}",
            [str(python), "-m", "refworld.runners.score_blendedmvs_pair", "--scene-root", str(scene_root), "--oracle-output", str(oracle_output), "--composition", str(composition), "--output", str(score_path)],
            cwd=repo_root,
        )
        score = json.loads(score_path.read_text(encoding="utf-8"))
        if bool(score["evaluation_inputs"]["target_depth_read"]):
            raise RuntimeError(f"scene {scene_order}: scorer read target depth")
        contrasts = score["contrasts"]
        full = score["metrics"]["full_frame"]
        observed = score["metrics"]["observed_support"]
        rows.append({
            "frozen_scene_order": scene_order,
            "scene_id": scene_id,
            "anchor_view_id": int(generated["anchor_view_id"]),
            "target_view_id": target_id,
            "B_psnr_full_db": float(full["B_unrestricted"]["psnr"]),
            "C_psnr_full_db": float(full["C_evidence_preserved"]["psnr"]),
            "C_minus_B_psnr_full_db": float(contrasts["C_minus_B_psnr_full_db"]),
            "B_psnr_observed_db": float(observed["B_unrestricted"]["psnr"]),
            "C_psnr_observed_db": float(observed["C_evidence_preserved"]["psnr"]),
            "C_minus_B_psnr_observed_db": float(contrasts["C_minus_B_psnr_observed_db"]),
            "score_path": score_path.relative_to(repo_root).as_posix(),
        })

    if [int(row["frozen_scene_order"]) for row in rows] != list(SCENE_ORDERS):
        raise RuntimeError("scored scene order mismatch")
    full_deltas = [float(row["C_minus_B_psnr_full_db"]) for row in rows]
    observed_deltas = [float(row["C_minus_B_psnr_observed_db"]) for row in rows]
    full_wins = sum(value > 0 for value in full_deltas)
    observed_wins = sum(value > 0 for value in observed_deltas)
    aggregate = {
        "version": "0.1",
        "stage": "refworld-lama-backend-independence-rank2-confirmation",
        "scope": "oracle-source-depth/camera diagnostic; not full single-image RefWorld-0",
        "primary_replication_unit": "scene",
        "protocol": {
            "scene_orders": list(SCENE_ORDERS),
            "target_source_order": TARGET_SOURCE_ORDER,
            "backend": "Big-LaMa Fourier-convolution inpainting",
            "model_sha256": MODEL_SHA256,
            "context_radius_original_pixels": CONTEXT_RADIUS,
            "all_candidates_hash_pinned_before_target_rgb_materialization": True,
            "target_depth_materialized_or_read": False,
            "per_scene_tuning": False,
        },
        "per_scene": rows,
        "aggregate": {
            "full_frame_C_minus_B_psnr_db": {**summarize(full_deltas), "positive_win_count": full_wins, "nonpositive_count": len(rows) - full_wins},
            "observed_support_C_minus_B_psnr_db": {**summarize(observed_deltas), "positive_win_count": observed_wins, "nonpositive_count": len(rows) - observed_wins},
        },
        "generation_complete_marker": marker_path.relative_to(repo_root).as_posix(),
    }
    aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    print("\nLAMA BACKEND-INDEPENDENCE CONFIRMATION COMPLETE", flush=True)
    print("scene target   C-B full dB   C-B observed dB", flush=True)
    for row in rows:
        print(f"{int(row['frozen_scene_order']):>5} {int(row['target_view_id']):>6} {float(row['C_minus_B_psnr_full_db']):>+13.4f} {float(row['C_minus_B_psnr_observed_db']):>+17.4f}", flush=True)
    full_summary = aggregate["aggregate"]["full_frame_C_minus_B_psnr_db"]
    obs_summary = aggregate["aggregate"]["observed_support_C_minus_B_psnr_db"]
    print("\nPRIMARY BACKEND-INDEPENDENCE RESULT — LaMa, frozen scenes 2-10, rank-2 targets", flush=True)
    print("Full-frame C-B PSNR:", flush=True)
    print(f"  wins:   {full_wins}/{len(rows)}", flush=True)
    print(f"  mean:   {full_summary['mean']:+.4f} dB", flush=True)
    print(f"  median: {full_summary['median']:+.4f} dB", flush=True)
    print(f"  worst:  {full_summary['minimum_worst_case']:+.4f} dB", flush=True)
    print("Observed-support C-B PSNR:", flush=True)
    print(f"  wins:   {observed_wins}/{len(rows)}", flush=True)
    print(f"  mean:   {obs_summary['mean']:+.4f} dB", flush=True)
    print(f"  median: {obs_summary['median']:+.4f} dB", flush=True)
    print(f"  worst:  {obs_summary['minimum_worst_case']:+.4f} dB", flush=True)
    print(f"Aggregate: {aggregate_path}", flush=True)
    print("Scope: oracle source-depth/camera diagnostic; backend independence only.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
