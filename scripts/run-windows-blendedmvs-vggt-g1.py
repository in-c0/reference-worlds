#!/usr/bin/env python3
"""Run the frozen reduced-resolution VGGT G1 rank-3 experiment on Windows.

All learned geometry, oracle ceiling warps, and frozen Big-LaMa B/C compositions
are generated and hash-pinned before any rank-3 target RGB is materialized.
Target depth is never fetched/read. VGGT 392px is explicitly diagnostic-only on
the RTX 2080 and does not replace the frozen 518px benchmark configuration.
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
from refworld.runners.lama_inpaint_candidate import MODEL_SHA256, MODEL_SIZE_BYTES, MODEL_URL

SCENE_ORDERS = tuple(range(2, 11))
TARGET_SOURCE_ORDER = 3
VGGT_MODEL_SIZE = 392
VGGT_PIN = "a288dd0f14786c93483e45524328726ab7b1b4ce"
CONTEXT_RADIUS = 16
SEAL_STAGE = "refworld-vggt-g1-rank3-all-generation-before-targets"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024, b"")):
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
    if any(order not in by_order for order in SCENE_ORDERS):
        raise RuntimeError("frozen manifest missing scene 2-10")
    return [by_order[order] for order in SCENE_ORDERS]


def artifact_record(path: Path, repo_root: Path) -> dict[str, str]:
    return {"path": path.resolve().relative_to(repo_root.resolve()).as_posix(), "sha256": sha256_file(path)}


def validate_seal(marker_path: Path, repo_root: Path) -> list[dict[str, Any]]:
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("stage") != SEAL_STAGE:
        raise RuntimeError("G1 generation marker stage mismatch")
    if int(marker.get("target_source_order", -1)) != TARGET_SOURCE_ORDER:
        raise RuntimeError("G1 generation marker rank mismatch")
    if int(marker.get("vggt_model_size", -1)) != VGGT_MODEL_SIZE:
        raise RuntimeError("G1 generation marker VGGT resolution mismatch")
    if marker.get("vggt_commit") != VGGT_PIN:
        raise RuntimeError("G1 generation marker VGGT commit mismatch")
    if marker.get("lama_model_sha256") != MODEL_SHA256:
        raise RuntimeError("G1 generation marker LaMa model mismatch")
    if bool(marker.get("target_rgb_present_when_marker_written")):
        raise RuntimeError("G1 marker reports target RGB already present")
    rows = marker.get("scenes")
    if not isinstance(rows, list) or [int(row["frozen_scene_order"]) for row in rows] != list(SCENE_ORDERS):
        raise RuntimeError("G1 generation marker scene order mismatch")
    for row in rows:
        for key in ("source_geometry_manifest", "learned_manifest", "oracle_manifest", "candidate_manifest", "composition_manifest"):
            record = row[key]
            artifact = repo_root / str(record["path"])
            if not artifact.is_file() or sha256_file(artifact) != str(record["sha256"]):
                raise RuntimeError(f"G1 sealed artifact mismatch: {artifact}")
    return rows


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    python = repo_root / ".venv-refworld" / "Scripts" / "python.exe"
    vggt_root = repo_root / ".upstream" / "vggt"
    if not python.is_file():
        raise RuntimeError(f"RefWorld venv not found: {python}")
    if not (vggt_root / ".git").exists():
        raise RuntimeError(f"Pinned VGGT checkout missing: {vggt_root}; run the existing Windows smoke setup first")
    head = subprocess.check_output(["git", "-C", str(vggt_root), "rev-parse", "HEAD"], text=True).strip()
    if head != VGGT_PIN:
        raise RuntimeError(f"VGGT checkout is {head}; expected {VGGT_PIN}")

    run_checked(
        "Checking CUDA/VGGT environment",
        [str(python), "-c", "import torch, cv2, PIL; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"],
        cwd=repo_root,
    )
    run_checked(
        "Installing/verifying RefWorld dependencies",
        [str(python), "-m", "pip", "install", "-e", ".[dataset,dev,method]"],
        cwd=repo_root,
    )
    run_checked(
        "Compiling sealed G1 scripts",
        [str(python), "-m", "py_compile", "scripts/materialize-blendedmvs-vggt-g1-rank3.py", "scripts/run-windows-blendedmvs-vggt-g1.py", "src/refworld/runners/score_blendedmvs_vggt_g1.py", "src/refworld/runners/blendedmvs_vggt_scaled_pair.py"],
        cwd=repo_root,
    )
    run_checked(
        "Running G1 geometry/archive/camera contract tests",
        [str(python), "-m", "pytest", "-q", "tests/test_geometry_scale.py", "tests/test_vggt_source.py", "tests/test_source_geometry.py", "tests/test_remote_zip.py", "tests/test_pfm.py", "tests/test_mvsnet_dataset.py", "tests/test_blendedmvs_oracle_selection.py"],
        cwd=repo_root,
    )

    frozen = load_manifest(repo_root / "datasets" / "blendedmvs-bootstrap-v0.json")
    selected = selected_scenes(frozen)
    data_root = repo_root / "private-data" / "blendedmvs-bootstrap"
    experiment_root = repo_root / "outputs" / "calibrated" / "blendedmvs" / "vggt-g1-rank3-392"
    experiment_root.mkdir(parents=True, exist_ok=True)
    marker_path = experiment_root / "ALL-G1-GENERATION-BEFORE-TARGETS.safe.json"
    aggregate_path = experiment_root / "VGGT-G1-RANK3-392.json"

    print("\nRefWorld EXP-002 VGGT G1 sealed rank-3 diagnostic", flush=True)
    print("Primary scenes: frozen 2-10; target source rank: 3", flush=True)
    print(f"VGGT: pinned {VGGT_PIN} at {VGGT_MODEL_SIZE}x{VGGT_MODEL_SIZE}", flush=True)
    print("Resolution scope: reduced-resolution diagnostic only; NOT the frozen 518px benchmark", flush=True)
    print("Scale: one positive scalar from anchor depth; target depth never fetched/read", flush=True)
    print(f"Repaint: Big-LaMa {MODEL_SHA256}; context {CONTEXT_RADIUS}px", flush=True)
    print("Seal: ALL VGGT/oracle/LaMa B/C outputs hash-pinned before rank-3 target RGB reveal", flush=True)

    generated_rows: list[dict[str, Any]]
    if marker_path.is_file():
        print("\nExisting sealed G1 marker found; validating resume...", flush=True)
        generated_rows = validate_seal(marker_path, repo_root)
        print("G1 generation seal valid.", flush=True)
    else:
        run_checked(
            "PHASE 1A: materialize rank-3 generation inputs (NO target RGB)",
            [str(python), "scripts/materialize-blendedmvs-vggt-g1-rank3.py", "--phase", "generation"],
            cwd=repo_root,
        )
        generated_rows = []
        print("\n========== PHASE 1B: GENERATE ALL NINE LEARNED/ORACLE/LAMA B-C OUTPUTS; TARGET RGB ABSENT ==========", flush=True)

        for entry in selected:
            scene_order = int(entry["order"])
            scene_id = str(entry["id"])
            scene_root = data_root / scene_id
            generation_manifest = scene_root / "vggt-g1-rank3-generation-inputs.safe.json"
            generation = json.loads(generation_manifest.read_text(encoding="utf-8"))
            if int(generation["target_source_order"]) != TARGET_SOURCE_ORDER:
                raise RuntimeError(f"scene {scene_order}: rank-3 generation selection mismatch")
            anchor_id = int(generation["anchor_view_id"])
            target_id = int(generation["target_view_id"])
            anchor_rgb = scene_root / "blended_images" / f"{anchor_id:08d}.jpg"
            target_rgb = scene_root / "blended_images" / f"{target_id:08d}.jpg"
            if target_rgb.exists():
                raise RuntimeError(f"scene {scene_order}: rank-3 target RGB exists before generation: {target_rgb}")

            scene_output = experiment_root / f"scene-{scene_order:02d}-{scene_id}-target-{target_id:08d}"
            source_output = scene_output / "source-geometry"
            learned_output = scene_output / "learned-warp"
            oracle_output = scene_output / "oracle-warp"
            candidate_output = scene_output / "lama"
            composition = candidate_output / "composition"

            run_checked(
                f"Scene {scene_order}/10: pinned VGGT source geometry",
                [str(python), "-m", "refworld.runners.vggt_source", "--vggt-root", str(vggt_root), "--reference", str(anchor_rgb), "--output", str(source_output), "--seed", "0", "--model-size", str(VGGT_MODEL_SIZE)],
                cwd=repo_root,
            )
            source_manifest = source_output / "source-geometry.safe.json"
            source_meta = json.loads(source_manifest.read_text(encoding="utf-8"))
            if int(source_meta["preprocessing"]["model_size"]) != VGGT_MODEL_SIZE:
                raise RuntimeError(f"scene {scene_order}: VGGT model size mismatch")
            if not bool(source_meta["preprocessing"]["reduced_resolution_smoke_only"]):
                raise RuntimeError(f"scene {scene_order}: 392px run must remain marked reduced-resolution")
            if source_meta["upstream"]["actual_commit"] != VGGT_PIN:
                raise RuntimeError(f"scene {scene_order}: VGGT pin mismatch")
            if target_rgb.exists():
                raise RuntimeError(f"scene {scene_order}: target RGB appeared during VGGT source inference")

            run_checked(
                f"Scene {scene_order}/10: G1 learned warp with one oracle scale scalar",
                [str(python), "-m", "refworld.runners.blendedmvs_vggt_scaled_pair", "--scene-root", str(scene_root), "--source-geometry", str(source_manifest), "--output", str(learned_output), "--held-out-rank", str(TARGET_SOURCE_ORDER)],
                cwd=repo_root,
            )
            learned_manifest = learned_output / "vggt-oracle-scale-pair.safe.json"
            learned = json.loads(learned_manifest.read_text(encoding="utf-8"))
            if int(learned["selection"]["target_view_id"]) != target_id or not bool(learned["primary_protocol"]):
                raise RuntimeError(f"scene {scene_order}: learned G1 selection mismatch")
            if bool(learned["method_inputs"]["target_rgb_read"]) or bool(learned["method_inputs"]["target_depth_read"]) or target_rgb.exists():
                raise RuntimeError(f"scene {scene_order}: held-out leakage during learned warp")

            run_checked(
                f"Scene {scene_order}/10: oracle geometry ceiling at same rank-3 target",
                [str(python), "-m", "refworld.runners.blendedmvs_oracle_pair", "--scene-root", str(scene_root), "--output", str(oracle_output), "--held-out-rank", str(TARGET_SOURCE_ORDER)],
                cwd=repo_root,
            )
            oracle_manifest = oracle_output / "oracle-pair.safe.json"
            oracle = json.loads(oracle_manifest.read_text(encoding="utf-8"))
            if int(oracle["selection"]["target_view_id"]) != target_id:
                raise RuntimeError(f"scene {scene_order}: oracle G1 target mismatch")
            if bool(oracle["method_inputs"]["target_rgb_read"]) or bool(oracle["method_inputs"]["target_depth_read"]) or target_rgb.exists():
                raise RuntimeError(f"scene {scene_order}: held-out leakage during oracle ceiling")

            learned_view = learned_output / str(learned["result"]["view_directory"])
            run_checked(
                f"Scene {scene_order}/10: frozen Big-LaMa candidate on learned geometry",
                [str(python), "-m", "refworld.runners.lama_inpaint_candidate", "--warp-view", str(learned_view), "--output", str(candidate_output), "--context-radius", str(CONTEXT_RADIUS)],
                cwd=repo_root,
            )
            candidate_manifest = candidate_output / "lama-inpaint.safe.json"
            candidate = json.loads(candidate_manifest.read_text(encoding="utf-8"))
            if candidate["backend"]["model_sha256"] != MODEL_SHA256 or candidate["backend"]["model_size_bytes"] != MODEL_SIZE_BYTES:
                raise RuntimeError(f"scene {scene_order}: LaMa model pin mismatch")
            if bool(candidate["input"]["held_out_evaluation_image_used"]) or target_rgb.exists():
                raise RuntimeError(f"scene {scene_order}: held-out leakage during LaMa generation")

            run_checked(
                f"Scene {scene_order}/10: learned-geometry evidence-preserving B/C composition",
                [str(python), "-m", "refworld.runners.compose_candidate", "--warp-view", str(learned_view), "--candidate", str(candidate_output / "candidate.png"), "--valid-mask-npy", str(candidate_output / "repaint-valid-mask.npy"), "--output", str(composition), "--backend", "big-lama-fourier-convolution-apache2", "--backend-run-id", MODEL_SHA256],
                cwd=repo_root,
            )
            composition_manifest = composition / "compose.safe.json"
            if target_rgb.exists():
                raise RuntimeError(f"scene {scene_order}: target RGB appeared before experiment-wide seal")

            generated_rows.append({
                "frozen_scene_order": scene_order,
                "scene_id": scene_id,
                "anchor_view_id": anchor_id,
                "target_view_id": target_id,
                "source_geometry_manifest": artifact_record(source_manifest, repo_root),
                "learned_manifest": artifact_record(learned_manifest, repo_root),
                "oracle_manifest": artifact_record(oracle_manifest, repo_root),
                "candidate_manifest": artifact_record(candidate_manifest, repo_root),
                "composition_manifest": artifact_record(composition_manifest, repo_root),
                "learned_output": learned_output.relative_to(repo_root).as_posix(),
                "oracle_output": oracle_output.relative_to(repo_root).as_posix(),
                "composition_output": composition.relative_to(repo_root).as_posix(),
            })

        for row in generated_rows:
            target = data_root / str(row["scene_id"]) / "blended_images" / f"{int(row['target_view_id']):08d}.jpg"
            if target.exists():
                raise RuntimeError(f"rank-3 target RGB exists before sealed marker: {target}")
        marker_path.write_text(json.dumps({
            "version": "0.1",
            "stage": SEAL_STAGE,
            "scene_orders": list(SCENE_ORDERS),
            "target_source_order": TARGET_SOURCE_ORDER,
            "vggt_commit": VGGT_PIN,
            "vggt_model_size": VGGT_MODEL_SIZE,
            "benchmark_default_vggt_model_size": 518,
            "reduced_resolution_diagnostic_only": True,
            "lama_model_url": MODEL_URL,
            "lama_model_size_bytes": MODEL_SIZE_BYTES,
            "lama_model_sha256": MODEL_SHA256,
            "context_radius_original_pixels": CONTEXT_RADIUS,
            "target_rgb_present_when_marker_written": False,
            "target_depth_materialized": False,
            "scenes": generated_rows,
        }, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        generated_rows = validate_seal(marker_path, repo_root)
        print(f"\nSEALED G1 GENERATION MARKER: {marker_path}", flush=True)

    targets_batch = data_root / "vggt-g1-rank3-targets-materialization.safe.json"
    if targets_batch.is_file():
        print("\nRank-3 targets already materialized after seal; validating resume inputs...", flush=True)
        for row in generated_rows:
            target = data_root / str(row["scene_id"]) / "blended_images" / f"{int(row['target_view_id']):08d}.jpg"
            if not target.is_file():
                raise FileNotFoundError(target)
    else:
        run_checked(
            "PHASE 2A: generation sealed; NOW materialize rank-3 target RGBs",
            [str(python), "scripts/materialize-blendedmvs-vggt-g1-rank3.py", "--phase", "targets", "--seal", str(marker_path)],
            cwd=repo_root,
        )

    print("\n========== PHASE 2B: SCORE NINE PREVIOUSLY SEALED RANK-3 TARGETS ==========", flush=True)
    rows: list[dict[str, Any]] = []
    for generated in generated_rows:
        scene_order = int(generated["frozen_scene_order"])
        scene_id = str(generated["scene_id"])
        target_id = int(generated["target_view_id"])
        scene_root = data_root / scene_id
        learned_output = repo_root / str(generated["learned_output"])
        oracle_output = repo_root / str(generated["oracle_output"])
        composition = repo_root / str(generated["composition_output"])
        score_path = learned_output.parent / "g1-score.json"
        run_checked(
            f"Scene {scene_order}/10: sealed held-out G1 scoring target {target_id}",
            [str(python), "-m", "refworld.runners.score_blendedmvs_vggt_g1", "--scene-root", str(scene_root), "--learned-output", str(learned_output), "--oracle-output", str(oracle_output), "--composition", str(composition), "--output", str(score_path)],
            cwd=repo_root,
        )
        score = json.loads(score_path.read_text(encoding="utf-8"))
        if bool(score["evaluation_inputs"]["target_depth_read"]):
            raise RuntimeError(f"scene {scene_order}: scorer read target depth")
        full = score["metrics"]["full_frame"]
        learned_support = score["metrics"]["learned_observed_support"]
        common = score["metrics"]["common_observed_support"]
        contrasts = score["contrasts"]
        rows.append({
            "frozen_scene_order": scene_order,
            "scene_id": scene_id,
            "target_view_id": target_id,
            "scale": float(score["scale_calibration"]["scale"]),
            "scale_relative_ratio_mad": float(score["scale_calibration"]["relative_ratio_mad"]),
            "learned_observed_fraction": float(score["support"]["learned_observed_fraction"]),
            "oracle_observed_fraction": float(score["support"]["oracle_observed_fraction"]),
            "learned_psnr_full_db": float(full["learned_vggt_warp"]["psnr"]),
            "oracle_psnr_full_db": float(full["oracle_warp"]["psnr"]),
            "learned_psnr_common_observed_db": float(common["learned_vggt_warp"]["psnr"]),
            "oracle_psnr_common_observed_db": float(common["oracle_warp"]["psnr"]),
            "learned_minus_oracle_psnr_common_observed_db": float(contrasts["learned_minus_oracle_psnr_common_observed_db"]),
            "C_minus_B_psnr_full_db": float(contrasts["C_minus_B_psnr_full_db"]),
            "C_minus_B_psnr_learned_observed_db": float(contrasts["C_minus_B_psnr_learned_observed_db"]),
            "B_psnr_learned_observed_db": float(learned_support["B_unrestricted"]["psnr"]),
            "C_psnr_learned_observed_db": float(learned_support["C_evidence_preserved"]["psnr"]),
            "score_path": score_path.relative_to(repo_root).as_posix(),
        })

    if [int(row["frozen_scene_order"]) for row in rows] != list(SCENE_ORDERS):
        raise RuntimeError("G1 scored scene order mismatch")
    geometry_gaps = [float(row["learned_minus_oracle_psnr_common_observed_db"]) for row in rows]
    full_bc = [float(row["C_minus_B_psnr_full_db"]) for row in rows]
    observed_bc = [float(row["C_minus_B_psnr_learned_observed_db"]) for row in rows]
    aggregate = {
        "version": "0.1",
        "stage": "refworld-vggt-g1-rank3-392-aggregate",
        "scope": {
            "scene_orders": list(SCENE_ORDERS),
            "target_source_order": TARGET_SOURCE_ORDER,
            "vggt_model_size": VGGT_MODEL_SIZE,
            "benchmark_default_vggt_model_size": 518,
            "reduced_resolution_diagnostic_only": True,
            "oracle_scale_scalar": True,
            "oracle_anchor_frame_placement": True,
            "target_depth_read": False,
            "full_single_image_method_result": False,
        },
        "generation_seal": artifact_record(marker_path, repo_root),
        "rows": rows,
        "summary": {
            "learned_minus_oracle_common_observed_psnr_db": summarize(geometry_gaps),
            "C_minus_B_full_psnr_db": {"wins": sum(v > 0 for v in full_bc), **summarize(full_bc)},
            "C_minus_B_learned_observed_psnr_db": {"wins": sum(v > 0 for v in observed_bc), **summarize(observed_bc)},
        },
    }
    aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    print("\nVGGT G1 RANK-3 392 DIAGNOSTIC COMPLETE", flush=True)
    print("scene target learned-oracle common dB   C-B full dB   C-B learned-observed dB", flush=True)
    for row in rows:
        print(f"{int(row['frozen_scene_order']):5d} {int(row['target_view_id']):6d} {float(row['learned_minus_oracle_psnr_common_observed_db']):+24.4f} {float(row['C_minus_B_psnr_full_db']):+13.4f} {float(row['C_minus_B_psnr_learned_observed_db']):+23.4f}", flush=True)
    print(f"\nGeometry learned-oracle common-observed median: {statistics.median(geometry_gaps):+.4f} dB", flush=True)
    print(f"Learned-geometry B/C full-frame wins: {sum(v > 0 for v in full_bc)}/9; median {statistics.median(full_bc):+.4f} dB", flush=True)
    print(f"Learned-geometry B/C observed wins: {sum(v > 0 for v in observed_bc)}/9; median {statistics.median(observed_bc):+.4f} dB", flush=True)
    print(f"Aggregate: {aggregate_path}", flush=True)
    print("Scope: reduced-resolution VGGT G1 with one oracle scale scalar + oracle frame placement; not end-to-end single-image benchmark evidence.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
