#!/usr/bin/env python3
"""Run the frozen 10-view BlendedMVS oracle B-vs-C replication on Windows.

Protocol was predeclared after the first held-out result and before viewing any
additional held-out scores:

- first frozen BlendedMVS scene only;
- first published pair.txt record only;
- all 10 published source views, in published order;
- oracle anchor depth/camera + target camera diagnostic;
- SD2 revision fixed to the exact revision used for the first calibrated result;
- seed/config fixed across every target;
- generate all 10 B/C candidates before invoking any replication score stage;
- then score every view and report all per-view contrasts plus aggregate
  mean/median/win-count/worst-case values.

This is still an oracle-geometry diagnostic, not the full single-image method.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
from pathlib import Path

from refworld.datasets.mvsnet import parse_pair_text


SCENE_ID = "5b7a3890fc8fcf6781e2593a"
EXPECTED_TARGETS = (136, 158, 20, 48, 190, 30, 165, 260, 29, 106)
FROZEN_SD2_REVISION = "5f74973cbb64c8568780732c17f43eb269d63a0d"
FROZEN_SEED = 42
FROZEN_STEPS = 30
FROZEN_GUIDANCE = 4.0
FROZEN_CONTEXT_RADIUS = 16
FROZEN_MAX_SIDE = 512


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


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    python = repo_root / ".venv-refworld" / "Scripts" / "python.exe"
    if not python.is_file():
        raise RuntimeError(f"RefWorld venv not found: {python}")

    scene_root = repo_root / "private-data" / "blendedmvs-bootstrap" / SCENE_ID
    pair_path = scene_root / "cams" / "pair.txt"
    materialization = scene_root / "materialization.safe.json"
    if not materialization.is_file() or not pair_path.is_file():
        raise RuntimeError(
            "frozen BlendedMVS scene is not materialized; run scripts/run-windows-blendedmvs-oracle.py once first"
        )

    records = parse_pair_text(pair_path.read_text(encoding="utf-8"))
    if not records:
        raise RuntimeError("pair.txt contains no records")
    first = records[0]
    actual_targets = tuple(int(value) for value in first.source_ids)
    if actual_targets != EXPECTED_TARGETS:
        raise RuntimeError(
            f"frozen published target order changed: {actual_targets} != {EXPECTED_TARGETS}"
        )

    experiment_root = (
        repo_root
        / "outputs"
        / "calibrated"
        / "blendedmvs"
        / SCENE_ID
        / "oracle-first-pair-10view"
    )
    experiment_root.mkdir(parents=True, exist_ok=True)

    print("RefWorld frozen 10-view calibrated replication", flush=True)
    print(f"Scene:          {SCENE_ID}", flush=True)
    print(f"Anchor:         {first.reference_id}", flush=True)
    print(f"Targets:        {list(EXPECTED_TARGETS)}", flush=True)
    print(f"SD2 revision:   {FROZEN_SD2_REVISION}", flush=True)
    print("SD2 config:     seed=42 steps=30 guidance=4.0 context=16 max-side=512", flush=True)
    print("Geometry:       ORACLE source depth/cameras diagnostic", flush=True)
    print("Generation rule: all 10 candidates before replication scoring", flush=True)

    run_checked(
        "Installing/verifying frozen replication dependencies",
        [str(python), "-m", "pip", "install", "-e", ".[dataset,repaint-sd2,dev]"],
        cwd=repo_root,
    )
    run_checked(
        "Running replication preflight contract tests",
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

    generated: list[dict[str, object]] = []
    print("\n========== PHASE 1: GENERATE ALL 10; NO BATCH SCORING ==========", flush=True)
    for rank, expected_target in enumerate(EXPECTED_TARGETS, start=1):
        rank_root = experiment_root / f"rank-{rank:02d}-target-{expected_target:08d}"
        oracle_output = rank_root / "warp"
        candidate_output = rank_root / "sd2"
        composition = candidate_output / "composition"

        run_checked(
            f"Rank {rank}/10: oracle warp to published target {expected_target}",
            [
                str(python),
                "-m",
                "refworld.runners.blendedmvs_oracle_pair",
                "--scene-root",
                str(scene_root),
                "--output",
                str(oracle_output),
                "--held-out-rank",
                str(rank),
            ],
            cwd=repo_root,
        )
        oracle = json.loads((oracle_output / "oracle-pair.safe.json").read_text(encoding="utf-8"))
        selection = oracle["selection"]
        if int(selection["held_out_source_order"]) != rank:
            raise RuntimeError(f"rank {rank}: oracle manifest source-order mismatch")
        if int(selection["target_view_id"]) != expected_target:
            raise RuntimeError(f"rank {rank}: oracle manifest target mismatch")
        if bool(oracle["method_inputs"]["target_rgb_read"]):
            raise RuntimeError(f"rank {rank}: oracle generation unexpectedly read target RGB")
        warp_view = oracle_output / str(oracle["result"]["view_directory"])

        run_checked(
            f"Rank {rank}/10: frozen SD2 candidate",
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
        candidate = json.loads((candidate_output / "sd2-inpaint.safe.json").read_text(encoding="utf-8"))
        if str(candidate["backend"]["resolved_revision"]) != FROZEN_SD2_REVISION:
            raise RuntimeError(f"rank {rank}: SD2 revision mismatch")
        if not bool(candidate["backend"].get("revision_was_explicitly_pinned")):
            raise RuntimeError(f"rank {rank}: SD2 revision was not explicitly pinned")
        cfg = candidate["configuration"]
        expected_cfg = {
            "seed": FROZEN_SEED,
            "num_inference_steps": FROZEN_STEPS,
            "guidance_scale": FROZEN_GUIDANCE,
            "context_radius_original_pixels": FROZEN_CONTEXT_RADIUS,
            "max_model_side": FROZEN_MAX_SIDE,
        }
        for key, expected in expected_cfg.items():
            if cfg[key] != expected:
                raise RuntimeError(f"rank {rank}: frozen config mismatch for {key}: {cfg[key]} != {expected}")
        if bool(candidate["input"]["held_out_evaluation_image_used"]):
            raise RuntimeError(f"rank {rank}: candidate generation unexpectedly used held-out RGB")

        run_checked(
            f"Rank {rank}/10: evidence-preserving B/C composition",
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

        generated.append(
            {
                "rank": rank,
                "target_view_id": expected_target,
                "rank_root": rank_root,
                "oracle_output": oracle_output,
                "composition": composition,
                "observed_fraction": float(oracle["result"]["observed_fraction"]),
                "unresolved_fraction": float(oracle["result"]["unresolved_fraction"]),
            }
        )

    print(
        "\nPHASE 1 COMPLETE: all 10 frozen candidates/compositions exist before the batch score phase.",
        flush=True,
    )

    print("\n========== PHASE 2: OPEN HELD-OUT RGB AND SCORE ==========", flush=True)
    rows: list[dict[str, object]] = []
    for item in generated:
        rank = int(item["rank"])
        target = int(item["target_view_id"])
        rank_root = Path(item["rank_root"])
        oracle_output = Path(item["oracle_output"])
        composition = Path(item["composition"])
        score_path = rank_root / "calibrated-score.json"

        run_checked(
            f"Rank {rank}/10: held-out scoring target {target}",
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
        if int(score["selection"]["target_view_id"]) != target:
            raise RuntimeError(f"rank {rank}: score target mismatch")
        full = score["metrics"]["full_frame"]
        observed = score["metrics"]["observed_support"]
        contrasts = score["contrasts"]
        rows.append(
            {
                "rank": rank,
                "target_view_id": target,
                "observed_fraction": item["observed_fraction"],
                "unresolved_fraction": item["unresolved_fraction"],
                "B_psnr_full_db": float(full["B_unrestricted"]["psnr"]),
                "C_psnr_full_db": float(full["C_evidence_preserved"]["psnr"]),
                "C_minus_B_psnr_full_db": float(contrasts["C_minus_B_psnr_full_db"]),
                "B_psnr_observed_db": float(observed["B_unrestricted"]["psnr"]),
                "C_psnr_observed_db": float(observed["C_evidence_preserved"]["psnr"]),
                "C_minus_B_psnr_observed_db": float(contrasts["C_minus_B_psnr_observed_db"]),
                "score_path": str(score_path),
            }
        )

    full_deltas = [float(row["C_minus_B_psnr_full_db"]) for row in rows]
    observed_deltas = [float(row["C_minus_B_psnr_observed_db"]) for row in rows]
    full_wins = sum(value > 0.0 for value in full_deltas)
    observed_wins = sum(value > 0.0 for value in observed_deltas)

    aggregate = {
        "version": "0.1",
        "stage": "refworld-blendedmvs-oracle-10view-replication",
        "scope": "oracle-source-depth diagnostic; not full single-image RefWorld-0",
        "protocol": {
            "scene_id": SCENE_ID,
            "pair_record_order": 1,
            "anchor_view_id": int(first.reference_id),
            "published_target_order": list(EXPECTED_TARGETS),
            "target_count": len(EXPECTED_TARGETS),
            "sd2_revision": FROZEN_SD2_REVISION,
            "seed": FROZEN_SEED,
            "steps": FROZEN_STEPS,
            "guidance_scale": FROZEN_GUIDANCE,
            "context_radius_original_pixels": FROZEN_CONTEXT_RADIUS,
            "max_model_side": FROZEN_MAX_SIDE,
            "all_candidates_generated_before_batch_scoring": True,
            "per_view_hyperparameter_tuning": False,
        },
        "per_view": rows,
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
    }
    aggregate_path = experiment_root / "TEN-VIEW-REPLICATION.json"
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("\nTEN-VIEW CALIBRATED ORACLE REPLICATION COMPLETE", flush=True)
    print("rank target   C-B full dB   C-B observed dB", flush=True)
    for row in rows:
        print(
            f"{int(row['rank']):>4} {int(row['target_view_id']):>6}"
            f" {float(row['C_minus_B_psnr_full_db']):>+13.4f}"
            f" {float(row['C_minus_B_psnr_observed_db']):>+17.4f}",
            flush=True,
        )
    full_summary = aggregate["aggregate"]["full_frame_C_minus_B_psnr_db"]
    obs_summary = aggregate["aggregate"]["observed_support_C_minus_B_psnr_db"]
    print("\nFull-frame C-B PSNR:", flush=True)
    print(f"  wins:   {full_wins}/10", flush=True)
    print(f"  mean:   {full_summary['mean']:+.4f} dB", flush=True)
    print(f"  median: {full_summary['median']:+.4f} dB", flush=True)
    print(f"  worst:  {full_summary['minimum_worst_case']:+.4f} dB", flush=True)
    print("Observed-support C-B PSNR:", flush=True)
    print(f"  wins:   {observed_wins}/10", flush=True)
    print(f"  mean:   {obs_summary['mean']:+.4f} dB", flush=True)
    print(f"  median: {obs_summary['median']:+.4f} dB", flush=True)
    print(f"  worst:  {obs_summary['minimum_worst_case']:+.4f} dB", flush=True)
    print(f"Aggregate: {aggregate_path}", flush=True)
    print("Scope: oracle source-depth/camera diagnostic; not full single-image RefWorld-0.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
