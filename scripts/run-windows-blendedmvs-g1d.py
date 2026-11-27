#!/usr/bin/env python3
"""Run EXP-002 G1-D decomposition on the already-opened G1 rank-3 cases.

No fresh target is consumed. This script reuses the existing G1 source geometry,
learned warp, oracle warp, and already-materialized rank-3 RGBs; target depth is
never read. It generates only the two hybrid ablations and scores all four
conditions on common OBSERVED support.
"""

from __future__ import annotations

import json
import statistics
import subprocess
from pathlib import Path
from typing import Any

from refworld.datasets.blendedmvs import load_manifest
from refworld.datasets.mvsnet import parse_pair_text

SCENE_ORDERS = tuple(range(2, 11))
RANK = 3
NEAR_ORACLE_DB = -1.0
SUBSTANTIALLY_BELOW_DB = -3.0


def run_checked(label: str, command: list[str], cwd: Path) -> None:
    print(f"\n== {label} ==", flush=True)
    print(" ".join(command), flush=True)
    result = subprocess.run(command, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")


def selected_scenes(frozen: dict[str, Any]) -> list[dict[str, Any]]:
    scenes = frozen.get("scenes")
    if not isinstance(scenes, list):
        raise RuntimeError("frozen manifest scenes missing")
    by_order = {int(item["order"]): item for item in scenes}
    if any(order not in by_order for order in SCENE_ORDERS):
        raise RuntimeError("frozen manifest missing scene 2-10")
    return [by_order[order] for order in SCENE_ORDERS]


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def classify(medians: dict[str, float]) -> str:
    depth = medians["vggt_depth_oracle_K"]
    intrinsics = medians["oracle_depth_vggt_K"]
    both = medians["vggt_both"]
    depth_poor = depth <= SUBSTANTIALLY_BELOW_DB
    intrinsics_poor = intrinsics <= SUBSTANTIALLY_BELOW_DB
    depth_near = depth >= NEAR_ORACLE_DB
    intrinsics_near = intrinsics >= NEAR_ORACLE_DB
    both_poor = both <= SUBSTANTIALLY_BELOW_DB
    if depth_poor and intrinsics_near:
        return "depth-shape-dominant"
    if intrinsics_poor and depth_near:
        return "intrinsics-dominant"
    if depth_poor and intrinsics_poor:
        return "both-components-failure"
    if depth_near and intrinsics_near and both_poor:
        return "coupling-registration-failure"
    return "mixed-inconclusive"


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    python = repo_root / ".venv-refworld" / "Scripts" / "python.exe"
    if not python.is_file():
        raise RuntimeError(f"RefWorld venv not found: {python}")

    run_checked(
        "Compiling G1-D scripts",
        [str(python), "-m", "py_compile", "src/refworld/runners/blendedmvs_g1d_hybrid.py", "src/refworld/runners/score_blendedmvs_g1d.py", "scripts/run-windows-blendedmvs-g1d.py"],
        repo_root,
    )
    run_checked(
        "Running focused geometry/camera tests",
        [str(python), "-m", "pytest", "-q", "tests/test_geometry_scale.py", "tests/test_source_geometry.py", "tests/test_pinhole_warp.py", "tests/test_mvsnet_dataset.py"],
        repo_root,
    )

    frozen = load_manifest(repo_root / "datasets" / "blendedmvs-bootstrap-v0.json")
    selected = selected_scenes(frozen)
    data_root = repo_root / "private-data" / "blendedmvs-bootstrap"
    g1_root = repo_root / "outputs" / "calibrated" / "blendedmvs" / "vggt-g1-rank3-392"
    g1d_root = repo_root / "outputs" / "calibrated" / "blendedmvs" / "vggt-g1d-rank3-392"
    g1d_root.mkdir(parents=True, exist_ok=True)

    print("\nEXP-002 G1-D: depth vs intrinsics decomposition", flush=True)
    print("Data: already-opened G1 rank-3 scenes 2-10; no fresh target consumed", flush=True)
    print("Primary metric: all-four common OBSERVED support", flush=True)
    print(f"Frozen bands: near >= {NEAR_ORACLE_DB:+.1f} dB; substantially below <= {SUBSTANTIALLY_BELOW_DB:+.1f} dB", flush=True)

    rows: list[dict[str, Any]] = []
    for entry in selected:
        scene_order = int(entry["order"])
        scene_id = str(entry["id"])
        scene_root = data_root / scene_id
        pair_path = scene_root / "cams" / "pair.txt"
        records = parse_pair_text(pair_path.read_text(encoding="utf-8"))
        if not records or len(records[0].source_ids) < RANK:
            raise RuntimeError(f"scene {scene_order}: missing rank-3 pair")
        anchor_id = int(records[0].reference_id)
        target_id = int(records[0].source_ids[RANK - 1])
        target_path = scene_root / "blended_images" / f"{target_id:08d}.jpg"
        target_depth = scene_root / "rendered_depth_maps" / f"{target_id:08d}.pfm"
        if not target_path.is_file():
            raise FileNotFoundError(f"scene {scene_order}: prior G1 rank-3 target RGB missing")
        # Do not read target depth. Its existence is irrelevant; the G1-D runner never receives it.

        g1_scene = g1_root / f"scene-{scene_order:02d}-{scene_id}-target-{target_id:08d}"
        source_geometry = g1_scene / "source-geometry" / "source-geometry.safe.json"
        learned_output = g1_scene / "learned-warp"
        oracle_output = g1_scene / "oracle-warp"
        for required in (source_geometry, learned_output / "vggt-oracle-scale-pair.safe.json", oracle_output / "oracle-pair.safe.json"):
            if not required.is_file():
                raise FileNotFoundError(required)

        scene_out = g1d_root / f"scene-{scene_order:02d}-{scene_id}-target-{target_id:08d}"
        hybrid_output = scene_out / "hybrids"
        score_path = scene_out / "g1d-score.json"

        run_checked(
            f"Scene {scene_order}/10: generate two fixed hybrid ablations",
            [str(python), "-m", "refworld.runners.blendedmvs_g1d_hybrid", "--scene-root", str(scene_root), "--source-geometry", str(source_geometry), "--output", str(hybrid_output)],
            repo_root,
        )
        run_checked(
            f"Scene {scene_order}/10: score all four conditions on common support",
            [str(python), "-m", "refworld.runners.score_blendedmvs_g1d", "--scene-root", str(scene_root), "--learned-output", str(learned_output), "--oracle-output", str(oracle_output), "--hybrid-output", str(hybrid_output), "--output", str(score_path)],
            repo_root,
        )
        score = json.loads(score_path.read_text(encoding="utf-8"))
        contrasts = {key: float(value) for key, value in score["contrasts_common_observed_psnr_db"].items()}
        rows.append({
            "frozen_scene_order": scene_order,
            "scene_id": scene_id,
            "anchor_view_id": anchor_id,
            "target_view_id": target_id,
            "common_observed_fraction": float(score["support"]["all_four_common_observed_fraction"]),
            "deltas_from_oracle_common_observed_psnr_db": contrasts,
            "score": score_path.relative_to(repo_root).as_posix(),
        })

    names = ("vggt_both", "vggt_depth_oracle_K", "oracle_depth_vggt_K")
    medians = {name: median([row["deltas_from_oracle_common_observed_psnr_db"][name] for row in rows]) for name in names}
    attribution = classify(medians)
    aggregate = {
        "version": "0.1",
        "stage": "refworld-exp002-g1d-aggregate",
        "scope": {
            "opened_rank3_reuse_only": True,
            "fresh_target_consumed": False,
            "target_depth_read": False,
            "vggt_resolution": 392,
            "benchmark_resolution_result": False,
        },
        "decision_bands_db": {"near_oracle_min": NEAR_ORACLE_DB, "substantially_below_oracle_max": SUBSTANTIALLY_BELOW_DB},
        "median_common_observed_delta_from_oracle_psnr_db": medians,
        "frozen_rule_attribution": attribution,
        "scenes": rows,
    }
    aggregate_path = g1d_root / "VGGT-G1D-RANK3-392.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    print("\nG1-D COMPLETE", flush=True)
    print("condition                         median delta from oracle on all-four common OBSERVED", flush=True)
    for name in names:
        print(f"{name:32s} {medians[name]:+9.4f} dB", flush=True)
    print(f"Frozen-rule attribution: {attribution}", flush=True)
    print(f"Aggregate: {aggregate_path}", flush=True)
    print("No fresh target consumed; rank-3 is diagnostic-only from this point onward.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
