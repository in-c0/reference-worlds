#!/usr/bin/env python3
"""Run frozen EXP-002 G1-B DA3-BASE geometry screen on Windows.

Uses only the already-opened rank-3 development targets from G1. DA3-BASE is
compared with an equalized VGGT reference and oracle geometry under the same
all-valid one-scalar oracle depth bridge. Rank 4 is never materialized.
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
from pathlib import Path
from typing import Any

from refworld.datasets.blendedmvs import load_manifest
from refworld.datasets.mvsnet import parse_pair_text

SCENE_ORDERS = tuple(range(2, 11))
RANK = 3
DA3_PIN = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
PROCESS_RES_LEVELS = (504, 392, 336)
PASS_ORACLE_GAP_DB = -3.0
PASS_VGGT_GAIN_DB = 3.0
PASS_MIN_WINS = 7


def run_checked(label: str, command: list[str], *, cwd: Path) -> None:
    print(f"\n== {label} ==", flush=True)
    print(" ".join(command), flush=True)
    result = subprocess.run(command, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")


def run_capture(label: str, command: list[str], *, cwd: Path) -> tuple[int, str]:
    print(f"\n== {label} ==", flush=True)
    print(" ".join(command), flush=True)
    result = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    text = result.stdout or ""
    if text:
        print(text, end="" if text.endswith("\n") else "\n", flush=True)
    return int(result.returncode), text


def is_cuda_oom(text: str) -> bool:
    lowered = text.lower()
    signatures = (
        "cuda out of memory",
        "outofmemoryerror",
        "cuda error: out of memory",
        "cublas_status_alloc_failed",
        "not enough memory",
    )
    return any(signature in lowered for signature in signatures)


def selected_scenes(frozen: dict[str, Any]) -> list[dict[str, Any]]:
    scenes = frozen.get("scenes")
    if not isinstance(scenes, list):
        raise RuntimeError("frozen BlendedMVS manifest scenes missing")
    by_order = {int(item["order"]): item for item in scenes}
    if any(order not in by_order for order in SCENE_ORDERS):
        raise RuntimeError("frozen manifest missing scene order 2-10")
    return [by_order[order] for order in SCENE_ORDERS]


def scene_selection(scene_root: Path) -> tuple[int, int]:
    pair_path = scene_root / "cams" / "pair.txt"
    if not pair_path.is_file():
        raise FileNotFoundError(pair_path)
    records = parse_pair_text(pair_path.read_text(encoding="utf-8"))
    if not records or len(records[0].source_ids) < RANK:
        raise RuntimeError(f"{scene_root.name}: first pair record has no rank-{RANK} target")
    return int(records[0].reference_id), int(records[0].source_ids[RANK - 1])


def ensure_da3_checkout(repo_root: Path) -> Path:
    upstream = repo_root / ".upstream"
    upstream.mkdir(parents=True, exist_ok=True)
    da3_root = upstream / "depth-anything-3"
    if not (da3_root / ".git").exists():
        run_checked(
            "Cloning Depth Anything 3 upstream",
            ["git", "clone", "https://github.com/ByteDance-Seed/Depth-Anything-3.git", str(da3_root)],
            cwd=repo_root,
        )
    run_checked(
        "Fetching pinned Depth Anything 3 upstream",
        ["git", "-C", str(da3_root), "fetch", "--all", "--tags"],
        cwd=repo_root,
    )
    run_checked(
        "Checking out frozen DA3 code",
        ["git", "-C", str(da3_root), "checkout", "--detach", DA3_PIN],
        cwd=repo_root,
    )
    head = subprocess.check_output(["git", "-C", str(da3_root), "rev-parse", "HEAD"], text=True).strip()
    if head != DA3_PIN:
        raise RuntimeError(f"DA3 checkout is {head}; expected {DA3_PIN}")
    return da3_root


def _da3_probe_code(da3_root: Path) -> str:
    src = str((da3_root / "src").resolve())
    return (
        "import sys; "
        f"sys.path.insert(0, {src!r}); "
        "from depth_anything_3.api import DepthAnything3; "
        "print('DA3 runtime import present')"
    )


def ensure_runtime_dependencies(python: Path, repo_root: Path, da3_root: Path) -> None:
    probe = subprocess.run(
        [str(python), "-c", _da3_probe_code(da3_root)],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    if probe.returncode == 0:
        if probe.stdout:
            print(probe.stdout, end="" if probe.stdout.endswith("\n") else "\n", flush=True)
        return
    print("DA3 runtime import probe failed; installing the frozen inference dependency subset.", flush=True)
    run_checked(
        "Installing DA3 inference runtime dependencies",
        [
            str(python),
            "-m",
            "pip",
            "install",
            "addict",
            "omegaconf",
            "evo",
            "einops",
            "tqdm",
            "safetensors",
            "trimesh",
            "imageio",
            "moviepy==1.0.3",
            "plyfile",
            "pillow_heif",
            "pycolmap",
        ],
        cwd=repo_root,
    )
    run_checked(
        "Verifying DA3 runtime import",
        [str(python), "-c", _da3_probe_code(da3_root)],
        cwd=repo_root,
    )


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    python = repo_root / ".venv-refworld" / "Scripts" / "python.exe"
    if not python.is_file():
        raise RuntimeError(f"RefWorld venv missing: {python}")

    run_checked(
        "Checking CUDA environment",
        [
            str(python),
            "-c",
            "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))",
        ],
        cwd=repo_root,
    )
    run_checked(
        "Compiling G1-B scripts",
        [
            str(python),
            "-m",
            "py_compile",
            "src/refworld/runners/da3_source.py",
            "src/refworld/runners/blendedmvs_learned_allvalid_pair.py",
            "src/refworld/runners/score_blendedmvs_g1b.py",
            "scripts/run-windows-blendedmvs-g1b-da3.py",
        ],
        cwd=repo_root,
    )
    run_checked(
        "Running focused G1-B geometry tests",
        [
            str(python),
            "-m",
            "pytest",
            "-q",
            "tests/test_da3_source.py",
            "tests/test_geometry_scale.py",
            "tests/test_source_geometry.py",
            "tests/test_pinhole_warp.py",
            "tests/test_mvsnet_dataset.py",
        ],
        cwd=repo_root,
    )

    da3_root = ensure_da3_checkout(repo_root)
    ensure_runtime_dependencies(python, repo_root, da3_root)

    frozen = load_manifest(repo_root / "datasets" / "blendedmvs-bootstrap-v0.json")
    selected = selected_scenes(frozen)
    data_root = repo_root / "private-data" / "blendedmvs-bootstrap"
    g1_root = repo_root / "outputs" / "calibrated" / "blendedmvs" / "vggt-g1-rank3-392"
    out_root = repo_root / "outputs" / "calibrated" / "blendedmvs" / "g1b-da3-rank3"
    out_root.mkdir(parents=True, exist_ok=True)

    scene_records: list[dict[str, Any]] = []
    for entry in selected:
        scene_order = int(entry["order"])
        scene_id = str(entry["id"])
        scene_root = data_root / scene_id
        anchor_id, target_id = scene_selection(scene_root)
        anchor_path = scene_root / "blended_images" / f"{anchor_id:08d}.jpg"
        target_path = scene_root / "blended_images" / f"{target_id:08d}.jpg"
        target_meta_path = scene_root / "vggt-g1-rank3-target-rgb.safe.json"
        if not anchor_path.is_file() or not target_path.is_file() or not target_meta_path.is_file():
            raise RuntimeError(f"scene {scene_order}: requires previously opened G1 rank-3 target state")
        target_meta = json.loads(target_meta_path.read_text(encoding="utf-8"))
        if int(target_meta.get("target_view_id", -1)) != target_id:
            raise RuntimeError(f"scene {scene_order}: rank-3 target id does not match G1 manifest")
        if bool(target_meta.get("target_depth_materialized")):
            raise RuntimeError(f"scene {scene_order}: G1 target manifest says target depth was materialized")

        g1_scene = g1_root / f"scene-{scene_order:02d}-{scene_id}-target-{target_id:08d}"
        vggt_source = g1_scene / "source-geometry" / "source-geometry.safe.json"
        oracle_output = g1_scene / "oracle-warp"
        if not vggt_source.is_file() or not (oracle_output / "oracle-pair.safe.json").is_file():
            raise RuntimeError(f"scene {scene_order}: frozen G1 geometry artifacts missing")
        scene_records.append(
            {
                "scene_order": scene_order,
                "scene_id": scene_id,
                "scene_root": scene_root,
                "anchor_id": anchor_id,
                "target_id": target_id,
                "anchor_path": anchor_path,
                "vggt_source": vggt_source,
                "oracle_output": oracle_output,
            }
        )

    print("\nEXP-002 G1-B: DA3-BASE geometry screen", flush=True)
    print("Data: already-opened rank-3 scenes 2-10; rank-4 untouched", flush=True)
    print("Scale bridge: all-valid one oracle multiplicative scalar for BOTH VGGT and DA3", flush=True)
    print("DA3 native scale/pose ignored; DA3 sky excluded from hard evidence", flush=True)
    print("Primary support: OBSERVED intersection of oracle + equalized VGGT + DA3", flush=True)
    print(
        f"PASS: median DA3-oracle > {PASS_ORACLE_GAP_DB:+.1f} dB; "
        f"median DA3-VGGT >= {PASS_VGGT_GAIN_DB:+.1f} dB; wins >= {PASS_MIN_WINS}/9",
        flush=True,
    )

    first = scene_records[0]
    first_out = out_root / f"scene-{first['scene_order']:02d}-{first['scene_id']}-target-{first['target_id']:08d}"
    first_da3_source = first_out / "da3-source"
    chosen_process_res: int | None = None
    for process_res in PROCESS_RES_LEVELS:
        if first_da3_source.exists():
            shutil.rmtree(first_da3_source)
        command = [
            str(python),
            "-m",
            "refworld.runners.da3_source",
            "--da3-root",
            str(da3_root),
            "--reference",
            str(first["anchor_path"]),
            "--output",
            str(first_da3_source),
            "--process-res",
            str(process_res),
        ]
        code, output_text = run_capture(
            f"Scene {first['scene_order']}/10: DA3-BASE hardware probe at process_res={process_res}",
            command,
            cwd=repo_root,
        )
        if code == 0:
            chosen_process_res = process_res
            break
        if not is_cuda_oom(output_text):
            raise RuntimeError(
                f"DA3 scene-{first['scene_order']} failed for a non-OOM reason at process_res={process_res}; "
                "frozen protocol forbids changing resolution for this failure"
            )
        print(f"CUDA OOM at process_res={process_res}; frozen protocol permits the next lower level.", flush=True)
    if chosen_process_res is None:
        raise RuntimeError("DA3 OOM at all predeclared process resolutions 504, 392, and 336")
    print(f"\nFROZEN DA3 PROCESS_RES FOR ALL NINE SCENES: {chosen_process_res}", flush=True)

    rows: list[dict[str, Any]] = []
    for record in scene_records:
        scene_order = int(record["scene_order"])
        scene_id = str(record["scene_id"])
        target_id = int(record["target_id"])
        scene_out = out_root / f"scene-{scene_order:02d}-{scene_id}-target-{target_id:08d}"
        da3_source = scene_out / "da3-source"
        vggt_equalized = scene_out / "vggt-equalized"
        da3_learned = scene_out / "da3-learned"
        score_path = scene_out / "g1b-score.json"

        if scene_order != int(first["scene_order"]):
            run_checked(
                f"Scene {scene_order}/10: pinned DA3-BASE source geometry",
                [
                    str(python),
                    "-m",
                    "refworld.runners.da3_source",
                    "--da3-root",
                    str(da3_root),
                    "--reference",
                    str(record["anchor_path"]),
                    "--output",
                    str(da3_source),
                    "--process-res",
                    str(chosen_process_res),
                ],
                cwd=repo_root,
            )
        if not (da3_source / "source-geometry.safe.json").is_file():
            raise RuntimeError(f"scene {scene_order}: DA3 source geometry missing")

        run_checked(
            f"Scene {scene_order}/10: equalized VGGT all-valid scalar warp",
            [
                str(python),
                "-m",
                "refworld.runners.blendedmvs_learned_allvalid_pair",
                "--scene-root",
                str(record["scene_root"]),
                "--source-geometry",
                str(record["vggt_source"]),
                "--output",
                str(vggt_equalized),
                "--held-out-rank",
                str(RANK),
            ],
            cwd=repo_root,
        )
        run_checked(
            f"Scene {scene_order}/10: DA3-BASE all-valid scalar warp",
            [
                str(python),
                "-m",
                "refworld.runners.blendedmvs_learned_allvalid_pair",
                "--scene-root",
                str(record["scene_root"]),
                "--source-geometry",
                str(da3_source / "source-geometry.safe.json"),
                "--output",
                str(da3_learned),
                "--held-out-rank",
                str(RANK),
            ],
            cwd=repo_root,
        )
        run_checked(
            f"Scene {scene_order}/10: score frozen DA3-BASE geometry screen",
            [
                str(python),
                "-m",
                "refworld.runners.score_blendedmvs_g1b",
                "--scene-root",
                str(record["scene_root"]),
                "--vggt-equalized",
                str(vggt_equalized),
                "--da3",
                str(da3_learned),
                "--oracle-output",
                str(record["oracle_output"]),
                "--output",
                str(score_path),
            ],
            cwd=repo_root,
        )
        score = json.loads(score_path.read_text(encoding="utf-8"))
        contrasts = score["contrasts_common_observed_psnr_db"]
        rows.append(
            {
                "frozen_scene_order": scene_order,
                "scene_id": scene_id,
                "anchor_view_id": int(record["anchor_id"]),
                "target_view_id": target_id,
                "common_observed_fraction": float(score["support"]["all_three_common_observed_fraction"]),
                "vggt_equalized_minus_oracle_db": float(contrasts["vggt_equalized_minus_oracle"]),
                "da3_base_minus_oracle_db": float(contrasts["da3_base_minus_oracle"]),
                "da3_base_minus_vggt_equalized_db": float(contrasts["da3_base_minus_vggt_equalized"]),
                "score": score_path.relative_to(repo_root).as_posix(),
            }
        )

    med_vggt_oracle = median([row["vggt_equalized_minus_oracle_db"] for row in rows])
    med_da3_oracle = median([row["da3_base_minus_oracle_db"] for row in rows])
    med_da3_vggt = median([row["da3_base_minus_vggt_equalized_db"] for row in rows])
    wins = sum(row["da3_base_minus_vggt_equalized_db"] > 0.0 for row in rows)
    passed = (
        med_da3_oracle > PASS_ORACLE_GAP_DB
        and med_da3_vggt >= PASS_VGGT_GAIN_DB
        and wins >= PASS_MIN_WINS
    )
    decision = "da3-pass-freeze-for-rank4-design" if passed else "da3-fail-stop-bounded-geometry-search"

    aggregate = {
        "version": "0.1",
        "stage": "refworld-exp002-g1b-da3-base-aggregate",
        "scope": {
            "opened_rank3_reuse_only": True,
            "fresh_target_consumed": False,
            "target_depth_read": False,
            "rank4_touched": False,
            "da3_code_commit": DA3_PIN,
            "da3_process_res": chosen_process_res,
            "process_res_method": "upper_bound_resize",
        },
        "frozen_pass_rule": {
            "median_da3_minus_oracle_gt_db": PASS_ORACLE_GAP_DB,
            "median_da3_minus_vggt_gte_db": PASS_VGGT_GAIN_DB,
            "minimum_da3_scene_wins": PASS_MIN_WINS,
        },
        "aggregate": {
            "median_vggt_equalized_minus_oracle_db": med_vggt_oracle,
            "median_da3_base_minus_oracle_db": med_da3_oracle,
            "median_da3_base_minus_vggt_equalized_db": med_da3_vggt,
            "da3_scene_wins": wins,
            "scene_count": len(rows),
        },
        "frozen_rule_decision": decision,
        "scenes": rows,
    }
    aggregate_path = out_root / "G1B-DA3-RANK3.json"
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("\nG1-B DA3-BASE SCREEN COMPLETE", flush=True)
    print("scene target  VGGT-oracle dB  DA3-oracle dB  DA3-VGGT dB", flush=True)
    for row in rows:
        print(
            f"{row['frozen_scene_order']:5d} {row['target_view_id']:6d} "
            f"{row['vggt_equalized_minus_oracle_db']:+15.4f} "
            f"{row['da3_base_minus_oracle_db']:+14.4f} "
            f"{row['da3_base_minus_vggt_equalized_db']:+13.4f}",
            flush=True,
        )
    print(f"Median equalized VGGT - oracle: {med_vggt_oracle:+.4f} dB", flush=True)
    print(f"Median DA3-BASE - oracle:        {med_da3_oracle:+.4f} dB", flush=True)
    print(f"Median DA3-BASE - VGGT:          {med_da3_vggt:+.4f} dB", flush=True)
    print(f"DA3-BASE scene wins: {wins}/9", flush=True)
    print(f"Frozen-rule decision: {decision}", flush=True)
    print(f"Aggregate: {aggregate_path}", flush=True)
    print("Rank-4 remains sealed and untouched.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
