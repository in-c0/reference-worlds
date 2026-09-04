#!/usr/bin/env python3
"""Run frozen EXP-002 G1-A MoGe-2 geometry screen on Windows.

Uses only the already-opened rank-3 development targets from G1. It compares
MoGe-2 ViT-B against an equalized VGGT baseline and oracle geometry under the
same all-valid one-scalar oracle depth bridge. Rank 4 is never materialized.
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
MOGE_PIN = "925b8ed835a7a9cdb7578ba15c658a0afc969030"
UTILS3D_PIN = "3fab839f0be9931dac7c8488eb0e1600c236e183"
MOGE_LEVELS = (9, 7, 5)
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
        "not enough memory",
        "cublas_status_alloc_failed",
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


def ensure_moge_checkout(repo_root: Path) -> Path:
    upstream = repo_root / ".upstream"
    upstream.mkdir(parents=True, exist_ok=True)
    moge_root = upstream / "moge"
    if not (moge_root / ".git").exists():
        run_checked(
            "Cloning MoGe upstream",
            ["git", "clone", "https://github.com/microsoft/MoGe.git", str(moge_root)],
            cwd=repo_root,
        )
    run_checked(
        "Fetching pinned MoGe upstream",
        ["git", "-C", str(moge_root), "fetch", "--all", "--tags"],
        cwd=repo_root,
    )
    run_checked(
        "Checking out frozen MoGe-2 code",
        ["git", "-C", str(moge_root), "checkout", "--detach", MOGE_PIN],
        cwd=repo_root,
    )
    head = subprocess.check_output(["git", "-C", str(moge_root), "rev-parse", "HEAD"], text=True).strip()
    if head != MOGE_PIN:
        raise RuntimeError(f"MoGe checkout is {head}; expected {MOGE_PIN}")
    return moge_root


def ensure_runtime_dependencies(python: Path, repo_root: Path) -> None:
    probe = subprocess.run(
        [str(python), "-c", "import huggingface_hub, utils3d; print('MoGe runtime dependencies present')"],
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
    print("MoGe runtime dependency probe failed; installing frozen utils3d dependency.", flush=True)
    run_checked(
        "Installing pinned utils3d runtime dependency",
        [
            str(python),
            "-m",
            "pip",
            "install",
            f"git+https://github.com/EasternJournalist/utils3d.git@{UTILS3D_PIN}",
        ],
        cwd=repo_root,
    )
    run_checked(
        "Verifying MoGe runtime dependencies",
        [str(python), "-c", "import huggingface_hub, utils3d; print('MoGe runtime dependencies present')"],
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
        "Compiling G1-A scripts",
        [
            str(python),
            "-m",
            "py_compile",
            "src/refworld/runners/moge2_source.py",
            "src/refworld/runners/blendedmvs_learned_allvalid_pair.py",
            "src/refworld/runners/score_blendedmvs_g1a.py",
            "scripts/run-windows-blendedmvs-g1a-moge2.py",
        ],
        cwd=repo_root,
    )
    run_checked(
        "Running focused G1-A geometry tests",
        [
            str(python),
            "-m",
            "pytest",
            "-q",
            "tests/test_moge2_source.py",
            "tests/test_geometry_scale.py",
            "tests/test_source_geometry.py",
            "tests/test_pinhole_warp.py",
            "tests/test_mvsnet_dataset.py",
        ],
        cwd=repo_root,
    )

    moge_root = ensure_moge_checkout(repo_root)
    ensure_runtime_dependencies(python, repo_root)

    frozen = load_manifest(repo_root / "datasets" / "blendedmvs-bootstrap-v0.json")
    selected = selected_scenes(frozen)
    data_root = repo_root / "private-data" / "blendedmvs-bootstrap"
    g1_root = repo_root / "outputs" / "calibrated" / "blendedmvs" / "vggt-g1-rank3-392"
    out_root = repo_root / "outputs" / "calibrated" / "blendedmvs" / "g1a-moge2-rank3"
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

    print("\nEXP-002 G1-A: MoGe-2 ViT-B geometry screen", flush=True)
    print("Data: already-opened rank-3 scenes 2-10; rank-4 untouched", flush=True)
    print("Scale bridge: all-valid one oracle multiplicative scalar for BOTH VGGT and MoGe", flush=True)
    print("Primary support: OBSERVED intersection of oracle + equalized VGGT + MoGe", flush=True)
    print(
        f"PASS: median MoGe-oracle > {PASS_ORACLE_GAP_DB:+.1f} dB; "
        f"median MoGe-VGGT >= {PASS_VGGT_GAIN_DB:+.1f} dB; wins >= {PASS_MIN_WINS}/9",
        flush=True,
    )

    # Hardware-fit probe on the first predeclared scene. The first level that
    # succeeds is frozen for every remaining scene. Only CUDA OOM permits fallback.
    first = scene_records[0]
    first_out = out_root / f"scene-{first['scene_order']:02d}-{first['scene_id']}-target-{first['target_id']:08d}"
    first_moge_source = first_out / "moge-source"
    chosen_level: int | None = None
    for level in MOGE_LEVELS:
        if first_moge_source.exists():
            shutil.rmtree(first_moge_source)
        command = [
            str(python),
            "-m",
            "refworld.runners.moge2_source",
            "--moge-root",
            str(moge_root),
            "--reference",
            str(first["anchor_path"]),
            "--output",
            str(first_moge_source),
            "--resolution-level",
            str(level),
        ]
        code, output_text = run_capture(
            f"Scene {first['scene_order']}/10: MoGe-2 hardware probe at resolution_level={level}",
            command,
            cwd=repo_root,
        )
        if code == 0:
            chosen_level = level
            break
        if not is_cuda_oom(output_text):
            raise RuntimeError(
                f"MoGe-2 scene-{first['scene_order']} failed for a non-OOM reason at resolution_level={level}; "
                "frozen protocol forbids changing resolution for this failure"
            )
        print(f"CUDA OOM at resolution_level={level}; frozen protocol permits the next lower level.", flush=True)
    if chosen_level is None:
        raise RuntimeError("MoGe-2 OOM at all predeclared resolution levels 9, 7, and 5")
    print(f"\nFROZEN MOGE RESOLUTION LEVEL FOR ALL NINE SCENES: {chosen_level}", flush=True)

    rows: list[dict[str, Any]] = []
    for record in scene_records:
        scene_order = int(record["scene_order"])
        scene_id = str(record["scene_id"])
        target_id = int(record["target_id"])
        scene_out = out_root / f"scene-{scene_order:02d}-{scene_id}-target-{target_id:08d}"
        moge_source = scene_out / "moge-source"
        vggt_equalized = scene_out / "vggt-equalized"
        moge_learned = scene_out / "moge-learned"
        score_path = scene_out / "g1a-score.json"

        if scene_order != int(first["scene_order"]):
            run_checked(
                f"Scene {scene_order}/10: pinned MoGe-2 source geometry",
                [
                    str(python),
                    "-m",
                    "refworld.runners.moge2_source",
                    "--moge-root",
                    str(moge_root),
                    "--reference",
                    str(record["anchor_path"]),
                    "--output",
                    str(moge_source),
                    "--resolution-level",
                    str(chosen_level),
                ],
                cwd=repo_root,
            )
        if not (moge_source / "source-geometry.safe.json").is_file():
            raise RuntimeError(f"scene {scene_order}: MoGe-2 source geometry missing")

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
            f"Scene {scene_order}/10: MoGe-2 all-valid scalar warp",
            [
                str(python),
                "-m",
                "refworld.runners.blendedmvs_learned_allvalid_pair",
                "--scene-root",
                str(record["scene_root"]),
                "--source-geometry",
                str(moge_source / "source-geometry.safe.json"),
                "--output",
                str(moge_learned),
                "--held-out-rank",
                str(RANK),
            ],
            cwd=repo_root,
        )
        run_checked(
            f"Scene {scene_order}/10: score frozen MoGe-2 geometry screen",
            [
                str(python),
                "-m",
                "refworld.runners.score_blendedmvs_g1a",
                "--scene-root",
                str(record["scene_root"]),
                "--vggt-equalized",
                str(vggt_equalized),
                "--moge",
                str(moge_learned),
                "--oracle-output",
                str(record["oracle_output"]),
                "--output",
                str(score_path),
            ],
            cwd=repo_root,
        )
        score = json.loads(score_path.read_text(encoding="utf-8"))
        contrasts = score["contrasts_common_observed_psnr_db"]
        source_meta = json.loads((moge_source / "source-geometry.safe.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "frozen_scene_order": scene_order,
                "scene_id": scene_id,
                "target_view_id": target_id,
                "common_observed_fraction": float(score["support"]["all_three_common_observed_fraction"]),
                "moge_valid_fraction": float(
                    np_mean_from_manifest(source_meta)
                ),
                "vggt_equalized_minus_oracle_db": float(contrasts["vggt_equalized_minus_oracle"]),
                "moge2_minus_oracle_db": float(contrasts["moge2_minus_oracle"]),
                "moge2_minus_vggt_equalized_db": float(contrasts["moge2_minus_vggt_equalized"]),
                "score": score_path.relative_to(repo_root).as_posix(),
            }
        )

    median_vggt_oracle = median([row["vggt_equalized_minus_oracle_db"] for row in rows])
    median_moge_oracle = median([row["moge2_minus_oracle_db"] for row in rows])
    median_moge_vggt = median([row["moge2_minus_vggt_equalized_db"] for row in rows])
    wins = sum(row["moge2_minus_vggt_equalized_db"] > 0.0 for row in rows)
    passed = (
        median_moge_oracle > PASS_ORACLE_GAP_DB
        and median_moge_vggt >= PASS_VGGT_GAIN_DB
        and wins >= PASS_MIN_WINS
    )
    decision = "moge2-pass-freeze-for-rank4" if passed else "moge2-fail-proceed-da3-development"

    aggregate = {
        "version": "0.1",
        "stage": "refworld-exp002-g1a-moge2-aggregate",
        "scope": {
            "opened_rank3_reuse_only": True,
            "fresh_target_consumed": False,
            "target_depth_read": False,
            "rank4_touched": False,
            "oracle_depth_scale_scalar": True,
            "oracle_anchor_frame_placement": True,
            "end_to_end_single_image_claim": False,
        },
        "moge_resolution_level": chosen_level,
        "frozen_pass_rule": {
            "median_moge_minus_oracle_strictly_greater_than_db": PASS_ORACLE_GAP_DB,
            "median_moge_minus_vggt_min_db": PASS_VGGT_GAIN_DB,
            "minimum_scene_wins": PASS_MIN_WINS,
        },
        "aggregate": {
            "median_vggt_equalized_minus_oracle_db": median_vggt_oracle,
            "median_moge2_minus_oracle_db": median_moge_oracle,
            "median_moge2_minus_vggt_equalized_db": median_moge_vggt,
            "moge2_scene_wins_vs_vggt_equalized": wins,
            "passed": passed,
            "decision": decision,
        },
        "scenes": rows,
    }
    aggregate_path = out_root / "G1A-MOGE2-RANK3.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    print("\nG1-A MOGE-2 SCREEN COMPLETE", flush=True)
    print("scene target  VGGT-oracle dB  MoGe-oracle dB  MoGe-VGGT dB", flush=True)
    for row in rows:
        print(
            f"{row['frozen_scene_order']:5d} {row['target_view_id']:6d} "
            f"{row['vggt_equalized_minus_oracle_db']:+15.4f} "
            f"{row['moge2_minus_oracle_db']:+14.4f} "
            f"{row['moge2_minus_vggt_equalized_db']:+13.4f}",
            flush=True,
        )
    print(f"Median equalized VGGT - oracle: {median_vggt_oracle:+.4f} dB", flush=True)
    print(f"Median MoGe-2 - oracle:          {median_moge_oracle:+.4f} dB", flush=True)
    print(f"Median MoGe-2 - equalized VGGT: {median_moge_vggt:+.4f} dB", flush=True)
    print(f"MoGe-2 scene wins: {wins}/9", flush=True)
    print(f"Frozen-rule decision: {decision}", flush=True)
    print(f"Aggregate: {aggregate_path}", flush=True)
    print("Rank-4 remains sealed and untouched.", flush=True)
    return 0


def np_mean_from_manifest(meta: dict[str, Any]) -> float:
    """Read candidate-valid fraction from the later scale bridge when available.

    The source manifest records the mask as an artifact, not an aggregate number.
    Keep this source-level field informational; the primary support is reported by
    the scorer. Returning NaN is avoided because aggregate JSON forbids it.
    """
    geometry = meta.get("geometry")
    if isinstance(geometry, dict) and "valid_fraction" in geometry:
        try:
            return float(geometry["valid_fraction"])
        except (TypeError, ValueError):
            pass
    return -1.0


if __name__ == "__main__":
    raise SystemExit(main())
