#!/usr/bin/env python3
"""Run the frozen G1-R0 VGGT 392-vs-518 resolution control on Windows.

Uses only frozen scene 2 and its already-opened rank-3 target. No target or depth
materialization occurs. The low-memory path converts pinned VGGT weights to FP16.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

SCENE_ID = "5c189f2326173c3a09ed7ef3"
SCENE_ORDER = 2
ANCHOR_ID = 0
TARGET_ID = 27
RANK = 3
VGGT_PIN = "a288dd0f14786c93483e45524328726ab7b1b4ce"


def run_checked(label: str, command: list[str], *, cwd: Path) -> None:
    print(f"\n== {label} ==", flush=True)
    print(" ".join(command), flush=True)
    result = subprocess.run(command, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    python = repo_root / ".venv-refworld" / "Scripts" / "python.exe"
    vggt_root = repo_root / ".upstream" / "vggt"
    if not python.is_file():
        raise RuntimeError(f"RefWorld venv missing: {python}")
    if not (vggt_root / ".git").exists():
        raise RuntimeError(f"pinned VGGT checkout missing: {vggt_root}")
    head = subprocess.check_output(["git", "-C", str(vggt_root), "rev-parse", "HEAD"], text=True).strip()
    if head != VGGT_PIN:
        raise RuntimeError(f"VGGT checkout is {head}; expected {VGGT_PIN}")

    scene_root = repo_root / "private-data" / "blendedmvs-bootstrap" / SCENE_ID
    anchor = scene_root / "blended_images" / f"{ANCHOR_ID:08d}.jpg"
    target = scene_root / "blended_images" / f"{TARGET_ID:08d}.jpg"
    target_meta = scene_root / "vggt-g1-rank3-target-rgb.safe.json"
    if not anchor.is_file() or not target.is_file() or not target_meta.is_file():
        raise RuntimeError("G1-R0 requires the already-opened scene-2 rank-3 G1 artifacts")
    meta = json.loads(target_meta.read_text(encoding="utf-8"))
    if int(meta["target_view_id"]) != TARGET_ID or bool(meta.get("target_depth_materialized")):
        raise RuntimeError("scene-2 rank-3 target state does not match frozen G1-R0 protocol")

    original_root = repo_root / "outputs" / "calibrated" / "blendedmvs" / "vggt-g1-rank3-392" / f"scene-02-{SCENE_ID}-target-{TARGET_ID:08d}"
    original_learned = original_root / "learned-warp"
    oracle_output = original_root / "oracle-warp"
    if not (original_learned / "vggt-oracle-scale-pair.safe.json").is_file() or not (oracle_output / "oracle-pair.safe.json").is_file():
        raise RuntimeError("original G1 learned/oracle outputs missing")

    control_root = repo_root / "outputs" / "calibrated" / "blendedmvs" / "vggt-g1r0-resolution-control"
    low392_source = control_root / "392-lowmem" / "source-geometry"
    low392_learned = control_root / "392-lowmem" / "learned-warp"
    low518_source = control_root / "518-lowmem" / "source-geometry"
    low518_learned = control_root / "518-lowmem" / "learned-warp"
    score_path = control_root / "G1R0-SCENE2-392-VS-518.json"
    control_root.mkdir(parents=True, exist_ok=True)

    run_checked(
        "Checking CUDA environment",
        [str(python), "-c", "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"],
        cwd=repo_root,
    )
    run_checked(
        "Compiling G1-R0 scripts",
        [str(python), "-m", "py_compile", "src/refworld/runners/vggt_source_lowmem.py", "src/refworld/runners/score_blendedmvs_g1r0.py", "scripts/run-windows-blendedmvs-g1r0.py"],
        cwd=repo_root,
    )
    run_checked(
        "Running focused source-geometry/scale/warp tests",
        [str(python), "-m", "pytest", "-q", "tests/test_vggt_source.py", "tests/test_source_geometry.py", "tests/test_geometry_scale.py", "tests/test_pinhole_warp.py"],
        cwd=repo_root,
    )

    print("\nEXP-002 G1-R0: first frozen scene only; rank-3 target already open; rank-4 untouched", flush=True)
    print("Equivalence guard: |392-lowmem - 392-original| <= 0.25 dB", flush=True)
    print("Strong resolution rescue: +3 dB vs 392-lowmem AND > -3 dB from oracle", flush=True)

    for size, source_out, learned_out in (
        (392, low392_source, low392_learned),
        (518, low518_source, low518_learned),
    ):
        run_checked(
            f"Pinned VGGT {size}x{size} with FP16 model weights",
            [str(python), "-m", "refworld.runners.vggt_source_lowmem", "--vggt-root", str(vggt_root), "--reference", str(anchor), "--output", str(source_out), "--seed", "0", "--model-size", str(size)],
            cwd=repo_root,
        )
        run_checked(
            f"Create rank-3 learned warp from {size}px low-memory geometry",
            [str(python), "-m", "refworld.runners.blendedmvs_vggt_scaled_pair", "--scene-root", str(scene_root), "--source-geometry", str(source_out / "source-geometry.safe.json"), "--output", str(learned_out), "--held-out-rank", str(RANK)],
            cwd=repo_root,
        )

    run_checked(
        "Score frozen resolution control on all-four common OBSERVED support",
        [str(python), "-m", "refworld.runners.score_blendedmvs_g1r0", "--scene-root", str(scene_root), "--original-392", str(original_learned), "--lowmem-392", str(low392_learned), "--lowmem-518", str(low518_learned), "--oracle-output", str(oracle_output), "--output", str(score_path)],
        cwd=repo_root,
    )
    print("\nG1-R0 COMPLETE", flush=True)
    print(score_path, flush=True)
    print("No fresh target consumed; rank-4 remains sealed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
