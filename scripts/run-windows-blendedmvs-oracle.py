#!/usr/bin/env python3
"""Run the first calibrated BlendedMVS oracle-geometry B-vs-C experiment on Windows.

Pipeline:
  offline dataset-reader + camera-import contract tests
  -> selective official split-ZIP materialization (first frozen scene only)
  -> first pair record / first held-out view
  -> oracle anchor-depth calibrated warp (held-out RGB still sealed)
  -> frozen SD2 repaint candidate
  -> unrestricted B vs evidence-preserved C
  -> first read of held-out RGB for scoring
  -> comparison sheet + score JSON

VGGT is not rerun. This is explicitly an oracle-source-depth diagnostic, not a
claim about the complete single-image method.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run first calibrated BlendedMVS oracle B-vs-C experiment")
    parser.add_argument("--no-open", action="store_true")
    return parser.parse_args()


def run_checked(label: str, command: list[str], *, cwd: Path) -> None:
    print(f"\n== {label} ==", flush=True)
    print(" ".join(command), flush=True)
    completed = subprocess.run(command, cwd=cwd)
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {completed.returncode}")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    python = repo_root / ".venv-refworld" / "Scripts" / "python.exe"
    if not python.is_file():
        raise RuntimeError(f"RefWorld venv not found: {python}")

    frozen = json.loads((repo_root / "datasets" / "blendedmvs-bootstrap-v0.json").read_text(encoding="utf-8"))
    scene_id = str(frozen["scenes"][0]["id"])
    scene_root = repo_root / "private-data" / "blendedmvs-bootstrap" / scene_id
    experiment_root = repo_root / "outputs" / "calibrated" / "blendedmvs" / scene_id / "oracle-first-pair"
    oracle_output = experiment_root / "warp"
    candidate_output = experiment_root / "sd2"
    composition = candidate_output / "composition"
    score_path = experiment_root / "calibrated-score.json"
    sheet_path = experiment_root / "CALIBRATED-BVSC-COMPARISON.png"

    print("RefWorld calibrated first-pair experiment", flush=True)
    print(f"Frozen scene: {scene_id}", flush=True)
    print("View rule:    first pair.txt record -> first published held-out source", flush=True)
    print("Geometry:     ORACLE anchor depth/camera diagnostic", flush=True)
    print("Held-out RGB: sealed until scoring stage", flush=True)
    print("VGGT:         NOT rerun", flush=True)

    run_checked(
        "Installing/verifying dataset + frozen repaint dependencies",
        [str(python), "-m", "pip", "install", "-e", ".[dataset,repaint-sd2,dev]"],
        cwd=repo_root,
    )

    run_checked(
        "Running offline split-ZIP/PFM/camera-import contract tests",
        [
            str(python),
            "-m",
            "pytest",
            "-q",
            "tests/test_remote_zip.py",
            "tests/test_pfm.py",
            "tests/test_mvsnet_dataset.py",
        ],
        cwd=repo_root,
    )

    run_checked(
        "Selectively materializing first frozen BlendedMVS scene",
        [str(python), str(repo_root / "scripts" / "materialize-blendedmvs-first-scene.py")],
        cwd=repo_root,
    )

    run_checked(
        "Creating calibrated oracle-source-depth warp (held-out RGB still sealed)",
        [
            str(python),
            "-m",
            "refworld.runners.blendedmvs_oracle_pair",
            "--scene-root",
            str(scene_root),
            "--output",
            str(oracle_output),
        ],
        cwd=repo_root,
    )
    oracle = json.loads((oracle_output / "oracle-pair.safe.json").read_text(encoding="utf-8"))
    warp_view = oracle_output / str(oracle["result"]["view_directory"])

    run_checked(
        "Generating same frozen SD2 repaint candidate",
        [
            str(python),
            "-m",
            "refworld.runners.sd2_inpaint_candidate",
            "--warp-view",
            str(warp_view),
            "--output",
            str(candidate_output),
            "--seed",
            "42",
            "--steps",
            "30",
            "--guidance-scale",
            "4.0",
            "--context-radius",
            "16",
            "--max-side",
            "512",
        ],
        cwd=repo_root,
    )
    candidate_manifest = json.loads((candidate_output / "sd2-inpaint.safe.json").read_text(encoding="utf-8"))
    revision = str(candidate_manifest["backend"]["resolved_revision"])

    run_checked(
        "Applying evidence-preserving compositor (same candidate for B and C)",
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
            "42",
            "--backend-run-id",
            revision,
        ],
        cwd=repo_root,
    )

    run_checked(
        "Opening sealed held-out RGB for final scoring",
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
    target_id = int(oracle["selection"]["target_view_id"])
    anchor_id = int(oracle["selection"]["anchor_view_id"])
    target_path = scene_root / "blended_images" / f"{target_id:08d}.jpg"
    anchor_path = scene_root / "blended_images" / f"{anchor_id:08d}.jpg"
    geometry_path = warp_view / "proposal.png"
    b_path = composition / "candidate-unrestricted.png"
    c_path = composition / "proposal-evidence-preserved.png"

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("comparison sheet requires Pillow") from exc

    images = [
        ("ANCHOR INPUT", Image.open(anchor_path).convert("RGB")),
        ("HELD-OUT TARGET", Image.open(target_path).convert("RGB")),
        ("ORACLE GEOMETRY", Image.open(geometry_path).convert("RGB")),
        ("B UNRESTRICTED", Image.open(b_path).convert("RGB")),
        ("C EVIDENCE-PRESERVED", Image.open(c_path).convert("RGB")),
    ]
    thumb_w = 360
    thumbs = []
    for label, image in images:
        scale = thumb_w / image.width
        thumb = image.resize((thumb_w, max(1, int(round(image.height * scale)))), Image.Resampling.LANCZOS)
        thumbs.append((label, thumb))
    label_h = 48
    gap = 8
    height = max(im.height for _, im in thumbs) + label_h
    width = len(thumbs) * thumb_w + (len(thumbs) - 1) * gap
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    x = 0
    for label, image in thumbs:
        sheet.paste(image, (x, label_h))
        draw.text((x + 8, 12), label, fill="black", font=font)
        x += thumb_w + gap
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(sheet_path)

    contrasts = score["contrasts"]
    full = score["metrics"]["full_frame"]
    observed = score["metrics"]["observed_support"]
    print("\nCALIBRATED ORACLE B-vs-C EXPERIMENT COMPLETE", flush=True)
    print(f"Scene:        {scene_id}", flush=True)
    print(f"Anchor view:  {anchor_id}", flush=True)
    print(f"Target view:  {target_id}", flush=True)
    print(f"Comparison:   {sheet_path}", flush=True)
    print(f"Score:        {score_path}", flush=True)
    print(f"B PSNR full:  {full['B_unrestricted']['psnr']:.4f} dB", flush=True)
    print(f"C PSNR full:  {full['C_evidence_preserved']['psnr']:.4f} dB", flush=True)
    print(f"C-B full:     {contrasts['C_minus_B_psnr_full_db']:+.4f} dB", flush=True)
    print(f"B PSNR obs:   {observed['B_unrestricted']['psnr']:.4f} dB", flush=True)
    print(f"C PSNR obs:   {observed['C_evidence_preserved']['psnr']:.4f} dB", flush=True)
    print(f"C-B observed: {contrasts['C_minus_B_psnr_observed_db']:+.4f} dB", flush=True)
    print("Scope: oracle source depth/camera diagnostic; not full single-image RefWorld-0.", flush=True)
    print(f"Manual open:  Start-Process \"{sheet_path}\"", flush=True)

    if not args.no_open:
        try:
            subprocess.Popen(["cmd.exe", "/c", "start", "", str(sheet_path)], cwd=repo_root)
        except OSError as exc:
            print(f"Could not launch default image viewer: {exc}", flush=True)
            try:
                subprocess.Popen(["explorer.exe", "/select,", str(sheet_path)], cwd=repo_root)
            except OSError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
