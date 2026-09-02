#!/usr/bin/env python3
"""Run the first machine-fit RefWorld B-vs-C novel-view repaint experiment on Windows.

This script reuses the latest successful VGGT smoke artifacts. It does *not*
rerun VGGT. It selects the predeclared +5 degree yaw warp, installs the pinned
lightweight SD2 inpainting stack into the existing RefWorld venv if needed,
generates one fixed-seed candidate, applies the existing evidence compositor,
creates a visual comparison sheet, and opens that sheet automatically.

This is an engineering/scientific baseline, not the final quality backend.
WorldForge + LongCat remains the intended higher-compute comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

MODEL_BACKEND_ID = "sd2-community-stable-diffusion-2-inpainting-openrailpp"
SEED = 42
TARGET_YAW_DEG = 5.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run first local RefWorld +5deg B-vs-C repaint candidate")
    parser.add_argument("--run", default="latest", help="successful windows-smoke run directory or 'latest'")
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--skip-dependency-install", action="store_true")
    parser.add_argument("--no-open", action="store_true", help="do not automatically open the result sheet")
    return parser.parse_args()


def run_checked(label: str, command: list[str], *, cwd: Path) -> None:
    print(f"\n== {label} ==", flush=True)
    print(" ".join(command), flush=True)
    completed = subprocess.run(command, cwd=cwd)
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {completed.returncode}")


def find_run(repo_root: Path, requested: str) -> Path:
    if requested != "latest":
        candidate = Path(requested)
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        candidate = candidate.resolve()
        if not candidate.is_dir():
            raise FileNotFoundError(candidate)
        return candidate

    base = repo_root / "outputs" / "windows-smoke"
    if not base.is_dir():
        raise FileNotFoundError(f"no Windows smoke output directory: {base}")
    candidates = [
        path
        for path in base.iterdir()
        if path.is_dir()
        and (path / "source-geometry" / "source-geometry.safe.json").is_file()
        and (path / "warp-only" / "warp-only.safe.json").is_file()
        and (path / "source-splat" / "anchor-score.json").is_file()
    ]
    if not candidates:
        raise RuntimeError("no successful smoke run with anchor score + warp-only outputs was found")
    return max(candidates, key=lambda p: p.stat().st_mtime).resolve()


def locate_reference(manifest: dict, explicit: Path | None) -> Path:
    expected_name = str(manifest["input"]["file_name"])
    expected_sha = str(manifest["input"]["sha256"])
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != expected_sha:
            raise RuntimeError("explicit reference SHA-256 does not match source-geometry manifest")
        return path

    home = Path.home()
    roots = [
        home / "Desktop",
        home / "Pictures",
        home / "Downloads",
        home / "OneDrive" / "Desktop",
        home / "OneDrive" / "Pictures",
        home,
    ]
    seen: set[Path] = set()
    for root in roots:
        candidate = (root / expected_name).resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file() and sha256_file(candidate) == expected_sha:
            return candidate
    raise RuntimeError("could not auto-locate original reference; pass --reference C:\\path\\to\\image.jpg")


def select_target(warp_manifest: dict) -> dict:
    matches = [
        view
        for view in warp_manifest.get("views", [])
        if view.get("displacement_kind") == "yaw_deg"
        and abs(float(view.get("displacement_value", 1e9)) - TARGET_YAW_DEG) < 1e-9
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one +{TARGET_YAW_DEG:g}deg yaw view, found {len(matches)}")
    return matches[0]


def normalized_mae(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32)) / 255.0
    return float(np.mean(diff[mask]))


def make_sheet(reference: Path, warp_path: Path, b_path: Path, c_path: Path, output: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    entries = [
        ("SOURCE C0 (real reference)", reference),
        ("+5 deg WARP ONLY (holes unresolved)", warp_path),
        ("B: UNRESTRICTED REPAINT", b_path),
        ("C: EVIDENCE-PRESERVED REPAINT", c_path),
    ]
    tile_w = 620
    tile_h = 620
    label_h = 44
    sheet = Image.new("RGB", (tile_w * 2, (tile_h + label_h) * 2), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for index, (label, path) in enumerate(entries):
        image = Image.open(path).convert("RGB")
        image.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        row = index // 2
        col = index % 2
        x0 = col * tile_w
        y0 = row * (tile_h + label_h)
        image_x = x0 + (tile_w - image.width) // 2
        image_y = y0 + label_h + (tile_h - image.height) // 2
        sheet.paste(image, (image_x, image_y))
        draw.text((x0 + 12, y0 + 14), label, fill="black", font=font)

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    if os.name != "nt":
        raise RuntimeError("this convenience orchestrator is intentionally Windows-only")

    run_dir = find_run(repo_root, args.run)
    geometry_path = run_dir / "source-geometry" / "source-geometry.safe.json"
    warp_manifest_path = run_dir / "warp-only" / "warp-only.safe.json"
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    warp_manifest = json.loads(warp_manifest_path.read_text(encoding="utf-8"))
    reference = locate_reference(geometry, args.reference)
    target = select_target(warp_manifest)
    view_id = str(target["view_id"])
    warp_view = run_dir / "warp-only" / view_id
    if not warp_view.is_dir():
        raise FileNotFoundError(warp_view)

    venv_python = repo_root / ".venv-refworld" / "Scripts" / "python.exe"
    if not venv_python.is_file():
        raise RuntimeError(f"RefWorld venv not found: {venv_python}")

    anchor = json.loads((run_dir / "source-splat" / "anchor-score.json").read_text(encoding="utf-8"))
    print("RefWorld first local novel-view candidate", flush=True)
    print(f"Run:            {run_dir}", flush=True)
    print(f"Reference:      {reference}", flush=True)
    print(f"Target:         +{TARGET_YAW_DEG:g} deg yaw ({view_id})", flush=True)
    print(f"Anchor PSNR:    {anchor['metrics'].get('psnr')}", flush=True)
    print("VGGT will NOT be rerun.", flush=True)
    print("Held-out evaluation RGB is NOT used.", flush=True)
    print("Frozen repaint config: seed=42, steps=30, guidance=4.0, context dilation=16px, max side=512.", flush=True)

    if not args.skip_dependency_install:
        run_checked(
            "Installing/verifying pinned lightweight repaint dependencies",
            [str(venv_python), "-m", "pip", "install", "-e", ".[repaint-sd2]"],
            cwd=repo_root,
        )

    candidate_dir = run_dir / "candidates" / "yaw-plus-5-sd2"
    composition_dir = candidate_dir / "composition"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    run_checked(
        "Generating fixed +5deg SD2 repaint candidate",
        [
            str(venv_python),
            "-m",
            "refworld.runners.sd2_inpaint_candidate",
            "--warp-view",
            str(warp_view),
            "--output",
            str(candidate_dir),
            "--seed",
            str(SEED),
        ],
        cwd=repo_root,
    )

    backend_manifest = json.loads((candidate_dir / "sd2-inpaint.safe.json").read_text(encoding="utf-8"))
    revision = str(backend_manifest["backend"]["resolved_revision"])
    run_checked(
        "Applying RefWorld evidence-preserving compositor (B vs C)",
        [
            str(venv_python),
            "-m",
            "refworld.runners.compose_candidate",
            "--warp-view",
            str(warp_view),
            "--candidate",
            str(candidate_dir / "candidate.png"),
            "--valid-mask-npy",
            str(candidate_dir / "repaint-valid-mask.npy"),
            "--output",
            str(composition_dir),
            "--backend",
            MODEL_BACKEND_ID,
            "--seed",
            str(SEED),
            "--backend-run-id",
            revision,
        ],
        cwd=repo_root,
    )

    from PIL import Image

    warp_rgb = np.asarray(Image.open(warp_view / "proposal.png").convert("RGB"), dtype=np.uint8)
    b_path = composition_dir / "candidate-unrestricted.png"
    c_path = composition_dir / "proposal-evidence-preserved.png"
    b_rgb = np.asarray(Image.open(b_path).convert("RGB"), dtype=np.uint8)
    c_rgb = np.asarray(Image.open(c_path).convert("RGB"), dtype=np.uint8)
    provenance = np.load(warp_view / "provenance.npy", allow_pickle=False)
    valid = np.load(candidate_dir / "repaint-valid-mask.npy", allow_pickle=False)
    observed = provenance == 1
    unresolved = provenance == 0
    overlap = valid & observed

    summary = {
        "version": "0.1",
        "experiment": "first-local-refworld-b-vs-c",
        "target": {
            "view_id": view_id,
            "displacement_kind": "yaw_deg",
            "displacement_value": TARGET_YAW_DEG,
        },
        "source_anchor_psnr": anchor["metrics"].get("psnr"),
        "backend": backend_manifest["backend"],
        "configuration": backend_manifest["configuration"],
        "metrics_without_held_out_rgb": {
            "observed_overlap_fraction": float(np.mean(overlap)),
            "observed_overlap_pixels": int(np.sum(overlap)),
            "unresolved_fraction": float(np.mean(unresolved)),
            "B_change_mae_on_observed_overlap_vs_geometric_warp": normalized_mae(b_rgb, warp_rgb, overlap),
            "C_change_mae_on_observed_overlap_vs_geometric_warp": normalized_mae(c_rgb, warp_rgb, overlap),
            "C_observed_pixels_bitwise_equal_to_geometric_warp": bool(np.array_equal(c_rgb[observed], warp_rgb[observed])),
            "B_and_C_equal_on_unresolved_support": bool(np.array_equal(b_rgb[unresolved], c_rgb[unresolved])),
        },
        "interpretation": {
            "held_out_quality_evaluated": False,
            "claim": "structural B-vs-C evidence-preservation smoke only; visual plausibility is inspectable but not a calibrated novel-view score",
        },
    }
    summary_path = candidate_dir / "first-candidate-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    sheet_path = candidate_dir / "FIRST-CANDIDATE-COMPARISON.png"
    make_sheet(reference, warp_view / "proposal.png", b_path, c_path, sheet_path)

    print("\nFIRST LOCAL NOVEL-VIEW CANDIDATE COMPLETE", flush=True)
    print(f"Comparison: {sheet_path}", flush=True)
    print(f"B:          {b_path}", flush=True)
    print(f"C:          {c_path}", flush=True)
    print(f"Summary:    {summary_path}", flush=True)
    for key, value in summary["metrics_without_held_out_rgb"].items():
        print(f"{key}: {value}", flush=True)

    if not args.no_open:
        try:
            os.startfile(sheet_path)  # type: ignore[attr-defined]
            print("Opened comparison image in the default Windows image viewer.", flush=True)
        except OSError as exc:
            print(f"Could not auto-open comparison image: {exc}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
