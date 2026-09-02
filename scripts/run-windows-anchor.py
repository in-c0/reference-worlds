#!/usr/bin/env python3
"""Render and score the latest successful Windows RefWorld source-only splat.

This stage is deterministic and does not run VGGT again. It finds a completed
outputs/windows-smoke run, verifies the original reference by SHA-256, renders
source-splat.ply with the emitted source-camera.json through the pinned
Spark/Three/Playwright harness, and writes exact-view anchor metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render + score a completed RefWorld Windows smoke anchor")
    parser.add_argument(
        "--run",
        default="latest",
        help="completed smoke run directory or 'latest' (default)",
    )
    parser.add_argument("--reference", type=Path, help="original reference image; auto-located by name+SHA if omitted")
    parser.add_argument("--skip-npm-install", action="store_true")
    parser.add_argument("--skip-browser-install", action="store_true")
    return parser.parse_args()


def find_run(repo_root: Path, requested: str) -> Path:
    if requested != "latest":
        candidate = Path(requested)
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        candidate = candidate.resolve()
        if not candidate.is_dir():
            raise FileNotFoundError(f"run directory not found: {candidate}")
        return candidate

    base = repo_root / "outputs" / "windows-smoke"
    if not base.is_dir():
        raise FileNotFoundError(f"no Windows smoke output directory: {base}")
    candidates = [
        p
        for p in base.iterdir()
        if p.is_dir()
        and (p / "source-geometry" / "source-geometry.safe.json").is_file()
        and (p / "source-splat" / "source-splat.ply").is_file()
        and (p / "source-splat" / "source-camera.json").is_file()
    ]
    if not candidates:
        raise RuntimeError("no completed Windows smoke run with source geometry + splat was found")
    return max(candidates, key=lambda p: p.stat().st_mtime).resolve()


def locate_reference(manifest: dict, explicit: Path | None) -> Path:
    expected_name = str(manifest["input"]["file_name"])
    expected_sha = str(manifest["input"]["sha256"])

    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        if digest != expected_sha:
            raise RuntimeError(f"reference SHA-256 mismatch for {path}: {digest} != {expected_sha}")
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

    raise RuntimeError(
        "could not auto-locate the original reference by filename + SHA-256; "
        "rerun with --reference C:\\path\\to\\image.jpg"
    )


def require_command(*names: str) -> str:
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    raise RuntimeError(
        f"required command not found ({', '.join(names)}). Install Node.js LTS, then rerun; no GPU/model rerun is needed."
    )


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

    run_dir = find_run(repo_root, args.run)
    geometry_manifest_path = run_dir / "source-geometry" / "source-geometry.safe.json"
    splat_dir = run_dir / "source-splat"
    splat_path = splat_dir / "source-splat.ply"
    camera_path = splat_dir / "source-camera.json"
    render_path = splat_dir / "source-render.png"
    render_meta_path = splat_dir / "source-render.meta.json"
    score_path = splat_dir / "anchor-score.json"

    manifest = json.loads(geometry_manifest_path.read_text(encoding="utf-8"))
    reference = locate_reference(manifest, args.reference)
    width = int(manifest["input"]["width"])
    height = int(manifest["input"]["height"])

    print("RefWorld source-anchor diagnostic", flush=True)
    print(f"Run:       {run_dir}", flush=True)
    print(f"Reference: {reference}", flush=True)
    print(f"Canvas:    {width}x{height}", flush=True)
    print("VGGT will NOT be rerun.", flush=True)

    node = require_command("node.exe", "node")
    npm = require_command("npm.cmd", "npm")
    npx = require_command("npx.cmd", "npx")

    renderer_dir = repo_root / "renderer"
    if not args.skip_npm_install:
        run_checked("Installing/verifying pinned renderer dependencies", [npm, "install"], cwd=renderer_dir)
    if not args.skip_browser_install:
        run_checked("Installing/verifying Playwright Chromium", [npx, "playwright", "install", "chromium"], cwd=renderer_dir)

    def rel(path: Path) -> str:
        try:
            return path.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError as exc:
            raise RuntimeError(f"renderer artifact escaped repository root: {path}") from exc

    capture_command = [
        node,
        str(renderer_dir / "capture.mjs"),
        "--asset",
        rel(splat_path),
        "--camera",
        rel(camera_path),
        "--out",
        rel(render_path),
        "--width",
        str(width),
        "--height",
        str(height),
    ]
    print("\n== Rendering source-only PLY at emitted C0 ==", flush=True)
    capture = subprocess.run(capture_command, cwd=repo_root, text=True, capture_output=True)
    if capture.stdout:
        print(capture.stdout, end="" if capture.stdout.endswith("\n") else "\n", flush=True)
    if capture.stderr:
        print(capture.stderr, file=sys.stderr, end="" if capture.stderr.endswith("\n") else "\n", flush=True)
    if capture.returncode != 0:
        raise RuntimeError(f"Spark source-anchor capture failed with exit code {capture.returncode}")
    try:
        capture_meta = json.loads(capture.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("renderer succeeded but did not emit valid JSON metadata") from exc
    render_meta_path.write_text(json.dumps(capture_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    venv_python = repo_root / ".venv-refworld" / "Scripts" / "python.exe"
    if not venv_python.is_file():
        raise RuntimeError(f"RefWorld venv Python not found: {venv_python}")
    score_command = [
        str(venv_python),
        "-m",
        "refworld.runners.score_anchor",
        "--reference",
        str(reference),
        "--render",
        str(render_path),
        "--output",
        str(score_path),
    ]
    run_checked("Scoring exact source-anchor reconstruction", score_command, cwd=repo_root)

    score = json.loads(score_path.read_text(encoding="utf-8"))
    metrics = score["metrics"]
    print("\nSOURCE-ANCHOR DIAGNOSTIC COMPLETE", flush=True)
    print(f"Render: {render_path}", flush=True)
    print(f"Score:  {score_path}", flush=True)
    for key in sorted(metrics):
        print(f"{key}: {metrics[key]}", flush=True)
    print(
        "Interpretation: this measures same-camera camera/depth/splat/renderer fidelity only; "
        "it is not a hidden-view or generation-quality result.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
