#!/usr/bin/env python3
"""Native Windows RefWorld smoke runner.

This intentionally uses only the Python stdlib in the outer process so Windows
PowerShell version quirks cannot affect subprocess streaming or exit-code
handling. It reuses the repo-local venv, pinned VGGT checkout and Hugging Face
cache across runs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

VGGT_PIN = "a288dd0f14786c93483e45524328726ab7b1b4ce"
VGGT_REPO = "https://github.com/facebookresearch/vggt.git"


def run(cmd: list[str], *, cwd: Path, label: str | None = None, check: bool = True, env: dict[str, str] | None = None) -> int:
    if label:
        print(f"\n== {label} ==", flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd), env=env)
    if check and proc.returncode != 0:
        raise RuntimeError(f"{label or cmd[0]} failed with exit code {proc.returncode}")
    return proc.returncode


def run_stream(cmd: list[str], *, cwd: Path, log_path: Path, label: str, env: dict[str, str] | None = None) -> tuple[int, str]:
    print(f"\n== {label} ==", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
            lines.append(line)
        code = proc.wait()
    return code, "".join(lines)


def git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def detect_gpu() -> tuple[str, int]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()[0]
    name, memory = [part.strip() for part in out.split(",", 1)]
    return name, int(memory)


def choose_model_size(vram_mib: int, requested: str) -> int:
    if requested != "Auto":
        return int(requested)
    if vram_mib >= 11800:
        return 518
    if vram_mib >= 9800:
        return 448
    if vram_mib >= 7600:
        return 392
    if vram_mib >= 6000:
        return 336
    return 280


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the native Windows RefWorld VGGT smoke path")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--model-size", choices=["Auto", "518", "448", "392", "336", "280"], default="Auto")
    parser.add_argument("--run-name", default=time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script = Path(__file__).resolve()
    repo = script.parent.parent
    reference = args.reference.expanduser().resolve()

    if os.name != "nt":
        raise RuntimeError("this runner is for native Windows only")
    windows_root = Path(os.environ.get("WINDIR", r"C:\Windows")).resolve()
    try:
        repo.relative_to(windows_root)
    except ValueError:
        pass
    else:
        raise RuntimeError(f"refusing to run from {repo}; clone under your user profile instead")
    if not reference.is_file():
        raise FileNotFoundError(reference)
    if shutil.which("git") is None:
        raise RuntimeError("git is required")
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("nvidia-smi is required; install/update the NVIDIA driver")
    if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
        raise RuntimeError(f"Python 3.10-3.12 required; current interpreter is {sys.version.split()[0]}")

    gpu_name, vram_mib = detect_gpu()
    model_size = choose_model_size(vram_mib, args.model_size)
    print("== RefWorld native Windows GPU smoke ==")
    print(f"Repository: {repo}")
    print(f"Reference:  {reference}")
    print(f"GPU:        {gpu_name} ({vram_mib} MiB VRAM)")
    print(f"VGGT input: {model_size} x {model_size}")
    if model_size != 518:
        print("WARNING: reduced resolution is a hardware smoke configuration, not the frozen benchmark baseline.")

    output = repo / "outputs" / "windows-smoke" / args.run_name
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / f"gpu-run-{model_size}.log"

    venv = repo / ".venv-refworld"
    venv_python = venv / "Scripts" / "python.exe"
    if not venv_python.is_file():
        run([sys.executable, "-m", "venv", str(venv)], cwd=repo, label="Creating repo-local Python environment")

    run([str(venv_python), "-m", "pip", "install", "--disable-pip-version-check", "--upgrade", "pip", "setuptools", "wheel"], cwd=repo, label="Bootstrapping pip/setuptools/wheel")
    run(
        [str(venv_python), "-m", "pip", "install", "torch==2.3.1", "torchvision==0.18.1", "--index-url", "https://download.pytorch.org/whl/cu121"],
        cwd=repo,
        label="Ensuring pinned PyTorch 2.3.1 CUDA 12.1 environment",
    )
    run(
        [str(venv_python), "-m", "pip", "install", "-e", ".[dev,method]", "huggingface_hub", "einops", "safetensors"],
        cwd=repo,
        label="Installing pinned RefWorld/VGGT dependencies",
    )
    run(
        [
            str(venv_python),
            "-c",
            "import numpy as np, torch; print('numpy', np.__version__); print('torch', torch.__version__); print('cuda runtime', torch.version.cuda); print('gpu', torch.cuda.get_device_name(0)); assert np.__version__ == '1.26.1'; assert torch.__version__.startswith('2.3.1'); assert torch.cuda.is_available()",
        ],
        cwd=repo,
        label="Verifying CUDA + NumPy compatibility",
    )

    upstream = repo / ".upstream"
    upstream.mkdir(exist_ok=True)
    vggt = upstream / "vggt"
    if not (vggt / ".git").is_dir():
        run(["git", "clone", VGGT_REPO, str(vggt)], cwd=repo, label="Cloning pinned VGGT source")
    if subprocess.run(["git", "-C", str(vggt), "cat-file", "-e", f"{VGGT_PIN}^{{commit}}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
        run(["git", "-C", str(vggt), "fetch", "origin", VGGT_PIN], cwd=repo, label="Fetching pinned VGGT commit")
    run(["git", "-C", str(vggt), "checkout", "--detach", VGGT_PIN], cwd=repo, label="Checking out pinned VGGT commit")
    if git_head(vggt) != VGGT_PIN:
        raise RuntimeError("VGGT pin verification failed")

    model_manifest = output / "vggt-model.local.json"
    code, _ = run_stream(
        [str(venv_python), "-m", "refworld.runners.prefetch_vggt", "--output", str(model_manifest)],
        cwd=repo,
        log_path=log_path,
        label="Prefetching VGGT-1B weights (~5.03 GB; progress is live)",
    )
    if code != 0:
        raise RuntimeError(f"VGGT model prefetch failed; see {log_path}")
    model_info = json.loads(model_manifest.read_text(encoding="utf-8"))
    checkpoint = Path(model_info["snapshot_path"])
    if not (checkpoint / "model.safetensors").is_file():
        raise RuntimeError(f"verified VGGT snapshot missing model.safetensors: {checkpoint}")

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    if not args.skip_tests:
        code, _ = run_stream(
            [
                str(venv_python), "-m", "pytest", "-q",
                "tests/test_vggt_source.py", "tests/test_source_geometry.py", "tests/test_pinhole_warp.py",
                "tests/test_proposals.py", "tests/test_splats.py", "tests/test_schemas.py",
            ],
            cwd=repo,
            log_path=log_path,
            label="Running focused contract tests",
            env=env,
        )
        if code != 0:
            raise RuntimeError(f"focused tests failed; see {log_path}")

    def inference_attempt(size: int) -> tuple[int, str]:
        source_dir = output / "source-geometry"
        splat_dir = output / "source-splat"
        warp_dir = output / "warp-only"
        for path in (source_dir, splat_dir, warp_dir):
            path.mkdir(parents=True, exist_ok=True)

        code, text = run_stream(
            [
                str(venv_python), "-m", "refworld.runners.vggt_source",
                "--vggt-root", str(vggt),
                "--checkpoint", str(checkpoint),
                "--reference", str(reference),
                "--output", str(source_dir),
                "--seed", "0",
                "--model-size", str(size),
            ],
            cwd=repo,
            log_path=log_path,
            label=f"Running REAL VGGT tensor inference at {size}x{size}",
            env=env,
        )
        if code != 0:
            return code, text

        code, text2 = run_stream(
            [
                str(venv_python), "-m", "refworld.runners.source_splat",
                "--reference", str(reference),
                "--source-geometry", str(source_dir / "source-geometry.safe.json"),
                "--output", str(splat_dir),
            ],
            cwd=repo,
            log_path=log_path,
            label="Building source-only 3DGS diagnostic",
            env=env,
        )
        if code != 0:
            return code, text + text2

        code, text3 = run_stream(
            [
                str(venv_python), "-m", "refworld.runners.warp_only",
                "--reference", str(reference),
                "--source-geometry", str(source_dir / "source-geometry.safe.json"),
                "--output", str(warp_dir),
            ],
            cwd=repo,
            log_path=log_path,
            label="Generating warp-only near-view neighborhood",
            env=env,
        )
        return code, text + text2 + text3

    code, text = inference_attempt(model_size)
    if code != 0 and model_size > 336 and any(token in text.lower() for token in ["cuda out of memory", "out of memory", "cudnn_status_alloc_failed"]):
        print(f"WARNING: CUDA OOM at {model_size}; retrying once at smoke-only 336x336.", flush=True)
        for name in ("source-geometry", "source-splat", "warp-only"):
            shutil.rmtree(output / name, ignore_errors=True)
        model_size = 336
        log_path = output / f"gpu-run-{model_size}.log"
        code, text = inference_attempt(model_size)
    if code != 0:
        raise RuntimeError(f"inference pipeline failed; see {log_path}")

    manifest_path = output / "source-geometry" / "source-geometry.safe.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    peak_gib = float(manifest["environment"]["peak_reserved_bytes"]) / (1024**3)
    print("\nREAL INFERENCE COMPLETE")
    print("Execution:     native Windows CUDA (Python orchestrator)")
    print(f"GPU:           {manifest['environment']['gpu_name']}")
    print(f"Model size:    {manifest['preprocessing']['model_size']}")
    print(f"Model load:    {manifest['timing']['model_load_seconds']:.2f} s")
    print(f"Inference:     {manifest['timing']['inference_seconds']:.2f} s")
    print(f"Peak reserved: {peak_gib:.2f} GiB")
    print(f"Output:        {output}")
    print(f"Manifest:      {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
