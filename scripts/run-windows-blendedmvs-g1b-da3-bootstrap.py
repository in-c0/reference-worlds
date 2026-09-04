#!/usr/bin/env python3
"""Repair the frozen NumPy ABI, then execute the unchanged G1-B DA3 screen.

The RefWorld environment is pinned to NumPy 1.26.1 and frozen DA3 requires
NumPy <2. A prior DA3 runtime dependency install can nevertheless leave NumPy
2.x in an existing Windows venv. PyTorch 2.3.1/torchvision in the frozen RTX
2080 lane cannot bridge that ABI and fails before model inference.

This bootstrap performs environment repair and regression preflight only. It
does not change target selection, DA3 model/configuration, process resolution
policy, geometry bridge, metrics, thresholds, or evidence access.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

NUMPY_PIN = "1.26.1"


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
    if not python.is_file():
        raise RuntimeError(f"RefWorld venv missing: {python}")

    probe_code = (
        "import numpy as np, torch; "
        f"assert np.__version__ == '{NUMPY_PIN}', np.__version__; "
        "x=torch.from_numpy(np.zeros((1,), dtype=np.float32)); "
        "print('NumPy/Torch ABI OK', np.__version__, x.item())"
    )
    probe = subprocess.run(
        [str(python), "-c", probe_code],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    if probe.returncode != 0:
        if probe.stdout:
            print(probe.stdout, end="" if probe.stdout.endswith("\n") else "\n", flush=True)
        run_checked(
            f"Restoring frozen NumPy {NUMPY_PIN}",
            [str(python), "-m", "pip", "install", "--upgrade", f"numpy=={NUMPY_PIN}"],
            cwd=repo_root,
        )

    run_checked(
        "Verifying NumPy/Torch ABI",
        [str(python), "-c", probe_code],
        cwd=repo_root,
    )
    run_checked(
        "Checking DA3 metadata placeholder regression",
        [str(python), "-m", "pytest", "-q", "tests/test_da3_metadata.py"],
        cwd=repo_root,
    )
    run_checked(
        "Running unchanged frozen G1-B DA3 screen",
        [str(python), str(repo_root / "scripts" / "run-windows-blendedmvs-g1b-da3.py")],
        cwd=repo_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
