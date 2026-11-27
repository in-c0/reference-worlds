#!/usr/bin/env python3
"""Prefetch the exact VGGT-1B safetensors payload with visible progress.

This separates model acquisition from CUDA inference so a network stall cannot be
misreported as GPU inference. Only config.json + model.safetensors are fetched;
model.pt is deliberately excluded to avoid downloading the weights twice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

VGGT_REPO = "facebook/VGGT-1B"
VGGT_SAFETENSORS_SHA256 = "f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e"
VGGT_SAFETENSORS_BYTES = 5_030_000_000  # documentation/display only; exact local size is recorded below


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prefetch pinned VGGT-1B safetensors weights")
    parser.add_argument("--output", type=Path, required=True, help="local JSON manifest path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import model_info, snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for VGGT prefetch") from exc

    info = model_info(VGGT_REPO)
    resolved_revision = str(info.sha)
    print(f"VGGT model repository: {VGGT_REPO}", flush=True)
    print(f"Resolved Hugging Face revision: {resolved_revision}", flush=True)
    print("Fetching config.json + model.safetensors (~5.03 GB).", flush=True)
    print("A cached partial/full download will be resumed/reused automatically.", flush=True)

    snapshot = Path(
        snapshot_download(
            repo_id=VGGT_REPO,
            revision=resolved_revision,
            allow_patterns=["config.json", "model.safetensors"],
        )
    ).resolve()

    weights = snapshot / "model.safetensors"
    config = snapshot / "config.json"
    if not weights.is_file() or not config.is_file():
        raise RuntimeError("VGGT snapshot completed but expected model.safetensors/config.json are missing")

    size_bytes = weights.stat().st_size
    print(f"Download present: {size_bytes / (1024**3):.2f} GiB", flush=True)
    print("Verifying model.safetensors SHA-256...", flush=True)
    digest = sha256_file(weights)
    if digest != VGGT_SAFETENSORS_SHA256:
        raise RuntimeError(
            "VGGT model.safetensors SHA-256 mismatch: "
            f"got {digest}, expected {VGGT_SAFETENSORS_SHA256}"
        )

    manifest = {
        "version": "0.1",
        "repo": VGGT_REPO,
        "resolved_revision": resolved_revision,
        "snapshot_path": str(snapshot),
        "model_safetensors": {
            "file_name": "model.safetensors",
            "size_bytes": size_bytes,
            "sha256": digest,
        },
        "config_file_name": "config.json",
        "note": "local execution manifest; snapshot_path is machine-local and must not be published as benchmark metadata",
    }
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"VGGT model prefetch verified: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
