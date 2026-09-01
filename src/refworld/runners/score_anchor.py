#!/usr/bin/env python3
"""Score a rendered anchor frame against the real supplied source image."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from refworld.metrics import anchor_metrics
from refworld.reporting import json_safe


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score one exact source-anchor render")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--render", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--renderer-id",
        default="spark-2.1.0-three-0.180.0-playwright-1.62.1",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reference = args.reference.resolve()
    rendered = args.render.resolve()
    output = args.output.resolve()
    if not reference.is_file():
        raise FileNotFoundError(reference)
    if not rendered.is_file():
        raise FileNotFoundError(rendered)
    if not args.renderer_id.strip():
        raise ValueError("renderer id cannot be empty")

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("refworld-score-anchor requires Pillow: pip install 'refworld-bench[method]'") from exc

    ref = np.asarray(Image.open(reference).convert("RGB"), dtype=np.uint8)
    got = np.asarray(Image.open(rendered).convert("RGB"), dtype=np.uint8)
    if ref.shape != got.shape:
        raise ValueError(
            f"anchor render must have exact source dimensions; got {got.shape}, expected {ref.shape}"
        )

    metrics = anchor_metrics(ref, got)
    payload = {
        "version": "0.1",
        "stage": "refworld-score-anchor",
        "reference": {
            "file_name": reference.name,
            "sha256": _sha256_file(reference),
            "width": int(ref.shape[1]),
            "height": int(ref.shape[0]),
        },
        "render": {
            "file_name": rendered.name,
            "sha256": _sha256_file(rendered),
            "renderer_id": args.renderer_id.strip(),
        },
        "metrics": metrics.as_dict(),
        "interpretation": {
            "scope": "exact same-camera source reconstruction diagnostic",
            "novel_view_claim": False,
            "generated_hidden_content_in_required_path": False,
            "note": "Poor source-only score implicates camera/depth/representation/rendering before repaint quality is considered.",
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
