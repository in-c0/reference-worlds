#!/usr/bin/env python3
"""Score one calibrated BlendedMVS oracle-geometry B-vs-C experiment.

Held-out target RGB is first read in this stage. The score reports full-frame and
OBSERVED-support MAE/MSE/PSNR for geometry-only warp, unrestricted repaint B,
and evidence-preserved C. Held-out depth is not read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from refworld.evidence import PixelProvenance
from refworld.metrics import anchor_metrics
from refworld.reporting import json_safe


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    if mask is None:
        return anchor_metrics(reference, candidate).as_dict()
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != reference.shape[:2] or not np.any(selected):
        raise ValueError("metric mask must contain at least one HxW selected pixel")
    return anchor_metrics(reference[selected], candidate[selected]).as_dict()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score calibrated BlendedMVS oracle B-vs-C pair")
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--oracle-output", type=Path, required=True)
    parser.add_argument("--composition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene_root = args.scene_root.resolve()
    oracle_output = args.oracle_output.resolve()
    composition = args.composition.resolve()
    output = args.output.resolve()

    oracle_manifest_path = oracle_output / "oracle-pair.safe.json"
    if not oracle_manifest_path.is_file():
        raise FileNotFoundError(oracle_manifest_path)
    oracle = json.loads(oracle_manifest_path.read_text(encoding="utf-8"))
    target_id = int(oracle["selection"]["target_view_id"])
    target_path = scene_root / "blended_images" / f"{target_id:08d}.jpg"
    if not target_path.is_file():
        raise FileNotFoundError(target_path)

    view_dir = oracle_output / str(oracle["result"]["view_directory"])
    warp_path = view_dir / "proposal.png"
    warp_provenance_path = view_dir / "provenance.npy"
    b_path = composition / "candidate-unrestricted.png"
    c_path = composition / "proposal-evidence-preserved.png"
    compose_manifest_path = composition / "compose.safe.json"
    for required in (warp_path, warp_provenance_path, b_path, c_path, compose_manifest_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("pair scoring requires Pillow") from exc

    target = np.asarray(Image.open(target_path).convert("RGB"), dtype=np.uint8)
    warp = np.asarray(Image.open(warp_path).convert("RGB"), dtype=np.uint8)
    b = np.asarray(Image.open(b_path).convert("RGB"), dtype=np.uint8)
    c = np.asarray(Image.open(c_path).convert("RGB"), dtype=np.uint8)
    if not (target.shape == warp.shape == b.shape == c.shape):
        raise ValueError(
            f"image shape mismatch target={target.shape} warp={warp.shape} B={b.shape} C={c.shape}"
        )

    provenance = np.load(warp_provenance_path, allow_pickle=False)
    if provenance.shape != target.shape[:2]:
        raise ValueError("warp provenance shape mismatch")
    observed = provenance == int(PixelProvenance.OBSERVED)
    unresolved = provenance == int(PixelProvenance.UNRESOLVED)
    if not np.any(observed) or not np.any(unresolved):
        raise RuntimeError("calibrated warp must contain both observed and unresolved support")

    full = {
        "geometry_only": _metrics(target, warp),
        "B_unrestricted": _metrics(target, b),
        "C_evidence_preserved": _metrics(target, c),
    }
    on_observed = {
        "geometry_only": _metrics(target, warp, observed),
        "B_unrestricted": _metrics(target, b, observed),
        "C_evidence_preserved": _metrics(target, c, observed),
    }
    on_unresolved = {
        "geometry_only": _metrics(target, warp, unresolved),
        "B_unrestricted": _metrics(target, b, unresolved),
        "C_evidence_preserved": _metrics(target, c, unresolved),
    }

    payload = {
        "version": "0.1",
        "stage": "refworld-score-blendedmvs-oracle-pair",
        "scene_id": scene_root.name,
        "selection": oracle["selection"],
        "evaluation_inputs": {
            "target_rgb_view_id": target_id,
            "target_rgb_sha256": _sha256(target_path),
            "target_depth_read": False,
            "held_out_rgb_first_read_stage": "scoring-only",
        },
        "diagnostic_scope": {
            "oracle_source_depth": True,
            "full_single_image_method_result": False,
            "purpose": "test evidence-preservation effect with calibrated target while factoring out monocular depth/scale error",
        },
        "support": {
            "observed_fraction": float(np.mean(observed)),
            "unresolved_fraction": float(np.mean(unresolved)),
            "observed_pixels": int(np.sum(observed)),
            "unresolved_pixels": int(np.sum(unresolved)),
        },
        "metrics": {
            "full_frame": full,
            "observed_support": on_observed,
            "unresolved_support": on_unresolved,
        },
        "contrasts": {
            "C_minus_B_psnr_full_db": float(full["C_evidence_preserved"]["psnr"] - full["B_unrestricted"]["psnr"]),
            "C_minus_B_mae_full": float(full["C_evidence_preserved"]["mae"] - full["B_unrestricted"]["mae"]),
            "C_minus_B_psnr_observed_db": float(on_observed["C_evidence_preserved"]["psnr"] - on_observed["B_unrestricted"]["psnr"]),
            "C_minus_B_mae_observed": float(on_observed["C_evidence_preserved"]["mae"] - on_observed["B_unrestricted"]["mae"]),
        },
        "artifacts": {
            "target_rgb": str(target_path),
            "geometry_only": str(warp_path),
            "B_unrestricted": str(b_path),
            "C_evidence_preserved": str(c_path),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(output, flush=True)
    print(f"B PSNR full: {full['B_unrestricted']['psnr']:.4f} dB", flush=True)
    print(f"C PSNR full: {full['C_evidence_preserved']['psnr']:.4f} dB", flush=True)
    print(f"C-B PSNR full: {payload['contrasts']['C_minus_B_psnr_full_db']:+.4f} dB", flush=True)
    print(f"B PSNR observed: {on_observed['B_unrestricted']['psnr']:.4f} dB", flush=True)
    print(f"C PSNR observed: {on_observed['C_evidence_preserved']['psnr']:.4f} dB", flush=True)
    print(f"C-B PSNR observed: {payload['contrasts']['C_minus_B_psnr_observed_db']:+.4f} dB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
