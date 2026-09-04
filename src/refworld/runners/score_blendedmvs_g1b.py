#!/usr/bin/env python3
"""Score one opened-rank3 EXP-002 G1-B DA3-BASE geometry screen scene."""

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
        raise ValueError("metric mask must contain selected HxW pixels")
    return anchor_metrics(reference[selected], candidate[selected]).as_dict()


def _load_learned(root: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    meta_path = root / "learned-allvalid-pair.safe.json"
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("stage") != "refworld-blendedmvs-learned-allvalid-scale-pair":
        raise RuntimeError("learned all-valid manifest stage mismatch")
    view = root / str(meta["result"]["view_directory"])
    from PIL import Image
    image = np.asarray(Image.open(view / "proposal.png").convert("RGB"), dtype=np.uint8)
    provenance = np.load(view / "provenance.npy", allow_pickle=False)
    return meta, image, provenance


def _load_oracle(root: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    meta_path = root / "oracle-pair.safe.json"
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("stage") != "refworld-blendedmvs-oracle-pair":
        raise RuntimeError("oracle manifest stage mismatch")
    view = root / str(meta["result"]["view_directory"])
    from PIL import Image
    image = np.asarray(Image.open(view / "proposal.png").convert("RGB"), dtype=np.uint8)
    provenance = np.load(view / "provenance.npy", allow_pickle=False)
    return meta, image, provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score one G1-B DA3-BASE geometry screen scene")
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--vggt-equalized", type=Path, required=True)
    parser.add_argument("--da3", type=Path, required=True)
    parser.add_argument("--oracle-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene_root = args.scene_root.resolve()
    vggt_meta, vggt, vggt_prov = _load_learned(args.vggt_equalized.resolve())
    da3_meta, da3, da3_prov = _load_learned(args.da3.resolve())
    oracle_meta, oracle, oracle_prov = _load_oracle(args.oracle_output.resolve())

    if vggt_meta.get("backend") != "vggt":
        raise RuntimeError(f"expected equalized VGGT backend, got {vggt_meta.get('backend')!r}")
    if da3_meta.get("backend") != "da3-base":
        raise RuntimeError(f"expected DA3-BASE backend, got {da3_meta.get('backend')!r}")

    selections = (vggt_meta["selection"], da3_meta["selection"], oracle_meta["selection"])
    target_ids = {int(item["target_view_id"]) for item in selections}
    ranks = {int(item["held_out_source_order"]) for item in selections}
    if len(target_ids) != 1 or ranks != {3}:
        raise RuntimeError("G1-B target/rank selection mismatch")
    target_id = target_ids.pop()

    for learned_meta in (vggt_meta, da3_meta):
        method_inputs = learned_meta["method_inputs"]
        if bool(method_inputs["target_rgb_read"]) or bool(method_inputs["target_depth_read"]):
            raise RuntimeError("learned generation manifest reports target leakage")
        scale_meta = learned_meta["scale_calibration"]
        if bool(scale_meta.get("confidence_used_for_selection")):
            raise RuntimeError("G1-B requires equalized all-valid scalar selection")
        if float(scale_meta.get("top_confidence_fraction")) != 1.0:
            raise RuntimeError("G1-B scalar bridge must use all candidate-valid pixels")

    target_path = scene_root / "blended_images" / f"{target_id:08d}.jpg"
    target_meta_path = scene_root / "vggt-g1-rank3-target-rgb.safe.json"
    if not target_path.is_file() or not target_meta_path.is_file():
        raise FileNotFoundError("G1-B requires already-opened G1 rank-3 target RGB")
    target_meta = json.loads(target_meta_path.read_text(encoding="utf-8"))
    if int(target_meta["target_view_id"]) != target_id or bool(target_meta.get("target_depth_materialized")):
        raise RuntimeError("prior G1 target materialization state mismatch")

    from PIL import Image
    target = np.asarray(Image.open(target_path).convert("RGB"), dtype=np.uint8)
    if not (target.shape == vggt.shape == da3.shape == oracle.shape):
        raise ValueError("G1-B image shape mismatch")
    if not (vggt_prov.shape == da3_prov.shape == oracle_prov.shape == target.shape[:2]):
        raise ValueError("G1-B provenance shape mismatch")

    vggt_obs = vggt_prov == int(PixelProvenance.OBSERVED)
    da3_obs = da3_prov == int(PixelProvenance.OBSERVED)
    oracle_obs = oracle_prov == int(PixelProvenance.OBSERVED)
    common = vggt_obs & da3_obs & oracle_obs
    if not np.any(common):
        raise RuntimeError("G1-B oracle/VGGT/DA3 common OBSERVED support is empty")

    full = {
        "oracle": _metrics(target, oracle),
        "vggt_equalized": _metrics(target, vggt),
        "da3_base": _metrics(target, da3),
    }
    common_metrics = {
        "oracle": _metrics(target, oracle, common),
        "vggt_equalized": _metrics(target, vggt, common),
        "da3_base": _metrics(target, da3, common),
    }
    oracle_psnr = float(common_metrics["oracle"]["psnr"])
    vggt_psnr = float(common_metrics["vggt_equalized"]["psnr"])
    da3_psnr = float(common_metrics["da3_base"]["psnr"])

    payload = {
        "version": "0.1",
        "stage": "refworld-score-exp002-g1b-da3-base",
        "scene_id": scene_root.name,
        "selection": da3_meta["selection"],
        "evaluation_inputs": {
            "target_rgb_view_id": target_id,
            "target_rgb_sha256": _sha256(target_path),
            "target_depth_read": False,
            "rank3_status": "already-opened-development-only",
        },
        "support": {
            "vggt_observed_fraction": float(np.mean(vggt_obs)),
            "da3_observed_fraction": float(np.mean(da3_obs)),
            "oracle_observed_fraction": float(np.mean(oracle_obs)),
            "all_three_common_observed_fraction": float(np.mean(common)),
            "all_three_common_observed_pixels": int(np.sum(common)),
        },
        "metrics": {
            "full_frame": full,
            "all_three_common_observed": common_metrics,
        },
        "contrasts_common_observed_psnr_db": {
            "vggt_equalized_minus_oracle": vggt_psnr - oracle_psnr,
            "da3_base_minus_oracle": da3_psnr - oracle_psnr,
            "da3_base_minus_vggt_equalized": da3_psnr - vggt_psnr,
        },
        "generation": {
            "vggt_backend": vggt_meta.get("backend"),
            "da3_backend": da3_meta.get("backend"),
            "scale_policy_equalized_all_valid": True,
            "oracle_depth_scalar": True,
            "oracle_anchor_frame_placement": True,
            "claim_as_end_to_end_single_image": False,
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(output, flush=True)
    print(f"VGGT equalized - oracle: {vggt_psnr - oracle_psnr:+.4f} dB", flush=True)
    print(f"DA3-BASE - oracle:       {da3_psnr - oracle_psnr:+.4f} dB", flush=True)
    print(f"DA3-BASE - VGGT:         {da3_psnr - vggt_psnr:+.4f} dB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
