#!/usr/bin/env python3
"""Score one sealed BlendedMVS VGGT G1 learned-geometry experiment.

Target RGB is first read here, after experiment-wide generation sealing. Target
depth is never read. Reports learned VGGT warp vs oracle warp and frozen Big-LaMa
B/C evidence-preservation metrics using the learned warp's OBSERVED support.
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
    parser = argparse.ArgumentParser(description="Score sealed BlendedMVS VGGT G1 pair")
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--learned-output", type=Path, required=True)
    parser.add_argument("--oracle-output", type=Path, required=True)
    parser.add_argument("--composition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene_root = args.scene_root.resolve()
    learned_output = args.learned_output.resolve()
    oracle_output = args.oracle_output.resolve()
    composition = args.composition.resolve()
    output = args.output.resolve()

    learned_manifest_path = learned_output / "vggt-oracle-scale-pair.safe.json"
    oracle_manifest_path = oracle_output / "oracle-pair.safe.json"
    compose_manifest_path = composition / "compose.safe.json"
    for required in (learned_manifest_path, oracle_manifest_path, compose_manifest_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    learned_meta = json.loads(learned_manifest_path.read_text(encoding="utf-8"))
    oracle_meta = json.loads(oracle_manifest_path.read_text(encoding="utf-8"))
    compose_meta = json.loads(compose_manifest_path.read_text(encoding="utf-8"))
    if learned_meta.get("stage") != "refworld-blendedmvs-vggt-oracle-scale-pair":
        raise RuntimeError("learned G1 manifest stage mismatch")
    if oracle_meta.get("stage") != "refworld-blendedmvs-oracle-pair":
        raise RuntimeError("oracle manifest stage mismatch")
    if not bool(learned_meta.get("primary_protocol")):
        raise RuntimeError("G1 scoring requires the frozen primary rank")

    learned_selection = learned_meta["selection"]
    oracle_selection = oracle_meta["selection"]
    target_id = int(learned_selection["target_view_id"])
    if int(learned_selection["held_out_source_order"]) != 3:
        raise RuntimeError("G1 primary scorer requires rank-3 target")
    if int(oracle_selection["target_view_id"]) != target_id or int(oracle_selection["held_out_source_order"]) != 3:
        raise RuntimeError("learned/oracle target selection mismatch")
    if bool(learned_meta["method_inputs"]["target_rgb_read"]) or bool(learned_meta["method_inputs"]["target_depth_read"]):
        raise RuntimeError("learned generation manifest reports held-out leakage")
    if bool(oracle_meta["method_inputs"]["target_rgb_read"]) or bool(oracle_meta["method_inputs"]["target_depth_read"]):
        raise RuntimeError("oracle generation manifest reports held-out leakage")

    target_path = scene_root / "blended_images" / f"{target_id:08d}.jpg"
    target_meta_path = scene_root / "vggt-g1-rank3-target-rgb.safe.json"
    if not target_path.is_file() or not target_meta_path.is_file():
        raise FileNotFoundError("sealed rank-3 target RGB/materialization manifest missing")
    target_meta = json.loads(target_meta_path.read_text(encoding="utf-8"))
    if int(target_meta["target_view_id"]) != target_id or not bool(target_meta["target_rgb_materialized"]):
        raise RuntimeError("target materialization selection mismatch")
    if bool(target_meta["target_depth_materialized"]):
        raise RuntimeError("target depth must never be materialized for G1")

    learned_view = learned_output / str(learned_meta["result"]["view_directory"])
    oracle_view = oracle_output / str(oracle_meta["result"]["view_directory"])
    learned_warp_path = learned_view / "proposal.png"
    learned_prov_path = learned_view / "provenance.npy"
    oracle_warp_path = oracle_view / "proposal.png"
    oracle_prov_path = oracle_view / "provenance.npy"
    b_path = composition / "candidate-unrestricted.png"
    c_path = composition / "proposal-evidence-preserved.png"
    for required in (learned_warp_path, learned_prov_path, oracle_warp_path, oracle_prov_path, b_path, c_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("G1 scoring requires Pillow") from exc

    target = np.asarray(Image.open(target_path).convert("RGB"), dtype=np.uint8)
    learned = np.asarray(Image.open(learned_warp_path).convert("RGB"), dtype=np.uint8)
    oracle = np.asarray(Image.open(oracle_warp_path).convert("RGB"), dtype=np.uint8)
    b = np.asarray(Image.open(b_path).convert("RGB"), dtype=np.uint8)
    c = np.asarray(Image.open(c_path).convert("RGB"), dtype=np.uint8)
    if not (target.shape == learned.shape == oracle.shape == b.shape == c.shape):
        raise ValueError("G1 image shape mismatch")

    learned_prov = np.load(learned_prov_path, allow_pickle=False)
    oracle_prov = np.load(oracle_prov_path, allow_pickle=False)
    if learned_prov.shape != target.shape[:2] or oracle_prov.shape != target.shape[:2]:
        raise ValueError("G1 provenance shape mismatch")
    learned_observed = learned_prov == int(PixelProvenance.OBSERVED)
    oracle_observed = oracle_prov == int(PixelProvenance.OBSERVED)
    common_observed = learned_observed & oracle_observed
    if not np.any(learned_observed) or not np.any(oracle_observed) or not np.any(common_observed):
        raise RuntimeError("G1 requires non-empty learned, oracle, and common observed support")

    full = {
        "learned_vggt_warp": _metrics(target, learned),
        "oracle_warp": _metrics(target, oracle),
        "B_unrestricted": _metrics(target, b),
        "C_evidence_preserved": _metrics(target, c),
    }
    learned_support = {
        "learned_vggt_warp": _metrics(target, learned, learned_observed),
        "B_unrestricted": _metrics(target, b, learned_observed),
        "C_evidence_preserved": _metrics(target, c, learned_observed),
    }
    oracle_support = {"oracle_warp": _metrics(target, oracle, oracle_observed)}
    common_support = {
        "learned_vggt_warp": _metrics(target, learned, common_observed),
        "oracle_warp": _metrics(target, oracle, common_observed),
        "B_unrestricted": _metrics(target, b, common_observed),
        "C_evidence_preserved": _metrics(target, c, common_observed),
    }

    payload = {
        "version": "0.1",
        "stage": "refworld-score-blendedmvs-vggt-g1",
        "scene_id": scene_root.name,
        "selection": learned_selection,
        "evaluation_inputs": {
            "target_rgb_view_id": target_id,
            "target_rgb_sha256": _sha256(target_path),
            "target_depth_read": False,
            "held_out_rgb_first_read_stage": "post-experiment-wide-generation-seal-scoring-only",
        },
        "diagnostic_scope": {
            "vggt_model_resolution": learned_meta.get("source_geometry", {}).get("model_size"),
            "oracle_depth_scale_scalar": True,
            "oracle_anchor_frame_placement": True,
            "full_single_image_method_result": False,
            "benchmark_resolution_result": False,
            "purpose": "G1 reduced-resolution depth-shape/intrinsics falsification plus frozen learned-geometry B/C test",
        },
        "scale_calibration": learned_meta["scale_calibration"],
        "camera_bridge": learned_meta["camera_bridge"],
        "support": {
            "learned_observed_fraction": float(np.mean(learned_observed)),
            "oracle_observed_fraction": float(np.mean(oracle_observed)),
            "common_observed_fraction": float(np.mean(common_observed)),
            "learned_observed_pixels": int(np.sum(learned_observed)),
            "oracle_observed_pixels": int(np.sum(oracle_observed)),
            "common_observed_pixels": int(np.sum(common_observed)),
        },
        "metrics": {
            "full_frame": full,
            "learned_observed_support": learned_support,
            "oracle_observed_support": oracle_support,
            "common_observed_support": common_support,
        },
        "contrasts": {
            "learned_minus_oracle_psnr_full_db": float(full["learned_vggt_warp"]["psnr"] - full["oracle_warp"]["psnr"]),
            "learned_minus_oracle_psnr_common_observed_db": float(common_support["learned_vggt_warp"]["psnr"] - common_support["oracle_warp"]["psnr"]),
            "C_minus_B_psnr_full_db": float(full["C_evidence_preserved"]["psnr"] - full["B_unrestricted"]["psnr"]),
            "C_minus_B_psnr_learned_observed_db": float(learned_support["C_evidence_preserved"]["psnr"] - learned_support["B_unrestricted"]["psnr"]),
        },
        "generation_manifests": {
            "learned": {"path": str(learned_manifest_path), "sha256": _sha256(learned_manifest_path)},
            "oracle": {"path": str(oracle_manifest_path), "sha256": _sha256(oracle_manifest_path)},
            "composition": {"path": str(compose_manifest_path), "sha256": _sha256(compose_manifest_path)},
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(output, flush=True)
    print(f"Learned warp PSNR full: {full['learned_vggt_warp']['psnr']:.4f} dB", flush=True)
    print(f"Oracle warp PSNR full:  {full['oracle_warp']['psnr']:.4f} dB", flush=True)
    print(f"Learned-oracle common-observed: {payload['contrasts']['learned_minus_oracle_psnr_common_observed_db']:+.4f} dB", flush=True)
    print(f"C-B PSNR full: {payload['contrasts']['C_minus_B_psnr_full_db']:+.4f} dB", flush=True)
    print(f"C-B PSNR learned observed: {payload['contrasts']['C_minus_B_psnr_learned_observed_db']:+.4f} dB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
