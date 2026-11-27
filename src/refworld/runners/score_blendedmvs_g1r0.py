#!/usr/bin/env python3
"""Score the frozen G1-R0 392-vs-518 VGGT resolution control.

Uses only the already-opened rank-3 target RGB. Target depth is never read.
Primary support is OBSERVED in oracle, original-392, lowmem-392 and lowmem-518.
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

EQUIVALENCE_DB = 0.25
RESCUE_MIN_IMPROVEMENT_DB = 3.0
PARTIAL_MIN_IMPROVEMENT_DB = 1.0
ORACLE_GAP_STRONG_DB = -3.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != reference.shape[:2] or not np.any(selected):
        raise ValueError("metric mask must contain selected pixels")
    return anchor_metrics(reference[selected], candidate[selected]).as_dict()


def _load_condition(root: Path, manifest_name: str) -> tuple[dict, np.ndarray, np.ndarray]:
    manifest_path = root / manifest_name
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    view = root / str(meta["result"]["view_directory"])
    from PIL import Image
    image = np.asarray(Image.open(view / "proposal.png").convert("RGB"), dtype=np.uint8)
    provenance = np.load(view / "provenance.npy", allow_pickle=False)
    return meta, image, provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score G1-R0 VGGT 392-vs-518 control")
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--original-392", type=Path, required=True)
    parser.add_argument("--lowmem-392", type=Path, required=True)
    parser.add_argument("--lowmem-518", type=Path, required=True)
    parser.add_argument("--oracle-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene_root = args.scene_root.resolve()
    original_meta, original, original_prov = _load_condition(args.original_392.resolve(), "vggt-oracle-scale-pair.safe.json")
    low392_meta, low392, low392_prov = _load_condition(args.lowmem_392.resolve(), "vggt-oracle-scale-pair.safe.json")
    low518_meta, low518, low518_prov = _load_condition(args.lowmem_518.resolve(), "vggt-oracle-scale-pair.safe.json")
    oracle_meta, oracle, oracle_prov = _load_condition(args.oracle_output.resolve(), "oracle-pair.safe.json")

    selections = [original_meta["selection"], low392_meta["selection"], low518_meta["selection"], oracle_meta["selection"]]
    target_ids = {int(item["target_view_id"]) for item in selections}
    held_out_ranks = {int(item["held_out_source_order"]) for item in selections}
    if target_ids != {27} or held_out_ranks != {3}:
        raise RuntimeError(f"G1-R0 frozen selection mismatch: targets={target_ids}, ranks={held_out_ranks}")
    target_id = 27
    target_path = scene_root / "blended_images" / f"{target_id:08d}.jpg"
    target_meta_path = scene_root / "vggt-g1-rank3-target-rgb.safe.json"
    if not target_path.is_file() or not target_meta_path.is_file():
        raise FileNotFoundError("already-opened rank-3 target/materialization manifest missing")
    target_meta = json.loads(target_meta_path.read_text(encoding="utf-8"))
    if bool(target_meta.get("target_depth_materialized")):
        raise RuntimeError("target depth must remain unmaterialized")

    from PIL import Image
    target = np.asarray(Image.open(target_path).convert("RGB"), dtype=np.uint8)
    if not (target.shape == original.shape == low392.shape == low518.shape == oracle.shape):
        raise ValueError("G1-R0 image shape mismatch")

    masks = []
    for provenance in (original_prov, low392_prov, low518_prov, oracle_prov):
        if provenance.shape != target.shape[:2]:
            raise ValueError("G1-R0 provenance shape mismatch")
        masks.append(provenance == int(PixelProvenance.OBSERVED))
    common = masks[0] & masks[1] & masks[2] & masks[3]
    if not np.any(common):
        raise RuntimeError("G1-R0 all-four common OBSERVED support is empty")

    scores = {
        "392_original": _metrics(target, original, common),
        "392_lowmem": _metrics(target, low392, common),
        "518_lowmem": _metrics(target, low518, common),
        "oracle": _metrics(target, oracle, common),
    }
    original_psnr = float(scores["392_original"]["psnr"])
    low392_psnr = float(scores["392_lowmem"]["psnr"])
    low518_psnr = float(scores["518_lowmem"]["psnr"])
    oracle_psnr = float(scores["oracle"]["psnr"])

    equivalence_delta = low392_psnr - original_psnr
    improvement_518 = low518_psnr - low392_psnr
    oracle_gap_518 = low518_psnr - oracle_psnr
    guard_pass = abs(equivalence_delta) <= EQUIVALENCE_DB
    if not guard_pass:
        attribution = "invalid-resolution-attribution-lowmem-equivalence-guard-failed"
    elif improvement_518 >= RESCUE_MIN_IMPROVEMENT_DB and oracle_gap_518 > ORACLE_GAP_STRONG_DB:
        attribution = "strong-resolution-rescue"
    elif improvement_518 >= PARTIAL_MIN_IMPROVEMENT_DB:
        attribution = "partial-resolution-rescue"
    else:
        attribution = "no-meaningful-resolution-rescue"

    payload = {
        "version": "0.1",
        "stage": "refworld-score-blendedmvs-g1r0-resolution-control",
        "scene_id": scene_root.name,
        "selection": original_meta["selection"],
        "evaluation_inputs": {
            "target_rgb_view_id": target_id,
            "target_rgb_sha256": _sha256(target_path),
            "target_depth_read": False,
            "target_status": "already-opened-rank3-diagnostic",
        },
        "common_observed_fraction": float(np.mean(common)),
        "metrics_common_observed": scores,
        "contrasts": {
            "392_lowmem_minus_392_original_db": equivalence_delta,
            "518_lowmem_minus_392_lowmem_db": improvement_518,
            "518_lowmem_minus_oracle_db": oracle_gap_518,
        },
        "frozen_thresholds": {
            "lowmem_equivalence_abs_db_max": EQUIVALENCE_DB,
            "strong_rescue_min_improvement_db": RESCUE_MIN_IMPROVEMENT_DB,
            "partial_rescue_min_improvement_db": PARTIAL_MIN_IMPROVEMENT_DB,
            "strong_rescue_oracle_gap_must_be_greater_than_db": ORACLE_GAP_STRONG_DB,
        },
        "equivalence_guard_pass": guard_pass,
        "attribution": attribution,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(output, flush=True)
    print(f"392 lowmem - 392 original: {equivalence_delta:+.4f} dB", flush=True)
    print(f"518 lowmem - 392 lowmem:    {improvement_518:+.4f} dB", flush=True)
    print(f"518 lowmem - oracle:        {oracle_gap_518:+.4f} dB", flush=True)
    print(f"Frozen-rule attribution: {attribution}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
