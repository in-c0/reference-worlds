#!/usr/bin/env python3
"""Score one EXP-002 G1-D geometry decomposition on an already-opened rank-3 target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from refworld.evidence import PixelProvenance
from refworld.metrics import anchor_metrics
from refworld.reporting import json_safe

CONDITION_ORDER = ("oracle", "vggt_both", "vggt_depth_oracle_K", "oracle_depth_vggt_K")


def _metrics(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    if mask is None:
        return anchor_metrics(reference, candidate).as_dict()
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != reference.shape[:2] or not np.any(selected):
        raise ValueError("metric mask must contain selected HxW pixels")
    return anchor_metrics(reference[selected], candidate[selected]).as_dict()


def _load_image(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _view_from_manifest(root: Path, manifest: dict, key: str = "result") -> Path:
    return root / str(manifest[key]["view_directory"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score EXP-002 G1-D depth/intrinsics decomposition")
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--learned-output", type=Path, required=True)
    parser.add_argument("--oracle-output", type=Path, required=True)
    parser.add_argument("--hybrid-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene_root = args.scene_root.resolve()
    learned_output = args.learned_output.resolve()
    oracle_output = args.oracle_output.resolve()
    hybrid_output = args.hybrid_output.resolve()
    output = args.output.resolve()

    learned_meta = json.loads((learned_output / "vggt-oracle-scale-pair.safe.json").read_text(encoding="utf-8"))
    oracle_meta = json.loads((oracle_output / "oracle-pair.safe.json").read_text(encoding="utf-8"))
    hybrid_meta = json.loads((hybrid_output / "g1d-hybrids.safe.json").read_text(encoding="utf-8"))
    target_id = int(learned_meta["selection"]["target_view_id"])
    if int(oracle_meta["selection"]["target_view_id"]) != target_id or int(hybrid_meta["selection"]["target_view_id"]) != target_id:
        raise RuntimeError("G1-D target selection mismatch")
    if int(learned_meta["selection"]["held_out_source_order"]) != 3:
        raise RuntimeError("G1-D requires the already-opened rank-3 G1 targets")
    if not bool(hybrid_meta["diagnostic_scope"]["opened_rank3_reuse_only"]) or bool(hybrid_meta["diagnostic_scope"]["fresh_target_consumed"]):
        raise RuntimeError("G1-D hybrid manifest scope mismatch")
    if bool(hybrid_meta["diagnostic_scope"]["target_rgb_read"]) or bool(hybrid_meta["diagnostic_scope"]["target_depth_read"]):
        raise RuntimeError("hybrid generation must not read target RGB/depth")

    target_path = scene_root / "blended_images" / f"{target_id:08d}.jpg"
    if not target_path.is_file():
        raise FileNotFoundError(target_path)
    target = _load_image(target_path)

    learned_view = _view_from_manifest(learned_output, learned_meta)
    oracle_view = _view_from_manifest(oracle_output, oracle_meta)
    condition_rows = {str(item["condition"]): item for item in hybrid_meta["conditions"]}
    for required in ("vggt_depth_oracle_K", "oracle_depth_vggt_K"):
        if required not in condition_rows:
            raise RuntimeError(f"G1-D hybrid manifest missing {required}")

    paths = {
        "oracle": (oracle_view / "proposal.png", oracle_view / "provenance.npy"),
        "vggt_both": (learned_view / "proposal.png", learned_view / "provenance.npy"),
        "vggt_depth_oracle_K": (
            hybrid_output / str(condition_rows["vggt_depth_oracle_K"]["view_directory"]) / "proposal.png",
            hybrid_output / str(condition_rows["vggt_depth_oracle_K"]["view_directory"]) / "provenance.npy",
        ),
        "oracle_depth_vggt_K": (
            hybrid_output / str(condition_rows["oracle_depth_vggt_K"]["view_directory"]) / "proposal.png",
            hybrid_output / str(condition_rows["oracle_depth_vggt_K"]["view_directory"]) / "provenance.npy",
        ),
    }

    images: dict[str, np.ndarray] = {}
    observed: dict[str, np.ndarray] = {}
    for name in CONDITION_ORDER:
        image_path, provenance_path = paths[name]
        if not image_path.is_file() or not provenance_path.is_file():
            raise FileNotFoundError(f"missing G1-D condition artifact for {name}")
        image = _load_image(image_path)
        provenance = np.load(provenance_path, allow_pickle=False)
        if image.shape != target.shape or provenance.shape != target.shape[:2]:
            raise ValueError(f"G1-D shape mismatch for {name}")
        mask = provenance == int(PixelProvenance.OBSERVED)
        if not np.any(mask):
            raise RuntimeError(f"G1-D condition has no OBSERVED support: {name}")
        images[name] = image
        observed[name] = mask

    common = np.logical_and.reduce([observed[name] for name in CONDITION_ORDER])
    if not np.any(common):
        raise RuntimeError("G1-D all-four common OBSERVED support is empty")

    full = {name: _metrics(target, images[name]) for name in CONDITION_ORDER}
    common_metrics = {name: _metrics(target, images[name], common) for name in CONDITION_ORDER}
    own_observed = {name: _metrics(target, images[name], observed[name]) for name in CONDITION_ORDER}
    oracle_common_psnr = float(common_metrics["oracle"]["psnr"])

    payload = {
        "version": "0.1",
        "stage": "refworld-score-exp002-g1d",
        "scene_id": scene_root.name,
        "target_view_id": target_id,
        "diagnostic_scope": {
            "opened_rank3_reuse_only": True,
            "fresh_target_consumed": False,
            "target_depth_read": False,
            "purpose": "decompose G1 failure into VGGT depth-shape and source-intrinsics contributions",
        },
        "support": {
            "all_four_common_observed_fraction": float(np.mean(common)),
            "all_four_common_observed_pixels": int(np.sum(common)),
            "own_observed_fraction": {name: float(np.mean(observed[name])) for name in CONDITION_ORDER},
        },
        "metrics": {
            "full_frame": full,
            "all_four_common_observed": common_metrics,
            "own_observed": own_observed,
        },
        "contrasts_common_observed_psnr_db": {
            name: float(common_metrics[name]["psnr"] - oracle_common_psnr) for name in CONDITION_ORDER if name != "oracle"
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    print(output, flush=True)
    print(f"Oracle common-observed PSNR: {oracle_common_psnr:.4f} dB", flush=True)
    for name in CONDITION_ORDER[1:]:
        delta = payload["contrasts_common_observed_psnr_db"][name]
        print(f"{name} - oracle common-observed: {delta:+.4f} dB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
