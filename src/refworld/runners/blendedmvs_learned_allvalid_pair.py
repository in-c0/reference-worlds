#!/usr/bin/env python3
"""Create a calibrated BlendedMVS warp from generic learned source geometry.

EXP-002 G1-A uses this bridge for both the equalized VGGT baseline and candidate
models. Every backend receives exactly one oracle multiplicative depth scalar fit
over all finite positive overlapping source pixels plus published anchor
extrinsics for benchmark-frame placement. Target RGB/depth are not read here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from refworld.adapters.base import Camera
from refworld.datasets.mvsnet import PairRecord, camera_pose_separation, parse_camera_text, parse_pair_text
from refworld.datasets.pfm import read_pfm
from refworld.geometry_scale import estimate_positive_depth_scale
from refworld.proposals import ObservationView, build_view_proposal
from refworld.repaints import NoRepaintBackend
from refworld.source_geometry import load_source_geometry
from refworld.warps import PinholeWarpBackend

PRIMARY_HELD_OUT_RANK = 3
ALL_VALID_FRACTION = 1.0
MIN_SCALE_PIXELS = 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path, root: Path, kind: str) -> dict[str, Any]:
    resolved = path.resolve()
    relative = resolved.relative_to(root.resolve())
    return {
        "kind": kind,
        "path": relative.as_posix(),
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _select_held_out(record: PairRecord, held_out_rank: int) -> tuple[int, float]:
    rank = int(held_out_rank)
    if rank < 1 or rank > len(record.source_ids):
        raise ValueError(f"held-out rank must be in [1,{len(record.source_ids)}], got {rank}")
    return int(record.source_ids[rank - 1]), float(record.scores[rank - 1])


def _optional_valid_mask(source_manifest_path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Load an optional hash-pinned source validity mask; default is all source pixels."""
    meta = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    records = meta.get("artifacts", [])
    matches = [item for item in records if isinstance(item, dict) and item.get("kind") == "source-valid-mask-npy"]
    if not matches:
        return np.ones(shape, dtype=bool)
    if len(matches) != 1:
        raise RuntimeError("source geometry declares multiple validity masks")
    record = matches[0]
    relative = Path(str(record.get("path", "")))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise RuntimeError("source validity mask path is not confined")
    path = (source_manifest_path.parent / relative).resolve()
    path.relative_to(source_manifest_path.parent.resolve())
    if not path.is_file() or _sha256(path) != str(record.get("sha256", "")):
        raise RuntimeError("source validity mask artifact hash mismatch")
    mask = np.load(path, allow_pickle=False)
    if mask.shape != shape:
        raise RuntimeError(f"source validity mask shape mismatch: {mask.shape} != {shape}")
    return np.asarray(mask, dtype=bool)


def _intrinsics_summary(learned_camera: Camera, benchmark_camera: Camera) -> dict[str, Any]:
    kl = np.asarray(learned_camera.intrinsics, dtype=np.float64).reshape(3, 3)
    kb = np.asarray(benchmark_camera.intrinsics, dtype=np.float64).reshape(3, 3)
    return {
        "learned": kl.reshape(-1).tolist(),
        "benchmark": kb.reshape(-1).tolist(),
        "delta": (kl - kb).reshape(-1).tolist(),
        "relative_focal_delta": {
            "fx": float((kl[0, 0] - kb[0, 0]) / kb[0, 0]),
            "fy": float((kl[1, 1] - kb[1, 1]) / kb[1, 1]),
        },
        "principal_point_delta_px": {
            "cx": float(kl[0, 2] - kb[0, 2]),
            "cy": float(kl[1, 2] - kb[1, 2]),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create G1-A all-valid learned-geometry BlendedMVS warp")
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--source-geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--held-out-rank", type=int, default=PRIMARY_HELD_OUT_RANK)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if int(args.held_out_rank) != PRIMARY_HELD_OUT_RANK:
        raise ValueError("G1-A development screen is frozen to already-opened rank 3")

    scene_root = args.scene_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_manifest_path = Path(args.source_geometry).resolve()
    if source_manifest_path.is_dir():
        source_manifest_path = source_manifest_path / "source-geometry.safe.json"

    pair_path = scene_root / "cams" / "pair.txt"
    records = parse_pair_text(pair_path.read_text(encoding="utf-8"))
    if not records or not records[0].source_ids:
        raise RuntimeError("first pair record has no target")
    first = records[0]
    anchor_id = int(first.reference_id)
    target_id, pair_score = _select_held_out(first, int(args.held_out_rank))
    anchor_stem = f"{anchor_id:08d}"
    target_stem = f"{target_id:08d}"

    anchor_image_path = scene_root / "blended_images" / f"{anchor_stem}.jpg"
    anchor_camera_path = scene_root / "cams" / f"{anchor_stem}_cam.txt"
    anchor_depth_path = scene_root / "rendered_depth_maps" / f"{anchor_stem}.pfm"
    target_camera_path = scene_root / "cams" / f"{target_stem}_cam.txt"
    for required in (anchor_image_path, anchor_camera_path, anchor_depth_path, target_camera_path, source_manifest_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    from PIL import Image
    anchor_rgb = np.asarray(Image.open(anchor_image_path).convert("RGB"), dtype=np.uint8)
    anchor_depth = read_pfm(anchor_depth_path)
    if anchor_depth.shape != anchor_rgb.shape[:2]:
        raise ValueError("anchor depth/image shape mismatch")

    source_geometry = load_source_geometry(source_manifest_path)
    anchor_sha = _sha256(anchor_image_path)
    if source_geometry.input_sha256 != anchor_sha:
        raise ValueError("source-geometry input hash does not match anchor RGB")
    if (source_geometry.height, source_geometry.width) != anchor_rgb.shape[:2]:
        raise ValueError("source-geometry dimensions do not match anchor RGB")

    benchmark_anchor = parse_camera_text(anchor_camera_path.read_text(encoding="utf-8")).camera
    target_camera = parse_camera_text(target_camera_path.read_text(encoding="utf-8")).camera
    valid_mask = _optional_valid_mask(source_manifest_path, anchor_rgb.shape[:2])
    if int(np.count_nonzero(valid_mask)) < MIN_SCALE_PIXELS:
        raise RuntimeError("candidate validity mask leaves too few scale pixels")

    depth_for_scale = np.asarray(source_geometry.depth, dtype=np.float64).copy()
    depth_for_scale[~valid_mask] = np.nan
    uniform_confidence = np.ones_like(depth_for_scale, dtype=np.float64)
    scale = estimate_positive_depth_scale(
        depth_for_scale,
        anchor_depth,
        uniform_confidence,
        top_fraction=ALL_VALID_FRACTION,
        min_selected=MIN_SCALE_PIXELS,
    )
    scaled_depth = np.asarray(source_geometry.depth, dtype=np.float32) * np.float32(scale.scale)
    scaled_depth[~valid_mask] = np.nan
    if np.any(np.isinf(scaled_depth)) or not np.any(np.isfinite(scaled_depth) & (scaled_depth > 0.0)):
        raise RuntimeError("scaled learned depth has invalid usable support")

    learned_source_camera = Camera(
        intrinsics=tuple(source_geometry.camera.intrinsics),
        extrinsics=tuple(benchmark_anchor.extrinsics),
        convention=benchmark_anchor.convention,
    )
    observation_id = f"blendedmvs-{source_geometry.backend}-anchor-{anchor_sha[:16]}"
    observation = ObservationView(observation_id=observation_id, image=anchor_rgb, camera=learned_source_camera)
    warper = PinholeWarpBackend(
        {observation_id: scaled_depth},
        name=f"blendedmvs-{source_geometry.backend}-allvalid-oracle-scale-pinhole@0.1",
    )
    warp = warper.warp([observation], target_camera)
    repaint = NoRepaintBackend().repaint(warp, target_camera, seed=None)
    proposal = build_view_proposal([observation], target_camera, warp, repaint, unresolved_value=0)

    view_dir = output / f"anchor-{anchor_stem}-target-{target_stem}"
    view_dir.mkdir(parents=True, exist_ok=True)
    image_path = view_dir / "proposal.png"
    provenance_path = view_dir / "provenance.npy"
    confidence_path = view_dir / "warp-confidence.npy"
    metadata_path = view_dir / "proposal.json"
    scaled_depth_path = view_dir / "learned-depth-scaled.npy"
    Image.fromarray(np.asarray(proposal.image, dtype=np.uint8), mode="RGB").save(image_path)
    np.save(provenance_path, proposal.provenance, allow_pickle=False)
    np.save(confidence_path, proposal.warp_confidence, allow_pickle=False)
    np.save(scaled_depth_path, scaled_depth, allow_pickle=False)
    metadata_path.write_text(json.dumps(proposal.metadata_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    manifest = {
        "version": "0.1",
        "stage": "refworld-blendedmvs-learned-allvalid-scale-pair",
        "role": "g1a-development-depth-shape-plus-intrinsics-screen-not-end-to-end-single-image",
        "backend": source_geometry.backend,
        "scene_id": scene_root.name,
        "selection": {
            "pair_record_order": 1,
            "held_out_source_order": PRIMARY_HELD_OUT_RANK,
            "anchor_view_id": anchor_id,
            "target_view_id": target_id,
            "pair_score": pair_score,
            "separation_from_anchor": camera_pose_separation(benchmark_anchor, target_camera),
        },
        "method_inputs": {
            "anchor_rgb_sha256": anchor_sha,
            "source_geometry_manifest_sha256": _sha256(source_manifest_path),
            "anchor_camera_sha256": _sha256(anchor_camera_path),
            "anchor_depth_sha256": _sha256(anchor_depth_path),
            "target_camera_sha256": _sha256(target_camera_path),
            "target_rgb_read": False,
            "target_depth_read": False,
        },
        "scale_calibration": {
            "estimator": "median(reference_depth / predicted_depth) over all finite positive overlapping candidate-valid pixels",
            "top_confidence_fraction": ALL_VALID_FRACTION,
            "confidence_used_for_selection": False,
            "candidate_valid_fraction": float(np.mean(valid_mask)),
            "min_selected_pixels": MIN_SCALE_PIXELS,
            **scale.as_dict(),
            "allowed_transform": "positive multiplicative depth scalar only",
            "offset": None,
            "spatial_correction": None,
            "pose_refinement": None,
            "focal_correction": None,
        },
        "camera_bridge": {
            "source_extrinsics": "published BlendedMVS anchor extrinsics used only for benchmark-frame placement",
            "source_intrinsics": f"{source_geometry.backend} predicted intrinsics mapped to original source pixels",
            "model_predicted_extrinsics_used": False,
            "intrinsics_diagnostic": _intrinsics_summary(source_geometry.camera, benchmark_anchor),
        },
        "oracle_status": {
            "uses_published_anchor_depth": True,
            "anchor_depth_use": "one global scale scalar only",
            "uses_published_anchor_extrinsics": True,
            "uses_published_anchor_intrinsics_for_warp": False,
            "uses_published_target_camera": True,
            "claim_as_end_to_end_single_image_result": False,
        },
        "result": {
            "proposal_id": proposal.proposal_id,
            "observed_fraction": float(proposal.summary["observed_fraction"]),
            "unresolved_fraction": float(proposal.summary["unresolved_fraction"]),
            "view_directory": view_dir.relative_to(output).as_posix(),
        },
        "artifacts": [
            _artifact(image_path, output, "learned-allvalid-warp-png"),
            _artifact(provenance_path, output, "learned-allvalid-provenance-npy"),
            _artifact(confidence_path, output, "learned-allvalid-warp-confidence-npy"),
            _artifact(scaled_depth_path, output, "learned-allvalid-depth-npy"),
            _artifact(metadata_path, output, "learned-allvalid-proposal-json"),
        ],
    }
    manifest_path = output / "learned-allvalid-pair.safe.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(manifest_path, flush=True)
    print(view_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
