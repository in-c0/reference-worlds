#!/usr/bin/env python3
"""Create one BlendedMVS warp using VGGT depth shape plus one oracle scale scalar.

This is the frozen EXP-002 G1 bridge diagnostic. It is intentionally *not* an
end-to-end single-image result: published anchor depth supplies exactly one
positive multiplicative scale scalar, and published anchor extrinsics place the
single-view prediction in the benchmark world frame. Target RGB/depth are never
read by this runner.
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
TOP_CONFIDENCE_FRACTION = 0.5
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
        raise ValueError(
            f"held-out rank must be in [1,{len(record.source_ids)}] for this pair record, got {rank}"
        )
    index = rank - 1
    return int(record.source_ids[index]), float(record.scores[index])


def _intrinsics_summary(vggt_camera: Camera, benchmark_camera: Camera) -> dict[str, Any]:
    kv = np.asarray(vggt_camera.intrinsics, dtype=np.float64).reshape(3, 3)
    kb = np.asarray(benchmark_camera.intrinsics, dtype=np.float64).reshape(3, 3)
    return {
        "vggt": kv.reshape(-1).tolist(),
        "benchmark": kb.reshape(-1).tolist(),
        "delta": (kv - kb).reshape(-1).tolist(),
        "relative_focal_delta": {
            "fx": float((kv[0, 0] - kb[0, 0]) / kb[0, 0]),
            "fy": float((kv[1, 1] - kb[1, 1]) / kb[1, 1]),
        },
        "principal_point_delta_px": {
            "cx": float(kv[0, 2] - kb[0, 2]),
            "cy": float(kv[1, 2] - kb[1, 2]),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create calibrated BlendedMVS warp from VGGT depth shape plus one oracle scale scalar"
    )
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument(
        "--source-geometry",
        type=Path,
        required=True,
        help="verified VGGT source-geometry.safe.json or its containing directory",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--held-out-rank",
        type=int,
        default=PRIMARY_HELD_OUT_RANK,
        help="1-based published source rank; frozen primary G1 protocol uses rank 3",
    )
    parser.add_argument(
        "--allow-nonprimary-rank",
        action="store_true",
        help="permit exploratory ranks other than the frozen primary rank 3; manifest remains non-primary",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    held_out_rank = int(args.held_out_rank)
    if held_out_rank != PRIMARY_HELD_OUT_RANK and not args.allow_nonprimary_rank:
        raise ValueError(
            f"frozen G1 primary protocol requires held-out rank {PRIMARY_HELD_OUT_RANK}; "
            "pass --allow-nonprimary-rank only for explicitly exploratory work"
        )

    scene_root = args.scene_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    pair_path = scene_root / "cams" / "pair.txt"
    if not pair_path.is_file():
        raise FileNotFoundError(pair_path)
    records = parse_pair_text(pair_path.read_text(encoding="utf-8"))
    if not records or not records[0].source_ids:
        raise RuntimeError("first pair record has no held-out target")
    first = records[0]
    anchor_id = int(first.reference_id)
    target_id, pair_score = _select_held_out(first, held_out_rank)

    anchor_stem = f"{anchor_id:08d}"
    target_stem = f"{target_id:08d}"
    anchor_image_path = scene_root / "blended_images" / f"{anchor_stem}.jpg"
    anchor_camera_path = scene_root / "cams" / f"{anchor_stem}_cam.txt"
    anchor_depth_path = scene_root / "rendered_depth_maps" / f"{anchor_stem}.pfm"
    target_camera_path = scene_root / "cams" / f"{target_stem}_cam.txt"
    for required in (anchor_image_path, anchor_camera_path, anchor_depth_path, target_camera_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("VGGT scaled pair runner requires Pillow") from exc

    anchor_rgb = np.asarray(Image.open(anchor_image_path).convert("RGB"), dtype=np.uint8)
    anchor_depth = read_pfm(anchor_depth_path)
    if anchor_depth.ndim != 2:
        raise ValueError(f"anchor depth must be grayscale HxW, got {anchor_depth.shape}")
    if anchor_depth.shape != anchor_rgb.shape[:2]:
        raise ValueError(
            f"anchor depth/image shape mismatch: {anchor_depth.shape} != {anchor_rgb.shape[:2]}"
        )

    source_geometry = load_source_geometry(args.source_geometry)
    if source_geometry.backend != "vggt":
        raise ValueError(f"G1 requires VGGT source geometry, got backend={source_geometry.backend!r}")
    anchor_sha = _sha256(anchor_image_path)
    if source_geometry.input_sha256 != anchor_sha:
        raise ValueError("VGGT source-geometry input hash does not match the BlendedMVS anchor RGB")
    if (source_geometry.height, source_geometry.width) != anchor_rgb.shape[:2]:
        raise ValueError(
            "VGGT source-geometry dimensions do not match anchor RGB: "
            f"{(source_geometry.height, source_geometry.width)} != {anchor_rgb.shape[:2]}"
        )

    benchmark_anchor = parse_camera_text(anchor_camera_path.read_text(encoding="utf-8")).camera
    target_camera = parse_camera_text(target_camera_path.read_text(encoding="utf-8")).camera

    scale = estimate_positive_depth_scale(
        source_geometry.depth,
        anchor_depth,
        source_geometry.confidence_raw,
        top_fraction=TOP_CONFIDENCE_FRACTION,
        min_selected=MIN_SCALE_PIXELS,
    )
    scaled_depth = np.asarray(source_geometry.depth, dtype=np.float32) * np.float32(scale.scale)
    if not np.all(np.isfinite(scaled_depth)) or np.any(scaled_depth <= 0.0):
        raise RuntimeError("scaled VGGT depth contains non-finite or non-positive values")

    learned_source_camera = Camera(
        intrinsics=tuple(source_geometry.camera.intrinsics),
        extrinsics=tuple(benchmark_anchor.extrinsics),
        convention=benchmark_anchor.convention,
    )

    observation_id = "blendedmvs-vggt-anchor-" + anchor_sha[:16]
    observation = ObservationView(
        observation_id=observation_id,
        image=anchor_rgb,
        camera=learned_source_camera,
    )
    warper = PinholeWarpBackend(
        {observation_id: scaled_depth},
        name="blendedmvs-vggt-oracle-scale-pinhole@0.1",
    )
    warp = warper.warp([observation], target_camera)
    repaint = NoRepaintBackend().repaint(warp, target_camera, seed=None)
    proposal = build_view_proposal(
        [observation],
        target_camera,
        warp,
        repaint,
        unresolved_value=0,
    )

    view_dir = output / f"anchor-{anchor_stem}-target-{target_stem}"
    view_dir.mkdir(parents=True, exist_ok=True)
    image_path = view_dir / "proposal.png"
    provenance_path = view_dir / "provenance.npy"
    confidence_path = view_dir / "warp-confidence.npy"
    metadata_path = view_dir / "proposal.json"
    scaled_depth_path = view_dir / "vggt-depth-scaled.npy"

    Image.fromarray(np.asarray(proposal.image, dtype=np.uint8), mode="RGB").save(image_path)
    np.save(provenance_path, proposal.provenance, allow_pickle=False)
    np.save(confidence_path, proposal.warp_confidence, allow_pickle=False)
    np.save(scaled_depth_path, scaled_depth, allow_pickle=False)
    metadata_path.write_text(
        json.dumps(proposal.metadata_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    source_manifest_path = Path(args.source_geometry).resolve()
    if source_manifest_path.is_dir():
        source_manifest_path = source_manifest_path / "source-geometry.safe.json"

    manifest = {
        "version": "0.1",
        "stage": "refworld-blendedmvs-vggt-oracle-scale-pair",
        "role": "g1-vggt-depth-shape-intrinsics-oracle-scale-and-frame-diagnostic-not-end-to-end-single-image",
        "scene_id": scene_root.name,
        "primary_protocol": held_out_rank == PRIMARY_HELD_OUT_RANK,
        "selection": {
            "pair_record_order": 1,
            "held_out_source_order": held_out_rank,
            "anchor_view_id": anchor_id,
            "target_view_id": target_id,
            "pair_score": pair_score,
            "separation_from_anchor": camera_pose_separation(benchmark_anchor, target_camera),
        },
        "method_inputs": {
            "anchor_rgb_sha256": anchor_sha,
            "vggt_source_geometry_manifest_sha256": _sha256(source_manifest_path),
            "anchor_camera_sha256": _sha256(anchor_camera_path),
            "anchor_depth_sha256": _sha256(anchor_depth_path),
            "target_camera_sha256": _sha256(target_camera_path),
            "target_rgb_read": False,
            "target_depth_read": False,
        },
        "scale_calibration": {
            "estimator": "median(reference_depth / vggt_depth) on exact top-50%-confidence valid pixels",
            "top_confidence_fraction": TOP_CONFIDENCE_FRACTION,
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
            "source_intrinsics": "VGGT-predicted intrinsics mapped to original source pixels",
            "vggt_predicted_extrinsics_used": False,
            "intrinsics_diagnostic": _intrinsics_summary(source_geometry.camera, benchmark_anchor),
        },
        "oracle_status": {
            "uses_published_anchor_depth": True,
            "anchor_depth_use": "one global depth-scale scalar only",
            "uses_published_anchor_extrinsics": True,
            "uses_published_anchor_intrinsics_for_warp": False,
            "uses_published_target_camera": True,
            "claim_as_end_to_end_single_image_result": False,
            "purpose": "falsify VGGT depth-shape/intrinsics adequacy before solving monocular scale identification",
        },
        "result": {
            "proposal_id": proposal.proposal_id,
            "observed_fraction": float(proposal.summary["observed_fraction"]),
            "unresolved_fraction": float(proposal.summary["unresolved_fraction"]),
            "view_directory": view_dir.relative_to(output).as_posix(),
        },
        "artifacts": [
            _artifact(image_path, output, "vggt-oracle-scale-warp-png"),
            _artifact(provenance_path, output, "vggt-oracle-scale-provenance-npy"),
            _artifact(confidence_path, output, "vggt-oracle-scale-warp-confidence-npy"),
            _artifact(scaled_depth_path, output, "vggt-oracle-scale-depth-npy"),
            _artifact(metadata_path, output, "vggt-oracle-scale-proposal-json"),
        ],
    }
    manifest_path = output / "vggt-oracle-scale-pair.safe.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(manifest_path, flush=True)
    print(view_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
