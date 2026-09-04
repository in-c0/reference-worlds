#!/usr/bin/env python3
"""Generate the two EXP-002 G1-D hybrid geometry ablations for one rank-3 pair.

This runner is diagnostic-only and uses already-opened G1 rank-3 cases. It does
not read target RGB or target depth. The two hybrids isolate learned depth shape
from learned source intrinsics while keeping the G1 scale estimator and benchmark
anchor frame placement fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from refworld.adapters.base import Camera
from refworld.datasets.mvsnet import PairRecord, parse_camera_text, parse_pair_text
from refworld.datasets.pfm import read_pfm
from refworld.geometry_scale import estimate_positive_depth_scale
from refworld.proposals import ObservationView, build_view_proposal
from refworld.repaints import NoRepaintBackend
from refworld.source_geometry import load_source_geometry
from refworld.warps import PinholeWarpBackend

HELD_OUT_RANK = 3
TOP_CONFIDENCE_FRACTION = 0.5
MIN_SCALE_PIXELS = 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_held_out(record: PairRecord) -> tuple[int, float]:
    if len(record.source_ids) < HELD_OUT_RANK:
        raise RuntimeError("first pair record has fewer than three source views")
    index = HELD_OUT_RANK - 1
    return int(record.source_ids[index]), float(record.scores[index])


def _write_condition(
    *,
    output: Path,
    name: str,
    anchor_rgb: np.ndarray,
    anchor_sha: str,
    source_camera: Camera,
    target_camera: Camera,
    depth: np.ndarray,
    anchor_stem: str,
    target_stem: str,
) -> dict[str, Any]:
    observation_id = f"g1d-{name}-" + anchor_sha[:16]
    observation = ObservationView(observation_id=observation_id, image=anchor_rgb, camera=source_camera)
    warper = PinholeWarpBackend({observation_id: np.asarray(depth, dtype=np.float32)}, name=f"g1d-{name}-pinhole@0.1")
    warp = warper.warp([observation], target_camera)
    repaint = NoRepaintBackend().repaint(warp, target_camera, seed=None)
    proposal = build_view_proposal([observation], target_camera, warp, repaint, unresolved_value=0)

    condition_dir = output / name / f"anchor-{anchor_stem}-target-{target_stem}"
    condition_dir.mkdir(parents=True, exist_ok=True)
    image_path = condition_dir / "proposal.png"
    provenance_path = condition_dir / "provenance.npy"
    confidence_path = condition_dir / "warp-confidence.npy"
    metadata_path = condition_dir / "proposal.json"

    from PIL import Image

    Image.fromarray(np.asarray(proposal.image, dtype=np.uint8), mode="RGB").save(image_path)
    np.save(provenance_path, proposal.provenance, allow_pickle=False)
    np.save(confidence_path, proposal.warp_confidence, allow_pickle=False)
    metadata_path.write_text(json.dumps(proposal.metadata_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    return {
        "condition": name,
        "view_directory": condition_dir.relative_to(output).as_posix(),
        "observed_fraction": float(proposal.summary["observed_fraction"]),
        "unresolved_fraction": float(proposal.summary["unresolved_fraction"]),
        "artifacts": {
            "proposal_png_sha256": _sha256(image_path),
            "provenance_npy_sha256": _sha256(provenance_path),
            "confidence_npy_sha256": _sha256(confidence_path),
            "proposal_json_sha256": _sha256(metadata_path),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate EXP-002 G1-D depth/intrinsics hybrid ablations")
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--source-geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene_root = args.scene_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    pair_path = scene_root / "cams" / "pair.txt"
    records = parse_pair_text(pair_path.read_text(encoding="utf-8"))
    if not records:
        raise RuntimeError("pair.txt has no records")
    first = records[0]
    anchor_id = int(first.reference_id)
    target_id, pair_score = _select_held_out(first)
    anchor_stem = f"{anchor_id:08d}"
    target_stem = f"{target_id:08d}"

    anchor_image_path = scene_root / "blended_images" / f"{anchor_stem}.jpg"
    anchor_camera_path = scene_root / "cams" / f"{anchor_stem}_cam.txt"
    anchor_depth_path = scene_root / "rendered_depth_maps" / f"{anchor_stem}.pfm"
    target_camera_path = scene_root / "cams" / f"{target_stem}_cam.txt"
    for required in (anchor_image_path, anchor_camera_path, anchor_depth_path, target_camera_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    from PIL import Image

    anchor_rgb = np.asarray(Image.open(anchor_image_path).convert("RGB"), dtype=np.uint8)
    anchor_depth = read_pfm(anchor_depth_path)
    if anchor_depth.shape != anchor_rgb.shape[:2]:
        raise ValueError("anchor RGB/depth shape mismatch")

    source_geometry = load_source_geometry(args.source_geometry)
    if source_geometry.backend != "vggt":
        raise ValueError(f"G1-D requires VGGT source geometry, got {source_geometry.backend!r}")
    anchor_sha = _sha256(anchor_image_path)
    if source_geometry.input_sha256 != anchor_sha:
        raise ValueError("source-geometry input hash does not match anchor RGB")
    if (source_geometry.height, source_geometry.width) != anchor_rgb.shape[:2]:
        raise ValueError("source-geometry dimensions do not match anchor RGB")

    benchmark_anchor = parse_camera_text(anchor_camera_path.read_text(encoding="utf-8")).camera
    target_camera = parse_camera_text(target_camera_path.read_text(encoding="utf-8")).camera

    scale = estimate_positive_depth_scale(
        source_geometry.depth,
        anchor_depth,
        source_geometry.confidence_raw,
        top_fraction=TOP_CONFIDENCE_FRACTION,
        min_selected=MIN_SCALE_PIXELS,
    )
    scaled_vggt_depth = np.asarray(source_geometry.depth, dtype=np.float32) * np.float32(scale.scale)
    if not np.all(np.isfinite(scaled_vggt_depth)) or np.any(scaled_vggt_depth <= 0.0):
        raise RuntimeError("scaled VGGT depth contains non-finite/non-positive values")

    source_camera_oracle_k = Camera(
        intrinsics=tuple(benchmark_anchor.intrinsics),
        extrinsics=tuple(benchmark_anchor.extrinsics),
        convention=benchmark_anchor.convention,
    )
    source_camera_vggt_k = Camera(
        intrinsics=tuple(source_geometry.camera.intrinsics),
        extrinsics=tuple(benchmark_anchor.extrinsics),
        convention=benchmark_anchor.convention,
    )

    depth_hybrid = _write_condition(
        output=output,
        name="vggt_depth_oracle_K",
        anchor_rgb=anchor_rgb,
        anchor_sha=anchor_sha,
        source_camera=source_camera_oracle_k,
        target_camera=target_camera,
        depth=scaled_vggt_depth,
        anchor_stem=anchor_stem,
        target_stem=target_stem,
    )
    intrinsics_hybrid = _write_condition(
        output=output,
        name="oracle_depth_vggt_K",
        anchor_rgb=anchor_rgb,
        anchor_sha=anchor_sha,
        source_camera=source_camera_vggt_k,
        target_camera=target_camera,
        depth=np.asarray(anchor_depth, dtype=np.float32),
        anchor_stem=anchor_stem,
        target_stem=target_stem,
    )

    source_manifest_path = Path(args.source_geometry).resolve()
    if source_manifest_path.is_dir():
        source_manifest_path = source_manifest_path / "source-geometry.safe.json"

    manifest = {
        "version": "0.1",
        "stage": "refworld-exp002-g1d-hybrid-generation",
        "scene_id": scene_root.name,
        "selection": {
            "pair_record_order": 1,
            "held_out_source_order": HELD_OUT_RANK,
            "anchor_view_id": anchor_id,
            "target_view_id": target_id,
            "pair_score": pair_score,
        },
        "diagnostic_scope": {
            "opened_rank3_reuse_only": True,
            "fresh_target_consumed": False,
            "target_rgb_read": False,
            "target_depth_read": False,
            "purpose": "decompose failed G1 geometry into depth-shape vs source-intrinsics contributions",
        },
        "fixed_inputs": {
            "anchor_rgb_sha256": anchor_sha,
            "anchor_camera_sha256": _sha256(anchor_camera_path),
            "anchor_depth_sha256": _sha256(anchor_depth_path),
            "target_camera_sha256": _sha256(target_camera_path),
            "vggt_source_geometry_manifest_sha256": _sha256(source_manifest_path),
            "scale_estimator": "same G1 median(reference_depth / vggt_depth) on exact top-50%-confidence valid pixels",
            "scale": scale.as_dict(),
            "anchor_extrinsics": "published BlendedMVS in both hybrids",
            "pose_refinement": None,
            "focal_fitting": None,
            "depth_offset": None,
            "spatial_depth_correction": None,
        },
        "conditions": [depth_hybrid, intrinsics_hybrid],
    }
    manifest_path = output / "g1d-hybrids.safe.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(manifest_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
