#!/usr/bin/env python3
"""Create one calibrated BlendedMVS oracle-source-depth warp for B-vs-C evaluation.

This is an explicit diagnostic ablation, not the full single-image RefWorld-0
method. It uses the published anchor camera + anchor depth and the first
published held-out camera from the first pair.txt record. The held-out RGB and
held-out depth are not read here; they remain evaluation-only evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from refworld.datasets.mvsnet import camera_pose_separation, parse_camera_text, parse_pair_text
from refworld.datasets.pfm import read_pfm
from refworld.proposals import ObservationView, build_view_proposal
from refworld.repaints import NoRepaintBackend
from refworld.warps import PinholeWarpBackend


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create first calibrated BlendedMVS oracle-depth warp")
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
    target_id = int(first.source_ids[0])
    pair_score = float(first.scores[0])

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
        raise RuntimeError("oracle pair runner requires Pillow") from exc

    anchor_rgb = np.asarray(Image.open(anchor_image_path).convert("RGB"), dtype=np.uint8)
    anchor_depth = read_pfm(anchor_depth_path)
    if anchor_depth.ndim != 2:
        raise ValueError(f"anchor depth must be grayscale HxW, got {anchor_depth.shape}")
    if anchor_depth.shape != anchor_rgb.shape[:2]:
        raise ValueError(
            f"anchor depth/image shape mismatch: {anchor_depth.shape} != {anchor_rgb.shape[:2]}"
        )

    anchor_camera = parse_camera_text(anchor_camera_path.read_text(encoding="utf-8")).camera
    target_camera = parse_camera_text(target_camera_path.read_text(encoding="utf-8")).camera

    observation_id = "blendedmvs-anchor-" + _sha256(anchor_image_path)[:16]
    observation = ObservationView(
        observation_id=observation_id,
        image=anchor_rgb,
        camera=anchor_camera,
    )
    warper = PinholeWarpBackend(
        {observation_id: np.asarray(anchor_depth, dtype=np.float32)},
        name="blendedmvs-oracle-source-depth-pinhole@0.1",
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

    Image.fromarray(np.asarray(proposal.image, dtype=np.uint8), mode="RGB").save(image_path)
    np.save(provenance_path, proposal.provenance, allow_pickle=False)
    np.save(confidence_path, proposal.warp_confidence, allow_pickle=False)
    metadata_path.write_text(
        json.dumps(proposal.metadata_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    separation = camera_pose_separation(anchor_camera, target_camera)
    valid_depth = np.isfinite(anchor_depth) & (anchor_depth > 1e-6)
    manifest = {
        "version": "0.1",
        "stage": "refworld-blendedmvs-oracle-pair",
        "role": "calibrated-oracle-source-depth-diagnostic-not-full-single-image-method",
        "scene_id": scene_root.name,
        "selection": {
            "pair_record_order": 1,
            "held_out_source_order": 1,
            "anchor_view_id": anchor_id,
            "target_view_id": target_id,
            "pair_score": pair_score,
            "separation_from_anchor": separation,
        },
        "method_inputs": {
            "anchor_rgb_sha256": _sha256(anchor_image_path),
            "anchor_camera_sha256": _sha256(anchor_camera_path),
            "anchor_depth_sha256": _sha256(anchor_depth_path),
            "target_camera_sha256": _sha256(target_camera_path),
            "target_rgb_read": False,
            "target_depth_read": False,
        },
        "oracle_status": {
            "uses_published_anchor_depth": True,
            "uses_published_anchor_camera": True,
            "uses_published_target_camera": True,
            "valid_anchor_depth_fraction": float(np.mean(valid_depth)),
            "claim_as_single_image_method_result": False,
            "purpose": "isolate evidence-preservation effect from monocular depth/scale error",
        },
        "result": {
            "proposal_id": proposal.proposal_id,
            "observed_fraction": float(proposal.summary["observed_fraction"]),
            "unresolved_fraction": float(proposal.summary["unresolved_fraction"]),
            "view_directory": view_dir.relative_to(output).as_posix(),
        },
        "artifacts": [
            _artifact(image_path, output, "oracle-warp-png"),
            _artifact(provenance_path, output, "oracle-warp-provenance-npy"),
            _artifact(confidence_path, output, "oracle-warp-confidence-npy"),
            _artifact(metadata_path, output, "oracle-warp-proposal-json"),
        ],
    }
    manifest_path = output / "oracle-pair.safe.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(manifest_path, flush=True)
    print(view_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
