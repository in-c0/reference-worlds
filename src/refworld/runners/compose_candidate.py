#!/usr/bin/env python3
"""Compose an external repaint candidate with a persisted warp-only proposal.

This stage is generator-agnostic. A GPU backend may produce a full-frame target
candidate in another environment; RefWorld then applies the hard evidence rule
locally so directly observed support cannot be overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from refworld.adapters.base import Camera
from refworld.evidence import PixelProvenance
from refworld.proposals import (
    RepaintResult,
    WarpResult,
    build_view_proposal_from_parent_ids,
    hash_array,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path, root: Path, kind: str) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"output artifact escaped run directory: {path}") from exc
    return {
        "kind": kind,
        "path": relative.as_posix(),
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _camera_from_metadata(metadata: dict[str, Any]) -> Camera:
    target = metadata.get("target_camera")
    if not isinstance(target, dict):
        raise ValueError("warp proposal metadata has no target_camera")
    return Camera(
        intrinsics=tuple(float(v) for v in target.get("intrinsics", [])),
        extrinsics=tuple(float(v) for v in target.get("extrinsics", [])),
        convention=str(target.get("convention", "")),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply RefWorld evidence preservation to an external repaint candidate"
    )
    parser.add_argument("--warp-view", type=Path, required=True, help="one view directory from refworld-warp-only")
    parser.add_argument("--candidate", type=Path, required=True, help="model-generated target RGB image")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", required=True, help="explicit generator/backend identifier")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--valid-mask-npy",
        type=Path,
        default=None,
        help="optional boolean HxW mask of accepted generated pixels; omitted means full-frame candidate",
    )
    parser.add_argument(
        "--backend-run-id",
        default=None,
        help="optional external run/checkpoint identifier for provenance",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    warp_view = args.warp_view.resolve()
    candidate_path = args.candidate.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    if not args.backend.strip():
        raise ValueError("--backend cannot be empty")
    if not warp_view.is_dir():
        raise NotADirectoryError(warp_view)
    if not candidate_path.is_file():
        raise FileNotFoundError(candidate_path)

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("refworld-compose-candidate requires Pillow: pip install 'refworld-bench[method]'") from exc

    warp_image_path = warp_view / "proposal.png"
    warp_provenance_path = warp_view / "provenance.npy"
    warp_confidence_path = warp_view / "warp-confidence.npy"
    warp_metadata_path = warp_view / "proposal.json"
    for required in (
        warp_image_path,
        warp_provenance_path,
        warp_confidence_path,
        warp_metadata_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    warp_meta = json.loads(warp_metadata_path.read_text())
    parent_ids = warp_meta.get("parent_observation_ids")
    if not isinstance(parent_ids, list) or not parent_ids:
        raise ValueError("warp proposal has no parent observation lineage")
    target_camera = _camera_from_metadata(warp_meta)

    warp_rgb = np.asarray(Image.open(warp_image_path).convert("RGB"), dtype=np.uint8)
    provenance = np.load(warp_provenance_path, allow_pickle=False)
    warp_confidence = np.load(warp_confidence_path, allow_pickle=False)
    if provenance.shape != warp_rgb.shape[:2]:
        raise ValueError("warp provenance shape does not match warp image")
    if warp_confidence.shape != warp_rgb.shape[:2]:
        raise ValueError("warp confidence shape does not match warp image")

    known_codes = {
        int(PixelProvenance.UNRESOLVED),
        int(PixelProvenance.OBSERVED),
        int(PixelProvenance.GENERATED),
    }
    values = {int(v) for v in np.unique(provenance)}
    if values - known_codes:
        raise ValueError(f"warp proposal contains unknown provenance codes: {sorted(values - known_codes)}")
    if int(PixelProvenance.GENERATED) in values:
        raise ValueError("compose-candidate requires a geometry-only warp proposal with no GENERATED pixels")
    observed_mask = provenance == int(PixelProvenance.OBSERVED)

    candidate_rgb = np.asarray(Image.open(candidate_path).convert("RGB"), dtype=np.uint8)
    if candidate_rgb.shape != warp_rgb.shape:
        raise ValueError(
            f"candidate shape {candidate_rgb.shape} does not match target warp shape {warp_rgb.shape}"
        )

    if args.valid_mask_npy is None:
        valid_mask = np.ones(warp_rgb.shape[:2], dtype=bool)
        valid_mask_policy = "full-frame-candidate"
        valid_mask_input_sha256 = None
    else:
        valid_mask_path = args.valid_mask_npy.resolve()
        if not valid_mask_path.is_file():
            raise FileNotFoundError(valid_mask_path)
        valid_mask = np.load(valid_mask_path, allow_pickle=False)
        if valid_mask.shape != warp_rgb.shape[:2] or valid_mask.dtype != np.bool_:
            raise ValueError("valid repaint mask must be boolean HxW matching target image")
        valid_mask_policy = "external-boolean-mask"
        valid_mask_input_sha256 = _sha256_file(valid_mask_path)

    warp_backend = str(warp_meta.get("backends", {}).get("warp", "")).strip()
    if not warp_backend:
        raise ValueError("warp proposal metadata is missing its warp backend")
    warp = WarpResult(
        rgb=warp_rgb,
        observed_mask=observed_mask,
        confidence=np.asarray(warp_confidence, dtype=np.float32),
        backend=warp_backend,
        metadata={
            **dict(warp_meta.get("backend_metadata", {}).get("warp", {})),
            "persisted_warp_proposal_id": warp_meta.get("proposal_id"),
            "persisted_warp_metadata_sha256": _sha256_file(warp_metadata_path),
        },
    )
    repaint = RepaintResult(
        rgb=candidate_rgb,
        valid_mask=valid_mask,
        backend=args.backend.strip(),
        seed=args.seed,
        metadata={
            "candidate_origin": "model-generated",
            "candidate_file_sha256": _sha256_file(candidate_path),
            "candidate_semantic_sha256": hash_array(candidate_rgb),
            "valid_mask_policy": valid_mask_policy,
            "valid_mask_input_sha256": valid_mask_input_sha256,
            "backend_run_id": args.backend_run_id,
            "held_out_evaluation_image_declared_as_input": False,
        },
    )

    proposal = build_view_proposal_from_parent_ids(
        parent_ids,
        target_camera,
        warp,
        repaint,
        unresolved_value=0,
    )

    unrestricted_path = output / "candidate-unrestricted.png"
    preserved_path = output / "proposal-evidence-preserved.png"
    provenance_out_path = output / "provenance.npy"
    metadata_out_path = output / "proposal.json"

    Image.fromarray(candidate_rgb).save(unrestricted_path)
    Image.fromarray(np.asarray(proposal.image, dtype=np.uint8)).save(preserved_path)
    np.save(provenance_out_path, proposal.provenance, allow_pickle=False)
    metadata_out_path.write_text(
        json.dumps(proposal.metadata_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )

    artifacts = [
        _artifact(unrestricted_path, output, "candidate-unrestricted-png"),
        _artifact(preserved_path, output, "proposal-evidence-preserved-png"),
        _artifact(provenance_out_path, output, "proposal-provenance-npy"),
        _artifact(metadata_out_path, output, "proposal-metadata-json"),
    ]
    manifest = {
        "version": "0.1",
        "stage": "refworld-compose-candidate",
        "ablation_pair": {
            "B": "unrestricted-repaint-candidate",
            "C": "evidence-preserving-repaint",
        },
        "inputs": {
            "warp_proposal_id": warp_meta.get("proposal_id"),
            "warp_metadata_sha256": _sha256_file(warp_metadata_path),
            "candidate_file_sha256": _sha256_file(candidate_path),
            "candidate_backend": args.backend.strip(),
            "candidate_seed": args.seed,
            "backend_run_id": args.backend_run_id,
        },
        "result": {
            "proposal_id": proposal.proposal_id,
            "observed_fraction": float(proposal.summary["observed_fraction"]),
            "generated_fraction": float(proposal.summary["generated_fraction"]),
            "unresolved_fraction": float(proposal.summary["unresolved_fraction"]),
            "overlap_attempt_pixels": int(proposal.summary["overlap_attempt_pixels"]),
            "observed_pixels_preserved_bitwise": True,
        },
        "guardrails": {
            "candidate_origin_declared": "model-generated",
            "held_out_evaluation_image_used": False,
            "observed_pixels_can_be_overwritten": False,
        },
        "artifacts": artifacts,
    }
    manifest_path = output / "compose.safe.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
