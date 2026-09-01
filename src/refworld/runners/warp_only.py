#!/usr/bin/env python3
"""Generate RefWorld-0 warp-only near-view proposals from verified source geometry.

This is ablation A: geometry only, no generative fill. Directly projected source
pixels become OBSERVED support; all disocclusions remain UNRESOLVED.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from refworld.neighborhood import (
    NearViewCamera,
    depth_normalized_translation_neighborhood,
    rotational_neighborhood,
)
from refworld.proposals import ObservationView, WarpResult, build_view_proposal
from refworld.repaints import NoRepaintBackend
from refworld.source_geometry import load_source_geometry
from refworld.warps import PinholeWarpBackend


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RefWorld-0 warp-only proposal neighborhood")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--source-geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--rotation-only",
        action="store_true",
        help="omit depth-normalized translations; useful for a scale-free diagnostic",
    )
    return parser.parse_args()


def _views(camera, depth: np.ndarray, *, rotation_only: bool) -> tuple[NearViewCamera, ...]:
    rotations = rotational_neighborhood(camera)
    if rotation_only:
        return rotations
    reference_depth = float(np.median(np.asarray(depth, dtype=np.float64)))
    translations = depth_normalized_translation_neighborhood(
        camera,
        reference_depth=reference_depth,
    )
    return tuple(rotations) + tuple(translations)


def main() -> int:
    args = parse_args()
    reference = args.reference.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    if not reference.is_file():
        raise FileNotFoundError(reference)

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("refworld-warp-only requires Pillow: pip install 'refworld-bench[method]'") from exc

    geometry = load_source_geometry(args.source_geometry)
    reference_sha = _sha256_file(reference)
    if reference_sha != geometry.input_sha256:
        raise ValueError("reference image SHA-256 does not match the verified source-geometry input")

    image = np.asarray(Image.open(reference).convert("RGB"), dtype=np.uint8)
    if image.shape[:2] != (geometry.height, geometry.width):
        raise ValueError(
            f"reference dimensions {image.shape[:2]} do not match source geometry "
            f"{(geometry.height, geometry.width)}"
        )

    observation_id = "obs-" + reference_sha[:16]
    observation = ObservationView(
        observation_id=observation_id,
        image=image,
        camera=geometry.camera,
    )
    warper = PinholeWarpBackend({observation_id: geometry.depth})
    no_repaint = NoRepaintBackend()

    view_records: list[dict[str, Any]] = []
    run_artifacts: list[dict[str, Any]] = []
    for near_view in _views(geometry.camera, geometry.depth, rotation_only=args.rotation_only):
        raw_warp = warper.warp([observation], near_view.camera)
        warp = WarpResult(
            rgb=raw_warp.rgb,
            observed_mask=raw_warp.observed_mask,
            confidence=raw_warp.confidence,
            backend=raw_warp.backend,
            metadata={
                **dict(raw_warp.metadata),
                "source_geometry_backend": geometry.backend,
                "source_geometry_input_sha256": geometry.input_sha256,
                "source_support_policy": "all-finite-positive-depth",
                "source_raw_confidence_policy": "recorded-but-not-used-for-support-or-weighting-v0",
            },
        )
        repaint = no_repaint.repaint(warp, near_view.camera, seed=None)
        proposal = build_view_proposal(
            [observation],
            near_view.camera,
            warp,
            repaint,
            unresolved_value=0,
        )

        view_dir = output / near_view.view_id
        view_dir.mkdir(parents=True, exist_ok=True)
        image_path = view_dir / "proposal.png"
        provenance_path = view_dir / "provenance.npy"
        confidence_path = view_dir / "warp-confidence.npy"
        metadata_path = view_dir / "proposal.json"

        Image.fromarray(np.asarray(proposal.image, dtype=np.uint8), mode="RGB").save(image_path)
        np.save(provenance_path, proposal.provenance, allow_pickle=False)
        np.save(confidence_path, proposal.warp_confidence, allow_pickle=False)
        metadata_path.write_text(
            json.dumps(proposal.metadata_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
        )

        artifacts = [
            _artifact(image_path, output, "proposal-png"),
            _artifact(provenance_path, output, "proposal-provenance-npy"),
            _artifact(confidence_path, output, "proposal-warp-confidence-npy"),
            _artifact(metadata_path, output, "proposal-metadata-json"),
        ]
        run_artifacts.extend(artifacts)
        view_records.append(
            {
                **near_view.metadata_dict(),
                "proposal_id": proposal.proposal_id,
                "proposal_metadata": metadata_path.relative_to(output).as_posix(),
                "observed_fraction": float(proposal.summary["observed_fraction"]),
                "unresolved_fraction": float(proposal.summary["unresolved_fraction"]),
                "artifacts": [item["path"] for item in artifacts],
            }
        )

    source_geometry_manifest = Path(args.source_geometry).resolve()
    if source_geometry_manifest.is_dir():
        source_geometry_manifest = source_geometry_manifest / "source-geometry.safe.json"

    manifest = {
        "version": "0.1",
        "stage": "refworld-warp-only",
        "ablation": "A-geometry-only",
        "input": {
            "reference_file_name": reference.name,
            "reference_sha256": reference_sha,
            "source_geometry_manifest_sha256": _sha256_file(source_geometry_manifest),
            "observation_id": observation_id,
        },
        "policies": {
            "repaint": "none",
            "source_support": "all-finite-positive-depth",
            "raw_vggt_confidence": "recorded-but-not-used-for-support-or-weighting-v0",
            "held_out_evaluation_images_used": False,
        },
        "neighborhood": {
            "rotation_only": bool(args.rotation_only),
            "translation_reference_depth_statistic": None
            if args.rotation_only
            else "median-positive-source-depth-model-units",
            "translation_reference_depth": None
            if args.rotation_only
            else float(np.median(geometry.depth)),
            "translation_units_metric": False,
        },
        "views": view_records,
        "artifacts": run_artifacts,
    }
    manifest_path = output / "warp-only.safe.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
