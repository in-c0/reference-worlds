#!/usr/bin/env python3
"""Export a deterministic RefWorld warp trajectory for WorldForge repaint guidance.

The exported masks are *generator guidance masks*: 255 means a pixel is supported
by RefWorld's strict source-geometry warp at that trajectory frame. They do not
change epistemic provenance and must never be interpreted as new observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from refworld.camera import pitch_camera, yaw_camera
from refworld.proposals import ObservationView
from refworld.source_geometry import load_source_geometry
from refworld.warps import PinholeWarpBackend

WORLD_FORGE_PIN = "ee573a051715a451b806a90e21462f23308faac4"
LONGCAT_PIN = "6b3f4b8582a8bc3f20f795735f5383716c4ba794"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare RefWorld warp frames for WorldForge")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--source-geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--axis", choices=("yaw", "pitch"), default="yaw")
    parser.add_argument("--degrees", type=float, default=5.0)
    parser.add_argument("--frames", type=int, default=49)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reference = args.reference.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    if not reference.is_file():
        raise FileNotFoundError(reference)
    if args.frames < 2 or args.frames > 241:
        raise ValueError("frames must lie in [2,241]")
    if not np.isfinite(args.degrees) or abs(args.degrees) > 45.0 or args.degrees == 0.0:
        raise ValueError("degrees must be finite, nonzero, and within [-45,45]")

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("refworld-worldforge-ref requires Pillow: pip install 'refworld-bench[method]'") from exc

    geometry = load_source_geometry(args.source_geometry)
    reference_sha = _sha256_file(reference)
    if reference_sha != geometry.input_sha256:
        raise ValueError("reference image SHA-256 does not match source geometry")

    image = np.asarray(Image.open(reference).convert("RGB"), dtype=np.uint8)
    if image.shape[:2] != (geometry.height, geometry.width):
        raise ValueError("reference dimensions do not match source geometry")

    observation_id = "obs-" + reference_sha[:16]
    observation = ObservationView(observation_id, image, geometry.camera)
    warper = PinholeWarpBackend({observation_id: geometry.depth})
    camera_fn = yaw_camera if args.axis == "yaw" else pitch_camera

    frame_records: list[dict[str, Any]] = []
    for index, degree in enumerate(np.linspace(0.0, float(args.degrees), args.frames)):
        target = camera_fn(geometry.camera, float(degree))
        warp = warper.warp([observation], target)

        frame_path = output / f"frame_{index:04d}.png"
        mask_path = output / f"mask_{index:04d}.png"
        Image.fromarray(np.asarray(warp.rgb, dtype=np.uint8)).save(frame_path)
        Image.fromarray((np.asarray(warp.observed_mask, dtype=np.uint8) * 255), mode="L").save(mask_path)

        frame_records.append(
            {
                "index": index,
                "axis": args.axis,
                "degrees": float(degree),
                "target_camera": {
                    "intrinsics": [float(v) for v in target.intrinsics],
                    "extrinsics": [float(v) for v in target.extrinsics],
                    "convention": target.convention,
                },
                "observed_fraction": float(np.mean(warp.observed_mask)),
                "frame": {
                    "path": frame_path.name,
                    "sha256": _sha256_file(frame_path),
                },
                "mask": {
                    "path": mask_path.name,
                    "sha256": _sha256_file(mask_path),
                    "semantics": "255=RefWorld geometric source-support guidance; 0=unsupported hole",
                },
            }
        )

    endpoint = frame_records[-1]
    manifest = {
        "version": "0.1",
        "stage": "refworld-worldforge-reference-trajectory",
        "purpose": "generator guidance only; does not create new observed evidence",
        "input": {
            "reference_file_name": reference.name,
            "reference_sha256": reference_sha,
            "observation_id": observation_id,
        },
        "trajectory": {
            "axis": args.axis,
            "endpoint_degrees": float(args.degrees),
            "frame_count": int(args.frames),
            "parameterization": "linear-angle-from-source-camera-center",
            "endpoint_target_camera": endpoint["target_camera"],
        },
        "guidance": {
            "frame_naming": "frame_####.png",
            "mask_naming": "mask_####.png",
            "mask_polarity": "255=geometric-support;0=hole",
            "mask_epistemic_status": "guidance-only; does not relabel generated pixels as observed",
            "crack_fill_applied": False,
            "raw_vggt_confidence_used": False,
        },
        "candidate_integration": {
            "expected_endpoint_frame_index": int(args.frames - 1),
            "compose_with": "refworld-compose-candidate",
            "held_out_rgb_allowed_as_input": False,
        },
        "planned_backend_pins": {
            "worldforge_commit": WORLD_FORGE_PIN,
            "longcat_commit": LONGCAT_PIN,
            "status": "provisional-until-checkpoint-terms-and-gpu-smoke-pass",
        },
        "frames": frame_records,
    }
    manifest_path = output / "worldforge-reference.safe.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
