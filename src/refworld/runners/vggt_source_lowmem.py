#!/usr/bin/env python3
"""Pinned VGGT source geometry with FP16 model weights for memory-fit diagnostics.

This is a G1-R0 diagnostic runner. It preserves the existing source-geometry
contract but converts the pinned VGGT model weights to FP16 before inference so
that the frozen 518px input may fit an RTX-2080-class GPU. It is not a new model
or benchmark baseline and must be compared against the existing 392px path.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np

from refworld.runners.vggt_source import (
    DEFAULT_MODEL_SIZE,
    PATCH_SIZE,
    VGGT_CHECKPOINT,
    VGGT_PIN,
    _original_from_square_transform,
    _remap_to_original,
    _squeeze_single_map,
    artifact_record,
    git_head,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract pinned VGGT source geometry with FP16 model weights")
    parser.add_argument("--vggt-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default=VGGT_CHECKPOINT)
    parser.add_argument("--model-size", type=int, choices=(392, 518), required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_size = int(args.model_size)
    if model_size > DEFAULT_MODEL_SIZE or model_size % PATCH_SIZE != 0:
        raise ValueError("invalid VGGT model size")

    vggt_root = args.vggt_root.resolve()
    source = args.reference.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not source.is_file():
        raise FileNotFoundError(source)
    head = git_head(vggt_root)
    if head != VGGT_PIN:
        raise RuntimeError(f"VGGT checkout is {head}; expected pinned {VGGT_PIN}")

    sys.path.insert(0, str(vggt_root))
    import torch
    from PIL import Image
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images_square
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from refworld.registration import opencv_w2c_to_camera

    if not torch.cuda.is_available():
        raise RuntimeError("G1-R0 low-memory VGGT requires CUDA")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    device = torch.device("cuda")
    dtype = torch.float16
    gpu = torch.cuda.get_device_properties(0)
    image = Image.open(source).convert("RGB")
    original_width, original_height = image.size

    checkpoint_source = str(args.checkpoint)
    checkpoint_is_local = Path(checkpoint_source).exists()
    print(f"Loading pinned VGGT then converting weights to FP16 for {model_size}x{model_size}...", flush=True)
    load_start = time.perf_counter()
    model = VGGT.from_pretrained(checkpoint_source).to(device=device, dtype=dtype).eval()
    model_load_seconds = time.perf_counter() - load_start

    images, original_coords = load_and_preprocess_images_square([str(source)], target_size=model_size)
    images = images.to(device=device, dtype=dtype)
    if tuple(images.shape[-2:]) != (model_size, model_size):
        raise RuntimeError(f"unexpected VGGT input shape {tuple(images.shape)}")

    inference_start = time.perf_counter()
    with torch.inference_mode():
        with torch.cuda.amp.autocast(dtype=dtype):
            batched = images[None]
            aggregated_tokens_list, ps_idx = model.aggregator(batched)
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, batched.shape[-2:])
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, batched, ps_idx)
    inference_seconds = time.perf_counter() - inference_start
    print(f"VGGT low-memory tensor inference completed in {inference_seconds:.2f}s.", flush=True)

    extrinsic_cv = np.asarray(extrinsic.squeeze(0)[0].detach().float().cpu().numpy(), dtype=np.float64)
    intrinsic_model = np.asarray(intrinsic.squeeze(0)[0].detach().float().cpu().numpy(), dtype=np.float64)
    if extrinsic_cv.shape != (3, 4) or intrinsic_model.shape != (3, 3):
        raise RuntimeError("unexpected VGGT camera output shapes")

    h_model_from_original, mapped_width, mapped_height = _original_from_square_transform(
        original_coords[0].detach().cpu().numpy()
    )
    if (mapped_width, mapped_height) != (original_width, original_height):
        raise RuntimeError("VGGT loader original dimensions disagree with source image")
    intrinsic_original = np.linalg.inv(h_model_from_original) @ intrinsic_model
    intrinsic_original /= intrinsic_original[2, 2]

    depth_model = _squeeze_single_map(depth_map, name="depth")
    confidence_model = _squeeze_single_map(depth_conf, name="depth confidence")
    if np.any(depth_model <= 0.0):
        raise RuntimeError("VGGT low-memory depth contains non-positive values")
    depth_original = _remap_to_original(depth_model, h_model_from_original, original_width, original_height)
    confidence_original = _remap_to_original(
        confidence_model, h_model_from_original, original_width, original_height
    )
    camera = opencv_w2c_to_camera(extrinsic_cv, intrinsic_original)

    depth_path = out / "depth.npy"
    confidence_path = out / "confidence-raw.npy"
    np.save(depth_path, depth_original, allow_pickle=False)
    np.save(confidence_path, confidence_original, allow_pickle=False)

    manifest = {
        "version": "0.1",
        "stage": "refworld-source-geometry",
        "backend": "vggt",
        "upstream": {
            "repo": "facebookresearch/vggt",
            "expected_commit": VGGT_PIN,
            "actual_commit": head,
            "unpinned_allowed": False,
            "checkpoint": VGGT_CHECKPOINT,
            "checkpoint_load_mode": "local-prefetched" if checkpoint_is_local else "hub",
        },
        "input": {
            "file_name": source.name,
            "sha256": sha256_file(source),
            "size_bytes": source.stat().st_size,
            "width": original_width,
            "height": original_height,
        },
        "camera": {
            "intrinsics": list(camera.intrinsics),
            "extrinsics": list(camera.extrinsics),
            "convention": camera.convention,
            "upstream_extrinsic_convention": "opencv-camera-from-world",
        },
        "geometry": {
            "depth_semantics": "positive optical-axis Z depth in the VGGT/OpenCV source camera; single-view scale is not assumed metric",
            "confidence_semantics": "raw VGGT depth confidence; ranking score only, not a calibrated probability",
            "confidence_calibration": None,
        },
        "preprocessing": {
            "loader": "load_and_preprocess_images_square",
            "model_size": model_size,
            "benchmark_default_model_size": DEFAULT_MODEL_SIZE,
            "reduced_resolution_smoke_only": model_size != DEFAULT_MODEL_SIZE,
            "model_from_original_pixel_homography": h_model_from_original.reshape(-1).tolist(),
            "intrinsics_mapped_back_to_original_pixels": True,
            "depth_confidence_mapped_back_to_original_pixels": True,
            "prediction_resampling": "bilinear model-to-original with existing RefWorld footprint guard",
        },
        "configuration": {
            "seed": int(args.seed),
            "compute_dtype": str(dtype),
            "model_weight_dtype": str(dtype),
            "g1r0_low_memory_diagnostic": True,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu_name": gpu.name,
            "gpu_total_bytes": int(gpu.total_memory),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "timing": {"model_load_seconds": model_load_seconds, "inference_seconds": inference_seconds},
        "artifacts": [
            artifact_record(depth_path, out, "source-depth-npy"),
            artifact_record(confidence_path, out, "source-confidence-raw-npy"),
        ],
    }
    manifest_path = out / "source-geometry.safe.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(manifest_path, flush=True)
    print(f"Peak reserved: {manifest['environment']['peak_reserved_bytes'] / (1024**3):.2f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
