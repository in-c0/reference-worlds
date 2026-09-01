#!/usr/bin/env python3
"""Extract one source camera, depth map and raw confidence with pinned VGGT.

This runner is executed inside a CUDA-capable VGGT environment. It deliberately
produces geometry evidence only; it does not generate novel views or repaint
holes. VGGT's OpenCV camera-from-world pose is converted to RefWorld's canonical
OpenGL camera-to-world convention.

The source image remains the epistemic anchor. VGGT preprocessing is square and
padding-preserving; predicted intrinsics/depth/confidence are mapped back into
the original image pixel coordinate system before being written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

VGGT_PIN = "a288dd0f14786c93483e45524328726ab7b1b4ce"
VGGT_CHECKPOINT = "facebook/VGGT-1B"
MODEL_SIZE = 518  # divisible by VGGT's 14-pixel patch size


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot verify VGGT git commit") from exc
    return out.strip()


def artifact_record(path: Path, root: Path, kind: str) -> dict[str, Any]:
    resolved = path.resolve()
    base = root.resolve()
    try:
        rel = resolved.relative_to(base)
    except ValueError as exc:
        raise RuntimeError(f"artifact escaped output directory: {path}") from exc
    return {
        "kind": kind,
        "path": rel.as_posix(),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _squeeze_single_map(value: Any, *, name: str) -> np.ndarray:
    array = value.detach().float().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise RuntimeError(f"expected single {name} map, got shape {array.shape}")
    array = np.asarray(array, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise RuntimeError(f"{name} contains non-finite values")
    return array


def _original_from_square_transform(coords: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Return original->model pixel-center homography from VGGT square-loader coords."""

    values = np.asarray(coords, dtype=np.float64).reshape(-1)
    if values.size != 6:
        raise RuntimeError(f"expected six square-loader coordinates, got {values.size}")
    x1, y1, x2, y2, width_f, height_f = values
    width = int(round(float(width_f)))
    height = int(round(float(height_f)))
    if width <= 0 or height <= 0:
        raise RuntimeError("invalid original image dimensions from VGGT loader")
    sx = (x2 - x1) / width
    sy = (y2 - y1) / height
    if sx <= 0.0 or sy <= 0.0 or not np.isfinite([sx, sy]).all():
        raise RuntimeError("invalid VGGT preprocessing scale")

    # PIL resize uses pixel-center sampling. Integer original pixel x maps to:
    #   x_model = sx * (x + 0.5) + x1 - 0.5
    tx = x1 + 0.5 * sx - 0.5
    ty = y1 + 0.5 * sy - 0.5
    h_model_from_original = np.asarray(
        [[sx, 0.0, tx], [0.0, sy, ty], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return h_model_from_original, width, height


def _remap_to_original(array: np.ndarray, h_model_from_original: np.ndarray, width: int, height: int) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("VGGT source runner requires OpenCV for calibrated resampling") from exc

    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    sx = float(h_model_from_original[0, 0])
    sy = float(h_model_from_original[1, 1])
    tx = float(h_model_from_original[0, 2])
    ty = float(h_model_from_original[1, 2])
    map_x = sx * xx + tx
    map_y = sy * yy + ty
    result = cv2.remap(
        np.asarray(array, dtype=np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float("nan"),
    )
    if result.shape != (height, width):
        raise RuntimeError(f"unexpected remapped shape {result.shape}, expected {(height, width)}")
    if not np.all(np.isfinite(result)):
        raise RuntimeError("VGGT prediction did not cover every original source pixel after inverse preprocessing")
    return np.asarray(result, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract canonical source geometry with pinned VGGT")
    parser.add_argument("--vggt-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-unpinned-vggt", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    vggt_root = args.vggt_root.resolve()
    source = args.reference.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not source.is_file():
        raise FileNotFoundError(source)
    head = git_head(vggt_root)
    if head != VGGT_PIN and not args.allow_unpinned_vggt:
        raise RuntimeError(
            f"VGGT checkout is {head}; expected pinned {VGGT_PIN}. "
            "Use --allow-unpinned-vggt only for explicitly non-baseline experiments."
        )

    sys.path.insert(0, str(vggt_root))

    import torch
    from PIL import Image
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images_square
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    # Import only after the lightweight package is available in the environment.
    from refworld.registration import opencv_w2c_to_camera

    if not torch.cuda.is_available():
        raise RuntimeError("VGGT source extraction requires CUDA; torch.cuda.is_available() is false")

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
    gpu = torch.cuda.get_device_properties(0)
    dtype = torch.bfloat16 if gpu.major >= 8 else torch.float16

    image = Image.open(source).convert("RGB")
    original_width, original_height = image.size

    load_start = time.perf_counter()
    model = VGGT.from_pretrained(VGGT_CHECKPOINT).to(device).eval()
    model_load_seconds = time.perf_counter() - load_start

    images, original_coords = load_and_preprocess_images_square([str(source)], target_size=MODEL_SIZE)
    images = images.to(device)
    if tuple(images.shape[-2:]) != (MODEL_SIZE, MODEL_SIZE):
        raise RuntimeError(f"unexpected VGGT source tensor shape {tuple(images.shape)}")

    inference_start = time.perf_counter()
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            batched = images[None]
            aggregated_tokens_list, ps_idx = model.aggregator(batched)
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, batched.shape[-2:])
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, batched, ps_idx)
    inference_seconds = time.perf_counter() - inference_start

    extrinsic_cv = np.asarray(extrinsic.squeeze(0)[0].detach().float().cpu().numpy(), dtype=np.float64)
    intrinsic_model = np.asarray(intrinsic.squeeze(0)[0].detach().float().cpu().numpy(), dtype=np.float64)
    if extrinsic_cv.shape != (3, 4):
        raise RuntimeError(f"unexpected VGGT extrinsic shape {extrinsic_cv.shape}")
    if intrinsic_model.shape != (3, 3):
        raise RuntimeError(f"unexpected VGGT intrinsic shape {intrinsic_model.shape}")

    h_model_from_original, mapped_width, mapped_height = _original_from_square_transform(
        original_coords[0].detach().cpu().numpy()
    )
    if (mapped_width, mapped_height) != (original_width, original_height):
        raise RuntimeError(
            "VGGT loader original-size metadata disagrees with source image: "
            f"{(mapped_width, mapped_height)} != {(original_width, original_height)}"
        )

    intrinsic_original = np.linalg.inv(h_model_from_original) @ intrinsic_model
    intrinsic_original /= intrinsic_original[2, 2]

    depth_model = _squeeze_single_map(depth_map, name="depth")
    confidence_model = _squeeze_single_map(depth_conf, name="depth confidence")
    if np.any(depth_model <= 0.0):
        # Do not invent depth for invalid source support. Fail now so a later warp
        # cannot silently reinterpret invalid geometry as observed evidence.
        raise RuntimeError("VGGT source depth contains non-positive values")

    depth_original = _remap_to_original(
        depth_model,
        h_model_from_original,
        original_width,
        original_height,
    )
    confidence_original = _remap_to_original(
        confidence_model,
        h_model_from_original,
        original_width,
        original_height,
    )

    camera = opencv_w2c_to_camera(extrinsic_cv, intrinsic_original)

    depth_path = out / "depth.npy"
    confidence_path = out / "confidence-raw.npy"
    np.save(depth_path, depth_original, allow_pickle=False)
    np.save(confidence_path, confidence_original, allow_pickle=False)

    artifacts = [
        artifact_record(depth_path, out, "source-depth-npy"),
        artifact_record(confidence_path, out, "source-confidence-raw-npy"),
    ]

    manifest = {
        "version": "0.1",
        "stage": "refworld-source-geometry",
        "backend": "vggt",
        "upstream": {
            "repo": "facebookresearch/vggt",
            "expected_commit": VGGT_PIN,
            "actual_commit": head,
            "unpinned_allowed": bool(args.allow_unpinned_vggt),
            "checkpoint": VGGT_CHECKPOINT,
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
            "confidence_semantics": "raw VGGT depth confidence; upstream uses it as a ranking/percentile score; it is not normalized or treated as a calibrated probability",
            "confidence_calibration": None,
        },
        "preprocessing": {
            "loader": "load_and_preprocess_images_square",
            "model_size": MODEL_SIZE,
            "model_from_original_pixel_homography": h_model_from_original.reshape(-1).tolist(),
            "intrinsics_mapped_back_to_original_pixels": True,
            "depth_confidence_mapped_back_to_original_pixels": True,
        },
        "configuration": {
            "seed": int(args.seed),
            "dtype": str(dtype),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu_name": gpu.name,
            "gpu_total_bytes": int(gpu.total_memory),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "timing": {
            "model_load_seconds": model_load_seconds,
            "inference_seconds": inference_seconds,
        },
        "artifacts": artifacts,
        "notes": [
            "No novel-view generation or crack filling occurs in this stage.",
            "Raw confidence is intentionally preserved so calibration policy is explicit and reproducible.",
            "The original supplied image remains the observation anchor; VGGT square preprocessing is inverted for camera/depth coordinates.",
        ],
    }

    safe_path = out / "source-geometry.safe.json"
    safe_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(safe_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
