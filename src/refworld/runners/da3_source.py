#!/usr/bin/env python3
"""Extract RefWorld source geometry from the frozen DA3-BASE candidate.

EXP-002 G1-B is development-only. DA3 predicts depth and camera intrinsics from
one anchor RGB. Its native relative scale and predicted pose are not used to earn
the G1-B gate; the generic later bridge fits the same one oracle scale scalar and
uses the same published anchor frame placement as the equalized VGGT reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from refworld.adapters.base import Camera
from refworld.camera import OPENGL_C2W

DA3_PIN = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
MODEL_REPO = "depth-anything/DA3-BASE"
MODEL_REVISION = "ee22d50d2aeb9a58c06b2079d2d27bc220e801aa"
MODEL_SHA256 = "e01067dc1659613083d9145a9a2547ccdbe6ccbbf83c4fe7b3e8a4e2bdae78b5"
MODEL_SIZE_BYTES = 541_518_028
ALLOWED_PROCESS_RES = (504, 392, 336)
UPSTREAM_SKY_THRESHOLD = 0.3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot verify DA3 git commit") from exc


def artifact_record(path: Path, root: Path, kind: str) -> dict[str, Any]:
    resolved = path.resolve()
    relative = resolved.relative_to(root.resolve())
    return {
        "kind": kind,
        "path": relative.as_posix(),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _map_intrinsics_to_original(
    intrinsics_processed: np.ndarray,
    *,
    processed_width: int,
    processed_height: int,
    original_width: int,
    original_height: int,
) -> np.ndarray:
    """Map DA3 pixel-space K from processed pixels to original image pixels."""
    if min(processed_width, processed_height, original_width, original_height) <= 0:
        raise ValueError("image dimensions must be positive")
    k = np.asarray(intrinsics_processed, dtype=np.float64).reshape(3, 3).copy()
    if not np.all(np.isfinite(k)):
        raise ValueError("DA3 intrinsics must be finite")
    sx = float(original_width) / float(processed_width)
    sy = float(original_height) / float(processed_height)
    k[0, :] *= sx
    k[1, :] *= sy
    if abs(float(k[0, 1])) <= 1e-6:
        k[0, 1] = 0.0
    if abs(float(k[1, 0])) <= 1e-6:
        k[1, 0] = 0.0
    if abs(float(k[0, 1])) > 1e-6 or abs(float(k[1, 0])) > 1e-6:
        raise ValueError("G1-B pinhole bridge requires zero-skew DA3 intrinsics")
    k[2, :] = [0.0, 0.0, 1.0]
    if k[0, 0] <= 0.0 or k[1, 1] <= 0.0:
        raise ValueError("DA3 focal lengths must be positive")
    return k


def _valid_aware_resize_depth(
    depth_processed: np.ndarray,
    valid_processed: np.ndarray,
    *,
    original_width: int,
    original_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize depth while refusing interpolation across invalid support.

    RefWorld source-geometry storage requires a finite positive array everywhere,
    so this helper returns the real validity mask separately. The caller may fill
    invalid storage positions only after this function; the G1-B bridge consumes
    the mask and excludes them from scale fitting and warping.
    """
    import cv2

    depth = np.asarray(depth_processed, dtype=np.float64)
    valid = np.asarray(valid_processed, dtype=bool)
    if depth.ndim != 2 or depth.shape != valid.shape or depth.size == 0:
        raise ValueError("processed depth and validity must be matching non-empty HxW arrays")
    if original_width <= 0 or original_height <= 0:
        raise ValueError("original dimensions must be positive")
    if np.any(valid & (~np.isfinite(depth) | (depth <= 0.0))):
        raise ValueError("DA3-valid depth must be finite and positive")
    if not np.any(valid):
        raise ValueError("DA3 validity mask is empty")

    target_size = (int(original_width), int(original_height))
    numerator = cv2.resize(
        np.where(valid, depth, 0.0).astype(np.float32),
        target_size,
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float64)
    support = cv2.resize(
        valid.astype(np.float32),
        target_size,
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float64)
    # Requiring virtually full interpolation support prevents a valid pixel from
    # borrowing geometry across an invalid/sky boundary.
    valid_original = support >= 0.999
    resized = np.full((original_height, original_width), np.nan, dtype=np.float64)
    safe = valid_original & np.isfinite(numerator) & (support > 0.0)
    resized[safe] = numerator[safe] / support[safe]
    valid_original = safe & np.isfinite(resized) & (resized > 0.0)
    resized[~valid_original] = np.nan
    if not np.any(valid_original):
        raise ValueError("DA3 validity vanished during original-pixel remapping")
    return resized.astype(np.float32), valid_original


def _resize_confidence(
    confidence_processed: np.ndarray,
    valid_processed: np.ndarray,
    valid_original: np.ndarray,
    *,
    original_width: int,
    original_height: int,
) -> np.ndarray:
    import cv2

    conf = np.asarray(confidence_processed, dtype=np.float64)
    valid = np.asarray(valid_processed, dtype=bool)
    if conf.shape != valid.shape:
        raise ValueError("DA3 confidence/validity shape mismatch")
    clean = np.where(valid & np.isfinite(conf), conf, 0.0).astype(np.float32)
    resized = cv2.resize(
        clean,
        (int(original_width), int(original_height)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    resized[~np.isfinite(resized)] = 0.0
    resized[~np.asarray(valid_original, dtype=bool)] = 0.0
    return resized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract pinned DA3-BASE source geometry")
    parser.add_argument("--da3-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--process-res", type=int, default=504)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    da3_root = args.da3_root.resolve()
    source = args.reference.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    process_res = int(args.process_res)
    if process_res not in ALLOWED_PROCESS_RES:
        raise ValueError(f"G1-B process resolution must be one of {ALLOWED_PROCESS_RES}")
    if not source.is_file():
        raise FileNotFoundError(source)
    actual_commit = git_head(da3_root)
    if actual_commit != DA3_PIN:
        raise RuntimeError(f"DA3 checkout is {actual_commit}; expected {DA3_PIN}")

    try:
        import torch
        from PIL import Image
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("DA3 adapter requires torch, Pillow and huggingface_hub") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("DA3 G1-B extraction requires CUDA")

    snapshot = Path(
        snapshot_download(
            repo_id=MODEL_REPO,
            revision=MODEL_REVISION,
            allow_patterns=["config.json", "model.safetensors"],
        )
    ).resolve()
    checkpoint = snapshot / "model.safetensors"
    config_path = snapshot / "config.json"
    if not checkpoint.is_file() or not config_path.is_file():
        raise RuntimeError("pinned DA3 snapshot is missing config.json or model.safetensors")
    if checkpoint.stat().st_size != MODEL_SIZE_BYTES:
        raise RuntimeError(f"DA3 checkpoint size mismatch: {checkpoint.stat().st_size}")
    actual_model_sha = sha256_file(checkpoint)
    if actual_model_sha != MODEL_SHA256:
        raise RuntimeError(f"DA3 checkpoint SHA mismatch: {actual_model_sha}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if str(config.get("model_name")) != "da3-base":
        raise RuntimeError(f"pinned DA3 config model_name mismatch: {config.get('model_name')!r}")

    sys.path.insert(0, str(da3_root / "src"))
    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError as exc:
        raise RuntimeError("cannot import pinned DA3 runtime; install the frozen G1-B runtime dependencies") from exc

    image = Image.open(source).convert("RGB")
    original_width, original_height = image.size

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    model = DepthAnything3.from_pretrained(str(snapshot)).to("cuda").eval()
    load_seconds = time.perf_counter() - load_start

    # Reproduce the pinned public inference path through its internal stages so
    # the raw sky tensor remains available. Public Prediction.sky thresholds at
    # 0.5, while DA3 itself uses 0.3 before replacing sky depth with a synthetic
    # max value. G1-B must exclude exactly the pixels upstream treated as sky.
    infer_start = time.perf_counter()
    imgs_cpu, ex_input, in_input = model._preprocess_inputs(
        [str(source)],
        None,
        None,
        process_res,
        "upper_bound_resize",
    )
    imgs, ex_t, in_t = model._prepare_model_inputs(imgs_cpu, ex_input, in_input)
    ex_t_norm = model._normalize_extrinsics(ex_t.clone() if ex_t is not None else None)
    raw_output = model._run_model_forward(
        imgs,
        ex_t_norm,
        in_t,
        [],
        False,
        False,
        "saddle_balanced",
    )
    raw_sky = raw_output.get("sky", None)
    prediction = model._convert_to_prediction(raw_output)
    prediction = model._add_processed_images(prediction, imgs_cpu)
    infer_seconds = time.perf_counter() - infer_start

    if prediction.intrinsics is None:
        raise RuntimeError("DA3-BASE returned no predicted intrinsics")
    depth_processed = np.asarray(prediction.depth[0], dtype=np.float32)
    if depth_processed.ndim != 2:
        raise RuntimeError(f"DA3 depth must be HxW, got {depth_processed.shape}")
    processed_height, processed_width = depth_processed.shape
    k_processed = np.asarray(prediction.intrinsics[0], dtype=np.float64)
    k_original = _map_intrinsics_to_original(
        k_processed,
        processed_width=processed_width,
        processed_height=processed_height,
        original_width=original_width,
        original_height=original_height,
    )

    confidence_processed = (
        np.asarray(prediction.conf[0], dtype=np.float32)
        if prediction.conf is not None
        else np.ones_like(depth_processed, dtype=np.float32)
    )
    if confidence_processed.shape != depth_processed.shape:
        raise RuntimeError("DA3 confidence/depth shape mismatch")
    valid_processed = (
        np.isfinite(depth_processed)
        & (depth_processed > 0.0)
        & np.isfinite(confidence_processed)
    )
    raw_sky_processed = None
    if raw_sky is not None:
        raw_sky_np = np.asarray(raw_sky.detach().float().cpu().numpy())
        if raw_sky_np.ndim != 4 or raw_sky_np.shape[0] != 1 or raw_sky_np.shape[1] != 1:
            raise RuntimeError(f"unexpected DA3 raw sky shape: {raw_sky_np.shape}")
        raw_sky_processed = raw_sky_np[0, 0]
        if raw_sky_processed.shape != depth_processed.shape:
            raise RuntimeError("DA3 raw sky/depth shape mismatch")
        valid_processed &= raw_sky_processed < UPSTREAM_SKY_THRESHOLD

    depth_original, valid_original = _valid_aware_resize_depth(
        depth_processed,
        valid_processed,
        original_width=original_width,
        original_height=original_height,
    )
    confidence_original = _resize_confidence(
        confidence_processed,
        valid_processed,
        valid_original,
        original_width=original_width,
        original_height=original_height,
    )

    valid_values = depth_original[valid_original]
    storage_fill = float(np.median(valid_values))
    if not np.isfinite(storage_fill) or storage_fill <= 0.0:
        raise RuntimeError("cannot derive positive DA3 storage fill from valid depth")
    stored_depth = np.where(valid_original, depth_original, storage_fill).astype(np.float32)
    if not np.all(np.isfinite(stored_depth)) or np.any(stored_depth <= 0.0):
        raise RuntimeError("DA3 source-geometry storage depth is not finite positive")

    # The G1-B bridge discards DA3 pose and substitutes published anchor
    # extrinsics. Identity is only a valid canonical placeholder for the generic
    # source-geometry contract.
    camera = Camera(
        intrinsics=tuple(k_original.reshape(-1).tolist()),
        extrinsics=tuple(np.eye(4, dtype=np.float64).reshape(-1).tolist()),
        convention=OPENGL_C2W,
    )

    depth_path = out / "depth.npy"
    confidence_path = out / "confidence-raw.npy"
    mask_path = out / "valid-mask.npy"
    np.save(depth_path, stored_depth, allow_pickle=False)
    np.save(confidence_path, confidence_original.astype(np.float32), allow_pickle=False)
    np.save(mask_path, valid_original.astype(bool), allow_pickle=False)

    gpu = torch.cuda.get_device_properties(0)
    manifest = {
        "version": "0.1",
        "stage": "refworld-source-geometry",
        "backend": "da3-base",
        "upstream": {
            "repo": "ByteDance-Seed/Depth-Anything-3",
            "expected_commit": DA3_PIN,
            "actual_commit": actual_commit,
            "checkpoint_repo": MODEL_REPO,
            "checkpoint_revision": MODEL_REVISION,
            "checkpoint_sha256": MODEL_SHA256,
            "checkpoint_size_bytes": MODEL_SIZE_BYTES,
            "config_sha256": sha256_file(config_path),
            "license": "Apache-2.0",
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
            "intrinsics_source": "DA3-BASE predicted pixel-space K mapped from processed pixels to original source pixels",
            "extrinsics_role": "canonical placeholder; G1-B bridge discards DA3 pose and uses published anchor extrinsics",
        },
        "geometry": {
            "depth_semantics": "positive camera-space depth from DA3; G1-B treats it as optical-axis depth under DA3 predicted K",
            "native_metric_scale_used": False,
            "predicted_pose_used": False,
            "confidence_semantics": "raw DA3 depth confidence remapped to original pixels; not used for G1-B scalar selection",
            "confidence_calibration": None,
            "validity_mask_artifact_kind": "source-valid-mask-npy",
            "validity_rule": "finite positive depth + finite confidence + raw DA3 sky < 0.3; original pixel valid only with >=0.999 interpolation support",
            "upstream_sky_threshold": UPSTREAM_SKY_THRESHOLD,
            "invalid_storage_fill": "median valid predicted depth; excluded by source-valid-mask before scale/warp",
        },
        "preprocessing": {
            "process_res": process_res,
            "process_res_method": "upper_bound_resize",
            "processed_width": processed_width,
            "processed_height": processed_height,
            "output_maps_remapped_to_original_resolution": True,
            "intrinsics_mapped_to_original_pixels": True,
        },
        "prediction": {
            "has_confidence": prediction.conf is not None,
            "has_raw_sky": raw_sky_processed is not None,
            "has_predicted_intrinsics": prediction.intrinsics is not None,
            "has_predicted_extrinsics": prediction.extrinsics is not None,
            "prediction_is_metric": int(prediction.is_metric),
            "valid_fraction_processed": float(np.mean(valid_processed)),
            "valid_fraction_original": float(np.mean(valid_original)),
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
        "timing": {
            "model_load_seconds": load_seconds,
            "inference_seconds": infer_seconds,
        },
        "artifacts": [
            artifact_record(depth_path, out, "source-depth-npy"),
            artifact_record(confidence_path, out, "source-confidence-raw-npy"),
            artifact_record(mask_path, out, "source-valid-mask-npy"),
        ],
    }
    manifest_path = out / "source-geometry.safe.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(manifest_path, flush=True)
    print(
        f"DA3-BASE process_res={process_res}; processed={processed_width}x{processed_height}; "
        f"valid_fraction={float(np.mean(valid_original)):.6f}",
        flush=True,
    )
    print(f"Peak reserved: {torch.cuda.max_memory_reserved() / (1024**3):.2f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
