#!/usr/bin/env python3
"""Extract RefWorld source geometry from the frozen MoGe-2 ViT-B candidate.

This runner is development-only for EXP-002 G1-A. It predicts depth and
intrinsics from the anchor RGB and writes the existing source-geometry contract.
MoGe's native metric scale is recorded but is not used as evidence in G1-A; the
later bridge fits the same one oracle global depth scalar for every candidate.
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

MOGE_PIN = "925b8ed835a7a9cdb7578ba15c658a0afc969030"
MODEL_REPO = "Ruicheng/moge-2-vitb-normal"
MODEL_REVISION = "54ad3a693e61907ea4633d13dec6ee682fa09419"
MODEL_SHA256 = "16b8110e86d5dc5a849db120ca96ef3a223fd30b0c9146d1d81db504073da5f6"
MODEL_SIZE_BYTES = 419_110_160


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot verify MoGe git commit") from exc


def artifact_record(path: Path, root: Path, kind: str) -> dict[str, Any]:
    resolved = path.resolve()
    relative = resolved.relative_to(root.resolve())
    return {
        "kind": kind,
        "path": relative.as_posix(),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _pixel_intrinsics_from_normalized(k_normalized: np.ndarray, width: int, height: int) -> np.ndarray:
    """Map MoGe normalized K to original +u-right/+v-down pixel coordinates."""
    k = np.asarray(k_normalized, dtype=np.float64).reshape(3, 3).copy()
    if width <= 0 or height <= 0 or not np.all(np.isfinite(k)):
        raise ValueError("invalid MoGe intrinsics or image dimensions")
    # MoGe constructs K in normalized image coordinates with principal point
    # (0.5, 0.5), so x quantities scale by width and y quantities by height.
    k[0, :] *= float(width)
    k[1, :] *= float(height)
    k[2, :] = [0.0, 0.0, 1.0]
    if k[0, 0] <= 0.0 or k[1, 1] <= 0.0:
        raise ValueError("MoGe focal lengths must be positive")
    if abs(float(k[0, 1])) > 1e-6 or abs(float(k[1, 0])) > 1e-6:
        raise ValueError("G1-A pinhole bridge requires zero-skew MoGe intrinsics")
    return k


def _storage_safe_depth(depth: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Preserve valid MoGe depth exactly; sanitize only pixels excluded by its mask.

    The generic RefWorld source-geometry container requires finite positive values
    everywhere. G1-A separately hash-pins MoGe's validity mask and the bridge sets
    excluded pixels to NaN before scale fitting/warping, so replacing invalid-only
    raw values here cannot enter the experiment's geometric evidence.
    """
    d = np.asarray(depth, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool)
    if d.shape != valid.shape:
        raise ValueError("depth/mask shape mismatch")
    if not np.any(valid):
        raise RuntimeError("MoGe validity mask is empty")
    if not np.all(np.isfinite(d[valid])) or np.any(d[valid] <= 0.0):
        raise RuntimeError("MoGe has non-finite or non-positive depth inside its declared valid mask")
    bad_storage = ~np.isfinite(d) | (d <= 0.0)
    bad_valid = bad_storage & valid
    if np.any(bad_valid):
        raise RuntimeError("MoGe invalid depth intersects declared valid support")
    storage = d.copy()
    sanitize = bad_storage & ~valid
    count = int(np.count_nonzero(sanitize))
    if count:
        storage[sanitize] = np.float32(np.median(d[valid]))
    if not np.all(np.isfinite(storage)) or np.any(storage <= 0.0):
        raise RuntimeError("storage-safe MoGe depth contract failed")
    return storage, count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract pinned MoGe-2 source geometry")
    parser.add_argument("--moge-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution-level", type=int, default=9)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    moge_root = args.moge_root.resolve()
    source = args.reference.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not source.is_file():
        raise FileNotFoundError(source)
    if git_head(moge_root) != MOGE_PIN:
        raise RuntimeError(f"MoGe checkout must be pinned to {MOGE_PIN}")
    if int(args.resolution_level) not in (5, 7, 9):
        raise ValueError("G1-A resolution level must be one of the predeclared {9,7,5}")

    try:
        import torch
        from PIL import Image
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("MoGe-2 adapter requires torch, Pillow and huggingface_hub") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("MoGe-2 G1-A extraction requires CUDA")

    sys.path.insert(0, str(moge_root))
    try:
        from moge.model.v2 import MoGeModel
    except ImportError as exc:
        raise RuntimeError("cannot import pinned MoGe-2; install its runtime dependency utils3d") from exc

    checkpoint = Path(
        hf_hub_download(
            repo_id=MODEL_REPO,
            filename="model.pt",
            revision=MODEL_REVISION,
        )
    ).resolve()
    if checkpoint.stat().st_size != MODEL_SIZE_BYTES:
        raise RuntimeError(f"MoGe-2 checkpoint size mismatch: {checkpoint.stat().st_size}")
    actual_model_sha = sha256_file(checkpoint)
    if actual_model_sha != MODEL_SHA256:
        raise RuntimeError(f"MoGe-2 checkpoint SHA mismatch: {actual_model_sha}")

    image = Image.open(source).convert("RGB")
    width, height = image.size
    rgb = np.asarray(image, dtype=np.uint8)
    image_tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).cuda()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    model = MoGeModel.from_pretrained(checkpoint).cuda().eval().half()
    load_seconds = time.perf_counter() - load_start
    infer_start = time.perf_counter()
    with torch.inference_mode():
        output = model.infer(
            image_tensor,
            resolution_level=int(args.resolution_level),
            use_fp16=True,
        )
    infer_seconds = time.perf_counter() - infer_start

    depth_raw = np.asarray(output["depth"].detach().float().cpu().numpy(), dtype=np.float32)
    mask = np.asarray(output["mask"].detach().cpu().numpy(), dtype=bool)
    k_norm = np.asarray(output["intrinsics"].detach().float().cpu().numpy(), dtype=np.float64)
    if depth_raw.shape != (height, width) or mask.shape != (height, width):
        raise RuntimeError(
            f"MoGe output shape mismatch: depth={depth_raw.shape}, mask={mask.shape}, expected={(height, width)}"
        )
    depth, sanitized_invalid_count = _storage_safe_depth(depth_raw, mask)
    k_pixel = _pixel_intrinsics_from_normalized(k_norm, width, height)

    # The G1-A bridge explicitly discards model pose and substitutes benchmark
    # anchor extrinsics. Store a valid canonical identity pose so the generic
    # source-geometry contract can carry MoGe depth + K without implying pose use.
    camera = Camera(
        intrinsics=tuple(k_pixel.reshape(-1).tolist()),
        extrinsics=tuple(np.eye(4, dtype=np.float64).reshape(-1).tolist()),
        convention=OPENGL_C2W,
    )
    confidence = np.ones((height, width), dtype=np.float32)

    depth_path = out / "depth.npy"
    confidence_path = out / "confidence-raw.npy"
    mask_path = out / "valid-mask.npy"
    np.save(depth_path, depth, allow_pickle=False)
    np.save(confidence_path, confidence, allow_pickle=False)
    np.save(mask_path, mask, allow_pickle=False)

    gpu = torch.cuda.get_device_properties(0)
    valid_fraction = float(np.mean(mask))
    manifest = {
        "version": "0.1",
        "stage": "refworld-source-geometry",
        "backend": "moge2-vitb-normal",
        "upstream": {
            "repo": "microsoft/MoGe",
            "expected_commit": MOGE_PIN,
            "actual_commit": git_head(moge_root),
            "checkpoint_repo": MODEL_REPO,
            "checkpoint_revision": MODEL_REVISION,
            "checkpoint_sha256": MODEL_SHA256,
            "checkpoint_size_bytes": MODEL_SIZE_BYTES,
            "license": "MIT",
        },
        "input": {
            "file_name": source.name,
            "sha256": sha256_file(source),
            "size_bytes": source.stat().st_size,
            "width": width,
            "height": height,
        },
        "camera": {
            "intrinsics": list(camera.intrinsics),
            "extrinsics": list(camera.extrinsics),
            "convention": camera.convention,
            "intrinsics_source": "MoGe-2 normalized K mapped to original source pixels",
            "extrinsics_role": "canonical placeholder; G1-A bridge discards model pose and uses published anchor extrinsics",
        },
        "geometry": {
            "depth_semantics": "positive optical-axis depth from MoGe-2; native model output is metric but G1-A ignores native scale",
            "native_metric_scale_claimed_by_upstream": True,
            "confidence_semantics": "synthetic uniform map; not used for G1-A all-valid scale fitting",
            "confidence_calibration": None,
            "validity_mask_artifact_kind": "source-valid-mask-npy",
            "valid_fraction": valid_fraction,
            "invalid_only_depth_values_sanitized_for_storage": sanitized_invalid_count,
            "sanitized_values_used_by_g1a_bridge": False,
        },
        "preprocessing": {
            "input_to_model": "original RGB tensor in [0,1]; MoGe internal resolution control",
            "resolution_level": int(args.resolution_level),
            "use_fp16": True,
            "output_maps_original_resolution": True,
            "intrinsics_mapped_to_original_pixels": True,
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
        "timing": {"model_load_seconds": load_seconds, "inference_seconds": infer_seconds},
        "artifacts": [
            artifact_record(depth_path, out, "source-depth-npy"),
            artifact_record(confidence_path, out, "source-confidence-raw-npy"),
            artifact_record(mask_path, out, "source-valid-mask-npy"),
        ],
    }
    manifest_path = out / "source-geometry.safe.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(manifest_path, flush=True)
    print(f"MoGe-2 resolution_level={args.resolution_level}; valid_fraction={valid_fraction:.6f}", flush=True)
    print(f"Peak reserved: {torch.cuda.max_memory_reserved() / (1024**3):.2f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
