#!/usr/bin/env python3
"""Generate one deterministic Big-LaMa repaint candidate for RefWorld.

This runner is the second-backend EXP-002 ablation. It is intentionally
architecturally distinct from the SD2 latent-diffusion baseline: Big-LaMa uses
Fourier-convolution image inpainting. The model is allowed to edit only the
UNRESOLVED support plus a fixed 16px context band; outside that declared
context the geometric warp is copied bit-for-bit. A later compose-candidate
stage produces B (unrestricted) and C (evidence-preserved) from this same
candidate.

Held-out evaluation imagery is never read by this runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

MODEL_URL = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
MODEL_SHA256 = "7ba7aa7ac37a4d41fdbbeba3a2af7ead18058552997e3a3cd1a3b2210c9e6b4c"
MODEL_SIZE_BYTES = 205_803_670
DEFAULT_CONTEXT_RADIUS = 16


def _sha256_file(path: Path) -> str:
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
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _ceil_modulo(value: int, modulo: int) -> int:
    return value if value % modulo == 0 else (value // modulo + 1) * modulo


def _download_verified_model(cache_path: Path) -> Path:
    import requests

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.is_file():
        size_ok = cache_path.stat().st_size == MODEL_SIZE_BYTES
        hash_ok = size_ok and _sha256_file(cache_path) == MODEL_SHA256
        if hash_ok:
            print(f"Using verified cached Big-LaMa model: {cache_path}", flush=True)
            return cache_path
        print("Cached Big-LaMa model failed size/hash verification; refetching.", flush=True)
        cache_path.unlink()

    temporary = cache_path.with_suffix(cache_path.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    print(f"Downloading pinned Big-LaMa TorchScript model ({MODEL_SIZE_BYTES / (1024**2):.1f} MiB)...", flush=True)
    with requests.get(MODEL_URL, stream=True, timeout=120) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            downloaded = 0
            next_report = 25 * 1024 * 1024
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    print(f"  downloaded {downloaded / (1024**2):.0f} MiB", flush=True)
                    next_report += 25 * 1024 * 1024
    actual_size = temporary.stat().st_size
    actual_sha = _sha256_file(temporary)
    if actual_size != MODEL_SIZE_BYTES or actual_sha != MODEL_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "Big-LaMa model verification failed: "
            f"size={actual_size} sha256={actual_sha}; expected size={MODEL_SIZE_BYTES} sha256={MODEL_SHA256}"
        )
    temporary.replace(cache_path)
    print(f"Big-LaMa model verified: sha256={MODEL_SHA256}", flush=True)
    return cache_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one pinned Big-LaMa RefWorld repaint candidate")
    parser.add_argument("--warp-view", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-radius", type=int, default=DEFAULT_CONTEXT_RADIUS)
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=Path(".model-cache/big-lama-v0.1.0.pt"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    warp_view = args.warp_view.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not warp_view.is_dir():
        raise NotADirectoryError(warp_view)
    if args.context_radius < 0 or args.context_radius > 256:
        raise ValueError("--context-radius must be in [0,256]")

    try:
        import cv2
        import torch
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Big-LaMa candidate runner requires OpenCV, PyTorch and Pillow") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Big-LaMa backend-independence run requires CUDA")

    warp_image_path = warp_view / "proposal.png"
    provenance_path = warp_view / "provenance.npy"
    warp_metadata_path = warp_view / "proposal.json"
    for required in (warp_image_path, provenance_path, warp_metadata_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    warp_rgb = np.asarray(Image.open(warp_image_path).convert("RGB"), dtype=np.uint8)
    provenance = np.load(provenance_path, allow_pickle=False)
    if provenance.shape != warp_rgb.shape[:2]:
        raise ValueError("warp provenance shape does not match warp image")
    values = {int(value) for value in np.unique(provenance)}
    if not values.issubset({0, 1}):
        raise ValueError(f"LaMa runner requires geometry-only provenance codes 0/1, got {sorted(values)}")
    unresolved = provenance == 0
    observed = provenance == 1
    if not np.any(unresolved) or not np.any(observed):
        raise RuntimeError("selected warp must contain both unresolved and observed support")

    if args.context_radius == 0:
        context = unresolved.copy()
    else:
        diameter = 2 * int(args.context_radius) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
        context = cv2.dilate(unresolved.astype(np.uint8), kernel, iterations=1).astype(bool)

    repo_root = Path(__file__).resolve().parents[3]
    cache_path = args.model_cache if args.model_cache.is_absolute() else repo_root / args.model_cache
    model_path = _download_verified_model(cache_path.resolve())

    height, width = warp_rgb.shape[:2]
    padded_height = _ceil_modulo(height, 8)
    padded_width = _ceil_modulo(width, 8)
    image_chw = np.transpose(warp_rgb.astype(np.float32) / 255.0, (2, 0, 1))
    mask_chw = context.astype(np.float32)[None, ...]
    image_chw = np.pad(
        image_chw,
        ((0, 0), (0, padded_height - height), (0, padded_width - width)),
        mode="symmetric",
    )
    mask_chw = np.pad(
        mask_chw,
        ((0, 0), (0, padded_height - height), (0, padded_width - width)),
        mode="symmetric",
    )

    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    model.to(device)
    load_seconds = time.perf_counter() - load_start
    print(f"Big-LaMa TorchScript loaded in {load_seconds:.2f}s; input {width}x{height} (padded {padded_width}x{padded_height}).", flush=True)

    image_tensor = torch.from_numpy(image_chw).unsqueeze(0).to(device)
    mask_tensor = (torch.from_numpy(mask_chw).unsqueeze(0).to(device) > 0).to(dtype=image_tensor.dtype)
    inference_start = time.perf_counter()
    with torch.inference_mode():
        predicted = model(image_tensor, mask_tensor)
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_start
    print(f"Big-LaMa inference completed in {inference_seconds:.2f}s.", flush=True)

    if isinstance(predicted, (tuple, list)):
        predicted = predicted[0]
    if not hasattr(predicted, "ndim") or predicted.ndim != 4 or predicted.shape[0] != 1 or predicted.shape[1] != 3:
        raise RuntimeError(f"unexpected Big-LaMa output shape/type: {getattr(predicted, 'shape', type(predicted))}")
    result = predicted[0, :, :height, :width].permute(1, 2, 0).detach().float().cpu().numpy()
    if not np.isfinite(result).all():
        raise RuntimeError("Big-LaMa output contains non-finite values")
    result_u8 = np.clip(result * 255.0, 0.0, 255.0).astype(np.uint8)

    candidate = warp_rgb.copy()
    candidate[context] = result_u8[context]
    candidate_path = output / "candidate.png"
    mask_path = output / "repaint-valid-mask.npy"
    model_output_path = output / "model-output.png"
    Image.fromarray(candidate, mode="RGB").save(candidate_path)
    Image.fromarray(result_u8, mode="RGB").save(model_output_path)
    np.save(mask_path, context.astype(np.bool_), allow_pickle=False)

    overlap = context & observed
    gpu = torch.cuda.get_device_properties(0)
    manifest = {
        "version": "0.1",
        "stage": "refworld-lama-inpaint-candidate",
        "role": "model-generated-repaint-candidate-not-observation",
        "backend": {
            "id": "big-lama-torchscript",
            "architecture_family": "LaMa Fourier-convolution image inpainting",
            "source_implementation": "https://github.com/advimman/lama",
            "source_code_license": "Apache-2.0",
            "distribution": "enesmsahin/simple-lama-inpainting release v0.1.0",
            "distribution_license": "Apache-2.0",
            "model_url": MODEL_URL,
            "model_size_bytes": MODEL_SIZE_BYTES,
            "model_sha256": MODEL_SHA256,
        },
        "input": {
            "warp_view_directory_name": warp_view.name,
            "warp_proposal_png_sha256": _sha256_file(warp_image_path),
            "warp_provenance_npy_sha256": _sha256_file(provenance_path),
            "warp_metadata_json_sha256": _sha256_file(warp_metadata_path),
            "held_out_evaluation_image_used": False,
        },
        "configuration": {
            "context_radius_original_pixels": int(args.context_radius),
            "context_shape": "elliptical-morphological-dilation",
            "model_input_policy": "native-resolution-symmetric-pad-to-modulo-8",
            "original_width": width,
            "original_height": height,
            "padded_width": padded_width,
            "padded_height": padded_height,
            "candidate_outside_context_policy": "copy-geometric-warp-bitwise",
            "stochastic_sampling": False,
        },
        "support": {
            "observed_fraction": float(np.mean(observed)),
            "unresolved_fraction": float(np.mean(unresolved)),
            "repaint_context_fraction": float(np.mean(context)),
            "repaint_observed_overlap_fraction": float(np.mean(overlap)),
            "repaint_observed_overlap_pixels": int(np.sum(overlap)),
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
            "inference_seconds": inference_seconds,
        },
        "artifacts": [
            _artifact(model_output_path, output, "raw-model-output-png"),
            _artifact(candidate_path, output, "candidate-png"),
            _artifact(mask_path, output, "repaint-valid-mask-npy"),
        ],
        "guardrails": {
            "generated_pixels_are_observations": False,
            "candidate_may_overwrite_observed_support_before_composition": True,
            "canonical_evidence_preservation_done_here": False,
            "held_out_rgb_used_for_config_or_selection": False,
            "configuration_frozen_before_backend-independence_scores": True,
        },
    }
    manifest_path = output / "lama-inpaint.safe.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(manifest_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
