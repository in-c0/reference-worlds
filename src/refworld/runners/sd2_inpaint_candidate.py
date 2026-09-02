#!/usr/bin/env python3
"""Generate one machine-fit RefWorld repaint candidate with SD2 inpainting.

This is intentionally a *repaint candidate* rather than evidence.  Geometry and
pixel provenance come from a persisted RefWorld warp-only view.  The generator
is allowed to edit unresolved pixels plus a fixed context band around them; the
separate compose-candidate stage decides whether those edits may survive on
OBSERVED support.

The first frozen local protocol is deliberately modest so it can run on an
8 GiB-class NVIDIA card:

- target: one predeclared +5 degree yaw warp selected by the orchestrator;
- model: sd2-community/stable-diffusion-2-inpainting (OpenRAIL++ weights);
- seed: 42;
- generic prompt, not scene-tuned;
- max model side: 512 px, aspect preserved and rounded to multiples of 8;
- unresolved mask dilated by 16 original-image pixels for repaint context;
- candidate outside that context is copied from the geometric warp exactly.

Held-out evaluation imagery is never read by this runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

MODEL_REPO = "sd2-community/stable-diffusion-2-inpainting"
DEFAULT_PROMPT = "photorealistic scene, coherent continuation of the visible image, consistent lighting and perspective"
DEFAULT_SEED = 42
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE = 4.0
DEFAULT_CONTEXT_RADIUS = 16
DEFAULT_MAX_SIDE = 512


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _artifact(path: Path, root: Path, kind: str) -> dict[str, Any]:
    resolved = path.resolve()
    base = root.resolve()
    try:
        relative = resolved.relative_to(base)
    except ValueError as exc:
        raise RuntimeError(f"artifact escaped output directory: {path}") from exc
    return {
        "kind": kind,
        "path": relative.as_posix(),
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _model_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("source dimensions must be positive")
    scale = min(1.0, float(max_side) / float(max(width, height)))
    model_width = max(64, int(round((width * scale) / 8.0)) * 8)
    model_height = max(64, int(round((height * scale) / 8.0)) * 8)
    model_width = min(max_side, model_width)
    model_height = min(max_side, model_height)
    return model_width, model_height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one lightweight SD2 RefWorld repaint candidate")
    parser.add_argument("--warp-view", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE)
    parser.add_argument("--context-radius", type=int, default=DEFAULT_CONTEXT_RADIUS)
    parser.add_argument("--max-side", type=int, default=DEFAULT_MAX_SIDE)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    warp_view = args.warp_view.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    if not warp_view.is_dir():
        raise NotADirectoryError(warp_view)
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if not math.isfinite(args.guidance_scale) or args.guidance_scale < 0.0:
        raise ValueError("--guidance-scale must be finite and non-negative")
    if args.context_radius < 0 or args.context_radius > 256:
        raise ValueError("--context-radius must be in [0,256]")
    if args.max_side < 256 or args.max_side > 768 or args.max_side % 8 != 0:
        raise ValueError("--max-side must be a multiple of 8 in [256,768]")
    if not args.prompt.strip():
        raise ValueError("prompt cannot be empty in the frozen first-candidate protocol")

    try:
        import cv2
        import torch
        from diffusers import StableDiffusionInpaintPipeline
        from huggingface_hub import model_info
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "SD2 repaint environment is incomplete; use scripts/run-windows-first-candidate.py"
        ) from exc

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
        raise ValueError(f"first-candidate runner requires geometry-only provenance codes 0/1, got {sorted(values)}")

    unresolved = provenance == 0
    observed = provenance == 1
    if not np.any(unresolved):
        raise RuntimeError("selected warp contains no unresolved support to repaint")
    if not np.any(observed):
        raise RuntimeError("selected warp contains no observed support")

    if args.context_radius == 0:
        context = unresolved.copy()
    else:
        diameter = 2 * int(args.context_radius) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
        context = cv2.dilate(unresolved.astype(np.uint8), kernel, iterations=1).astype(bool)

    height, width = warp_rgb.shape[:2]
    model_width, model_height = _model_size(width, height, int(args.max_side))
    warp_pil = Image.fromarray(warp_rgb, mode="RGB")
    input_pil = warp_pil.resize((model_width, model_height), Image.Resampling.LANCZOS)
    mask_u8 = np.where(context, 255, 0).astype(np.uint8)
    mask_pil = Image.fromarray(mask_u8, mode="L")
    model_mask = mask_pil.resize((model_width, model_height), Image.Resampling.NEAREST)

    input_path = output / "model-input.png"
    mask_path = output / "model-mask.png"
    input_pil.save(input_path)
    model_mask.save(mask_path)

    if not torch.cuda.is_available():
        raise RuntimeError("SD2 repaint candidate requires CUDA")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print(f"Resolving exact model revision for {MODEL_REPO}...", flush=True)
    info = model_info(MODEL_REPO)
    resolved_revision = str(info.sha)
    print(f"Resolved model revision: {resolved_revision}", flush=True)
    print(
        f"Loading SD2 inpainting fp16 with CPU offload; model canvas {model_width}x{model_height}...",
        flush=True,
    )

    load_start = time.perf_counter()
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        MODEL_REPO,
        revision=resolved_revision,
        variant="fp16",
        torch_dtype=torch.float16,
        use_safetensors=True,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.enable_model_cpu_offload(gpu_id=0)
    pipe.set_progress_bar_config(disable=False)
    load_seconds = time.perf_counter() - load_start
    print(f"SD2 pipeline ready in {load_seconds:.2f}s; starting frozen candidate generation.", flush=True)

    generator = torch.Generator(device="cuda").manual_seed(int(args.seed))
    inference_start = time.perf_counter()
    result = pipe(
        prompt=args.prompt,
        image=input_pil,
        mask_image=model_mask,
        width=model_width,
        height=model_height,
        num_inference_steps=int(args.steps),
        guidance_scale=float(args.guidance_scale),
        generator=generator,
    ).images[0].convert("RGB")
    inference_seconds = time.perf_counter() - inference_start
    print(f"SD2 candidate generation completed in {inference_seconds:.2f}s.", flush=True)

    model_output_path = output / "model-output.png"
    result.save(model_output_path)

    # Upscaling is permitted only inside the declared repaint context.  Outside
    # that context the persisted geometric warp is copied byte-for-byte, avoiding
    # a global resampling confound in the B-vs-C experiment.
    upscaled = np.asarray(result.resize((width, height), Image.Resampling.LANCZOS), dtype=np.uint8)
    candidate = warp_rgb.copy()
    candidate[context] = upscaled[context]

    candidate_path = output / "candidate.png"
    valid_mask_path = output / "repaint-valid-mask.npy"
    Image.fromarray(candidate, mode="RGB").save(candidate_path)
    np.save(valid_mask_path, context.astype(np.bool_), allow_pickle=False)

    overlap = context & observed
    unresolved_in_context = context & unresolved
    gpu = torch.cuda.get_device_properties(0)
    manifest = {
        "version": "0.1",
        "stage": "refworld-sd2-inpaint-candidate",
        "role": "model-generated-repaint-candidate-not-observation",
        "backend": {
            "id": "sd2-community-stable-diffusion-2-inpainting",
            "repo": MODEL_REPO,
            "resolved_revision": resolved_revision,
            "weights_license": "CreativeML Open RAIL++-M",
            "checkpoint_variant": "fp16",
            "safety_checker_loaded": False,
        },
        "input": {
            "warp_view_directory_name": warp_view.name,
            "warp_proposal_png_sha256": _sha256_file(warp_image_path),
            "warp_provenance_npy_sha256": _sha256_file(provenance_path),
            "warp_metadata_json_sha256": _sha256_file(warp_metadata_path),
            "held_out_evaluation_image_used": False,
        },
        "configuration": {
            "seed": int(args.seed),
            "prompt": args.prompt,
            "prompt_sha256": _sha256_text(args.prompt),
            "num_inference_steps": int(args.steps),
            "guidance_scale": float(args.guidance_scale),
            "context_radius_original_pixels": int(args.context_radius),
            "context_shape": "elliptical-morphological-dilation",
            "max_model_side": int(args.max_side),
            "model_width": model_width,
            "model_height": model_height,
            "candidate_outside_context_policy": "copy-geometric-warp-bitwise",
            "model_output_resampling_to_original": "PIL-LANCZOS-inside-context-only",
        },
        "support": {
            "original_width": width,
            "original_height": height,
            "observed_fraction": float(np.mean(observed)),
            "unresolved_fraction": float(np.mean(unresolved)),
            "repaint_context_fraction": float(np.mean(context)),
            "repaint_observed_overlap_fraction": float(np.mean(overlap)),
            "unresolved_covered_by_context_fraction": float(np.sum(unresolved_in_context) / np.sum(unresolved)),
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
            "pipeline_load_seconds": load_seconds,
            "inference_seconds": inference_seconds,
        },
        "artifacts": [
            _artifact(input_path, output, "model-input-png"),
            _artifact(mask_path, output, "model-mask-png"),
            _artifact(model_output_path, output, "raw-model-output-png"),
            _artifact(candidate_path, output, "candidate-png"),
            _artifact(valid_mask_path, output, "repaint-valid-mask-npy"),
        ],
        "guardrails": {
            "generated_pixels_are_observations": False,
            "candidate_may_overwrite_observed_support_before_composition": True,
            "canonical_evidence_preservation_done_here": False,
            "held_out_rgb_used_for_prompt_config_or_selection": False,
            "configuration_frozen_before_first_candidate": True,
        },
    }

    manifest_path = output / "sd2-inpaint.safe.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(manifest_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
