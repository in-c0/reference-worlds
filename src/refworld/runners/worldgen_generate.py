#!/usr/bin/env python3
"""Run the pinned WorldGen image-to-scene baseline with explicit provenance.

This script is intentionally executed inside a WorldGen-capable CUDA environment.
It preserves WorldGen's current image-to-scene algorithm while exposing the
otherwise implicit panorama-fill seed and retaining the panorama intermediate.
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

WORLDGEN_PIN = "7ce7b2767fdf31e2727b69a2e61e2e950e3a017f"
BASE_CHECKPOINTS = {
    "depth": "haodongli/DA-2",
    "panorama_base": "black-forest-labs/FLUX.1-Fill-dev",
    "panorama_lora_repo": "LeoXie/WorldGen",
    "panorama_lora_file": "models--WorldGen-Flux-Lora/worldgen_img2scene.safetensors",
}


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
        raise RuntimeError("cannot verify WorldGen git commit") from exc
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--worldgen-root", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resolution", type=int, default=1600)
    p.add_argument("--prompt", default="")
    p.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-sharp", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--inpaint-bg", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--return-mesh", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--allow-unpinned-worldgen", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    wg_root = args.worldgen_root.resolve()
    source = args.reference.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not source.is_file():
        raise FileNotFoundError(source)
    if args.resolution < 512 or args.resolution % 2:
        raise ValueError("resolution must be an even integer >= 512")

    head = git_head(wg_root)
    if head != WORLDGEN_PIN and not args.allow_unpinned_worldgen:
        raise RuntimeError(
            f"WorldGen checkout is {head}; expected pinned {WORLDGEN_PIN}. "
            "Use --allow-unpinned-worldgen only for explicitly non-baseline experiments."
        )

    src_dir = wg_root / "src"
    if not src_dir.is_dir():
        raise RuntimeError(f"WorldGen src directory not found: {src_dir}")
    sys.path.insert(0, str(src_dir))

    import numpy as np
    import torch
    from PIL import Image
    from worldgen import WorldGen
    from worldgen.pano_depth import pred_depth
    from worldgen.pano_gen import gen_pano_fill_image
    from worldgen.utils.general_utils import map_image_to_pano, resize_img

    if not torch.cuda.is_available():
        raise RuntimeError("EXP-000 requires CUDA; torch.cuda.is_available() is false")

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

    init_start = time.perf_counter()
    worldgen = WorldGen(
        mode="i2s",
        device=device,
        low_vram=args.low_vram,
        use_sharp=args.use_sharp,
        inpaint_bg=args.inpaint_bg,
        resolution=args.resolution,
    )
    init_seconds = time.perf_counter() - init_start

    generation_start = time.perf_counter()
    image = Image.open(source).convert("RGB")
    resized = resize_img(image)

    # Mirrors WorldGen.generate_pano(mode="i2s") at WORLDGEN_PIN but passes
    # seed explicitly instead of relying on gen_pano_fill_image(seed=42).
    predictions = pred_depth(worldgen.depth_model, resized)
    pano_cond_img, cond_mask = map_image_to_pano(predictions, device=worldgen.device)
    pano_image = gen_pano_fill_image(
        worldgen.pano_gen_model,
        image=pano_cond_img,
        mask=cond_mask,
        prompt=args.prompt,
        seed=args.seed,
        height=args.resolution // 2,
        width=args.resolution,
    )

    # Preserve WorldGen's explicit source-pixel remap after generative fill.
    map_height, map_width = pano_cond_img.height, pano_cond_img.width
    pano_image = pano_image.resize((map_width, map_height))
    observed = np.asarray(pano_cond_img)
    mask = np.asarray(cond_mask, dtype=np.float32) / 255.0
    generated = np.asarray(pano_image, dtype=np.float32)
    merged = generated * mask[:, :, None] + observed * (1.0 - mask[:, :, None])
    pano_image = Image.fromarray(np.clip(merged, 0, 255).astype(np.uint8))

    pano_path = out / "panorama.png"
    pano_image.save(pano_path)

    scene = worldgen._generate_world(pano_image=pano_image, return_mesh=args.return_mesh)

    artifacts = [artifact_record(pano_path, out, "panorama")]
    if args.return_mesh:
        import open3d as o3d

        scene_path = out / "world-mesh.ply"
        if not o3d.io.write_triangle_mesh(str(scene_path), scene):
            raise RuntimeError("Open3D failed to write mesh")
        artifacts.append(artifact_record(scene_path, out, "mesh"))
    else:
        scene_path = out / "world-splat.ply"
        scene.save(str(scene_path))
        artifacts.append(artifact_record(scene_path, out, "splat-ply"))

    generation_seconds = time.perf_counter() - generation_start
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())

    checkpoints: dict[str, Any] = dict(BASE_CHECKPOINTS)
    if args.low_vram:
        try:
            from nunchaku.utils import get_precision

            precision = str(get_precision())
            checkpoints["nunchaku_precision"] = precision
            checkpoints["quantized_transformer"] = (
                f"mit-han-lab/svdq-{precision}-flux.1-fill-dev"
            )
        except Exception:
            checkpoints["nunchaku_precision"] = "unknown"
            checkpoints["quantized_transformer"] = (
                "mit-han-lab/svdq-<runtime-precision>-flux.1-fill-dev"
            )

    if args.use_sharp:
        from worldgen.pano_sharp import DEFAULT_MODEL_URL

        checkpoints["sharp_model_url"] = DEFAULT_MODEL_URL

    if args.inpaint_bg:
        from worldgen.models.inpaint_model import LAMA_MODEL_MD5, LAMA_MODEL_URL

        checkpoints["background_inpaint_lama_url"] = LAMA_MODEL_URL
        checkpoints["background_inpaint_lama_md5"] = LAMA_MODEL_MD5

    manifest = {
        "version": "0.1",
        "experiment": "EXP-000",
        "baseline": "worldgen",
        "upstream": {
            "repo": "ZiYang-xie/WorldGen",
            "expected_commit": WORLDGEN_PIN,
            "actual_commit": head,
            "unpinned_allowed": bool(args.allow_unpinned_worldgen),
        },
        "input": {
            "file_name": source.name,
            "sha256": sha256_file(source),
            "size_bytes": source.stat().st_size,
            "prompt_empty": args.prompt == "",
            "prompt_sha256": hashlib.sha256(args.prompt.encode("utf-8")).hexdigest(),
        },
        "configuration": {
            "mode": "i2s",
            "seed": args.seed,
            "resolution": args.resolution,
            "low_vram": args.low_vram,
            "use_sharp": args.use_sharp,
            "inpaint_bg": args.inpaint_bg,
            "return_mesh": args.return_mesh,
            "explicit_seed_patch": True,
        },
        "checkpoints": checkpoints,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu_name": gpu.name,
            "gpu_total_bytes": int(gpu.total_memory),
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        },
        "timing": {
            "model_init_seconds": init_seconds,
            "generation_seconds": generation_seconds,
        },
        "artifacts": artifacts,
        "notes": [
            "Image-to-panorama logic mirrors the pinned WorldGen i2s path but exposes the panorama-fill seed explicitly.",
            "The panorama intermediate is retained so hidden-view completion can be evaluated separately from 3D reconstruction.",
            "Checkpoint entries describe models actually activated by this configuration; external checkpoint licenses/terms remain upstream-owned.",
        ],
    }

    safe_path = out / "run.safe.json"
    safe_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(safe_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
