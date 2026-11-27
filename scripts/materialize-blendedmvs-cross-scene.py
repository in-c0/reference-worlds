#!/usr/bin/env python3
"""Selectively materialize frozen BlendedMVS cross-scene confirmation inputs.

This script supports the predeclared EXP-002 cross-scene oracle diagnostic.
For frozen scenes 2-10 it has two explicit phases:

1. generation: fetch only pair.txt, anchor RGB/camera/depth and the first
   published target camera. The held-out target RGB must not already exist.
2. targets: after every candidate/composition has been generated, fetch only
   the held-out target RGB needed for scoring. Target depth is never fetched.

The official low-resolution release is a split ZIP. RemoteZip performs HTTP
range reads only; this script never downloads the full archive.
"""

from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from refworld.datasets.blendedmvs import load_manifest
from refworld.datasets.mvsnet import parse_camera_text, parse_pair_text
from refworld.datasets.pfm import read_pfm
from refworld.datasets.remote_zip import RemoteZip

RELEASE_URL = "https://github.com/YoYo000/BlendedMVS/releases/download/v1.0.0/BlendedMVS.zip"
DEFAULT_SCENE_ORDERS = tuple(range(2, 11))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_orders(value: str) -> tuple[int, ...]:
    try:
        orders = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("scene orders must be comma-separated integers") from exc
    if not orders:
        raise argparse.ArgumentTypeError("at least one scene order is required")
    if len(set(orders)) != len(orders):
        raise argparse.ArgumentTypeError("scene orders must be unique")
    if any(order < 1 for order in orders):
        raise argparse.ArgumentTypeError("scene orders are 1-based and must be positive")
    return orders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Selectively materialize frozen cross-scene BlendedMVS inputs")
    parser.add_argument("--phase", choices=("generation", "targets"), required=True)
    parser.add_argument(
        "--scene-orders",
        type=parse_orders,
        default=DEFAULT_SCENE_ORDERS,
        help="comma-separated 1-based frozen scene orders (default: 2-10)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("datasets/blendedmvs-bootstrap-v0.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("private-data/blendedmvs-bootstrap"),
    )
    parser.add_argument("--url", default=RELEASE_URL)
    return parser.parse_args()


def select_scenes(frozen: dict[str, Any], orders: tuple[int, ...]) -> list[dict[str, Any]]:
    scenes = frozen.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise RuntimeError("frozen manifest has no scenes")
    by_order: dict[int, dict[str, Any]] = {}
    for entry in scenes:
        if not isinstance(entry, dict) or "id" not in entry or "order" not in entry:
            raise RuntimeError("malformed frozen scene entry")
        order = int(entry["order"])
        if order in by_order:
            raise RuntimeError(f"duplicate frozen scene order {order}")
        by_order[order] = entry
    missing = [order for order in orders if order not in by_order]
    if missing:
        raise RuntimeError(f"requested frozen scene orders are absent: {missing}")
    return [by_order[order] for order in orders]


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = args.manifest if args.manifest.is_absolute() else repo_root / args.manifest
    output_root = args.output_root if args.output_root.is_absolute() else repo_root / args.output_root
    output_root = output_root.resolve()

    frozen = load_manifest(manifest_path)
    selected = select_scenes(frozen, tuple(args.scene_orders))

    print("RefWorld cross-scene selective BlendedMVS materialization", flush=True)
    print(f"Phase:        {args.phase}", flush=True)
    print(f"Scene orders: {[int(item['order']) for item in selected]}", flush=True)
    print(f"Archive:      {args.url}", flush=True)
    print("Policy:       HTTP ranges only; full archive download forbidden.", flush=True)
    if args.phase == "generation":
        print("Seal:         target RGB must not exist and will not be fetched.", flush=True)
    else:
        print("Seal:         target RGB fetched only after generation manifests exist.", flush=True)

    batch_records: list[dict[str, Any]] = []

    with RemoteZip(args.url, timeout_seconds=120.0) as archive:
        print(f"Remote logical archive size: {archive.size_bytes / (1024**3):.2f} GiB", flush=True)
        print("Reading ZIP central directory...", flush=True)
        entries = archive.entries()
        print(f"ZIP file entries indexed: {len(entries)}", flush=True)

        for scene_entry in selected:
            scene_order = int(scene_entry["order"])
            scene_id = str(scene_entry["id"])
            scene_root = output_root / scene_id
            scene_root.mkdir(parents=True, exist_ok=True)

            pair_suffix = f"/{scene_id}/cams/pair.txt"
            pair_matches = [
                entry
                for name, entry in entries.items()
                if name.endswith(pair_suffix) or name == pair_suffix.lstrip("/")
            ]
            if len(pair_matches) != 1:
                raise RuntimeError(f"scene {scene_id}: expected one pair.txt, found {len(pair_matches)}")
            pair_entry = pair_matches[0]
            pair_bytes = archive.read(pair_entry)
            records = parse_pair_text(pair_bytes.decode("utf-8"))
            if not records or not records[0].source_ids:
                raise RuntimeError(f"scene {scene_id}: first pair record has no published source")
            first = records[0]
            anchor_id = int(first.reference_id)
            target_id = int(first.source_ids[0])
            pair_score = float(first.scores[0])
            anchor_stem = f"{anchor_id:08d}"
            target_stem = f"{target_id:08d}"
            pair_prefix = pair_entry.name[: -len("cams/pair.txt")]

            target_rgb_path = scene_root / "blended_images" / f"{target_stem}.jpg"
            generation_manifest = scene_root / "cross-scene-generation-inputs.safe.json"
            target_manifest = scene_root / "cross-scene-target-rgb.safe.json"

            print(
                f"\nScene {scene_order}/10 {scene_id}: anchor={anchor_id}, first target={target_id}",
                flush=True,
            )

            if args.phase == "generation" and target_rgb_path.exists():
                raise RuntimeError(
                    f"scene {scene_id}: target RGB already exists before sealed generation phase: {target_rgb_path}"
                )
            if args.phase == "targets" and not generation_manifest.is_file():
                raise RuntimeError(
                    f"scene {scene_id}: generation materialization manifest missing before target phase"
                )

            fetched: list[dict[str, Any]] = []

            def materialize(archive_name: str, relative_path: Path) -> dict[str, Any]:
                if archive_name not in entries:
                    raise FileNotFoundError(f"required archive entry missing: {archive_name}")
                entry = entries[archive_name]
                destination = scene_root / relative_path
                if destination.is_file() and destination.stat().st_size == entry.uncompressed_size:
                    existing = destination.read_bytes()
                    crc = binascii.crc32(existing) & 0xFFFFFFFF
                    if crc == entry.crc32:
                        print(f"cached: {relative_path.as_posix()}", flush=True)
                        record = {
                            "archive_entry": entry.name,
                            "path": relative_path.as_posix(),
                            "size_bytes": len(existing),
                            "sha256": sha256_bytes(existing),
                            "reused": True,
                        }
                        fetched.append(record)
                        return record
                    print(f"cache CRC mismatch; refetching {relative_path.as_posix()}", flush=True)
                print(
                    f"fetch:  {relative_path.as_posix()} ({entry.uncompressed_size / (1024**2):.2f} MiB)",
                    flush=True,
                )
                data = archive.read(entry)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
                record = {
                    "archive_entry": entry.name,
                    "path": relative_path.as_posix(),
                    "size_bytes": len(data),
                    "sha256": sha256_bytes(data),
                    "reused": False,
                }
                fetched.append(record)
                return record

            if args.phase == "generation":
                pair_relative = Path("cams") / "pair.txt"
                pair_destination = scene_root / pair_relative
                pair_destination.parent.mkdir(parents=True, exist_ok=True)
                pair_destination.write_bytes(pair_bytes)
                fetched.append(
                    {
                        "archive_entry": pair_entry.name,
                        "path": pair_relative.as_posix(),
                        "size_bytes": len(pair_bytes),
                        "sha256": sha256_bytes(pair_bytes),
                        "reused": False,
                    }
                )

                generation_required = (
                    (
                        f"{pair_prefix}blended_images/{anchor_stem}.jpg",
                        Path("blended_images") / f"{anchor_stem}.jpg",
                    ),
                    (
                        f"{pair_prefix}cams/{anchor_stem}_cam.txt",
                        Path("cams") / f"{anchor_stem}_cam.txt",
                    ),
                    (
                        f"{pair_prefix}rendered_depth_maps/{anchor_stem}.pfm",
                        Path("rendered_depth_maps") / f"{anchor_stem}.pfm",
                    ),
                    (
                        f"{pair_prefix}cams/{target_stem}_cam.txt",
                        Path("cams") / f"{target_stem}_cam.txt",
                    ),
                )
                for archive_name, relative in generation_required:
                    materialize(archive_name, relative)

                anchor_rgb_path = scene_root / "blended_images" / f"{anchor_stem}.jpg"
                anchor_camera_path = scene_root / "cams" / f"{anchor_stem}_cam.txt"
                anchor_depth_path = scene_root / "rendered_depth_maps" / f"{anchor_stem}.pfm"
                target_camera_path = scene_root / "cams" / f"{target_stem}_cam.txt"

                try:
                    from PIL import Image
                except ImportError as exc:
                    raise RuntimeError("generation materialization validation requires Pillow") from exc

                with Image.open(anchor_rgb_path) as image:
                    width, height = image.convert("RGB").size
                depth = read_pfm(anchor_depth_path)
                if depth.ndim != 2 or depth.shape != (height, width):
                    raise RuntimeError(
                        f"scene {scene_id}: anchor RGB/depth mismatch: RGB={(height, width)} depth={depth.shape}"
                    )
                if not np.any(np.isfinite(depth) & (depth > 1e-6)):
                    raise RuntimeError(f"scene {scene_id}: anchor depth has no positive finite support")
                parse_camera_text(anchor_camera_path.read_text(encoding="utf-8"))
                parse_camera_text(target_camera_path.read_text(encoding="utf-8"))
                if target_rgb_path.exists():
                    raise RuntimeError(f"scene {scene_id}: target RGB appeared during generation materialization")

                payload = {
                    "version": "0.1",
                    "stage": "refworld-cross-scene-generation-materialization",
                    "scene_id": scene_id,
                    "frozen_scene_order": scene_order,
                    "pair_record_order": 1,
                    "anchor_view_id": anchor_id,
                    "target_source_order": 1,
                    "target_view_id": target_id,
                    "pair_score": pair_score,
                    "target_rgb_materialized": False,
                    "target_depth_materialized": False,
                    "download_policy": "HTTP-range-selective; no full archive download",
                    "files": fetched,
                }
                generation_manifest.write_text(
                    json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
                print(f"generation manifest: {generation_manifest}", flush=True)
            else:
                prior = json.loads(generation_manifest.read_text(encoding="utf-8"))
                if int(prior["frozen_scene_order"]) != scene_order:
                    raise RuntimeError(f"scene {scene_id}: generation manifest scene-order mismatch")
                if int(prior["anchor_view_id"]) != anchor_id or int(prior["target_view_id"]) != target_id:
                    raise RuntimeError(f"scene {scene_id}: generation manifest pair selection mismatch")
                if bool(prior.get("target_rgb_materialized")):
                    raise RuntimeError(f"scene {scene_id}: generation manifest incorrectly says target RGB was materialized")

                materialize(
                    f"{pair_prefix}blended_images/{target_stem}.jpg",
                    Path("blended_images") / f"{target_stem}.jpg",
                )
                if not target_rgb_path.is_file():
                    raise RuntimeError(f"scene {scene_id}: target RGB missing after target materialization")

                payload = {
                    "version": "0.1",
                    "stage": "refworld-cross-scene-target-rgb-materialization",
                    "scene_id": scene_id,
                    "frozen_scene_order": scene_order,
                    "pair_record_order": 1,
                    "anchor_view_id": anchor_id,
                    "target_source_order": 1,
                    "target_view_id": target_id,
                    "pair_score": pair_score,
                    "target_rgb_materialized": True,
                    "target_rgb_sha256": sha256_file(target_rgb_path),
                    "target_depth_materialized": False,
                    "download_policy": "HTTP-range-selective; no full archive download",
                    "files": fetched,
                }
                target_manifest.write_text(
                    json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
                print(f"target manifest: {target_manifest}", flush=True)

            batch_records.append(
                {
                    "scene_id": scene_id,
                    "frozen_scene_order": scene_order,
                    "anchor_view_id": anchor_id,
                    "target_view_id": target_id,
                    "phase": args.phase,
                }
            )

    batch_manifest = output_root / f"cross-scene-{args.phase}-materialization.safe.json"
    batch_manifest.write_text(
        json.dumps(
            {
                "version": "0.1",
                "stage": f"refworld-cross-scene-{args.phase}-materialization-batch",
                "source_manifest_id": frozen.get("id"),
                "scene_orders": [int(item["frozen_scene_order"]) for item in batch_records],
                "scene_count": len(batch_records),
                "records": batch_records,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nCROSS-SCENE {args.phase.upper()} MATERIALIZATION COMPLETE", flush=True)
    print(f"Batch manifest: {batch_manifest}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr, flush=True)
        raise
