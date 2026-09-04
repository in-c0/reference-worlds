#!/usr/bin/env python3
"""Selective BlendedMVS materializer for sealed VGGT G1 rank-3 evaluation.

Generation phase materializes only anchor RGB/camera/depth + rank-3 target camera.
Target RGB must remain absent. Target depth is never materialized. Target phase
requires the experiment-wide sealed-generation marker before any target RGB read.
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
SCENE_ORDERS = tuple(range(2, 11))
TARGET_SOURCE_ORDER = 3
SEAL_STAGE = "refworld-vggt-g1-rank3-all-generation-before-targets"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize sealed VGGT G1 rank-3 BlendedMVS inputs")
    parser.add_argument("--phase", choices=("generation", "targets"), required=True)
    parser.add_argument("--url", default=RELEASE_URL)
    parser.add_argument("--seal", type=Path, help="required experiment-wide seal for target phase")
    return parser.parse_args()


def select_scenes(frozen: dict[str, Any]) -> list[dict[str, Any]]:
    scenes = frozen.get("scenes")
    if not isinstance(scenes, list):
        raise RuntimeError("frozen manifest has no scenes")
    by_order = {int(item["order"]): item for item in scenes}
    if any(order not in by_order for order in SCENE_ORDERS):
        raise RuntimeError("frozen manifest missing one of scene orders 2-10")
    return [by_order[order] for order in SCENE_ORDERS]


def validate_seal(path: Path, repo_root: Path) -> None:
    seal_path = path.resolve()
    if not seal_path.is_file():
        raise FileNotFoundError(seal_path)
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("stage") != SEAL_STAGE:
        raise RuntimeError("G1 seal stage mismatch")
    if seal.get("target_source_order") != TARGET_SOURCE_ORDER:
        raise RuntimeError("G1 seal target rank mismatch")
    rows = seal.get("scenes")
    if not isinstance(rows, list) or [int(row["frozen_scene_order"]) for row in rows] != list(SCENE_ORDERS):
        raise RuntimeError("G1 seal scene set/order mismatch")
    if bool(seal.get("target_rgb_present_when_marker_written")):
        raise RuntimeError("G1 seal claims target RGB was already present")
    for row in rows:
        for key in ("source_geometry_manifest", "learned_manifest", "oracle_manifest", "candidate_manifest", "composition_manifest"):
            record = row.get(key)
            if not isinstance(record, dict):
                raise RuntimeError(f"G1 seal missing {key}")
            artifact = repo_root / str(record.get("path", ""))
            if not artifact.is_file() or sha256_file(artifact) != str(record.get("sha256", "")):
                raise RuntimeError(f"G1 sealed artifact mismatch: {artifact}")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "private-data" / "blendedmvs-bootstrap"
    frozen = load_manifest(repo_root / "datasets" / "blendedmvs-bootstrap-v0.json")
    selected = select_scenes(frozen)

    if args.phase == "targets":
        if args.seal is None:
            raise ValueError("--seal is required for target phase")
        validate_seal(args.seal, repo_root)

    print("RefWorld VGGT G1 rank-3 selective materialization", flush=True)
    print(f"Phase: {args.phase}", flush=True)
    print("Scenes: 2-10 | pair record: 1 | published source rank: 3", flush=True)
    print("Policy: HTTP ranges only; target depth is never fetched.", flush=True)
    if args.phase == "generation":
        print("Seal: every rank-3 target RGB must remain absent.", flush=True)
    else:
        print("Seal verified: all VGGT/oracle/LaMa B/C generation artifacts were hash-pinned first.", flush=True)

    batch: list[dict[str, Any]] = []
    with RemoteZip(args.url, timeout_seconds=120.0) as archive:
        entries = archive.entries()
        print(f"Remote logical archive: {archive.size_bytes / (1024**3):.2f} GiB; entries={len(entries)}", flush=True)

        for scene_entry in selected:
            scene_order = int(scene_entry["order"])
            scene_id = str(scene_entry["id"])
            scene_root = data_root / scene_id
            scene_root.mkdir(parents=True, exist_ok=True)
            suffix = f"/{scene_id}/cams/pair.txt"
            matches = [entry for name, entry in entries.items() if name.endswith(suffix) or name == suffix.lstrip("/")]
            if len(matches) != 1:
                raise RuntimeError(f"scene {scene_id}: expected one pair.txt, found {len(matches)}")
            pair_entry = matches[0]
            pair_bytes = archive.read(pair_entry)
            records = parse_pair_text(pair_bytes.decode("utf-8"))
            if not records or len(records[0].source_ids) < TARGET_SOURCE_ORDER:
                raise RuntimeError(f"scene {scene_id}: first pair record has fewer than three source views")
            first = records[0]
            anchor_id = int(first.reference_id)
            target_id = int(first.source_ids[TARGET_SOURCE_ORDER - 1])
            pair_score = float(first.scores[TARGET_SOURCE_ORDER - 1])
            anchor_stem = f"{anchor_id:08d}"
            target_stem = f"{target_id:08d}"
            prefix = pair_entry.name[: -len("cams/pair.txt")]
            target_rgb = scene_root / "blended_images" / f"{target_stem}.jpg"
            generation_manifest = scene_root / "vggt-g1-rank3-generation-inputs.safe.json"
            target_manifest = scene_root / "vggt-g1-rank3-target-rgb.safe.json"

            print(f"\nScene {scene_order}/10 {scene_id}: anchor={anchor_id}, rank3 target={target_id}", flush=True)
            if args.phase == "generation" and target_rgb.exists():
                raise RuntimeError(f"scene {scene_order}: rank-3 target RGB already exists before sealed generation: {target_rgb}")
            if args.phase == "targets" and not generation_manifest.is_file():
                raise RuntimeError(f"scene {scene_order}: generation manifest missing before target phase")

            fetched: list[dict[str, Any]] = []

            def materialize(archive_name: str, relative: Path) -> None:
                if archive_name not in entries:
                    raise FileNotFoundError(archive_name)
                entry = entries[archive_name]
                destination = scene_root / relative
                if destination.is_file() and destination.stat().st_size == entry.uncompressed_size:
                    existing = destination.read_bytes()
                    if (binascii.crc32(existing) & 0xFFFFFFFF) == entry.crc32:
                        fetched.append({"archive_entry": entry.name, "path": relative.as_posix(), "size_bytes": len(existing), "sha256": sha256_bytes(existing), "reused": True})
                        print(f"cached: {relative.as_posix()}", flush=True)
                        return
                data = archive.read(entry)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
                fetched.append({"archive_entry": entry.name, "path": relative.as_posix(), "size_bytes": len(data), "sha256": sha256_bytes(data), "reused": False})
                print(f"fetch:  {relative.as_posix()} ({len(data) / (1024**2):.2f} MiB)", flush=True)

            if args.phase == "generation":
                pair_path = scene_root / "cams" / "pair.txt"
                pair_path.parent.mkdir(parents=True, exist_ok=True)
                pair_path.write_bytes(pair_bytes)
                for archive_name, relative in (
                    (f"{prefix}blended_images/{anchor_stem}.jpg", Path("blended_images") / f"{anchor_stem}.jpg"),
                    (f"{prefix}cams/{anchor_stem}_cam.txt", Path("cams") / f"{anchor_stem}_cam.txt"),
                    (f"{prefix}rendered_depth_maps/{anchor_stem}.pfm", Path("rendered_depth_maps") / f"{anchor_stem}.pfm"),
                    (f"{prefix}cams/{target_stem}_cam.txt", Path("cams") / f"{target_stem}_cam.txt"),
                ):
                    materialize(archive_name, relative)

                from PIL import Image
                anchor_rgb_path = scene_root / "blended_images" / f"{anchor_stem}.jpg"
                anchor_depth_path = scene_root / "rendered_depth_maps" / f"{anchor_stem}.pfm"
                with Image.open(anchor_rgb_path) as image:
                    width, height = image.convert("RGB").size
                depth = read_pfm(anchor_depth_path)
                if depth.ndim != 2 or depth.shape != (height, width):
                    raise RuntimeError(f"scene {scene_order}: anchor RGB/depth shape mismatch")
                if not np.any(np.isfinite(depth) & (depth > 1e-6)):
                    raise RuntimeError(f"scene {scene_order}: anchor depth has no valid support")
                parse_camera_text((scene_root / "cams" / f"{anchor_stem}_cam.txt").read_text(encoding="utf-8"))
                parse_camera_text((scene_root / "cams" / f"{target_stem}_cam.txt").read_text(encoding="utf-8"))
                if target_rgb.exists():
                    raise RuntimeError(f"scene {scene_order}: rank-3 target RGB appeared during generation materialization")
                generation_manifest.write_text(json.dumps({
                    "version": "0.1",
                    "stage": "refworld-vggt-g1-rank3-generation-materialization",
                    "scene_id": scene_id,
                    "frozen_scene_order": scene_order,
                    "pair_record_order": 1,
                    "anchor_view_id": anchor_id,
                    "target_source_order": TARGET_SOURCE_ORDER,
                    "target_view_id": target_id,
                    "pair_score": pair_score,
                    "target_rgb_materialized": False,
                    "target_depth_materialized": False,
                    "files": fetched,
                }, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
            else:
                prior = json.loads(generation_manifest.read_text(encoding="utf-8"))
                if int(prior["target_source_order"]) != TARGET_SOURCE_ORDER or int(prior["target_view_id"]) != target_id:
                    raise RuntimeError(f"scene {scene_order}: generation selection mismatch")
                materialize(f"{prefix}blended_images/{target_stem}.jpg", Path("blended_images") / f"{target_stem}.jpg")
                target_manifest.write_text(json.dumps({
                    "version": "0.1",
                    "stage": "refworld-vggt-g1-rank3-target-rgb-materialization",
                    "scene_id": scene_id,
                    "frozen_scene_order": scene_order,
                    "pair_record_order": 1,
                    "anchor_view_id": anchor_id,
                    "target_source_order": TARGET_SOURCE_ORDER,
                    "target_view_id": target_id,
                    "pair_score": pair_score,
                    "target_rgb_materialized": True,
                    "target_rgb_sha256": sha256_file(target_rgb),
                    "target_depth_materialized": False,
                    "files": fetched,
                }, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

            batch.append({"frozen_scene_order": scene_order, "scene_id": scene_id, "anchor_view_id": anchor_id, "target_view_id": target_id})

    batch_path = data_root / f"vggt-g1-rank3-{args.phase}-materialization.safe.json"
    batch_path.write_text(json.dumps({
        "version": "0.1",
        "stage": f"refworld-vggt-g1-rank3-{args.phase}-materialization-batch",
        "scene_orders": list(SCENE_ORDERS),
        "target_source_order": TARGET_SOURCE_ORDER,
        "target_depth_materialized": False,
        "records": batch,
    }, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(f"\nVGGT G1 RANK-3 {args.phase.upper()} MATERIALIZATION COMPLETE", flush=True)
    print(f"Batch manifest: {batch_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr, flush=True)
        raise
