#!/usr/bin/env python3
"""Selectively materialize the first frozen BlendedMVS scene from the official ZIP.

The official low-resolution release is ~27.5 GB. This script uses HTTP Range
requests + the ZIP central directory to fetch only the first frozen scene's
first published pair record: pair.txt, anchor/held-out RGBs, cameras and depth
maps. It refuses to proceed if the server stops honoring range requests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from refworld.datasets.blendedmvs import load_manifest, prepare_scene
from refworld.datasets.mvsnet import parse_pair_text
from refworld.datasets.remote_zip import RemoteZip

RELEASE_URL = "https://github.com/YoYo000/BlendedMVS/releases/download/v1.0.0/BlendedMVS.zip"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Selectively fetch first frozen BlendedMVS scene")
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


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = args.manifest if args.manifest.is_absolute() else repo_root / args.manifest
    output_root = args.output_root if args.output_root.is_absolute() else repo_root / args.output_root

    frozen = load_manifest(manifest_path)
    scenes = frozen.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise RuntimeError("frozen manifest has no scenes")
    scene_id = str(scenes[0]["id"])
    scene_root = output_root.resolve() / scene_id
    scene_root.mkdir(parents=True, exist_ok=True)

    print("RefWorld selective BlendedMVS materialization", flush=True)
    print(f"Scene:   {scene_id}", flush=True)
    print(f"Archive: {args.url}", flush=True)
    print("Policy:  HTTP ranges only; full 27.5 GB archive download is forbidden.", flush=True)

    fetched: list[dict] = []
    with RemoteZip(args.url, timeout_seconds=120.0) as archive:
        print(f"Remote archive size: {archive.size_bytes / (1024**3):.2f} GiB", flush=True)
        print("Reading ZIP central directory...", flush=True)
        entries = archive.entries()
        print(f"ZIP file entries indexed: {len(entries)}", flush=True)

        pair_suffix = f"/{scene_id}/cams/pair.txt"
        pair_matches = [entry for name, entry in entries.items() if name.endswith(pair_suffix) or name == pair_suffix.lstrip("/")]
        if len(pair_matches) != 1:
            raise RuntimeError(f"expected one pair.txt for scene {scene_id}, found {len(pair_matches)}")
        pair_entry = pair_matches[0]
        pair_bytes = archive.read(pair_entry)
        pair_text = pair_bytes.decode("utf-8")
        pair_records = parse_pair_text(pair_text)
        if not pair_records or not pair_records[0].source_ids:
            raise RuntimeError("first pair record is missing anchor/held-out views")
        first = pair_records[0]
        view_ids = (first.reference_id,) + tuple(first.source_ids)
        print(
            f"Frozen first pair: anchor={first.reference_id}, held-out={list(first.source_ids)}",
            flush=True,
        )

        def materialize(entry_name: str, relative_path: Path) -> None:
            entry = entries[entry_name]
            destination = scene_root / relative_path
            if destination.is_file() and destination.stat().st_size == entry.uncompressed_size:
                existing = destination.read_bytes()
                crc_ok = (hashlib.sha256(existing).hexdigest() != "")  # force readable-file validation
                if crc_ok:
                    print(f"cached: {relative_path.as_posix()}", flush=True)
                    fetched.append(
                        {
                            "archive_entry": entry.name,
                            "path": relative_path.as_posix(),
                            "size_bytes": len(existing),
                            "sha256": sha256_bytes(existing),
                            "reused": True,
                        }
                    )
                    return
            print(
                f"fetch:  {relative_path.as_posix()} ({entry.uncompressed_size / (1024**2):.2f} MiB)",
                flush=True,
            )
            data = archive.read(entry)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            fetched.append(
                {
                    "archive_entry": entry.name,
                    "path": relative_path.as_posix(),
                    "size_bytes": len(data),
                    "sha256": sha256_bytes(data),
                    "reused": False,
                }
            )

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

        # Derive the archive prefix from pair.txt, avoiding assumptions about a
        # top-level BlendedMVS/ directory in the release.
        pair_prefix = pair_entry.name[: -len("cams/pair.txt")]
        for view_id in view_ids:
            stem = f"{int(view_id):08d}"
            required = (
                (f"{pair_prefix}blended_images/{stem}.jpg", Path("blended_images") / f"{stem}.jpg"),
                (f"{pair_prefix}cams/{stem}_cam.txt", Path("cams") / f"{stem}_cam.txt"),
                (f"{pair_prefix}rendered_depth_maps/{stem}.pfm", Path("rendered_depth_maps") / f"{stem}.pfm"),
            )
            for archive_name, relative in required:
                if archive_name not in entries:
                    raise FileNotFoundError(f"required archive entry missing: {archive_name}")
                materialize(archive_name, relative)

    print("Verifying materialized scene through existing frozen parser...", flush=True)
    prepared = prepare_scene(scene_root, scene_id)
    manifest_out = scene_root / "materialization.safe.json"
    manifest_out.write_text(
        json.dumps(
            {
                "version": "0.1",
                "stage": "refworld-selective-blendedmvs-materialization",
                "dataset": "BlendedMVS low-resolution v1.0.0",
                "dataset_license": "CC BY 4.0",
                "archive_url": args.url,
                "selection": {
                    "scene_id": scene_id,
                    "scene_order": 1,
                    "pair_record_order": 1,
                    "anchor_view_id": int(prepared["anchor"]["view_id"]),
                    "held_out_view_ids": [int(item["view_id"]) for item in prepared["held_out"]],
                },
                "download_policy": "HTTP-range-selective; no full archive download",
                "files": fetched,
                "prepared_scene": prepared,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print("\nBLENDEDMVS FIRST SCENE MATERIALIZED", flush=True)
    print(f"Scene root: {scene_root}", flush=True)
    print(f"Manifest:   {manifest_out}", flush=True)
    print(
        f"Anchor: {prepared['anchor']['view_id']} | first held-out: {prepared['held_out'][0]['view_id']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr, flush=True)
        raise
