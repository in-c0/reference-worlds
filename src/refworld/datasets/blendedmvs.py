"""Deterministic preparation of the RefWorldBench BlendedMVS bootstrap.

This module never copies dataset imagery into the repository. Given a local
BlendedMVS root and the frozen scene manifest, it verifies required files and
emits metadata/hashes/camera separations only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .mvsnet import camera_pose_separation, parse_camera_text, parse_pair_text


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def _view_metadata(scene_root: Path, view_id: int) -> dict[str, Any]:
    stem = f"{view_id:08d}"
    image = _required(scene_root / "blended_images" / f"{stem}.jpg", "image")
    camera_path = _required(scene_root / "cams" / f"{stem}_cam.txt", "camera")
    depth = _required(scene_root / "rendered_depth_maps" / f"{stem}.pfm", "depth map")
    camera_file = parse_camera_text(camera_path.read_text(encoding="utf-8"))
    return {
        "view_id": view_id,
        "image": {
            "path": str(image.relative_to(scene_root)),
            "sha256": _sha256(image),
            "bytes": image.stat().st_size,
        },
        "camera": {
            "path": str(camera_path.relative_to(scene_root)),
            "sha256": _sha256(camera_path),
            "convention": camera_file.camera.convention,
            "source_convention": camera_file.source_convention,
            "intrinsics": list(camera_file.camera.intrinsics),
            "extrinsics": list(camera_file.camera.extrinsics),
            "depth_min": camera_file.depth_min,
            "depth_interval": camera_file.depth_interval,
            "depth_num": camera_file.depth_num,
            "depth_max": camera_file.depth_max,
            "rotation_orthonormalization_frobenius": camera_file.rotation_orthonormalization_frobenius,
        },
        "depth": {
            "path": str(depth.relative_to(scene_root)),
            "sha256": _sha256(depth),
            "bytes": depth.stat().st_size,
        },
        "_camera_object": camera_file.camera,
    }


def prepare_scene(scene_root: str | Path, scene_id: str) -> dict[str, Any]:
    root = Path(scene_root)
    pair_path = _required(root / "cams" / "pair.txt", "pair.txt")
    records = parse_pair_text(pair_path.read_text(encoding="utf-8"))
    if not records:
        raise ValueError(f"scene {scene_id} has no pair records")

    first = records[0]
    if not first.source_ids:
        raise ValueError(f"scene {scene_id} first pair record has no held-out sources")

    anchor = _view_metadata(root, first.reference_id)
    anchor_camera = anchor.pop("_camera_object")
    held_out: list[dict[str, Any]] = []

    for source_id, score in zip(first.source_ids, first.scores, strict=True):
        source = _view_metadata(root, source_id)
        source_camera = source.pop("_camera_object")
        source["pair_score"] = score
        source["separation_from_anchor"] = camera_pose_separation(anchor_camera, source_camera)
        held_out.append(source)

    return {
        "scene_id": scene_id,
        "scale_status": "unknown-source-units",
        "pair_file": {
            "path": "cams/pair.txt",
            "sha256": _sha256(pair_path),
            "reference_record_count": len(records),
        },
        "anchor": anchor,
        "held_out": held_out,
    }


def prepare_bootstrap(
    dataset_root: str | Path,
    frozen_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(dataset_root)
    if not root.is_dir():
        raise NotADirectoryError(root)
    scenes = frozen_manifest.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("frozen manifest must contain a non-empty scenes list")

    prepared = []
    seen: set[str] = set()
    for entry in scenes:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("id"), str):
            raise ValueError("every frozen scene entry must contain a string id")
        scene_id = entry["id"]
        if scene_id in seen:
            raise ValueError(f"duplicate scene id in frozen manifest: {scene_id}")
        seen.add(scene_id)
        prepared.append(prepare_scene(root / scene_id, scene_id))

    return {
        "version": "0.1",
        "source_manifest_id": frozen_manifest.get("id"),
        "dataset": frozen_manifest.get("dataset", {}),
        "scene_count": len(prepared),
        "selection_rule": frozen_manifest.get("selection_rule", {}),
        "scenes": prepared,
    }


def load_manifest(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("manifest root must be an object")
    return value
