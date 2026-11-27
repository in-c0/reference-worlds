import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from refworld.source_geometry import load_source_geometry


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(root: Path) -> Path:
    depth = np.full((2, 3), 2.0, dtype=np.float32)
    confidence = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    depth_path = root / "depth.npy"
    confidence_path = root / "confidence-raw.npy"
    np.save(depth_path, depth, allow_pickle=False)
    np.save(confidence_path, confidence, allow_pickle=False)

    manifest = {
        "version": "0.1",
        "stage": "refworld-source-geometry",
        "backend": "vggt",
        "input": {
            "sha256": "a" * 64,
            "width": 3,
            "height": 2,
        },
        "camera": {
            "intrinsics": [100.0, 0.0, 1.0, 0.0, 100.0, 0.5, 0.0, 0.0, 1.0],
            "extrinsics": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "convention": "opengl-camera-to-world",
        },
        "geometry": {
            "depth_semantics": "positive optical-axis Z depth",
            "confidence_semantics": "raw VGGT depth confidence",
            "confidence_calibration": None,
        },
        "artifacts": [
            {"kind": "source-depth-npy", "path": "depth.npy", "sha256": _sha(depth_path)},
            {"kind": "source-confidence-raw-npy", "path": "confidence-raw.npy", "sha256": _sha(confidence_path)},
        ],
    }
    manifest_path = root / "source-geometry.safe.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


def test_load_source_geometry_verifies_and_loads_arrays(tmp_path):
    manifest_path = _write_fixture(tmp_path)

    geometry = load_source_geometry(manifest_path)

    assert geometry.backend == "vggt"
    assert geometry.depth.shape == (2, 3)
    assert geometry.confidence_raw.shape == (2, 3)
    np.testing.assert_allclose(geometry.depth, 2.0)
    np.testing.assert_allclose(geometry.confidence_raw[1], [4.0, 5.0, 6.0])


def test_load_source_geometry_rejects_tampered_artifact(tmp_path):
    manifest_path = _write_fixture(tmp_path)
    with (tmp_path / "depth.npy").open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_source_geometry(manifest_path)


def test_load_source_geometry_rejects_nonpositive_depth(tmp_path):
    manifest_path = _write_fixture(tmp_path)
    depth_path = tmp_path / "depth.npy"
    depth = np.load(depth_path, allow_pickle=False)
    depth[0, 0] = 0.0
    np.save(depth_path, depth, allow_pickle=False)

    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"][0]["sha256"] = _sha(depth_path)
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="positive optical-axis depth"):
        load_source_geometry(manifest_path)


def test_load_source_geometry_rejects_silent_confidence_calibration(tmp_path):
    manifest_path = _write_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["geometry"]["confidence_calibration"] = {"method": "per-image-minmax"}
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="raw uncalibrated VGGT confidence"):
        load_source_geometry(manifest_path)


def test_load_source_geometry_rejects_path_escape(tmp_path):
    manifest_path = _write_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"][0]["path"] = "../depth.npy"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="output-relative"):
        load_source_geometry(manifest_path)
