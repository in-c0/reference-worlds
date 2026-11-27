from pathlib import Path

from refworld.datasets.blendedmvs import prepare_bootstrap


CAMERA = """extrinsic
1 0 0 {tx}
0 1 0 0
0 0 1 0
0 0 0 1

intrinsic
800 0 320
0 800 240
0 0 1

0.5 0.01 128 1.77
"""


def _write_view(scene: Path, view_id: int, tx: float):
    stem = f"{view_id:08d}"
    (scene / "blended_images").mkdir(parents=True, exist_ok=True)
    (scene / "cams").mkdir(parents=True, exist_ok=True)
    (scene / "rendered_depth_maps").mkdir(parents=True, exist_ok=True)
    (scene / "blended_images" / f"{stem}.jpg").write_bytes(f"image-{view_id}".encode())
    (scene / "cams" / f"{stem}_cam.txt").write_text(CAMERA.format(tx=tx))
    (scene / "rendered_depth_maps" / f"{stem}.pfm").write_bytes(f"depth-{view_id}".encode())


def test_prepare_bootstrap_uses_first_pair_record_and_emits_metadata_only(tmp_path: Path):
    scene_id = "scene-a"
    scene = tmp_path / scene_id
    _write_view(scene, 0, 0.0)
    _write_view(scene, 1, -1.0)
    _write_view(scene, 2, -2.0)
    (scene / "cams" / "pair.txt").write_text(
        "3\n0\n2 1 0.9 2 0.8\n1\n1 0 0.9\n2\n1 0 0.8\n"
    )

    frozen = {
        "id": "fixture",
        "dataset": {"name": "BlendedMVS"},
        "selection_rule": {"view_rule": "first pair record"},
        "scenes": [{"id": scene_id}],
    }
    prepared = prepare_bootstrap(tmp_path, frozen)

    assert prepared["scene_count"] == 1
    got = prepared["scenes"][0]
    assert got["anchor"]["view_id"] == 0
    assert [view["view_id"] for view in got["held_out"]] == [1, 2]
    assert [view["pair_score"] for view in got["held_out"]] == [0.9, 0.8]
    assert got["held_out"][0]["separation_from_anchor"]["center_distance_source_units"] == 1.0
    assert got["held_out"][1]["separation_from_anchor"]["center_distance_source_units"] == 2.0

    # The output is metadata-only: hashes/paths/calibration, never dataset bytes.
    assert got["anchor"]["image"]["sha256"]
    assert "image-0" not in repr(prepared)
    assert "_camera_object" not in repr(prepared)


def test_prepare_bootstrap_fails_closed_on_missing_required_file(tmp_path: Path):
    scene = tmp_path / "scene-a"
    _write_view(scene, 0, 0.0)
    (scene / "cams" / "pair.txt").write_text("1\n0\n1 1 0.9\n")
    frozen = {"id": "fixture", "scenes": [{"id": "scene-a"}]}

    try:
        prepare_bootstrap(tmp_path, frozen)
    except FileNotFoundError as exc:
        assert "00000001.jpg" in str(exc)
    else:
        raise AssertionError("missing held-out view should fail preparation")
