import hashlib
import json
from pathlib import Path

import pytest

from refworld.adapters.base import Unsupported
from refworld.adapters.worldgen_local import (
    WORLDGEN_PIN,
    WorldGenLocalAdapter,
    WorldGenLocalConfig,
    build_worldgen_command,
)
from refworld.runners.worldgen_generate import (
    WORLDGEN_PIN as RUNNER_WORLDGEN_PIN,
    artifact_record,
)


def test_runner_and_adapter_pin_same_worldgen_commit():
    assert WORLDGEN_PIN == RUNNER_WORLDGEN_PIN


def test_build_command_exposes_seed_and_quality_switches(tmp_path: Path):
    config = WorldGenLocalConfig(
        worldgen_root=tmp_path / "WorldGen",
        output_dir=tmp_path / "out",
        python_executable="python-gpu",
        seed=17,
        resolution=1200,
        low_vram=False,
        use_sharp=True,
        inpaint_bg=True,
        return_mesh=False,
        prompt="",
    )
    command = build_worldgen_command(tmp_path / "reference.jpg", config)
    assert command[:3] == ["python-gpu", "-m", "refworld.runners.worldgen_generate"]
    assert command[command.index("--seed") + 1] == "17"
    assert command[command.index("--resolution") + 1] == "1200"
    assert "--no-low-vram" in command
    assert "--use-sharp" in command
    assert "--inpaint-bg" in command
    assert "--no-return-mesh" in command
    assert "--allow-unpinned-worldgen" not in command


def test_adapter_normalizes_safe_runner_manifest_and_existing_exports(tmp_path: Path):
    reference = tmp_path / "reference.jpg"
    reference.write_bytes(b"reference")
    output = tmp_path / "output"
    expected_hash = hashlib.sha256(reference.read_bytes()).hexdigest()
    seen = []

    def fake_run(command, check):
        assert check is True
        seen.append(command)
        output.mkdir(parents=True, exist_ok=True)
        (output / "panorama.png").write_bytes(b"pano")
        (output / "world-splat.ply").write_bytes(b"ply")
        manifest = {
            "version": "0.1",
            "experiment": "EXP-000",
            "baseline": "worldgen",
            "upstream": {
                "repo": "ZiYang-xie/WorldGen",
                "expected_commit": WORLDGEN_PIN,
                "actual_commit": WORLDGEN_PIN,
                "unpinned_allowed": False,
            },
            "input": {"sha256": expected_hash},
            "configuration": {"seed": 42},
            "artifacts": [
                {"kind": "panorama", "path": "panorama.png", "sha256": "a", "size_bytes": 4},
                {"kind": "splat-ply", "path": "world-splat.ply", "sha256": "b", "size_bytes": 3},
            ],
        }
        (output / "run.safe.json").write_text(json.dumps(manifest))

    adapter = WorldGenLocalAdapter(run=fake_run)
    world = adapter.generate(
        reference,
        {
            "worldgen_root": tmp_path / "WorldGen",
            "output_dir": output,
            "seed": 42,
        },
    )

    assert seen
    assert world.system == "worldgen-local"
    assert world.world_id == f"worldgen-{expected_hash[:12]}-seed42"

    bundle = adapter.export(world, tmp_path / "unused-copy-destination")
    assert bundle.root == output.resolve()
    assert bundle.assets["panorama"].read_bytes() == b"pano"
    assert bundle.assets["splat-ply"].read_bytes() == b"ply"

    assert isinstance(adapter.anchor_camera(world), Unsupported)
    assert isinstance(adapter.render(world, None, (768, 576)), Unsupported)
    assert isinstance(adapter.entities(world), Unsupported)
    assert isinstance(adapter.snapshot(world), Unsupported)


def test_adapter_rejects_unpinned_runner_manifest(tmp_path: Path):
    reference = tmp_path / "reference.jpg"
    reference.write_bytes(b"reference")
    output = tmp_path / "output"

    def fake_run(command, check):
        output.mkdir(parents=True, exist_ok=True)
        (output / "run.safe.json").write_text(
            json.dumps(
                {
                    "experiment": "EXP-000",
                    "baseline": "worldgen",
                    "upstream": {"actual_commit": "deadbeef"},
                    "input": {"sha256": "a" * 64},
                    "artifacts": [],
                }
            )
        )

    adapter = WorldGenLocalAdapter(run=fake_run)
    with pytest.raises(RuntimeError, match="unpinned"):
        adapter.generate(
            reference,
            {"worldgen_root": tmp_path / "WorldGen", "output_dir": output},
        )


def test_runner_artifact_record_is_output_relative_and_rejects_escape(tmp_path: Path):
    root = tmp_path / "out"
    root.mkdir()
    inside = root / "world.ply"
    inside.write_bytes(b"world")
    record = artifact_record(inside, root, "splat-ply")
    assert record["path"] == "world.ply"
    assert record["sha256"] == hashlib.sha256(b"world").hexdigest()

    outside = tmp_path / "secret.txt"
    outside.write_text("secret")
    with pytest.raises(RuntimeError, match="escaped"):
        artifact_record(outside, root, "bad")
