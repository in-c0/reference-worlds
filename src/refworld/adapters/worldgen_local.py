"""Local WorldGen baseline adapter.

The benchmark package stays lightweight. Heavy WorldGen/CUDA dependencies live
in a separate environment; this adapter invokes the RefWorld runner in that
environment and normalizes its hash-based output into the neutral WorldAdapter
contract.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .base import Camera, ExportBundle, RenderFrame, Unsupported, WorldRef


WORLDGEN_PIN = "7ce7b2767fdf31e2727b69a2e61e2e950e3a017f"


@dataclass(frozen=True)
class WorldGenLocalConfig:
    worldgen_root: Path
    output_dir: Path
    python_executable: str = sys.executable
    seed: int = 42
    resolution: int = 1600
    low_vram: bool = True
    use_sharp: bool = False
    inpaint_bg: bool = False
    return_mesh: bool = False
    allow_unpinned_worldgen: bool = False
    prompt: str = ""

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "WorldGenLocalConfig":
        try:
            worldgen_root = Path(config["worldgen_root"])
            output_dir = Path(config["output_dir"])
        except KeyError as exc:
            raise ValueError(f"missing WorldGen config field: {exc.args[0]}") from exc
        return cls(
            worldgen_root=worldgen_root,
            output_dir=output_dir,
            python_executable=str(config.get("python_executable") or sys.executable),
            seed=int(config.get("seed", 42)),
            resolution=int(config.get("resolution", 1600)),
            low_vram=bool(config.get("low_vram", True)),
            use_sharp=bool(config.get("use_sharp", False)),
            inpaint_bg=bool(config.get("inpaint_bg", False)),
            return_mesh=bool(config.get("return_mesh", False)),
            allow_unpinned_worldgen=bool(config.get("allow_unpinned_worldgen", False)),
            prompt=str(config.get("prompt", "")),
        )


def build_worldgen_command(reference: Path, config: WorldGenLocalConfig) -> list[str]:
    cmd = [
        config.python_executable,
        "-m",
        "refworld.runners.worldgen_generate",
        "--worldgen-root",
        str(config.worldgen_root),
        "--reference",
        str(reference),
        "--output",
        str(config.output_dir),
        "--seed",
        str(config.seed),
        "--resolution",
        str(config.resolution),
        "--prompt",
        config.prompt,
        "--low-vram" if config.low_vram else "--no-low-vram",
        "--use-sharp" if config.use_sharp else "--no-use-sharp",
        "--inpaint-bg" if config.inpaint_bg else "--no-inpaint-bg",
        "--return-mesh" if config.return_mesh else "--no-return-mesh",
    ]
    if config.allow_unpinned_worldgen:
        cmd.append("--allow-unpinned-worldgen")
    return cmd


class WorldGenLocalAdapter:
    name = "worldgen-local"

    def __init__(self, *, run: Callable[..., Any] = subprocess.run) -> None:
        self._run = run

    def generate(self, input_ref: str | Path, config: Mapping[str, Any]) -> WorldRef:
        source = Path(input_ref)
        if not source.is_file():
            raise FileNotFoundError(source)
        cfg = WorldGenLocalConfig.from_mapping(config)
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        cmd = build_worldgen_command(source, cfg)
        self._run(cmd, check=True)

        manifest_path = cfg.output_dir / "run.safe.json"
        if not manifest_path.is_file():
            raise RuntimeError("WorldGen runner completed without run.safe.json")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("experiment") != "EXP-000" or manifest.get("baseline") != "worldgen":
            raise RuntimeError("unexpected WorldGen runner manifest")
        actual = manifest.get("upstream", {}).get("actual_commit")
        if actual != WORLDGEN_PIN and not cfg.allow_unpinned_worldgen:
            raise RuntimeError(f"runner returned unpinned WorldGen commit: {actual!r}")

        input_hash = manifest.get("input", {}).get("sha256")
        if not isinstance(input_hash, str) or len(input_hash) != 64:
            raise RuntimeError("WorldGen runner manifest has no valid input sha256")

        return WorldRef(
            system=self.name,
            world_id=f"worldgen-{input_hash[:12]}-seed{cfg.seed}",
            metadata={
                "output_dir": str(cfg.output_dir.resolve()),
                "manifest": manifest,
            },
        )

    def export(self, world: WorldRef, destination: Path) -> ExportBundle | Unsupported:
        if world.system != self.name:
            raise ValueError(f"WorldRef belongs to {world.system!r}, not {self.name!r}")
        raw_root = world.metadata.get("output_dir")
        manifest = world.metadata.get("manifest")
        if not isinstance(raw_root, str) or not isinstance(manifest, Mapping):
            raise RuntimeError("WorldRef is missing local WorldGen provenance")
        root = Path(raw_root)
        assets: dict[str, Path] = {}
        for item in manifest.get("artifacts", []):
            if not isinstance(item, Mapping):
                continue
            kind = item.get("kind")
            rel = item.get("path")
            if isinstance(kind, str) and isinstance(rel, str):
                path = (root / rel).resolve()
                try:
                    path.relative_to(root.resolve())
                except ValueError as exc:
                    raise RuntimeError("artifact path escaped WorldGen output root") from exc
                if not path.is_file():
                    raise FileNotFoundError(path)
                assets[kind] = path
        if not assets:
            return Unsupported("export", "WorldGen run contains no materialized artifacts")

        # The generation runner has already materialized potentially large PLYs.
        # The destination parameter is intentionally not used to duplicate them.
        return ExportBundle(
            root=root,
            assets=assets,
            metadata={
                "upstream": manifest.get("upstream", {}),
                "configuration": manifest.get("configuration", {}),
                "artifacts": manifest.get("artifacts", []),
            },
        )

    def anchor_camera(self, world: WorldRef) -> Camera | Unsupported:
        return Unsupported(
            "anchor_camera",
            "WorldGen does not expose the source pinhole camera; recover it with RefWorld registration.",
        )

    def render(
        self,
        world: WorldRef,
        camera: Camera,
        resolution: tuple[int, int],
    ) -> RenderFrame | Unsupported:
        return Unsupported(
            "render",
            "WorldGen baseline exports PLY/mesh; use the pinned benchmark renderer for scoring.",
        )

    def navigate(
        self, world: WorldRef, path: Sequence[Camera]
    ) -> Sequence[RenderFrame] | Unsupported:
        return Unsupported("navigate", "navigation is evaluated through the benchmark renderer")

    def entities(self, world: WorldRef) -> Sequence[Mapping[str, Any]] | Unsupported:
        return Unsupported("entities", "WorldGen does not expose native semantic entities")

    def snapshot(self, world: WorldRef) -> Mapping[str, Any] | Unsupported:
        return Unsupported("snapshot", "static exported WorldGen assets have no native snapshot API")

    def edit(
        self, world: WorldRef, edit_spec: Mapping[str, Any]
    ) -> Mapping[str, Any] | Unsupported:
        return Unsupported("edit", "WorldGen does not expose a native persistent edit API")

    def restore(
        self, world: WorldRef, snapshot: Mapping[str, Any]
    ) -> Mapping[str, Any] | Unsupported:
        return Unsupported("restore", "WorldGen does not expose a native restore API")
