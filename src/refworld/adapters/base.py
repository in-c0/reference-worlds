"""Neutral adapter contract for RefWorldBench baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class Unsupported:
    """Explicitly marks a capability that a baseline does not expose."""

    capability: str
    reason: str


@dataclass(frozen=True)
class Camera:
    """Renderer-neutral pinhole camera payload.

    Matrices are flattened row-major arrays. Coordinate convention belongs in
    ``convention`` so adapters cannot silently mix OpenCV/OpenGL frames.
    """

    intrinsics: tuple[float, ...]
    extrinsics: tuple[float, ...]
    convention: str

    def __post_init__(self) -> None:
        if len(self.intrinsics) != 9:
            raise ValueError("intrinsics must contain 9 values")
        if len(self.extrinsics) != 16:
            raise ValueError("extrinsics must contain 16 values")
        if not self.convention.strip():
            raise ValueError("camera convention must be explicit")


@dataclass(frozen=True)
class WorldRef:
    system: str
    world_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExportBundle:
    root: Path
    assets: Mapping[str, Path]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RenderFrame:
    image: Any
    camera: Camera
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class WorldAdapter(Protocol):
    """Minimum seam between a world system and the neutral benchmark core."""

    name: str

    def generate(self, input_ref: str | Path, config: Mapping[str, Any]) -> WorldRef:
        ...

    def export(self, world: WorldRef, destination: Path) -> ExportBundle | Unsupported:
        ...

    def anchor_camera(self, world: WorldRef) -> Camera | Unsupported:
        ...

    def render(
        self,
        world: WorldRef,
        camera: Camera,
        resolution: tuple[int, int],
    ) -> RenderFrame:
        ...

    def navigate(self, world: WorldRef, path: Sequence[Camera]) -> Sequence[RenderFrame] | Unsupported:
        ...

    def entities(self, world: WorldRef) -> Sequence[Mapping[str, Any]] | Unsupported:
        ...

    def snapshot(self, world: WorldRef) -> Mapping[str, Any] | Unsupported:
        ...

    def edit(self, world: WorldRef, edit_spec: Mapping[str, Any]) -> Mapping[str, Any] | Unsupported:
        ...

    def restore(self, world: WorldRef, snapshot: Mapping[str, Any]) -> Mapping[str, Any] | Unsupported:
        ...
