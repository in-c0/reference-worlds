"""Strict Portable Float Map reader for MVSNet/BlendedMVS depth maps."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_DIMENSIONS = re.compile(rb"^(\d+)\s+(\d+)\s*$")


def read_pfm(path: str | Path) -> np.ndarray:
    """Read a PFM file and return a top-to-bottom float32 array.

    PFM stores rows bottom-to-top. ``Pf`` yields HxW grayscale depth; ``PF``
    yields HxWx3. The scale sign determines endianess; its magnitude is applied
    to the decoded samples as specified by common MVSNet tooling.
    """

    file_path = Path(path)
    with file_path.open("rb") as handle:
        header = handle.readline().strip()
        if header not in {b"Pf", b"PF"}:
            raise ValueError(f"unsupported PFM header {header!r}")
        color = header == b"PF"

        dimensions = handle.readline().strip()
        match = _DIMENSIONS.match(dimensions)
        if match is None:
            raise ValueError("PFM dimensions line is malformed")
        width = int(match.group(1))
        height = int(match.group(2))
        if width <= 0 or height <= 0:
            raise ValueError("PFM dimensions must be positive")

        scale_line = handle.readline().strip()
        try:
            scale = float(scale_line)
        except ValueError as exc:
            raise ValueError("PFM scale is not numeric") from exc
        if not np.isfinite(scale) or scale == 0.0:
            raise ValueError("PFM scale must be finite and non-zero")

        endian = "<" if scale < 0 else ">"
        channels = 3 if color else 1
        count = width * height * channels
        data = np.fromfile(handle, dtype=np.dtype(endian + "f4"), count=count)
        if data.size != count:
            raise ValueError(f"PFM payload is truncated: expected {count} floats, got {data.size}")
        trailing = handle.read(1)
        if trailing:
            raise ValueError("PFM contains unexpected trailing payload")

    shape = (height, width, channels) if color else (height, width)
    array = data.reshape(shape)
    array = np.flipud(array).astype(np.float32, copy=False)
    magnitude = abs(scale)
    if magnitude != 1.0:
        array = array * np.float32(magnitude)
    if not np.all(np.isfinite(array) | np.isnan(array)):
        raise ValueError("PFM contains invalid non-finite samples")
    return np.asarray(array, dtype=np.float32)
