from __future__ import annotations

import io
import zipfile

from refworld.datasets.remote_zip import RemoteZip


def _memory_remote_zip(payload: bytes) -> RemoteZip:
    remote = object.__new__(RemoteZip)
    remote.url = "memory://fixture.zip"
    remote.timeout_seconds = 1.0
    remote.size_bytes = len(payload)
    remote._entries = None
    remote._range = lambda start, end: payload[start : end + 1]
    return remote


def test_remote_zip_reads_stored_and_deflated_entries_without_full_download():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", allowZip64=True) as archive:
        archive.writestr("dataset/scene/cams/pair.txt", "1\n0\n1 1 0.9\n", compress_type=zipfile.ZIP_STORED)
        archive.writestr(
            "dataset/scene/blended_images/00000000.jpg",
            b"not-a-real-jpeg-but-byte-extraction-is-the-contract" * 100,
            compress_type=zipfile.ZIP_DEFLATED,
        )
    remote = _memory_remote_zip(buffer.getvalue())

    entries = remote.entries()
    assert set(entries) == {
        "dataset/scene/cams/pair.txt",
        "dataset/scene/blended_images/00000000.jpg",
    }
    assert remote.read("dataset/scene/cams/pair.txt") == b"1\n0\n1 1 0.9\n"
    assert remote.read("dataset/scene/blended_images/00000000.jpg").startswith(b"not-a-real-jpeg")


def test_remote_zip_unique_suffix_selection():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("root/abc/cams/pair.txt", "x")
    remote = _memory_remote_zip(buffer.getvalue())

    entry = remote.require_unique_suffix("abc/cams/pair.txt")
    assert entry.name == "root/abc/cams/pair.txt"
