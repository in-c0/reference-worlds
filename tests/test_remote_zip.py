from __future__ import annotations

import io
import struct
import zipfile

from refworld.datasets.remote_zip import RemoteZip, RemoteZipVolume


def _memory_remote_zip(*payloads: bytes) -> RemoteZip:
    remote = object.__new__(RemoteZip)
    remote.url = "memory://fixture.zip"
    remote.timeout_seconds = 1.0
    remote._entries = None
    remote._session = None
    remote._volumes = tuple(
        RemoteZipVolume(index, f"memory://fixture-{index}", len(payload))
        for index, payload in enumerate(payloads)
    )
    remote.size_bytes = sum(len(payload) for payload in payloads)
    remote._read_volume = lambda disk, start, end: payloads[disk][start : end + 1]
    return remote


def _make_two_disk_split_zip() -> tuple[bytes, bytes, bytes]:
    expected = b"split-payload-" * 100
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", allowZip64=True) as archive:
        archive.writestr("dataset/scene/file.bin", expected, compress_type=zipfile.ZIP_STORED)
    payload = bytearray(buffer.getvalue())

    eocd = payload.rfind(b"PK\x05\x06")
    assert eocd >= 0
    cd_offset = struct.unpack_from("<L", payload, eocd + 16)[0]
    assert 0 < cd_offset < eocd

    # Split through the file payload so extraction must cross disk 0 -> disk 1.
    split = cd_offset // 2
    assert split > 30
    first = bytearray(payload[:split])
    second = bytearray(payload[split:])
    cd_rel = cd_offset - split
    eocd_rel = eocd - split
    assert second[cd_rel : cd_rel + 4] == b"PK\x01\x02"
    assert second[eocd_rel : eocd_rel + 4] == b"PK\x05\x06"

    # Classic split-ZIP metadata: final disk=1, central directory starts on disk 1,
    # while the entry's local header starts on disk 0 at offset 0.
    struct.pack_into("<H", second, cd_rel + 34, 0)  # disk number start
    struct.pack_into("<L", second, cd_rel + 42, 0)  # local-header offset on disk 0
    struct.pack_into("<H", second, eocd_rel + 4, 1)  # this disk
    struct.pack_into("<H", second, eocd_rel + 6, 1)  # central-directory start disk
    struct.pack_into("<H", second, eocd_rel + 8, 1)  # entries on this disk
    struct.pack_into("<H", second, eocd_rel + 10, 1)  # total entries
    struct.pack_into("<L", second, eocd_rel + 16, cd_rel)  # CD offset on disk 1
    return bytes(first), bytes(second), expected


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


def test_remote_zip_reads_file_spanning_classic_split_volumes():
    first, second, expected = _make_two_disk_split_zip()
    remote = _memory_remote_zip(first, second)

    entries = remote.entries()
    entry = entries["dataset/scene/file.bin"]
    assert entry.disk_start == 0
    assert entry.local_header_offset == 0
    assert remote.read(entry) == expected


def test_remote_zip_unique_suffix_selection():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("root/abc/cams/pair.txt", "x")
    remote = _memory_remote_zip(buffer.getvalue())

    entry = remote.require_unique_suffix("abc/cams/pair.txt")
    assert entry.name == "root/abc/cams/pair.txt"
