"""Minimal HTTP-range ZIP reader for large public benchmark archives.

The reader is intentionally conservative: it refuses servers that do not honor
Range requests, so a selective materialization command can never accidentally
download a multi-GB archive in full. ZIP32 and ZIP64 central-directory metadata
are supported; file extraction supports stored and deflated entries.
"""

from __future__ import annotations

import binascii
import struct
import zlib
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class RemoteZipEntry:
    name: str
    compression_method: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    local_header_offset: int


class RemoteZip:
    def __init__(self, url: str, *, timeout_seconds: float = 60.0):
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError("RemoteZip requires requests; install refworld-bench[dataset]") from exc

        self.url = str(url)
        self.timeout_seconds = float(timeout_seconds)
        self._requests = requests
        self._session = requests.Session()
        self.size_bytes = self._probe_size()
        self._entries: dict[str, RemoteZipEntry] | None = None

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "RemoteZip":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _probe_size(self) -> int:
        response = self._session.get(
            self.url,
            headers={"Range": "bytes=0-0", "Accept-Encoding": "identity"},
            allow_redirects=True,
            timeout=self.timeout_seconds,
            stream=True,
        )
        try:
            if response.status_code != 206:
                raise RuntimeError(
                    "remote archive server did not honor HTTP Range requests; "
                    f"status={response.status_code}. Refusing full-archive download."
                )
            content_range = response.headers.get("Content-Range", "")
            if "/" not in content_range:
                raise RuntimeError("range response has no total length in Content-Range")
            total_text = content_range.rsplit("/", 1)[1]
            total = int(total_text)
            if total <= 0:
                raise RuntimeError("remote archive reported non-positive size")
            return total
        finally:
            response.close()

    def _range(self, start: int, end_inclusive: int) -> bytes:
        if start < 0 or end_inclusive < start or end_inclusive >= self.size_bytes:
            raise ValueError(f"invalid byte range [{start},{end_inclusive}] for archive size {self.size_bytes}")
        response = self._session.get(
            self.url,
            headers={"Range": f"bytes={start}-{end_inclusive}", "Accept-Encoding": "identity"},
            allow_redirects=True,
            timeout=self.timeout_seconds,
        )
        if response.status_code != 206:
            raise RuntimeError(
                "remote archive stopped honoring HTTP Range requests; "
                f"status={response.status_code}. Refusing full-archive download."
            )
        data = response.content
        expected = end_inclusive - start + 1
        if len(data) != expected:
            raise RuntimeError(f"short range read: expected {expected} bytes, got {len(data)}")
        return data

    @staticmethod
    def _zip64_values(extra: bytes, *, need_uncompressed: bool, need_compressed: bool, need_offset: bool) -> tuple[int | None, int | None, int | None]:
        cursor = 0
        while cursor + 4 <= len(extra):
            header_id, size = struct.unpack_from("<HH", extra, cursor)
            cursor += 4
            payload = extra[cursor : cursor + size]
            cursor += size
            if header_id != 0x0001:
                continue
            pos = 0
            values: list[int | None] = [None, None, None]
            for index, needed in enumerate((need_uncompressed, need_compressed, need_offset)):
                if not needed:
                    continue
                if pos + 8 > len(payload):
                    raise RuntimeError("truncated ZIP64 extended-information field")
                values[index] = struct.unpack_from("<Q", payload, pos)[0]
                pos += 8
            return values[0], values[1], values[2]
        if need_uncompressed or need_compressed or need_offset:
            raise RuntimeError("ZIP64 values required but ZIP64 extra field is missing")
        return None, None, None

    def _central_directory_location(self) -> tuple[int, int, int]:
        tail_size = min(self.size_bytes, 131_072)
        tail_start = self.size_bytes - tail_size
        tail = self._range(tail_start, self.size_bytes - 1)
        eocd_sig = b"PK\x05\x06"
        eocd_rel = tail.rfind(eocd_sig)
        if eocd_rel < 0 or eocd_rel + 22 > len(tail):
            raise RuntimeError("ZIP end-of-central-directory record not found")
        eocd = struct.unpack_from("<4s4H2LH", tail, eocd_rel)
        _, disk_no, cd_disk, disk_entries, total_entries, cd_size32, cd_offset32, _ = eocd
        if disk_no != 0 or cd_disk != 0:
            raise RuntimeError("multi-disk ZIP archives are not supported")

        needs_zip64 = (
            disk_entries == 0xFFFF
            or total_entries == 0xFFFF
            or cd_size32 == 0xFFFFFFFF
            or cd_offset32 == 0xFFFFFFFF
        )
        if not needs_zip64:
            return int(cd_offset32), int(cd_size32), int(total_entries)

        locator_sig = b"PK\x06\x07"
        locator_rel = tail.rfind(locator_sig, 0, eocd_rel)
        if locator_rel < 0 or locator_rel + 20 > len(tail):
            raise RuntimeError("ZIP64 locator not found")
        _, zip64_disk, zip64_offset, total_disks = struct.unpack_from("<4sLQL", tail, locator_rel)
        if zip64_disk != 0 or total_disks != 1:
            raise RuntimeError("multi-disk ZIP64 archives are not supported")
        zip64_head = self._range(int(zip64_offset), int(zip64_offset) + 55)
        fields = struct.unpack_from("<4sQ2H2L4Q", zip64_head, 0)
        sig, _, _, _, disk_num, disk_cd, _, total_entries64, cd_size64, cd_offset64 = fields
        if sig != b"PK\x06\x06" or disk_num != 0 or disk_cd != 0:
            raise RuntimeError("invalid ZIP64 end-of-central-directory record")
        return int(cd_offset64), int(cd_size64), int(total_entries64)

    def entries(self) -> dict[str, RemoteZipEntry]:
        if self._entries is not None:
            return dict(self._entries)
        cd_offset, cd_size, expected_entries = self._central_directory_location()
        if cd_size <= 0:
            raise RuntimeError("ZIP central directory is empty")
        directory = self._range(cd_offset, cd_offset + cd_size - 1)
        cursor = 0
        parsed: dict[str, RemoteZipEntry] = {}
        while cursor < len(directory):
            if cursor + 46 > len(directory):
                raise RuntimeError("truncated ZIP central-directory header")
            fields = struct.unpack_from("<4s6H3L5H2L", directory, cursor)
            (
                sig,
                _,
                _,
                flags,
                method,
                _,
                _,
                crc32,
                compressed32,
                uncompressed32,
                name_len,
                extra_len,
                comment_len,
                _,
                _,
                _,
                offset32,
            ) = fields
            if sig != b"PK\x01\x02":
                raise RuntimeError(f"invalid ZIP central-directory signature at byte {cursor}")
            start = cursor + 46
            name_bytes = directory[start : start + name_len]
            extra = directory[start + name_len : start + name_len + extra_len]
            encoding = "utf-8" if flags & 0x800 else "cp437"
            name = name_bytes.decode(encoding).replace("\\", "/")
            need_u = uncompressed32 == 0xFFFFFFFF
            need_c = compressed32 == 0xFFFFFFFF
            need_o = offset32 == 0xFFFFFFFF
            zip_u, zip_c, zip_o = self._zip64_values(
                extra,
                need_uncompressed=need_u,
                need_compressed=need_c,
                need_offset=need_o,
            )
            uncompressed = int(zip_u if need_u else uncompressed32)
            compressed = int(zip_c if need_c else compressed32)
            local_offset = int(zip_o if need_o else offset32)
            if name and not name.endswith("/"):
                parsed[name] = RemoteZipEntry(
                    name=name,
                    compression_method=int(method),
                    crc32=int(crc32),
                    compressed_size=compressed,
                    uncompressed_size=uncompressed,
                    local_header_offset=local_offset,
                )
            cursor = start + name_len + extra_len + comment_len
        if len(parsed) > expected_entries:
            raise RuntimeError("parsed more ZIP file entries than declared")
        self._entries = parsed
        return dict(parsed)

    def names(self) -> tuple[str, ...]:
        return tuple(self.entries().keys())

    def read(self, entry: RemoteZipEntry | str) -> bytes:
        if isinstance(entry, str):
            try:
                record = self.entries()[entry]
            except KeyError as exc:
                raise KeyError(f"ZIP entry not found: {entry}") from exc
        else:
            record = entry
        local = self._range(record.local_header_offset, record.local_header_offset + 29)
        sig, _, _, _, method, _, _, _, _, _, name_len, extra_len = struct.unpack_from("<4s5H3L2H", local, 0)
        if sig != b"PK\x03\x04":
            raise RuntimeError(f"invalid local ZIP header for {record.name}")
        if int(method) != record.compression_method:
            raise RuntimeError(f"compression-method mismatch for {record.name}")
        data_start = record.local_header_offset + 30 + int(name_len) + int(extra_len)
        if record.compressed_size == 0:
            compressed = b""
        else:
            compressed = self._range(data_start, data_start + record.compressed_size - 1)
        if record.compression_method == 0:
            data = compressed
        elif record.compression_method == 8:
            data = zlib.decompress(compressed, -15)
        else:
            raise RuntimeError(
                f"unsupported ZIP compression method {record.compression_method} for {record.name}"
            )
        if len(data) != record.uncompressed_size:
            raise RuntimeError(
                f"uncompressed-size mismatch for {record.name}: {len(data)} != {record.uncompressed_size}"
            )
        crc = binascii.crc32(data) & 0xFFFFFFFF
        if crc != record.crc32:
            raise RuntimeError(f"CRC32 mismatch for {record.name}")
        return data

    def find_suffix(self, suffix: str) -> tuple[RemoteZipEntry, ...]:
        normalized = suffix.replace("\\", "/").lstrip("/")
        matches = [entry for name, entry in self.entries().items() if name.endswith(normalized)]
        return tuple(matches)

    def require_unique_suffix(self, suffix: str) -> RemoteZipEntry:
        matches = self.find_suffix(suffix)
        if len(matches) != 1:
            raise RuntimeError(f"expected one ZIP entry ending with {suffix!r}, found {len(matches)}")
        return matches[0]
