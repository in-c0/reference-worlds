"""HTTP-range ZIP reader for large public benchmark archives.

The reader is deliberately conservative: every remote read must be satisfied by
HTTP 206 Range responses, so selective benchmark materialization can never
silently turn into a full multi-GB download.

Both ordinary ZIPs and classic split ZIPs (``.z01`` ... ``.zip``) are supported.
For split archives, central-directory offsets and file offsets are interpreted
relative to their declared disk, and compressed payloads may span disk
boundaries. ZIP32 is fully supported; ZIP64 metadata is supported where the
standard records provide explicit disk/offset information.
"""

from __future__ import annotations

import binascii
import struct
import zlib
from dataclasses import dataclass


@dataclass(frozen=True)
class RemoteZipVolume:
    disk_index: int
    url: str
    size_bytes: int


@dataclass(frozen=True)
class RemoteZipEntry:
    name: str
    compression_method: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    disk_start: int
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
        self._entries: dict[str, RemoteZipEntry] | None = None

        final_size = self._probe_url_size(self.url)
        disk_no = self._final_disk_number(self.url, final_size)
        self._volumes = self._discover_volumes(disk_no, final_size)
        self.size_bytes = sum(volume.size_bytes for volume in self._volumes)

    def close(self) -> None:
        session = getattr(self, "_session", None)
        if session is not None:
            session.close()

    def __enter__(self) -> "RemoteZip":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _probe_url_size(self, url: str) -> int:
        response = self._session.get(
            url,
            headers={"Range": "bytes=0-0", "Accept-Encoding": "identity"},
            allow_redirects=True,
            timeout=self.timeout_seconds,
            stream=True,
        )
        try:
            if response.status_code != 206:
                raise RuntimeError(
                    "remote archive server did not honor HTTP Range requests; "
                    f"status={response.status_code} for {url}. Refusing full-archive download."
                )
            content_range = response.headers.get("Content-Range", "")
            if "/" not in content_range:
                raise RuntimeError(f"range response has no total length for {url}")
            total = int(content_range.rsplit("/", 1)[1])
            if total <= 0:
                raise RuntimeError(f"remote archive volume reported non-positive size: {url}")
            return total
        finally:
            response.close()

    def _read_url_range(self, url: str, start: int, end_inclusive: int, size_bytes: int) -> bytes:
        if start < 0 or end_inclusive < start or end_inclusive >= size_bytes:
            raise ValueError(f"invalid byte range [{start},{end_inclusive}] for {url} size {size_bytes}")
        response = self._session.get(
            url,
            headers={"Range": f"bytes={start}-{end_inclusive}", "Accept-Encoding": "identity"},
            allow_redirects=True,
            timeout=self.timeout_seconds,
        )
        try:
            if response.status_code != 206:
                raise RuntimeError(
                    "remote archive stopped honoring HTTP Range requests; "
                    f"status={response.status_code} for {url}. Refusing full-archive download."
                )
            data = response.content
        finally:
            response.close()
        expected = end_inclusive - start + 1
        if len(data) != expected:
            raise RuntimeError(f"short range read from {url}: expected {expected} bytes, got {len(data)}")
        return data

    @staticmethod
    def _find_eocd(tail: bytes) -> tuple[int, tuple]:
        rel = tail.rfind(b"PK\x05\x06")
        if rel < 0 or rel + 22 > len(tail):
            raise RuntimeError("ZIP end-of-central-directory record not found")
        fields = struct.unpack_from("<4s4H2LH", tail, rel)
        comment_len = int(fields[-1])
        if rel + 22 + comment_len > len(tail):
            raise RuntimeError("truncated ZIP end-of-central-directory comment")
        return rel, fields

    def _final_disk_number(self, final_url: str, final_size: int) -> int:
        tail_size = min(final_size, 131_072)
        tail = self._read_url_range(final_url, final_size - tail_size, final_size - 1, final_size)
        eocd_rel, eocd = self._find_eocd(tail)
        _, disk_no, _, _, _, _, _, _ = eocd
        if disk_no != 0xFFFF:
            return int(disk_no)

        locator_rel = tail.rfind(b"PK\x06\x07", 0, eocd_rel)
        if locator_rel < 0 or locator_rel + 20 > len(tail):
            raise RuntimeError("ZIP64 locator required to determine split archive disk count")
        _, _, _, total_disks = struct.unpack_from("<4sLQL", tail, locator_rel)
        if total_disks < 1:
            raise RuntimeError("ZIP64 locator reported no archive disks")
        return int(total_disks - 1)

    def _split_volume_url(self, disk_index: int) -> str:
        if not self.url.lower().endswith(".zip"):
            raise RuntimeError(
                "multi-disk archive detected but final URL does not end in .zip; "
                "cannot derive .zNN volume URLs safely"
            )
        suffix_number = disk_index + 1
        if suffix_number > 99:
            raise RuntimeError("split ZIP with more than 99 .zNN volumes is not supported")
        return self.url[:-4] + f".z{suffix_number:02d}"

    def _discover_volumes(self, final_disk_index: int, final_size: int) -> tuple[RemoteZipVolume, ...]:
        if final_disk_index < 0:
            raise RuntimeError("invalid final ZIP disk number")
        volumes: list[RemoteZipVolume] = []
        for disk in range(final_disk_index):
            volume_url = self._split_volume_url(disk)
            volume_size = self._probe_url_size(volume_url)
            volumes.append(RemoteZipVolume(disk, volume_url, volume_size))
        volumes.append(RemoteZipVolume(final_disk_index, self.url, final_size))
        return tuple(volumes)

    def _read_volume(self, disk_index: int, start: int, end_inclusive: int) -> bytes:
        if disk_index < 0 or disk_index >= len(self._volumes):
            raise ValueError(f"ZIP disk index out of range: {disk_index}")
        volume = self._volumes[disk_index]
        return self._read_url_range(volume.url, start, end_inclusive, volume.size_bytes)

    def _read_spanning(self, disk_index: int, offset: int, length: int) -> bytes:
        if disk_index < 0 or disk_index >= len(self._volumes):
            raise ValueError(f"ZIP disk index out of range: {disk_index}")
        if offset < 0 or length < 0:
            raise ValueError("ZIP offsets/lengths must be non-negative")
        if length == 0:
            return b""

        disk = int(disk_index)
        local_offset = int(offset)
        while disk < len(self._volumes) and local_offset >= self._volumes[disk].size_bytes:
            local_offset -= self._volumes[disk].size_bytes
            disk += 1
        if disk >= len(self._volumes):
            raise RuntimeError("ZIP logical offset escaped available split volumes")

        remaining = int(length)
        chunks: list[bytes] = []
        while remaining:
            if disk >= len(self._volumes):
                raise RuntimeError("ZIP payload extends past final split volume")
            volume = self._volumes[disk]
            available = volume.size_bytes - local_offset
            if available <= 0:
                disk += 1
                local_offset = 0
                continue
            take = min(remaining, available)
            chunks.append(self._read_volume(disk, local_offset, local_offset + take - 1))
            remaining -= take
            disk += 1
            local_offset = 0
        return b"".join(chunks)

    @staticmethod
    def _zip64_values(
        extra: bytes,
        *,
        need_uncompressed: bool,
        need_compressed: bool,
        need_offset: bool,
        need_disk: bool,
    ) -> tuple[int | None, int | None, int | None, int | None]:
        cursor = 0
        while cursor + 4 <= len(extra):
            header_id, size = struct.unpack_from("<HH", extra, cursor)
            cursor += 4
            payload = extra[cursor : cursor + size]
            cursor += size
            if header_id != 0x0001:
                continue
            pos = 0
            values: list[int | None] = [None, None, None, None]
            for index, (needed, width) in enumerate(
                (
                    (need_uncompressed, 8),
                    (need_compressed, 8),
                    (need_offset, 8),
                    (need_disk, 4),
                )
            ):
                if not needed:
                    continue
                if pos + width > len(payload):
                    raise RuntimeError("truncated ZIP64 extended-information field")
                fmt = "<Q" if width == 8 else "<L"
                values[index] = int(struct.unpack_from(fmt, payload, pos)[0])
                pos += width
            return values[0], values[1], values[2], values[3]
        if need_uncompressed or need_compressed or need_offset or need_disk:
            raise RuntimeError("ZIP64 values required but ZIP64 extra field is missing")
        return None, None, None, None

    def _central_directory_location(self) -> tuple[int, int, int, int]:
        final_disk = len(self._volumes) - 1
        final_size = self._volumes[final_disk].size_bytes
        tail_size = min(final_size, 131_072)
        tail = self._read_volume(final_disk, final_size - tail_size, final_size - 1)
        eocd_rel, eocd = self._find_eocd(tail)
        _, disk_no, cd_disk, disk_entries, total_entries, cd_size32, cd_offset32, _ = eocd

        if disk_no != 0xFFFF and int(disk_no) != final_disk:
            raise RuntimeError(
                f"split ZIP final disk mismatch: EOCD says {disk_no}, discovered {final_disk}"
            )

        needs_zip64 = (
            disk_no == 0xFFFF
            or cd_disk == 0xFFFF
            or disk_entries == 0xFFFF
            or total_entries == 0xFFFF
            or cd_size32 == 0xFFFFFFFF
            or cd_offset32 == 0xFFFFFFFF
        )
        if not needs_zip64:
            return int(cd_disk), int(cd_offset32), int(cd_size32), int(total_entries)

        locator_rel = tail.rfind(b"PK\x06\x07", 0, eocd_rel)
        if locator_rel < 0 or locator_rel + 20 > len(tail):
            raise RuntimeError("ZIP64 locator not found")
        _, zip64_disk, zip64_offset, total_disks = struct.unpack_from("<4sLQL", tail, locator_rel)
        if int(total_disks) != len(self._volumes):
            raise RuntimeError(
                f"ZIP64 volume count mismatch: locator={total_disks}, discovered={len(self._volumes)}"
            )
        zip64_head = self._read_spanning(int(zip64_disk), int(zip64_offset), 56)
        fields = struct.unpack_from("<4sQ2H2L4Q", zip64_head, 0)
        sig, _, _, _, disk_num, disk_cd, _, total_entries64, cd_size64, cd_offset64 = fields
        if sig != b"PK\x06\x06":
            raise RuntimeError("invalid ZIP64 end-of-central-directory record")
        if int(disk_num) != final_disk:
            raise RuntimeError("ZIP64 EOCD final-disk number does not match discovered volumes")
        return int(disk_cd), int(cd_offset64), int(cd_size64), int(total_entries64)

    def entries(self) -> dict[str, RemoteZipEntry]:
        if self._entries is not None:
            return dict(self._entries)

        cd_disk, cd_offset, cd_size, expected_entries = self._central_directory_location()
        if cd_size <= 0:
            raise RuntimeError("ZIP central directory is empty")
        directory = self._read_spanning(cd_disk, cd_offset, cd_size)

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
                disk_start32,
                _,
                _,
                offset32,
            ) = fields
            if sig != b"PK\x01\x02":
                raise RuntimeError(f"invalid ZIP central-directory signature at byte {cursor}")

            start = cursor + 46
            end = start + name_len + extra_len + comment_len
            if end > len(directory):
                raise RuntimeError("truncated ZIP central-directory variable fields")
            name_bytes = directory[start : start + name_len]
            extra = directory[start + name_len : start + name_len + extra_len]
            encoding = "utf-8" if flags & 0x800 else "cp437"
            name = name_bytes.decode(encoding).replace("\\", "/")

            need_u = uncompressed32 == 0xFFFFFFFF
            need_c = compressed32 == 0xFFFFFFFF
            need_o = offset32 == 0xFFFFFFFF
            need_d = disk_start32 == 0xFFFF
            zip_u, zip_c, zip_o, zip_d = self._zip64_values(
                extra,
                need_uncompressed=need_u,
                need_compressed=need_c,
                need_offset=need_o,
                need_disk=need_d,
            )
            uncompressed = int(zip_u if need_u else uncompressed32)
            compressed = int(zip_c if need_c else compressed32)
            local_offset = int(zip_o if need_o else offset32)
            disk_start = int(zip_d if need_d else disk_start32)
            if disk_start < 0 or disk_start >= len(self._volumes):
                raise RuntimeError(f"entry {name!r} starts on unavailable ZIP disk {disk_start}")

            if name and not name.endswith("/"):
                parsed[name] = RemoteZipEntry(
                    name=name,
                    compression_method=int(method),
                    crc32=int(crc32),
                    compressed_size=compressed,
                    uncompressed_size=uncompressed,
                    disk_start=disk_start,
                    local_header_offset=local_offset,
                )
            cursor = end

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

        local = self._read_spanning(record.disk_start, record.local_header_offset, 30)
        sig, _, _, method, _, _, _, _, _, name_len, extra_len = struct.unpack_from(
            "<4s5H3L2H", local, 0
        )
        if sig != b"PK\x03\x04":
            raise RuntimeError(f"invalid local ZIP header for {record.name}")
        if int(method) != record.compression_method:
            raise RuntimeError(f"compression-method mismatch for {record.name}")

        data_offset = record.local_header_offset + 30 + int(name_len) + int(extra_len)
        compressed = self._read_spanning(record.disk_start, data_offset, record.compressed_size)
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
