from __future__ import annotations

from pathlib import Path


_JPEG_SOF_MARKERS = frozenset({0xC0, 0xC1, 0xC2})
_JPEG_STANDALONE_MARKERS = frozenset({0x01, *range(0xD0, 0xD8), 0xD8, 0xD9})


def _positive_dimensions(width: int, height: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    return width, height


def _read_exact(stream: object, size: int) -> bytes:
    data = stream.read(size)  # type: ignore[attr-defined]
    if len(data) != size:
        raise ValueError("truncated image header")
    return data


def _jpeg_dimensions(stream: object) -> tuple[int, int]:
    if _read_exact(stream, 2) != b"\xff\xd8":
        raise ValueError("file is not a supported image")
    while True:
        marker_prefix = _read_exact(stream, 1)
        while marker_prefix != b"\xff":
            marker_prefix = _read_exact(stream, 1)
        marker = _read_exact(stream, 1)[0]
        while marker == 0xFF:
            marker = _read_exact(stream, 1)[0]
        if marker == 0x00 or marker == 0xDA:
            break
        if marker in _JPEG_STANDALONE_MARKERS:
            continue
        segment_length = int.from_bytes(_read_exact(stream, 2), "big")
        if segment_length < 2:
            raise ValueError("invalid JPEG segment")
        payload_size = segment_length - 2
        if marker in _JPEG_SOF_MARKERS:
            if payload_size < 6:
                raise ValueError("invalid JPEG frame header")
            frame = _read_exact(stream, 5)
            height = int.from_bytes(frame[1:3], "big")
            width = int.from_bytes(frame[3:5], "big")
            return _positive_dimensions(width, height)
        _read_exact(stream, payload_size)
    raise ValueError("JPEG has no supported frame header")


def _webp_dimensions(header: bytes) -> tuple[int, int]:
    if len(header) < 16 or header[:4] != b"RIFF" or header[8:12] != b"WEBP":
        raise ValueError("file is not a supported image")
    chunk = header[12:16]
    if chunk == b"VP8 ":
        if len(header) < 30 or header[23:26] != b"\x9d\x01\x2a":
            raise ValueError("invalid VP8 image header")
        width = int.from_bytes(header[26:28], "little") & 0x3FFF
        height = int.from_bytes(header[28:30], "little") & 0x3FFF
        return _positive_dimensions(width, height)
    if chunk == b"VP8L":
        if len(header) < 25 or header[20] != 0x2F:
            raise ValueError("invalid VP8L image header")
        bits = int.from_bytes(header[21:25], "little")
        width = 1 + (bits & 0x3FFF)
        height = 1 + ((bits >> 14) & 0x3FFF)
        return _positive_dimensions(width, height)
    if chunk == b"VP8X":
        if len(header) < 30:
            raise ValueError("truncated VP8X image header")
        width = 1 + int.from_bytes(header[24:27], "little")
        height = 1 + int.from_bytes(header[27:30], "little")
        return _positive_dimensions(width, height)
    raise ValueError("unsupported WebP image header")


def read_image_dimensions(path: Path) -> tuple[int, int]:
    """Return native pixel dimensions for supported raster image files.

    Header-only formats are decoded from a short prefix. JPEG requires a
    segment walk because its SOF marker can follow metadata segments.
    """
    with path.open("rb") as stream:
        header = stream.read(2)
        if len(header) < 2:
            raise ValueError("file is not a supported image")
        stream.seek(0)
        if header == b"\xff\xd8":
            return _jpeg_dimensions(stream)
        prefix = stream.read(30)
    if prefix.startswith(b"\x89PNG\r\n\x1a\n"):
        if len(prefix) < 24 or prefix[12:16] != b"IHDR":
            raise ValueError("invalid PNG header")
        return _positive_dimensions(int.from_bytes(prefix[16:20], "big"), int.from_bytes(prefix[20:24], "big"))
    if prefix.startswith((b"GIF87a", b"GIF89a")):
        if len(prefix) < 10:
            raise ValueError("truncated GIF header")
        return _positive_dimensions(int.from_bytes(prefix[6:8], "little"), int.from_bytes(prefix[8:10], "little"))
    return _webp_dimensions(prefix)
