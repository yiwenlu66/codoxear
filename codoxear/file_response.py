from __future__ import annotations

import http.server
import urllib.parse
from pathlib import Path


def single_byte_range(header: str | None, size: int) -> tuple[int, int] | None:
    if not header:
        return None
    raw = header.strip()
    if not raw.startswith("bytes="):
        raise ValueError("unsupported range unit")
    spec = raw[len("bytes=") :].strip()
    if "," in spec or "-" not in spec:
        raise ValueError("unsupported byte range")
    start_raw, end_raw = spec.split("-", 1)
    if start_raw == "":
        if not end_raw.isdigit():
            raise ValueError("invalid byte range")
        suffix = int(end_raw)
        if suffix <= 0 or size <= 0:
            raise ValueError("unsatisfiable byte range")
        return max(0, size - suffix), size - 1
    if not start_raw.isdigit() or (end_raw and not end_raw.isdigit()):
        raise ValueError("invalid byte range")
    start = int(start_raw)
    end = int(end_raw) if end_raw else size - 1
    if start >= size or start > end:
        raise ValueError("unsatisfiable byte range")
    return start, min(end, size - 1)


def _stream_file_bytes(handler: http.server.BaseHTTPRequestHandler, path: Path, *, start: int = 0, length: int | None = None) -> None:
    with path.open("rb") as f:
        if start:
            f.seek(start)
        remaining = length
        while remaining is None or remaining > 0:
            max_read = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
            chunk = f.read(max_read)
            if not chunk:
                break
            handler.wfile.write(chunk)
            if remaining is not None:
                remaining -= len(chunk)


def send_inline_file_response(handler: http.server.BaseHTTPRequestHandler, path: Path, content_type: str) -> None:
    size = int(path.stat().st_size)
    try:
        byte_range = single_byte_range(handler.headers.get("Range"), size)
    except ValueError:
        handler.send_response(416)
        handler.send_header("Content-Range", f"bytes */{size}")
        handler.send_header("Accept-Ranges", "bytes")
        handler.end_headers()
        return
    start = 0 if byte_range is None else byte_range[0]
    end = size - 1 if byte_range is None else byte_range[1]
    length = max(0, end - start + 1)
    handler.send_response(206 if byte_range is not None else 200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(length))
    handler.send_header("Accept-Ranges", "bytes")
    if byte_range is not None:
        handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
    handler.send_header("Content-Disposition", f"inline; filename*=UTF-8''{urllib.parse.quote(path.name, safe='')}")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Pragma", "no-cache")
    handler.send_header("Expires", "0")
    handler.end_headers()
    _stream_file_bytes(handler, path, start=start, length=length)


def send_attachment_file_response(
    handler: http.server.BaseHTTPRequestHandler,
    path: Path,
    *,
    size: int,
    content_disposition: str,
) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", "application/octet-stream")
    handler.send_header("Content-Length", str(size))
    handler.send_header("Content-Disposition", content_disposition)
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Pragma", "no-cache")
    handler.send_header("Expires", "0")
    handler.end_headers()
    _stream_file_bytes(handler, path)
