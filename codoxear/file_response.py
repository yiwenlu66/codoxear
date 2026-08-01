from __future__ import annotations

import http.server
import os
import sys
import urllib.parse
from pathlib import Path
from typing import Any, BinaryIO

from .file_text import open_regular_file_no_symlink
from .git_ops import path_json_text as _path_json_text


def _safe_name(name: str) -> str:
    # A raw-byte filename surfaces as lone surrogates that urllib.parse.quote
    # cannot UTF-8 encode; serialize through the surrogate-safe display codec.
    return _path_json_text(name)


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


def _send_file_open_error(handler: http.server.BaseHTTPRequestHandler, exc: BaseException) -> None:
    if isinstance(exc, FileNotFoundError):
        handler.send_error(404, str(exc))
        return
    if isinstance(exc, PermissionError):
        handler.send_error(403, str(exc))
        return
    if isinstance(exc, ValueError):
        handler.send_error(400, str(exc))
        return
    raise exc


class FileResponseStream:
    def __init__(self, context: Any, file_obj: BinaryIO) -> None:
        self.context = context
        self.file_obj = file_obj

    def __enter__(self) -> BinaryIO:
        return self.file_obj

    def __exit__(self, exc_type: object, exc: object, tb: object) -> object:
        return self.context.__exit__(exc_type, exc, tb)


def _open_file_for_response(handler: http.server.BaseHTTPRequestHandler, path: Path) -> FileResponseStream | None:
    try:
        opened = open_regular_file_no_symlink(path)
        file_obj, _stat_result = opened.__enter__()
    except (FileNotFoundError, PermissionError, ValueError) as e:
        _send_file_open_error(handler, e)
        return None
    return FileResponseStream(opened, file_obj)


def _open_file_size(f: BinaryIO) -> int:
    return int(os.fstat(f.fileno()).st_size)


def _log_late_stream_error(handler: http.server.BaseHTTPRequestHandler, exc: OSError) -> None:
    message = f"file response stream failed after headers: {type(exc).__name__}: {exc}"
    log_error = getattr(handler, "log_error", None)
    if callable(log_error):
        try:
            log_error("%s", message)
        except Exception:
            pass
    try:
        sys.stderr.write(f"error: {message}\n")
        sys.stderr.flush()
    except Exception:
        pass


def _stream_open_file_bytes(handler: http.server.BaseHTTPRequestHandler, f: BinaryIO, *, start: int = 0, length: int | None = None) -> None:
    try:
        if start:
            f.seek(start)
    except OSError as exc:
        _log_late_stream_error(handler, exc)
        return
    remaining = length
    while remaining is None or remaining > 0:
        max_read = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
        try:
            chunk = f.read(max_read)
        except OSError as exc:
            _log_late_stream_error(handler, exc)
            break
        if not chunk:
            break
        try:
            handler.wfile.write(chunk)
        except OSError as exc:
            _log_late_stream_error(handler, exc)
            break
        if remaining is not None:
            remaining -= len(chunk)


def _stream_file_bytes(handler: http.server.BaseHTTPRequestHandler, path: Path, *, start: int = 0, length: int | None = None) -> None:
    stream = _open_file_for_response(handler, path)
    if stream is None:
        return
    with stream as f:
        _stream_open_file_bytes(handler, f, start=start, length=length)


def send_inline_file_response(handler: http.server.BaseHTTPRequestHandler, path: Path, content_type: str) -> None:
    stream = _open_file_for_response(handler, path)
    if stream is None:
        return
    with stream as opened_stream:
        size = _open_file_size(opened_stream)
        try:
            byte_range = single_byte_range(handler.headers.get("Range"), size)
        except ValueError:
            handler.send_response(416)
            handler.send_header("Content-Range", f"bytes */{size}")
            handler.send_header("Accept-Ranges", "bytes")
            handler.send_header("Content-Length", "0")
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
        handler.send_header("Content-Disposition", f"inline; filename*=UTF-8''{urllib.parse.quote(_safe_name(path.name), safe='')}")
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("Pragma", "no-cache")
        handler.send_header("Expires", "0")
        handler.end_headers()
        _stream_open_file_bytes(handler, opened_stream, start=start, length=length)


def send_attachment_file_response(
    handler: http.server.BaseHTTPRequestHandler,
    path: Path,
    *,
    size: int,
    content_disposition: str,
) -> None:
    stream = _open_file_for_response(handler, path)
    if stream is None:
        return
    with stream as opened_stream:
        actual_size = _open_file_size(opened_stream)
        length = min(max(0, int(size)), actual_size)
        handler.send_response(200)
        handler.send_header("Content-Type", "application/octet-stream")
        handler.send_header("Content-Length", str(length))
        handler.send_header("Content-Disposition", content_disposition)
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("Pragma", "no-cache")
        handler.send_header("Expires", "0")
        handler.end_headers()
        _stream_open_file_bytes(handler, opened_stream, length=length)
