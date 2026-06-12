from __future__ import annotations

import urllib.parse
from dataclasses import dataclass
from pathlib import Path

from .file_text import FILE_READ_MAX_BYTES
from .file_text import decode_text_view_for_client
from .file_text import markdown_kind
from .file_types import file_kind


@dataclass(frozen=True)
class ClientFileView:
    kind: str
    size: int
    content_type: str | None = None
    text: str | None = None
    editable: bool = False
    version: str | None = None
    blocked_reason: str | None = None
    viewer_max_bytes: int | None = None


def inspect_openable_file(path_obj: Path) -> tuple[bytes, int, str, str | None]:
    view = read_client_file_view(path_obj)
    if view.kind == "directory":
        raise ValueError("path is not a file")
    if view.kind == "download_only":
        if view.blocked_reason == "too_large":
            raise ValueError(f"file too large (max {FILE_READ_MAX_BYTES} bytes)")
        raise ValueError("binary file not supported")
    raw = path_obj.read_bytes()
    return raw, view.size, view.kind, view.content_type


def inspect_path_metadata(path_obj: Path) -> tuple[int, str, str | None]:
    view = read_client_file_view(path_obj)
    return view.size, view.kind, view.content_type


def read_client_file_view(path_obj: Path) -> ClientFileView:
    if not path_obj.exists():
        raise FileNotFoundError("file not found")
    if path_obj.is_dir():
        return ClientFileView(kind="directory", size=0)
    if not path_obj.is_file():
        raise ValueError("path is not a file")
    try:
        size = int(path_obj.stat().st_size)
        with path_obj.open("rb") as f:
            prefix = f.read(4096)
    except PermissionError as e:
        raise PermissionError("permission denied") from e
    kind, content_type = file_kind(path_obj, prefix)
    if kind in {"image", "pdf", "video"}:
        return ClientFileView(kind=kind, size=size, content_type=content_type)
    if size > FILE_READ_MAX_BYTES:
        return ClientFileView(
            kind="download_only",
            size=size,
            blocked_reason="too_large",
            viewer_max_bytes=FILE_READ_MAX_BYTES,
        )
    raw = path_obj.read_bytes()
    text_payload = decode_text_view_for_client(path_obj, raw)
    if text_payload is None:
        return ClientFileView(kind="download_only", size=size, blocked_reason="binary")
    text, editable, version = text_payload
    return ClientFileView(
        kind=markdown_kind(path_obj),
        size=size,
        text=text,
        editable=editable,
        version=version,
    )


def read_text_or_image(path_obj: Path) -> tuple[str, int, str | None, bytes | None]:
    view = read_client_file_view(path_obj)
    if view.kind in {"image", "pdf", "video", "download_only", "directory"}:
        return view.kind, view.size, view.content_type, None
    raw = path_obj.read_bytes()
    return view.kind, view.size, view.content_type, raw


def inspect_downloadable_file(path_obj: Path) -> int:
    if not path_obj.exists():
        raise FileNotFoundError("file not found")
    if not path_obj.is_file():
        raise ValueError("path is not a file")
    try:
        size = int(path_obj.stat().st_size)
        with path_obj.open("rb"):
            pass
    except PermissionError as e:
        raise PermissionError("permission denied") from e
    return size


def inspect_client_path(path_obj: Path) -> tuple[int, str, str | None]:
    view = read_client_file_view(path_obj)
    return view.size, view.kind, view.content_type


def download_disposition(path_obj: Path) -> str:
    return f"attachment; filename*=UTF-8''{urllib.parse.quote(path_obj.name, safe='')}"
