from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping
import urllib.parse

from .file_text import FILE_READ_MAX_BYTES
from .file_text import read_text_file_for_write
from .file_text import write_new_text_file_atomic
from .file_text import write_text_file_atomic
from .file_view import ClientFileView
from .video_preview import video_response_payload


@dataclass(frozen=True)
class FileRouteResponse:
    status: int
    payload: dict[str, Any]


class FileRouteError(Exception):
    def __init__(self, status: int, payload: dict[str, Any]) -> None:
        super().__init__(str(payload.get("error", "file route error")))
        self.status = int(status)
        self.payload = payload


@dataclass(frozen=True)
class SessionFileWriteRequest:
    path: str
    text: str
    create: bool
    git_path: bool
    version: str | None


FileWriteLock = Callable[[Path], AbstractContextManager[None]]
GitFileResolver = Callable[[str], tuple[Path, str]]
SessionPathResolver = Callable[[Path, str], Path]
FileRecorder = Callable[[str], None]


def body_flag(obj: Mapping[str, Any], name: str) -> bool:
    raw = obj.get(name)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return False


def resolve_session_write_update_path(base: Path, raw_path: str) -> Path:
    if not isinstance(raw_path, str) or raw_path == "":
        raise ValueError("path required")
    if "\x00" in raw_path:
        raise ValueError("invalid path")
    p = Path(raw_path)
    if p.is_absolute():
        return _expanduser_path(p).resolve()
    resolved_base = _expanduser_path(base)
    if not resolved_base.is_absolute():
        resolved_base = resolved_base.resolve()
    resolved_base = resolved_base.resolve()
    resolved = (resolved_base / p).resolve()
    try:
        resolved.relative_to(resolved_base)
    except ValueError as e:
        raise ValueError("path escapes session cwd") from e
    return resolved


def session_file_read_payload(
    *,
    session_id: str,
    path_obj: Path,
    rel: str,
    view: ClientFileView,
    git_path: bool,
) -> dict[str, Any]:
    git_suffix = "&git_path=1" if git_path else ""
    rel_for_url = urllib.parse.quote(rel)
    if view.kind == "image":
        return {
            "ok": True,
            "kind": "image",
            "content_type": view.content_type,
            "path": str(path_obj),
            "rel": str(rel),
            "size": int(view.size),
            "image_url": f"/api/sessions/{session_id}/file/blob?path={rel_for_url}{git_suffix}",
        }
    if view.kind == "pdf":
        return {
            "ok": True,
            "kind": "pdf",
            "content_type": view.content_type,
            "path": str(path_obj),
            "rel": str(rel),
            "size": int(view.size),
            "pdf_url": f"/api/sessions/{session_id}/file/blob?path={rel_for_url}{git_suffix}",
        }
    if view.kind == "video":
        return video_response_payload(
            path_obj=path_obj,
            rel=str(rel),
            size=int(view.size),
            content_type=view.content_type,
            video_url=f"/api/sessions/{session_id}/file/blob?path={rel_for_url}{git_suffix}",
            preview_url=f"/api/sessions/{session_id}/file/video_preview?path={rel_for_url}{git_suffix}",
        )
    if view.kind == "download_only":
        return {
            "ok": True,
            "kind": "download_only",
            "path": str(path_obj),
            "rel": str(rel),
            "size": int(view.size),
            "reason": view.blocked_reason,
            "viewer_max_bytes": view.viewer_max_bytes,
        }
    return {
        "ok": True,
        "kind": view.kind,
        "path": str(path_obj),
        "rel": str(rel),
        "size": int(view.size),
        "text": view.text,
        "editable": bool(view.editable),
        "version": view.version,
    }


def parse_session_file_write_request(obj: Mapping[str, Any]) -> SessionFileWriteRequest:
    path_raw = obj.get("path")
    if not isinstance(path_raw, str) or path_raw == "":
        raise FileRouteError(400, {"error": "path required"})
    text_raw = obj.get("text")
    if not isinstance(text_raw, str):
        raise FileRouteError(400, {"error": "text must be a string"})
    create_raw = obj.get("create")
    create = create_raw if isinstance(create_raw, bool) else False
    git_path = body_flag(obj, "git_path")
    if create and git_path:
        raise FileRouteError(400, {"error": "git_path is only supported for existing files"})
    version_raw = obj.get("version")
    version = version_raw if isinstance(version_raw, str) else None
    if not create and (version is None or not version.strip()):
        raise FileRouteError(400, {"error": "version required"})
    return SessionFileWriteRequest(
        path=path_raw,
        text=text_raw,
        create=create,
        git_path=git_path,
        version=version,
    )


def session_file_write_response(
    *,
    session_base: Path,
    resolve_create_path: SessionPathResolver,
    resolve_git_existing_regular_file: GitFileResolver,
    file_write_lock: FileWriteLock,
    body: Mapping[str, Any] | None = None,
    request: SessionFileWriteRequest | None = None,
    record_file: FileRecorder | None = None,
) -> FileRouteResponse:
    try:
        if request is None:
            if body is None:
                raise TypeError("body or request required")
            request = parse_session_file_write_request(body)
        payload = write_session_file(
            request=request,
            session_base=session_base,
            resolve_create_path=resolve_create_path,
            resolve_git_existing_regular_file=resolve_git_existing_regular_file,
            file_write_lock=file_write_lock,
            record_file=record_file,
        )
    except FileRouteError as e:
        return FileRouteResponse(e.status, e.payload)
    return FileRouteResponse(200, payload)


def write_session_file(
    *,
    request: SessionFileWriteRequest,
    session_base: Path,
    resolve_create_path: SessionPathResolver,
    resolve_git_existing_regular_file: GitFileResolver,
    file_write_lock: FileWriteLock,
    record_file: FileRecorder | None = None,
) -> dict[str, Any]:
    if request.create:
        path_obj, size, next_version = _create_session_file(
            request=request,
            session_base=session_base,
            resolve_create_path=resolve_create_path,
        )
    else:
        path_obj, size, next_version = _update_session_file(
            request=request,
            session_base=session_base,
            resolve_git_existing_regular_file=resolve_git_existing_regular_file,
            file_write_lock=file_write_lock,
        )
    if record_file is not None:
        try:
            record_file(str(path_obj))
        except KeyError:
            pass
    return {
        "ok": True,
        "path": str(path_obj),
        "rel": str(request.path),
        "size": int(size),
        "version": next_version,
        "editable": True,
    }


def _create_session_file(
    *,
    request: SessionFileWriteRequest,
    session_base: Path,
    resolve_create_path: SessionPathResolver,
) -> tuple[Path, int, str]:
    try:
        path_obj = resolve_create_path(session_base, request.path)
    except ValueError as e:
        raise FileRouteError(400, {"error": str(e)}) from e
    try:
        size, next_version = write_new_text_file_atomic(path_obj, text=request.text)
    except FileExistsError as e:
        payload: dict[str, Any] = {"error": "file already exists", "conflict": True, "path": str(path_obj)}
        if path_obj.is_file():
            try:
                _current_text, _current_size, current_version = read_text_file_for_write(path_obj, max_bytes=FILE_READ_MAX_BYTES)
                payload["version"] = current_version
            except (FileNotFoundError, PermissionError, ValueError):
                pass
        raise FileRouteError(409, payload) from e
    except FileNotFoundError as e:
        raise FileRouteError(404, {"error": str(e)}) from e
    except PermissionError as e:
        raise FileRouteError(403, {"error": str(e)}) from e
    except ValueError as e:
        raise FileRouteError(400, {"error": str(e)}) from e
    return path_obj, size, next_version


def _update_session_file(
    *,
    request: SessionFileWriteRequest,
    session_base: Path,
    resolve_git_existing_regular_file: GitFileResolver,
    file_write_lock: FileWriteLock,
) -> tuple[Path, int, str]:
    try:
        if request.git_path:
            path_obj, _rel = resolve_git_existing_regular_file(request.path)
        else:
            path_obj = resolve_session_write_update_path(session_base, request.path)
    except FileNotFoundError as e:
        raise FileRouteError(404, {"error": str(e)}) from e
    except PermissionError as e:
        raise FileRouteError(403, {"error": str(e)}) from e
    except ValueError as e:
        raise FileRouteError(400, {"error": str(e)}) from e
    except RuntimeError as e:
        raise FileRouteError(409, {"error": str(e)}) from e
    with file_write_lock(path_obj):
        try:
            _current_text, _current_size, current_version = read_text_file_for_write(path_obj, max_bytes=FILE_READ_MAX_BYTES)
        except FileNotFoundError as e:
            raise FileRouteError(404, {"error": str(e)}) from e
        except PermissionError as e:
            raise FileRouteError(403, {"error": str(e)}) from e
        except ValueError as e:
            raise FileRouteError(400, {"error": str(e)}) from e
        if current_version != request.version:
            raise FileRouteError(
                409,
                {"error": "file changed on disk", "conflict": True, "path": str(path_obj), "version": current_version},
            )
        try:
            size, next_version = write_text_file_atomic(path_obj, text=request.text)
        except FileNotFoundError as e:
            raise FileRouteError(404, {"error": str(e)}) from e
        except PermissionError as e:
            raise FileRouteError(403, {"error": str(e)}) from e
        except ValueError as e:
            raise FileRouteError(400, {"error": str(e)}) from e
    return path_obj, size, next_version


def _expanduser_path(path: Path) -> Path:
    try:
        return path.expanduser()
    except RuntimeError as e:
        raise ValueError(str(e)) from e
