from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .file_route_common import FileRouteError
from .file_route_common import FileRouteResponse
from .file_route_common import SessionFileWriteRequest
from .file_route_common import body_flag
from .file_route_common import resolve_session_write_update_path
from .file_text import FILE_READ_MAX_BYTES
from .file_text import read_text_file_for_write
from .file_text import write_new_text_file_atomic
from .file_text import write_text_file_atomic


FileWriteLock = Callable[[Path], AbstractContextManager[None]]
GitFileResolver = Callable[[str], tuple[Path, str]]
SessionPathResolver = Callable[[Path, str], Path]
FileRecorder = Callable[[str], None]
JsonResponse = Callable[[Any, int, dict[str, Any]], None]
RouteMatcher = Callable[..., str | None]


@dataclass(frozen=True)
class FileWriteRouteDeps:
    require_auth: Callable[[Any], bool]
    json_response: JsonResponse
    read_json_body: Callable[..., Mapping[str, Any]]
    resolve_session_cwd: Callable[[str], Path]
    resolve_create_path: SessionPathResolver
    resolve_git_existing_regular_file: Callable[..., tuple[Path, str]]
    file_write_lock: FileWriteLock


def handle_file_write_post_route(
    handler: Any,
    *,
    path: str,
    manager: Any,
    deps: FileWriteRouteDeps,
    match_session_route: RouteMatcher,
) -> bool:
    session_id = match_session_route(path, "file", "write")
    if session_id is None:
        return False
    if not deps.require_auth(handler):
        handler._unauthorized()
        return True
    body = deps.read_json_body(handler)
    try:
        request = parse_session_file_write_request(body)
    except FileRouteError as e:
        deps.json_response(handler, e.status, e.payload)
        return True
    manager.refresh_session_meta(session_id)
    session = manager.get_session(session_id)
    if not session:
        deps.json_response(handler, 404, {"error": "unknown session"})
        return True
    try:
        base = deps.resolve_session_cwd(session.cwd)
    except ValueError as e:
        deps.json_response(handler, 400, {"error": str(e)})
        return True
    response = session_file_write_response(
        session_base=base,
        request=request,
        resolve_create_path=deps.resolve_create_path,
        resolve_git_existing_regular_file=lambda raw_path: deps.resolve_git_existing_regular_file(
            session_id=session_id,
            raw_path=raw_path,
        ),
        file_write_lock=deps.file_write_lock,
        record_file=lambda path_value: manager.files_add(session_id, path_value),
    )
    deps.json_response(handler, response.status, response.payload)
    return True


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
