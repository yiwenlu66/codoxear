from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping
import urllib.parse

from .git_ops import git_path_from_token
from .git_ops import path_has_surrogate_bytes
from .git_ops import path_json_text
from .git_ops import path_token_query
from .git_ops import path_token_response_fields
from .git_ops import git_path_token
from .file_route_common import body_flag
from .file_view import ClientFileView
from .video_preview import video_response_payload


JsonResponse = Callable[[Any, int, dict[str, Any]], None]


@dataclass(frozen=True)
class GlobalFileRequest:
    path: str
    session_id: str
    git_path: bool


@dataclass(frozen=True)
class GlobalFileRouteDeps:
    require_auth: Callable[[Any], bool]
    json_response: JsonResponse
    read_json_body: Callable[..., Mapping[str, Any]]
    resolve_git_client_file_view: Callable[..., tuple[Path, str, ClientFileView]]
    resolve_client_file_path: Callable[..., Path]
    read_client_file_view: Callable[[Path], ClientFileView]


def handle_global_file_post_route(
    handler: Any,
    *,
    path: str,
    manager: Any,
    deps: GlobalFileRouteDeps,
) -> bool:
    if path == "/api/files/read":
        _handle_global_file_read(handler, manager=manager, deps=deps)
        return True
    if path == "/api/files/inspect":
        _handle_global_file_inspect(handler, deps=deps)
        return True
    return False


def _map_resolve_error(handler: Any, exc: BaseException, deps: GlobalFileRouteDeps, *, runtime_status: int = 409) -> None:
    if isinstance(exc, FileNotFoundError):
        deps.json_response(handler, 404, {"error": str(exc)})
        return
    if isinstance(exc, PermissionError):
        deps.json_response(handler, 403, {"error": str(exc)})
        return
    if isinstance(exc, ValueError):
        deps.json_response(handler, 400, {"error": str(exc)})
        return
    if isinstance(exc, RuntimeError):
        deps.json_response(handler, runtime_status, {"error": str(exc)})
        return
    raise exc


def _global_file_request(handler: Any, deps: GlobalFileRouteDeps) -> GlobalFileRequest | None:
    body = deps.read_json_body(handler)
    raw_path = body.get("path")
    raw_token = body.get("path_token")
    if not isinstance(raw_path, str) or raw_path == "":
        deps.json_response(handler, 400, {"error": "path required"})
        return None
    session_id_raw = body.get("session_id")
    if session_id_raw is not None and not isinstance(session_id_raw, str):
        deps.json_response(handler, 400, {"error": "session_id must be a string"})
        return None
    session_id = session_id_raw if isinstance(session_id_raw, str) and session_id_raw else ""
    git_path = body_flag(body, "git_path")
    # path_token is accepted for plain files too: a non-UTF absolute path
    # returned to the client as a JSON-safe display string can only be reopened
    # via its reversible token.
    if isinstance(raw_token, str) and raw_token:
        try:
            raw_path = git_path_from_token(raw_token)
        except ValueError as e:
            deps.json_response(handler, 400, {"error": str(e)})
            return None
    return GlobalFileRequest(path=raw_path, session_id=session_id, git_path=git_path)


def _global_file_view(request: GlobalFileRequest, deps: GlobalFileRouteDeps) -> tuple[Path, str, ClientFileView]:
    if request.git_path:
        return deps.resolve_git_client_file_view(session_id=request.session_id, raw_path=request.path)
    path_obj = deps.resolve_client_file_path(session_id=request.session_id, raw_path=request.path)
    return path_obj, "", deps.read_client_file_view(path_obj)


def _handle_global_file_read(handler: Any, *, manager: Any, deps: GlobalFileRouteDeps) -> None:
    if not deps.require_auth(handler):
        handler._unauthorized()
        return
    request = _global_file_request(handler, deps)
    if request is None:
        return
    try:
        path_obj, rel_for_url, view = _global_file_view(request, deps)
    except (FileNotFoundError, PermissionError, ValueError, RuntimeError) as e:
        _map_resolve_error(handler, e, deps)
        return
    if request.session_id:
        try:
            manager.files_add(request.session_id, path_json_text(path_obj))
        except KeyError:
            pass
    deps.json_response(handler, 200, global_file_read_payload(request=request, path_obj=path_obj, rel_for_url=rel_for_url, view=view))


def _handle_global_file_inspect(handler: Any, *, deps: GlobalFileRouteDeps) -> None:
    if not deps.require_auth(handler):
        handler._unauthorized()
        return
    request = _global_file_request(handler, deps)
    if request is None:
        return
    try:
        path_obj, _rel_for_url, view = _global_file_view(request, deps)
    except (FileNotFoundError, PermissionError, ValueError, RuntimeError) as e:
        _map_resolve_error(handler, e, deps)
        return
    deps.json_response(
        handler,
        200,
        {
            "ok": True,
            "path": path_json_text(path_obj),
            **path_token_response_fields(str(path_obj)),
            "kind": view.kind,
            "content_type": view.content_type,
            "size": int(view.size),
            "reason": view.blocked_reason,
            "viewer_max_bytes": view.viewer_max_bytes,
        },
    )


def _absolute_media_query(path_obj: Path) -> str:
    display = path_json_text(path_obj)
    query = f"path={urllib.parse.quote(display, safe='')}"
    if path_has_surrogate_bytes(str(path_obj)):
        query += f"&path_token={urllib.parse.quote(git_path_token(str(path_obj)))}"
    return query


def global_file_read_payload(
    *,
    request: GlobalFileRequest,
    path_obj: Path,
    rel_for_url: str,
    view: ClientFileView,
) -> dict[str, Any]:
    token_fields = path_token_response_fields(str(path_obj))
    if request.git_path and request.session_id and rel_for_url:
        media_blob_url = f"/api/sessions/{request.session_id}/file/blob?{path_token_query(rel_for_url)}&git_path=1"
        media_preview_url = f"/api/sessions/{request.session_id}/file/video_preview?{path_token_query(rel_for_url)}&git_path=1"
    else:
        abs_query = _absolute_media_query(path_obj)
        media_blob_url = f"/api/files/blob?{abs_query}"
        media_preview_url = f"/api/files/video_preview?{abs_query}"
    if view.kind == "image":
        return {
            "ok": True,
            "kind": "image",
            "content_type": view.content_type,
            "path": path_json_text(path_obj),
            **token_fields,
            "size": int(view.size),
            "image_url": media_blob_url,
        }
    if view.kind == "pdf":
        return {
            "ok": True,
            "kind": "pdf",
            "content_type": view.content_type,
            "path": path_json_text(path_obj),
            **token_fields,
            "size": int(view.size),
            "pdf_url": media_blob_url,
        }
    if view.kind == "video":
        payload = video_response_payload(
            path_obj=path_obj,
            size=int(view.size),
            content_type=view.content_type,
            video_url=media_blob_url,
            preview_url=media_preview_url,
        )
        payload.update(token_fields)
        return payload
    if view.kind == "download_only":
        return {
            "ok": True,
            "kind": "download_only",
            "path": path_json_text(path_obj),
            **token_fields,
            "size": int(view.size),
            "reason": view.blocked_reason,
            "viewer_max_bytes": view.viewer_max_bytes,
        }
    return {
        "ok": True,
        "kind": view.kind,
        "path": path_json_text(path_obj),
        **token_fields,
        "size": int(view.size),
        "text": view.text,
        "editable": bool(view.editable),
        "version": view.version,
    }
