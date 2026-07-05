from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping
import urllib.parse

from .git_ops import git_path_from_token
from .git_ops import git_path_token
from .git_ops import path_has_surrogate_bytes
from .git_ops import path_json_text
from .git_ops import path_token_query
from .git_ops import path_token_response_fields
from .file_view import ClientFileView
from .video_preview import video_response_payload


def session_file_read_payload(
    *,
    session_id: str,
    path_obj: Path,
    rel: str,
    view: ClientFileView,
    git_path: bool,
) -> dict[str, Any]:
    git_suffix = "&git_path=1" if git_path else ""
    # path_token_query emits the JSON-safe display ``path`` plus a reversible
    # ``path_token`` whenever the raw rel carries surrogate bytes, so non-UTF
    # plain files (not just git paths) can be re-opened through blob/download.
    rel_for_url = path_token_query(rel)
    token_fields = path_token_response_fields(rel)
    path_text = path_json_text(path_obj)
    rel_text = path_json_text(rel)
    if view.kind == "image":
        return {
            "ok": True,
            "kind": "image",
            "content_type": view.content_type,
            "path": path_text,
            "rel": rel_text,
            **token_fields,
            "size": int(view.size),
            "image_url": f"/api/sessions/{session_id}/file/blob?{rel_for_url}{git_suffix}",
        }
    if view.kind == "pdf":
        return {
            "ok": True,
            "kind": "pdf",
            "content_type": view.content_type,
            "path": path_text,
            "rel": rel_text,
            **token_fields,
            "size": int(view.size),
            "pdf_url": f"/api/sessions/{session_id}/file/blob?{rel_for_url}{git_suffix}",
        }
    if view.kind == "video":
        payload = video_response_payload(
            path_obj=path_obj,
            rel=rel_text,
            size=int(view.size),
            content_type=view.content_type,
            video_url=f"/api/sessions/{session_id}/file/blob?{rel_for_url}{git_suffix}",
            preview_url=f"/api/sessions/{session_id}/file/video_preview?{rel_for_url}{git_suffix}",
        )
        payload.update(token_fields)
        return payload
    if view.kind == "download_only":
        return {
            "ok": True,
            "kind": "download_only",
            "path": path_text,
            "rel": rel_text,
            **token_fields,
            "size": int(view.size),
            "reason": view.blocked_reason,
            "viewer_max_bytes": view.viewer_max_bytes,
        }
    return {
        "ok": True,
        "kind": view.kind,
        "path": path_text,
        "rel": rel_text,
        **token_fields,
        "size": int(view.size),
        "text": view.text,
        "editable": bool(view.editable),
        "version": view.version,
    }


JsonResponse = Callable[[Any, int, dict[str, Any]], None]
RouteMatcher = Callable[..., str | None]


@dataclass(frozen=True)
class FileGetRouteDeps:
    require_auth: Callable[[Any], bool]
    json_response: JsonResponse
    resolve_session_cwd: Callable[[str], Path]
    resolve_existing_session_file: Callable[[Path, str], Path]
    resolve_session_path: Callable[[Path, str], Path]
    resolve_git_client_file_view: Callable[..., tuple[Path, str, ClientFileView]]
    resolve_git_existing_regular_file: Callable[..., tuple[Path, str]]
    resolve_existing_absolute_file: Callable[[str], Path]
    read_client_file_view: Callable[[Path], ClientFileView]
    read_regular_file_prefix: Callable[[Path, int], tuple[bytes, int]]
    search_session_relative_files: Callable[..., dict[str, Any]]
    list_session_relative_files: Callable[[Path], list[str]]
    list_session_relative_file_entries: Callable[[Path], list[dict[str, Any]]]
    file_kind: Callable[[Path, bytes], tuple[str, str | None]]
    ensure_video_preview: Callable[[Path], Path]
    inspect_downloadable_file: Callable[[Path], int]
    download_disposition: Callable[[Path], str]
    send_inline_file_response: Callable[[Any, Path, str], None]
    send_attachment_file_response: Callable[..., None]
    file_search_limit: int


def handle_file_get_route(
    handler: Any,
    *,
    path: str,
    query: str,
    manager: Any,
    deps: FileGetRouteDeps,
    match_session_route: RouteMatcher,
) -> bool:
    session_id = match_session_route(path, "file", "read")
    if session_id is not None:
        _handle_session_file_read(handler, session_id=session_id, query=query, manager=manager, deps=deps)
        return True

    session_id = match_session_route(path, "file", "search")
    if session_id is not None:
        _handle_session_file_search(handler, session_id=session_id, query=query, manager=manager, deps=deps)
        return True

    session_id = match_session_route(path, "file", "list")
    if session_id is not None:
        _handle_session_file_list(handler, session_id=session_id, manager=manager, deps=deps)
        return True

    session_id = match_session_route(path, "file", "blob")
    if session_id is not None:
        _handle_session_file_blob(handler, session_id=session_id, query=query, manager=manager, deps=deps)
        return True

    session_id = match_session_route(path, "file", "video_preview")
    if session_id is not None:
        _handle_session_file_video_preview(handler, session_id=session_id, query=query, manager=manager, deps=deps)
        return True

    if handle_absolute_file_preview_route(handler, path=path, query=query, deps=deps):
        return True

    session_id = match_session_route(path, "file", "download")
    if session_id is not None:
        _handle_session_file_download(handler, session_id=session_id, query=query, manager=manager, deps=deps)
        return True

    return False


def handle_absolute_file_preview_route(
    handler: Any,
    *,
    path: str,
    query: str,
    deps: FileGetRouteDeps,
) -> bool:
    if path == "/api/files/blob":
        _handle_absolute_file_blob(handler, query=query, deps=deps)
        return True
    if path == "/api/files/video_preview":
        _handle_absolute_file_video_preview(handler, query=query, deps=deps)
        return True
    return False


def _authorized(handler: Any, deps: FileGetRouteDeps) -> bool:
    if deps.require_auth(handler):
        return True
    handler._unauthorized()
    return False


def _session(handler: Any, *, session_id: str, manager: Any, deps: FileGetRouteDeps) -> Any | None:
    if not _authorized(handler, deps):
        return None
    manager.refresh_session_meta(session_id)
    session = manager.get_session(session_id)
    if not session:
        deps.json_response(handler, 404, {"error": "unknown session"})
        return None
    return session


def _query_values(query: str) -> dict[str, list[str]]:
    return urllib.parse.parse_qs(query)


def _query_flag(values: Mapping[str, list[str]], name: str) -> bool:
    raw = values.get(name, [""])[0]
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _required_query_value(handler: Any, values: Mapping[str, list[str]], deps: FileGetRouteDeps, name: str) -> str | None:
    value = values.get(name)
    if not value or not value[0]:
        deps.json_response(handler, 400, {"error": f"{name} required"})
        return None
    return value[0]


def _git_or_plain_query_path(
    handler: Any,
    values: Mapping[str, list[str]],
    deps: FileGetRouteDeps,
    *,
    git_path: bool,
) -> str | None:
    # The reversible path_token channel is accepted for plain file routes as
    # well as git_path routes: a non-UTF filename surfaced by list/search can
    # only be re-opened through its token, since the JSON-safe display ``path``
    # is not byte-roundtrippable.
    token = values.get("path_token")
    if token and token[0]:
        try:
            return git_path_from_token(token[0])
        except ValueError as e:
            deps.json_response(handler, 400, {"error": str(e)})
            return None
    return _required_query_value(handler, values, deps, "path")


def _map_resolve_error(handler: Any, exc: BaseException, deps: FileGetRouteDeps, *, runtime_status: int = 409) -> None:
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


def _handle_session_file_read(handler: Any, *, session_id: str, query: str, manager: Any, deps: FileGetRouteDeps) -> None:
    session = _session(handler, session_id=session_id, manager=manager, deps=deps)
    if session is None:
        return
    qs = _query_values(query)
    git_path = _query_flag(qs, "git_path")
    rel = _git_or_plain_query_path(handler, qs, deps, git_path=git_path)
    if rel is None:
        return
    try:
        if git_path:
            path_obj, rel, view = deps.resolve_git_client_file_view(session_id=session_id, raw_path=rel)
        else:
            base = deps.resolve_session_cwd(session.cwd)
            path_obj = deps.resolve_existing_session_file(base, rel)
            view = deps.read_client_file_view(path_obj)
    except (FileNotFoundError, PermissionError, ValueError, RuntimeError) as e:
        _map_resolve_error(handler, e, deps)
        return
    try:
        rel_token = git_path_token(rel) if path_has_surrogate_bytes(rel) else ""
        if rel_token:
            manager.files_add(session_id, path_json_text(path_obj), api_path=rel_token)
        else:
            manager.files_add(session_id, path_json_text(path_obj))
    except KeyError:
        pass
    deps.json_response(
        handler,
        200,
        session_file_read_payload(
            session_id=session_id,
            path_obj=path_obj,
            rel=str(rel),
            view=view,
            git_path=git_path,
        ),
    )


def _handle_session_file_search(handler: Any, *, session_id: str, query: str, manager: Any, deps: FileGetRouteDeps) -> None:
    session = _session(handler, session_id=session_id, manager=manager, deps=deps)
    if session is None:
        return
    qs = _query_values(query)
    query_raw = qs.get("q")
    if not query_raw or not query_raw[0].strip():
        deps.json_response(handler, 400, {"error": "q required"})
        return
    limit_raw = qs.get("limit", [str(deps.file_search_limit)])[0]
    try:
        limit = int(str(limit_raw).strip() or str(deps.file_search_limit))
    except ValueError:
        deps.json_response(handler, 400, {"error": "limit must be an integer"})
        return
    if limit < 1:
        deps.json_response(handler, 400, {"error": "limit must be >= 1"})
        return
    try:
        base = deps.resolve_session_cwd(session.cwd)
        result = deps.search_session_relative_files(base, query=query_raw[0], limit=limit)
    except FileNotFoundError as e:
        deps.json_response(handler, 404, {"error": str(e)})
        return
    except PermissionError as e:
        deps.json_response(handler, 403, {"error": str(e)})
        return
    except (RuntimeError, ValueError) as e:
        deps.json_response(handler, 400, {"error": str(e)})
        return
    deps.json_response(
        handler,
        200,
        {
            "ok": True,
            "cwd": path_json_text(base),
            "query": result["query"],
            "mode": result["mode"],
            "matches": result["matches"],
            "scanned": result["scanned"],
            "truncated": result["truncated"],
        },
    )


def _handle_session_file_list(handler: Any, *, session_id: str, manager: Any, deps: FileGetRouteDeps) -> None:
    session = _session(handler, session_id=session_id, manager=manager, deps=deps)
    if session is None:
        return
    try:
        base = deps.resolve_session_cwd(session.cwd)
        entries = deps.list_session_relative_file_entries(base)
    except FileNotFoundError as e:
        deps.json_response(handler, 404, {"error": str(e)})
        return
    except PermissionError as e:
        deps.json_response(handler, 403, {"error": str(e)})
        return
    except ValueError as e:
        deps.json_response(handler, 400, {"error": str(e)})
        return
    # ``files`` keeps the legacy UTF-8-safe display-string list; ``entries`` is
    # the additive parallel structure carrying api_path/non_utf8_path for any
    # raw-byte filename so the picker can open it via path_token.
    files = [str(entry.get("path", "")) for entry in entries]
    deps.json_response(handler, 200, {"ok": True, "cwd": path_json_text(base), "files": files, "entries": entries})


def _session_file_path_for_preview(
    handler: Any,
    *,
    session_id: str,
    query: str,
    manager: Any,
    deps: FileGetRouteDeps,
    for_download: bool = False,
) -> Path | None:
    session = _session(handler, session_id=session_id, manager=manager, deps=deps)
    if session is None:
        return None
    qs = _query_values(query)
    git_path = _query_flag(qs, "git_path")
    rel = _git_or_plain_query_path(handler, qs, deps, git_path=git_path)
    if rel is None:
        return None
    try:
        if git_path:
            path_obj, _rel = deps.resolve_git_existing_regular_file(session_id=session_id, raw_path=rel)
        else:
            base = deps.resolve_session_cwd(session.cwd)
            if for_download:
                path_obj = deps.resolve_session_path(base, rel)
            else:
                path_obj = deps.resolve_existing_session_file(base, rel)
    except (FileNotFoundError, PermissionError, ValueError, RuntimeError) as e:
        _map_resolve_error(handler, e, deps)
        return None
    return path_obj


def _handle_session_file_blob(handler: Any, *, session_id: str, query: str, manager: Any, deps: FileGetRouteDeps) -> None:
    path_obj = _session_file_path_for_preview(handler, session_id=session_id, query=query, manager=manager, deps=deps)
    if path_obj is None:
        return
    _send_preview_blob(handler, path_obj=path_obj, deps=deps)


def _handle_session_file_video_preview(handler: Any, *, session_id: str, query: str, manager: Any, deps: FileGetRouteDeps) -> None:
    path_obj = _session_file_path_for_preview(handler, session_id=session_id, query=query, manager=manager, deps=deps)
    if path_obj is None:
        return
    _send_video_preview(handler, path_obj=path_obj, deps=deps)


def _handle_absolute_file_blob(handler: Any, *, query: str, deps: FileGetRouteDeps) -> None:
    if not _authorized(handler, deps):
        return
    path_obj = _absolute_file_path(handler, query=query, deps=deps)
    if path_obj is None:
        return
    _send_preview_blob(handler, path_obj=path_obj, deps=deps)


def _handle_absolute_file_video_preview(handler: Any, *, query: str, deps: FileGetRouteDeps) -> None:
    if not _authorized(handler, deps):
        return
    path_obj = _absolute_file_path(handler, query=query, deps=deps)
    if path_obj is None:
        return
    _send_video_preview(handler, path_obj=path_obj, deps=deps)


def _absolute_file_path(handler: Any, *, query: str, deps: FileGetRouteDeps) -> Path | None:
    qs = _query_values(query)
    token = qs.get("path_token")
    if token and token[0]:
        try:
            raw_path = git_path_from_token(token[0])
        except ValueError as e:
            deps.json_response(handler, 400, {"error": str(e)})
            return None
    else:
        raw_path = _required_query_value(handler, qs, deps, "path")
        if raw_path is None:
            return None
    try:
        return deps.resolve_existing_absolute_file(raw_path)
    except (FileNotFoundError, PermissionError, ValueError) as e:
        _map_resolve_error(handler, e, deps)
        return None


def _read_prefix(handler: Any, *, path_obj: Path, deps: FileGetRouteDeps) -> bytes | None:
    try:
        prefix, _size = deps.read_regular_file_prefix(path_obj, 4096)
        return prefix
    except FileNotFoundError as e:
        deps.json_response(handler, 404, {"error": str(e)})
        return None
    except PermissionError as e:
        deps.json_response(handler, 403, {"error": str(e)})
        return None
    except ValueError as e:
        deps.json_response(handler, 400, {"error": str(e)})
        return None


def _send_preview_blob(handler: Any, *, path_obj: Path, deps: FileGetRouteDeps) -> None:
    prefix = _read_prefix(handler, path_obj=path_obj, deps=deps)
    if prefix is None:
        return
    kind, content_type = deps.file_kind(path_obj, prefix)
    if kind == "video":
        deps.send_inline_file_response(handler, path_obj, content_type or "application/octet-stream")
        return
    if kind not in {"image", "pdf"} or content_type is None:
        deps.json_response(handler, 400, {"error": "file is not previewable inline"})
        return
    deps.send_inline_file_response(handler, path_obj, content_type)


def _send_video_preview(handler: Any, *, path_obj: Path, deps: FileGetRouteDeps) -> None:
    prefix = _read_prefix(handler, path_obj=path_obj, deps=deps)
    if prefix is None:
        return
    kind, _content_type = deps.file_kind(path_obj, prefix)
    if kind != "video":
        deps.json_response(handler, 400, {"error": "file is not a video"})
        return
    try:
        preview = deps.ensure_video_preview(path_obj)
    except FileNotFoundError as e:
        deps.json_response(handler, 404, {"error": str(e)})
        return
    except PermissionError as e:
        deps.json_response(handler, 403, {"error": str(e)})
        return
    except RuntimeError as e:
        deps.json_response(handler, 500, {"error": f"video preview failed: {e}"})
        return
    deps.send_inline_file_response(handler, preview, "video/mp4")


def _handle_session_file_download(handler: Any, *, session_id: str, query: str, manager: Any, deps: FileGetRouteDeps) -> None:
    path_obj = _session_file_path_for_preview(
        handler,
        session_id=session_id,
        query=query,
        manager=manager,
        deps=deps,
        for_download=True,
    )
    if path_obj is None:
        return
    try:
        size = deps.inspect_downloadable_file(path_obj)
    except (FileNotFoundError, PermissionError, ValueError, RuntimeError) as e:
        _map_resolve_error(handler, e, deps)
        return
    deps.send_attachment_file_response(handler, path_obj, size=size, content_disposition=deps.download_disposition(path_obj))
