from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable


JsonResponse = Callable[[Any, int, dict[str, Any]], None]
RouteMatcher = Callable[..., str | None]
ReadBody = Callable[[Any], bytes]
ReadJsonBody = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class ControlRouteDeps:
    require_auth: Callable[[Any], bool]
    json_response: JsonResponse
    read_body: ReadBody
    read_json_body: ReadJsonBody
    attach_upload_body_max_bytes: int
    attach_upload_max_bytes: int
    stage_uploaded_file: Callable[[str, str, bytes], Path]
    attachment_inject_text: Callable[[int, Path], str]
    clean_unattended_cooldown_minutes: Callable[[Any], int]
    clean_unattended_remaining_injections: Callable[..., int]
    session_not_ready_error: type[BaseException]
    session_injection_error: type[BaseException]
    session_commit_unknown_error: type[BaseException]


def _authorized(handler: Any, deps: ControlRouteDeps) -> bool:
    if deps.require_auth(handler):
        return True
    handler._unauthorized()
    return False


def handle_control_get_route(
    handler: Any,
    *,
    path: str,
    manager: Any,
    deps: ControlRouteDeps,
    match_session_route: RouteMatcher,
) -> bool:
    session_id = match_session_route(path, "attachments")
    if session_id is None:
        return False
    if not _authorized(handler, deps):
        return True
    try:
        res = manager.list_staged_attachments(session_id)
    except KeyError:
        deps.json_response(handler, 404, {"error": "unknown session"})
        return True
    deps.json_response(handler, 200, res)
    return True


def handle_control_post_route(
    handler: Any,
    *,
    path: str,
    manager: Any,
    deps: ControlRouteDeps,
    match_session_route: RouteMatcher,
) -> bool:
    route_handlers = (
        ("delete", None, _handle_delete),
        ("edit", None, _handle_edit),
        ("rename", None, _handle_rename),
        ("pending_attachment", "clear", _handle_pending_attachment_clear),
        ("attachments", "delete", _handle_attachment_delete),
        ("attachments", "clear", _handle_attachments_clear),
        ("commit_unknown_send", "clear", _handle_commit_unknown_send_clear),
        ("send", None, _handle_send),
        ("unattended", None, _handle_unattended),
        ("interrupt", None, _handle_interrupt),
    )
    for route, suffix, route_handler in route_handlers:
        session_id = match_session_route(path, route) if suffix is None else match_session_route(path, route, suffix)
        if session_id is not None:
            route_handler(handler, session_id=session_id, manager=manager, deps=deps)
            return True
    session_id = match_session_route(path, "inject_file")
    if session_id is None:
        session_id = match_session_route(path, "inject_image")
    if session_id is not None:
        _handle_inject_attachment(handler, session_id=session_id, manager=manager, deps=deps)
        return True
    return False


def _handle_delete(handler: Any, *, session_id: str, manager: Any, deps: ControlRouteDeps) -> None:
    if not _authorized(handler, deps):
        return
    deps.read_body(handler)
    ok = manager.delete_session(session_id)
    if not ok:
        deps.json_response(handler, 404, {"error": "unknown session"})
        return
    deps.json_response(handler, 200, {"ok": True})


def _handle_edit(handler: Any, *, session_id: str, manager: Any, deps: ControlRouteDeps) -> None:
    if not _authorized(handler, deps):
        return
    obj = deps.read_json_body(handler)
    name = obj.get("name")
    if not isinstance(name, str):
        deps.json_response(handler, 400, {"error": "name required"})
        return
    try:
        alias, sidebar_meta = manager.edit_session(
            session_id,
            name=name,
            priority_offset=obj.get("priority_offset"),
            snooze_until=obj.get("snooze_until"),
            dependency_session_id=obj.get("dependency_session_id"),
        )
    except KeyError:
        deps.json_response(handler, 404, {"error": "unknown session"})
        return
    except ValueError as e:
        deps.json_response(handler, 400, {"error": str(e)})
        return
    deps.json_response(handler, 200, {"ok": True, "alias": alias, **sidebar_meta})


def _handle_rename(handler: Any, *, session_id: str, manager: Any, deps: ControlRouteDeps) -> None:
    if not _authorized(handler, deps):
        return
    obj = deps.read_json_body(handler)
    name = obj.get("name")
    if not isinstance(name, str):
        deps.json_response(handler, 400, {"error": "name required"})
        return
    try:
        alias = manager.alias_set(session_id, name)
    except KeyError:
        deps.json_response(handler, 404, {"error": "unknown session"})
        return
    deps.json_response(handler, 200, {"ok": True, "alias": alias})


def _handle_pending_attachment_clear(handler: Any, *, session_id: str, manager: Any, deps: ControlRouteDeps) -> None:
    if not _authorized(handler, deps):
        return
    try:
        res = manager.clear_pending_attachment(session_id)
    except KeyError:
        deps.json_response(handler, 404, {"error": "unknown session"})
        return
    except ValueError as e:
        deps.json_response(handler, 400, {"error": str(e)})
        return
    deps.json_response(handler, 200, res)


def _handle_attachment_delete(handler: Any, *, session_id: str, manager: Any, deps: ControlRouteDeps) -> None:
    if not _authorized(handler, deps):
        return
    obj = deps.read_json_body(handler)
    attachment_id = obj.get("id")
    if not isinstance(attachment_id, str) or not attachment_id.strip():
        deps.json_response(handler, 400, {"error": "attachment id required"})
        return
    try:
        res = manager.remove_staged_attachment(session_id, attachment_id.strip())
    except KeyError:
        deps.json_response(handler, 404, {"error": "unknown session"})
        return
    except ValueError as e:
        status = 404 if str(e) == "unknown attachment" else 400
        deps.json_response(handler, status, {"error": str(e)})
        return
    deps.json_response(handler, 200, res)


def _handle_attachments_clear(handler: Any, *, session_id: str, manager: Any, deps: ControlRouteDeps) -> None:
    if not _authorized(handler, deps):
        return
    deps.read_body(handler)
    try:
        res = manager.clear_staged_attachments(session_id)
    except KeyError:
        deps.json_response(handler, 404, {"error": "unknown session"})
        return
    except ValueError as e:
        deps.json_response(handler, 400, {"error": str(e)})
        return
    deps.json_response(handler, 200, res)


def _handle_commit_unknown_send_clear(handler: Any, *, session_id: str, manager: Any, deps: ControlRouteDeps) -> None:
    if not _authorized(handler, deps):
        return
    try:
        res = manager.clear_commit_unknown_send(session_id)
    except KeyError:
        deps.json_response(handler, 404, {"error": "unknown session"})
        return
    deps.json_response(handler, 200, res)


def _handle_send(handler: Any, *, session_id: str, manager: Any, deps: ControlRouteDeps) -> None:
    if not _authorized(handler, deps):
        return
    obj = deps.read_json_body(handler)
    text = obj.get("text")
    if not isinstance(text, str) or not text.strip():
        deps.json_response(handler, 400, {"error": "text required"})
        return
    allow_pending_attachment = bool(obj.get("allow_pending_attachment"))
    try:
        res = manager.send(session_id, text, allow_pending_attachment=allow_pending_attachment)
    except KeyError:
        deps.json_response(handler, 404, {"error": "unknown session"})
        return
    except deps.session_not_ready_error as e:
        deps.json_response(handler, 409, {"error": str(e)})
        return
    except deps.session_injection_error as e:
        deps.json_response(handler, 502, {"error": str(e)})
        return
    except deps.session_commit_unknown_error as e:
        deps.json_response(handler, 504, {"error": str(e), "commit_unknown": True})
        return
    deps.json_response(handler, 200, res)


def _handle_unattended(handler: Any, *, session_id: str, manager: Any, deps: ControlRouteDeps) -> None:
    if not _authorized(handler, deps):
        return
    obj = deps.read_json_body(handler)
    enabled_raw = obj.get("enabled", None)
    request_raw = obj.get("request", None)
    cooldown_minutes_raw = obj.get("cooldown_minutes", None)
    remaining_injections_raw = obj.get("remaining_injections", None)
    if "text" in obj:
        deps.json_response(handler, 400, {"error": "unknown field: text (use request)"})
        return
    if enabled_raw is None:
        enabled: bool | None = None
    elif isinstance(enabled_raw, bool):
        enabled = enabled_raw
    else:
        deps.json_response(handler, 400, {"error": "enabled must be a boolean"})
        return
    if request_raw is not None and (not isinstance(request_raw, str)):
        deps.json_response(handler, 400, {"error": "request must be a string"})
        return
    request = request_raw if request_raw is not None else None
    if cooldown_minutes_raw is not None:
        try:
            cooldown_minutes = deps.clean_unattended_cooldown_minutes(cooldown_minutes_raw)
        except ValueError as e:
            deps.json_response(handler, 400, {"error": str(e)})
            return
    else:
        cooldown_minutes = None
    if remaining_injections_raw is not None:
        try:
            remaining_injections = deps.clean_unattended_remaining_injections(remaining_injections_raw, allow_zero=True)
        except ValueError as e:
            deps.json_response(handler, 400, {"error": str(e)})
            return
    else:
        remaining_injections = None
    cfg = manager.unattended_set(
        session_id,
        enabled=enabled,
        request=request,
        cooldown_minutes=cooldown_minutes,
        remaining_injections=remaining_injections,
    )
    deps.json_response(handler, 200, {"ok": True, **cfg})


def _handle_interrupt(handler: Any, *, session_id: str, manager: Any, deps: ControlRouteDeps) -> None:
    if not _authorized(handler, deps):
        return
    deps.read_body(handler)
    try:
        resp = manager.inject_keys(session_id, "\\x1b", interrupt=True)
    except KeyError:
        deps.json_response(handler, 404, {"error": "unknown session"})
        return
    deps.json_response(handler, 200, {"ok": True, "broker": resp})


def _handle_inject_attachment(handler: Any, *, session_id: str, manager: Any, deps: ControlRouteDeps) -> None:
    if not _authorized(handler, deps):
        return
    obj = deps.read_json_body(
        handler,
        limit=deps.attach_upload_body_max_bytes,
        too_large_error=f"file too large (max {deps.attach_upload_max_bytes} bytes)",
    )
    data_b64 = obj.get("data_b64")
    filename = obj.get("filename")
    attachment_index = obj.get("attachment_index")
    if not isinstance(filename, str) or (not filename.strip()):
        deps.json_response(handler, 400, {"error": "filename required"})
        return
    if isinstance(attachment_index, bool) or not isinstance(attachment_index, int):
        deps.json_response(handler, 400, {"error": "attachment_index must be an integer"})
        return
    if not isinstance(data_b64, str) or not data_b64:
        deps.json_response(handler, 400, {"error": "data_b64 required"})
        return
    try:
        ready_for_attachment = manager.attachment_staging_ready(session_id)
    except KeyError:
        deps.json_response(handler, 404, {"error": "unknown session"})
        return
    except deps.session_not_ready_error as e:
        deps.json_response(handler, 409, {"error": str(e)})
        return
    except Exception:
        deps.json_response(handler, 409, {"error": "session state unavailable; wait before attaching a file"})
        return
    if not ready_for_attachment:
        deps.json_response(handler, 409, {"error": "session is busy; wait before attaching a file"})
        return
    try:
        raw = base64.b64decode(data_b64.encode("ascii"), validate=True)
    except Exception:
        deps.json_response(handler, 400, {"error": "invalid base64"})
        return
    try:
        out_path = deps.stage_uploaded_file(session_id, filename, raw)
    except ValueError as e:
        status = 413 if str(e).startswith("file too large") else 400
        deps.json_response(handler, status, {"error": str(e)})
        return
    try:
        res = manager.add_staged_attachment(
            session_id,
            display_name=filename,
            filename=out_path.name,
            path=out_path,
            size=len(raw),
            created_ts=time.time(),
        )
    except KeyError:
        try:
            out_path.unlink()
        except OSError:
            pass
        deps.json_response(handler, 404, {"error": "unknown session"})
        return
    except ValueError as e:
        try:
            out_path.unlink()
        except OSError:
            pass
        deps.json_response(handler, 400, {"error": str(e)})
        return
    deps.json_response(handler, 200, {"ok": True, "path": str(out_path), **res})
