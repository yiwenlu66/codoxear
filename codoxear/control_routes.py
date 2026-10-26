from __future__ import annotations

import base64
import re
import json
import sys
from urllib.parse import parse_qs, unquote, urlsplit
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable

from .attachment_resource import AttachmentPersistenceUncertain, upload_id_value
from .quick_workspace import release_workspace
from .server_http import BadRequestError, RequestPayloadTooLargeError
from .session_store import public_staged_attachment


JsonResponse = Callable[[Any, int, dict[str, Any]], None]
RouteMatcher = Callable[..., str | None]
ReadBody = Callable[..., bytes]
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


def _project_attachment_value(value: Any) -> Any:
    if isinstance(value, dict) and "path" in value:
        projected = public_staged_attachment(value)
        if projected is not None:
            return projected
        redacted = dict(value)
        redacted.pop("path", None)
        return redacted
    return value


def _project_attachment_response(res: dict[str, Any]) -> dict[str, Any]:
    projected = dict(res)
    if isinstance(projected.get("attachments"), list):
        projected["attachments"] = [_project_attachment_value(item) for item in projected["attachments"]]
    if "attachment" in projected:
        projected["attachment"] = _project_attachment_value(projected["attachment"])
    if "removed" in projected:
        projected["removed"] = _project_attachment_value(projected["removed"])
    projected.pop("path", None)
    return projected


def handle_control_get_route(
    handler: Any,
    *,
    path: str,
    manager: Any,
    deps: ControlRouteDeps,
    match_session_route: RouteMatcher,
) -> bool:
    status_id = match_session_route(path, "attachments", "upload-status")
    session_id = status_id if status_id is not None else match_session_route(path, "attachments")
    if session_id is None:
        return False
    if not _authorized(handler, deps):
        return True
    try:
        if status_id is not None:
            handler._codoxear_response_no_store = True
            query = parse_qs(urlsplit(handler.path).query)
            upload_id = upload_id_value(query.get("upload_id", [None])[0])
            res = manager.upload_status(session_id, upload_id)
        else:
            res = manager.list_staged_attachments(session_id)
    except KeyError:
        deps.json_response(handler, 404, {"error": "unknown session"})
        return True
    except ValueError as exc:
        deps.json_response(handler, 400, {"error": str(exc)})
        return True
    deps.json_response(handler, 200, _project_attachment_response(res))
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
        ("attachments", "upload-cancel", _handle_upload_cancel),
        ("commit_unknown_send", "clear", _handle_commit_unknown_send_clear),
        ("send", None, _handle_send),
        ("settings", None, _handle_settings),
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
    session = manager.get_session(session_id)
    session_cwd = getattr(session, "cwd", "") if session is not None else ""
    ok = manager.delete_session(session_id)
    if not ok:
        deps.json_response(handler, 404, {"error": "unknown session"})
        return
    # A quick workspace is machine-managed scratch space, so it is trimmed once
    # the session that owned it is gone (rmdir only when empty, never recursive).
    if isinstance(session_cwd, str) and session_cwd:
        release_workspace(session_cwd)
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
    deps.json_response(handler, 200, _project_attachment_response(res))


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
    deps.json_response(handler, 200, _project_attachment_response(res))


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
    deps.json_response(handler, 200, _project_attachment_response(res))


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
    raw_send_id = obj.get("send_id")
    send_id = raw_send_id.strip() if isinstance(raw_send_id, str) else None
    if not isinstance(send_id, str) or re.fullmatch(r"[A-Za-z0-9._:-]{1,120}", send_id) is None:
        send_id = None
    try:
        send_kwargs: dict[str, Any] = {"allow_pending_attachment": allow_pending_attachment}
        if send_id is not None:
            send_kwargs["send_id"] = send_id
        res = manager.send(session_id, text, **send_kwargs)
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


def _handle_settings(handler: Any, *, session_id: str, manager: Any, deps: ControlRouteDeps) -> None:
    if not _authorized(handler, deps):
        return
    obj = deps.read_json_body(handler)
    model_raw = obj.get("model")
    effort_raw = obj.get("effort")
    model = model_raw.strip() if isinstance(model_raw, str) and model_raw.strip() else None
    effort = effort_raw.strip().lower() if isinstance(effort_raw, str) and effort_raw.strip() else None
    if bool(model) == bool(effort):
        deps.json_response(handler, 400, {"error": "exactly one of model or effort is required"})
        return
    try:
        result = manager.update_live_settings(session_id, model=model, effort=effort)
    except KeyError:
        deps.json_response(handler, 404, {"error": "unknown session"})
        return
    except ValueError as exc:
        deps.json_response(handler, 409, {"error": str(exc)})
        return
    except Exception as exc:
        deps.json_response(handler, 502, {"error": f"Codex live settings failed: {exc}"})
        return
    deps.json_response(handler, 200, result)


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


def _handle_upload_cancel(handler: Any, *, session_id: str, manager: Any, deps: ControlRouteDeps) -> None:
    if not _authorized(handler, deps):
        return
    obj = deps.read_json_body(handler)
    try:
        upload_id = upload_id_value(obj.get("upload_id"))
        result = manager.cancel_upload(session_id, upload_id)
    except KeyError:
        deps.json_response(handler, 404, {"error": "unknown session"})
        return
    except ValueError as exc:
        deps.json_response(handler, 400, {"error": str(exc)})
        return
    deps.json_response(handler, 200, _project_attachment_response(result))


def _handle_inject_attachment(handler: Any, *, session_id: str, manager: Any, deps: ControlRouteDeps) -> None:
    if not _authorized(handler, deps):
        return
    started = time.monotonic()
    upload_id = None
    writer = False
    expected = None
    received = 0
    phase = "validation"
    outcome = "uncommitted"
    error_category = None
    raw_mode = False
    try:
        raw_mode = str(getattr(handler, "headers", {}).get("Content-Type", "")).split(";", 1)[0].strip().lower() == "application/octet-stream"
        if raw_mode:
            upload_id = upload_id_value(handler.headers.get("X-Codoxear-Upload-Id"))
            encoded_name = handler.headers.get("X-Codoxear-Filename")
            if not isinstance(encoded_name, str) or re.search(r"%(?![0-9a-fA-F]{2})", encoded_name):
                raise ValueError("invalid filename encoding")
            filename = unquote(encoded_name, encoding="utf-8", errors="strict")
            length = handler.headers.get("Content-Length")
            if length is None or re.fullmatch(r"[0-9]+", str(length).strip()) is None:
                raise ValueError("known content-length required")
            expected = int(length)
            if expected > deps.attach_upload_max_bytes:
                raise RequestPayloadTooLargeError(f"file too large (max {deps.attach_upload_max_bytes} bytes)")
        else:
            phase = "receiving"
            obj = deps.read_json_body(handler, limit=deps.attach_upload_body_max_bytes,
                                      too_large_error=f"file too large (max {deps.attach_upload_max_bytes} bytes)")
            phase = "validation"
            filename = obj.get("filename")
            if not isinstance(filename, str) or not filename.strip():
                raise ValueError("filename required")
            if "upload_id" in obj:
                upload_id = upload_id_value(obj["upload_id"])
            data_b64 = obj.get("data_b64")
            if not isinstance(data_b64, str) or not data_b64:
                raise ValueError("data_b64 required")
            if upload_id is None:
                _require_attachment_ready(manager, session_id, deps)
            try:
                raw = base64.b64decode(data_b64.encode("ascii"), validate=True)
            except Exception as exc:
                raise ValueError("invalid base64") from exc
            expected = received = len(raw)
            if expected > deps.attach_upload_max_bytes:
                raise RequestPayloadTooLargeError(f"file too large (max {deps.attach_upload_max_bytes} bytes)")
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("filename required")
        if len(filename) > 256 or any(ord(ch) < 32 or ch in "/\\" for ch in filename) or filename in (".", ".."):
            raise ValueError("invalid filename")
        if upload_id is not None:
            # Committed/retired IDs remain recoverable even if the session is
            # temporarily not ready for a new upload.
            existing = manager.upload_status(session_id, upload_id)
            if existing["state"] in ("absent", "failed"):
                _require_attachment_ready(manager, session_id, deps)
            reservation = manager.begin_upload(session_id, upload_id, filename=filename, size=expected)
            writer = reservation.pop("writer")
        phase = "receiving"
        if raw_mode:
            raw = deps.read_body(handler, limit=deps.attach_upload_max_bytes)
            received = len(raw)
            if received != expected:
                raise BadRequestError("incomplete upload body")
        phase = "commit"
        stage = lambda: deps.stage_uploaded_file(session_id, filename, raw)
        if upload_id is not None:
            if writer or reservation["state"] == "committed":
                res = manager.finish_upload(session_id, upload_id, filename=filename, raw=raw, stage=stage)
            else:
                res = manager.upload_status(session_id, upload_id)
            outcome = res["state"]
        else:
            out_path = stage()
            try:
                res = manager.add_staged_attachment(session_id, display_name=filename, filename=out_path.name,
                                                    path=out_path, size=len(raw), created_ts=time.time())
            except AttachmentPersistenceUncertain:
                # The envelope already references these bytes, even though its
                # directory fsync could not confirm crash durability.
                raise
            except Exception:
                out_path.unlink(missing_ok=True)
                raise
            outcome = "committed"
        phase = "response"
        deps.json_response(handler, 200, _project_attachment_response(res))
    except Exception as exc:
        error_category = type(exc).__name__
        if raw_mode and expected is not None:
            remaining = getattr(handler, "_codoxear_body_remaining", None)
            if isinstance(remaining, int):
                received = expected - remaining
        if writer and outcome != "committed" and not isinstance(exc, AttachmentPersistenceUncertain):
            try:
                manager.fail_upload(session_id, upload_id)
            except (KeyError, OSError):
                # Status reconciliation remains authoritative if recording the
                # failed receive itself could not be persisted.
                pass
        if isinstance(exc, OSError) and phase in ("validation", "commit"):
            outcome = "unknown" if isinstance(exc, AttachmentPersistenceUncertain) else "uncommitted"
            deps.json_response(handler, 503, {"error": "upload storage unavailable; check upload status", "upload_id": upload_id, "stage": phase})
        elif isinstance(exc, KeyError):
            deps.json_response(handler, 404, {"error": "unknown session"})
        elif isinstance(exc, deps.session_not_ready_error):
            deps.json_response(handler, 409, {"error": str(exc)})
        elif isinstance(exc, RequestPayloadTooLargeError):
            deps.json_response(handler, 413, {"error": str(exc)})
        elif isinstance(exc, ValueError):
            status = 413 if str(exc).startswith("file too large") else 409 if str(exc).startswith("upload_id reused") else 400
            deps.json_response(handler, status, {"error": str(exc)})
        else:
            raise
    finally:
        # Bounded metadata only: no filenames, paths, request data or auth.
        print(json.dumps({"event": "attachment_upload", "upload_id": upload_id,
                          "stage": phase, "expected_bytes": expected,
                          "received_bytes": received, "commit_outcome": outcome,
                          "duration_ms": round((time.monotonic() - started) * 1000),
                          "error_category": error_category}), file=sys.stderr)


def _require_attachment_ready(manager: Any, session_id: str, deps: ControlRouteDeps) -> None:
    try:
        ready = manager.attachment_staging_ready(session_id)
    except (KeyError, deps.session_not_ready_error):
        raise
    except Exception as exc:
        raise deps.session_not_ready_error("session state unavailable; wait before attaching a file") from exc
    if not ready:
        raise deps.session_not_ready_error("wait for the current queue drain before attaching a file")
