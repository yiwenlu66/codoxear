from __future__ import annotations

import json
from typing import Any

_SERVER = None


def bind_server_runtime(runtime: Any) -> None:
    global _SERVER
    _SERVER = runtime



def _sv() -> Any:
    if _SERVER is None:
        raise RuntimeError("server runtime not bound")
    return _SERVER



def handle_post(handler: Any, path: str, _u: Any) -> bool:
    sv = _sv()
    if path == "/api/cwd_groups/edit":
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        try:
            body = sv._read_body(handler)
            body_text = body.decode("utf-8")
            if not body_text.strip():
                raise ValueError("empty request body")
            obj = json.loads(body_text)
            if not isinstance(obj, dict):
                raise ValueError("invalid json body (expected object)")
            cwd, entry = sv.MANAGER.cwd_group_set(
                cwd=obj.get("cwd"),
                label=obj.get("label"),
                collapsed=obj.get("collapsed"),
            )
        except ValueError as e:
            sv._json_response(handler, 400, {"error": str(e)})
            return True
        sv._json_response(handler, 200, {"ok": True, "cwd": cwd, **entry})
        return True
    if path == "/api/sessions":
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        body = sv._read_body(handler)
        body_text = body.decode("utf-8")
        if not body_text.strip():
            raise ValueError("empty request body")
        obj = json.loads(body_text)
        if not isinstance(obj, dict):
            raise ValueError("invalid json body (expected object)")
        try:
            payload = sv._parse_create_session_request(obj)
        except ValueError as e:
            err = str(e)
            out = {"error": err}
            if err == "cwd required":
                out["field"] = "cwd"
            sv._json_response(handler, 400, out)
            return True
        try:
            res = sv.MANAGER.spawn_web_session(
                cwd=payload["cwd"],
                args=payload["args"],
                resume_session_id=payload["resume_session_id"],
                worktree_branch=payload["worktree_branch"],
                model_provider=payload["model_provider"],
                preferred_auth_method=payload["preferred_auth_method"],
                model=payload["model"],
                reasoning_effort=payload["reasoning_effort"],
                service_tier=payload["service_tier"],
                create_in_tmux=payload["create_in_tmux"],
                backend=payload["backend"],
            )
            alias = sv.MANAGER.set_created_session_name(
                session_id=res.get("session_id"),
                runtime_id=res.get("runtime_id"),
                backend=res.get("backend") or payload["backend"],
                name=payload["name"],
            )
        except ValueError as e:
            response_payload: dict[str, Any] = {"error": str(e)}
            if str(e).startswith("cwd "):
                response_payload["field"] = "cwd"
            sv._json_response(handler, 400, response_payload)
            return True
        except KeyError:
            sv._json_response(handler, 404, {"error": "unknown session"})
            return True
        response_payload = {"ok": True, **res}
        if alias:
            response_payload["alias"] = alias
        sv._publish_sessions_invalidate(reason="session_created")
        sv._json_response(handler, 200, response_payload)
        return True
    session_id = sv._match_session_route(path, "delete")
    if session_id is not None:
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        sv._read_body(handler)
        ok = sv.MANAGER.delete_session(session_id)
        if not ok:
            sv._json_response(handler, 404, {"error": "unknown session"})
            return True
        sv._publish_sessions_invalidate(reason="session_deleted")
        sv._json_response(handler, 200, {"ok": True})
        return True
    if path.startswith("/api/sessions/") and path.endswith("/edit"):
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        parts = path.split("/")
        session_id = parts[3] if len(parts) >= 4 else ""
        body = sv._read_body(handler)
        body_text = body.decode("utf-8")
        if not body_text.strip():
            raise ValueError("empty request body")
        obj = json.loads(body_text)
        if not isinstance(obj, dict):
            raise ValueError("invalid json body (expected object)")
        name = obj.get("name")
        if not isinstance(name, str):
            sv._json_response(handler, 400, {"error": "name required"})
            return True
        try:
            alias, sidebar_meta = sv.MANAGER.edit_session(
                session_id,
                name=name,
                priority_offset=obj.get("priority_offset"),
                snooze_until=obj.get("snooze_until"),
                dependency_session_id=obj.get("dependency_session_id"),
            )
        except KeyError:
            sv._json_response(handler, 404, {"error": "unknown session"})
            return True
        except ValueError as e:
            sv._json_response(handler, 400, {"error": str(e)})
            return True
        sv._json_response(handler, 200, {"ok": True, "alias": alias, **sidebar_meta})
        return True
    if path.startswith("/api/sessions/") and path.endswith("/rename"):
        session_id = sv._match_session_route(path, "rename")
        if session_id is None:
            sv._json_response(handler, 404, {"error": "unknown session"})
            return True
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        body = sv._read_body(handler)
        body_text = body.decode("utf-8")
        if not body_text.strip():
            raise ValueError("empty request body")
        obj = json.loads(body_text)
        if not isinstance(obj, dict):
            raise ValueError("invalid json body (expected object)")
        name = obj.get("name")
        if not isinstance(name, str):
            sv._json_response(handler, 400, {"error": "name required"})
            return True
        try:
            alias = sv.MANAGER.alias_set(session_id, name)
        except KeyError:
            sv._json_response(handler, 404, {"error": "unknown session"})
            return True
        sv._json_response(handler, 200, {"ok": True, "alias": alias})
        return True
    if path.startswith("/api/sessions/") and path.endswith("/focus"):
        session_id = sv._match_session_route(path, "focus")
        if session_id is None:
            sv._json_response(handler, 404, {"error": "unknown session"})
            return True
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        body = sv._read_body(handler)
        body_text = body.decode("utf-8")
        if not body_text.strip():
            raise ValueError("empty request body")
        obj = json.loads(body_text)
        if not isinstance(obj, dict):
            raise ValueError("invalid json body (expected object)")
        try:
            focused = sv.MANAGER.focus_set(session_id, obj.get("focused"))
        except KeyError:
            sv._json_response(handler, 404, {"error": "unknown session"})
            return True
        except ValueError as e:
            sv._json_response(handler, 400, {"error": str(e)})
            return True
        sv._json_response(handler, 200, {"ok": True, "focused": focused})
        return True
    if path.startswith("/api/sessions/") and path.endswith("/send"):
        session_id = sv._match_session_route(path, "send")
        if session_id is None:
            sv._json_response(handler, 404, {"error": "unknown session"})
            return True
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        body = sv._read_body(handler)
        body_text = body.decode("utf-8")
        if not body_text.strip():
            raise ValueError("empty request body")
        obj = json.loads(body_text)
        if not isinstance(obj, dict):
            raise ValueError("invalid json body (expected object)")
        text = obj.get("text")
        if not isinstance(text, str) or not text.strip():
            sv._json_response(handler, 400, {"error": "text required"})
            return True
        try:
            res = sv.MANAGER.send(session_id, text)
        except KeyError:
            sv._json_response(handler, 404, {"error": "unknown session"})
            return True
        except ValueError as e:
            sv._json_response(handler, 502, {"error": str(e)})
            return True
        sv._json_response(handler, 200, res)
        return True
    if path.startswith("/api/sessions/") and path.endswith("/ui_response"):
        session_id = sv._match_session_route(path, "ui_response")
        if session_id is None:
            sv._json_response(handler, 404, {"error": "unknown session"})
            return True
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        body = sv._read_body(handler)
        body_text = body.decode("utf-8")
        if not body_text.strip():
            raise ValueError("empty request body")
        obj = json.loads(body_text)
        if not isinstance(obj, dict):
            raise ValueError("invalid json body (expected object)")
        try:
            sv.MANAGER.submit_ui_response(session_id, obj)
        except KeyError:
            sv._json_response(handler, 404, {"error": "unknown session"})
            return True
        except ValueError as e:
            sv._json_response(handler, 502, {"error": str(e)})
            return True
        durable_session_id = sv.MANAGER._durable_session_id_for_identifier(session_id) or session_id
        runtime_id = sv.MANAGER._runtime_session_id_for_identifier(session_id)
        sv._publish_session_workspace_invalidate(
            durable_session_id,
            runtime_id=runtime_id,
            reason="ui_response",
        )
        sv._json_response(handler, 200, {"ok": True})
        return True
    if path.startswith("/api/sessions/") and path.endswith("/enqueue"):
        session_id = sv._match_session_route(path, "enqueue")
        if session_id is None:
            sv._json_response(handler, 404, {"error": "unknown session"})
            return True
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        body = sv._read_body(handler)
        body_text = body.decode("utf-8")
        if not body_text.strip():
            raise ValueError("empty request body")
        obj = json.loads(body_text)
        if not isinstance(obj, dict):
            raise ValueError("invalid json body (expected object)")
        text = obj.get("text")
        if not isinstance(text, str) or not text.strip():
            sv._json_response(handler, 400, {"error": "text required"})
            return True
        try:
            res = sv.MANAGER.enqueue(session_id, text)
        except KeyError:
            sv._json_response(handler, 404, {"error": "unknown session"})
            return True
        except ValueError as e:
            sv._json_response(handler, 502, {"error": str(e)})
            return True
        sv._json_response(handler, 200, res)
        return True
    if path.startswith("/api/sessions/") and path.endswith("/queue/delete"):
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        parts = path.split("/")
        session_id = parts[3] if len(parts) >= 4 else ""
        body = sv._read_body(handler)
        body_text = body.decode("utf-8")
        if not body_text.strip():
            raise ValueError("empty request body")
        obj = json.loads(body_text)
        if not isinstance(obj, dict):
            raise ValueError("invalid json body (expected object)")
        idx = obj.get("index")
        if not isinstance(idx, int):
            sv._json_response(handler, 400, {"error": "index required"})
            return True
        try:
            res = sv.MANAGER.queue_delete(session_id, idx)
        except KeyError:
            sv._json_response(handler, 404, {"error": "unknown session"})
            return True
        except ValueError as e:
            sv._json_response(handler, 502, {"error": str(e)})
            return True
        sv._json_response(handler, 200, res)
        return True
    if path.startswith("/api/sessions/") and path.endswith("/queue/update"):
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        parts = path.split("/")
        session_id = parts[3] if len(parts) >= 4 else ""
        body = sv._read_body(handler)
        body_text = body.decode("utf-8")
        if not body_text.strip():
            raise ValueError("empty request body")
        obj = json.loads(body_text)
        if not isinstance(obj, dict):
            raise ValueError("invalid json body (expected object)")
        idx = obj.get("index")
        text = obj.get("text")
        if not isinstance(idx, int):
            sv._json_response(handler, 400, {"error": "index required"})
            return True
        if not isinstance(text, str) or not text.strip():
            sv._json_response(handler, 400, {"error": "text required"})
            return True
        try:
            res = sv.MANAGER.queue_update(session_id, idx, text)
        except KeyError:
            sv._json_response(handler, 404, {"error": "unknown session"})
            return True
        except ValueError as e:
            sv._json_response(handler, 502, {"error": str(e)})
            return True
        sv._json_response(handler, 200, res)
        return True
    if path.startswith("/api/sessions/") and path.endswith("/harness"):
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        parts = path.split("/")
        session_id = parts[3] if len(parts) >= 4 else ""
        body = sv._read_body(handler)
        body_text = body.decode("utf-8")
        if not body_text.strip():
            raise ValueError("empty request body")
        obj = json.loads(body_text)
        if not isinstance(obj, dict):
            raise ValueError("invalid json body (expected object)")
        enabled_raw = obj.get("enabled", None)
        request_raw = obj.get("request", None)
        cooldown_minutes_raw = obj.get("cooldown_minutes", None)
        remaining_injections_raw = obj.get("remaining_injections", None)
        if "text" in obj:
            sv._json_response(handler, 400, {"error": "unknown field: text (use request)"})
            return True
        enabled = None if enabled_raw is None else bool(enabled_raw)
        if request_raw is not None and not isinstance(request_raw, str):
            sv._json_response(handler, 400, {"error": "request must be a string"})
            return True
        request = request_raw if request_raw is not None else None
        if cooldown_minutes_raw is not None:
            try:
                cooldown_minutes = sv._clean_harness_cooldown_minutes(cooldown_minutes_raw)
            except ValueError as e:
                sv._json_response(handler, 400, {"error": str(e)})
                return True
        else:
            cooldown_minutes = None
        if remaining_injections_raw is not None:
            try:
                remaining_injections = sv._clean_harness_remaining_injections(remaining_injections_raw, allow_zero=True)
            except ValueError as e:
                sv._json_response(handler, 400, {"error": str(e)})
                return True
        else:
            remaining_injections = None
        cfg = sv.MANAGER.harness_set(
            session_id,
            enabled=enabled,
            request=request,
            cooldown_minutes=cooldown_minutes,
            remaining_injections=remaining_injections,
        )
        sv._json_response(handler, 200, {"ok": True, **cfg})
        return True
    if path.startswith("/api/sessions/") and path.endswith("/interrupt"):
        if not sv._require_auth(handler):
            handler._unauthorized()
            return True
        parts = path.split("/")
        session_id = parts[3] if len(parts) >= 4 else ""
        sv._read_body(handler)
        try:
            resp = sv.MANAGER.inject_keys(session_id, "\\x1b")
        except KeyError:
            sv._json_response(handler, 404, {"error": "unknown session"})
            return True
        except ValueError as e:
            sv._json_response(handler, 502, {"error": str(e)})
            return True
        durable_session_id = sv.MANAGER._durable_session_id_for_identifier(session_id) or session_id
        runtime_id = sv.MANAGER._runtime_session_id_for_identifier(session_id)
        sv._publish_session_live_invalidate(
            durable_session_id,
            runtime_id=runtime_id,
            reason="interrupt",
        )
        sv._json_response(handler, 200, {"ok": True, "broker": resp})
        return True
    return False
