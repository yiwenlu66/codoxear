from __future__ import annotations

import time
from pathlib import Path
from typing import Any

_SERVER = None


def bind_server_runtime(runtime) -> None:
    global _SERVER
    _SERVER = runtime



def _sv():
    if _SERVER is None:
        raise RuntimeError("server runtime not bound")
    return _SERVER



def get_ui_state(manager, session_id: str) -> dict[str, Any]:
    sv = _sv()
    runtime_id = manager._runtime_session_id_for_identifier(session_id)
    if runtime_id is None:
        raise KeyError("unknown session")
    manager.refresh_session_meta(runtime_id, strict=False)
    with manager._lock:
        s = manager._sessions.get(runtime_id)
        if not s:
            raise KeyError("unknown session")
        if s.backend != "pi":
            raise ValueError("ui interactions are only supported for pi sessions")
        sock = s.sock_path
    try:
        resp = manager._sock_call(sock, {"cmd": "ui_state"}, timeout_s=1.5)
    except Exception:
        if not sv._pid_alive(s.broker_pid) and not sv._pid_alive(s.codex_pid):
            with manager._lock:
                manager._sessions.pop(runtime_id, None)
            manager._clear_deleted_session_state(runtime_id)
            sv._unlink_quiet(sock)
            sv._unlink_quiet(sock.with_suffix(".json"))
            raise KeyError("unknown session")
        raise ValueError("broker unavailable")
    if resp.get("error") == "unknown cmd":
        if sv._session_supports_live_pi_ui(s):
            raise ValueError("live ui interactions are unavailable for this pi session")
        return {"requests": []}
    error = resp.get("error")
    if isinstance(error, str) and error:
        raise ValueError(error)
    return sv._sanitize_pi_ui_state_payload(resp)



def get_session_commands(manager, session_id: str) -> dict[str, Any]:
    sv = _sv()
    runtime_id = manager._runtime_session_id_for_identifier(session_id)
    if runtime_id is None:
        raise KeyError("unknown session")
    manager.refresh_session_meta(runtime_id, strict=False)
    with manager._lock:
        s = manager._sessions.get(runtime_id)
        if not s:
            raise KeyError("unknown session")
        if s.backend != "pi":
            raise ValueError("command listing is only supported for pi sessions")
        now_ts = time.time()
        session_path_key = str(s.session_path.resolve()) if isinstance(s.session_path, Path) else None
        cache = getattr(manager, "_pi_commands_cache", None)
        cached = cache.get(runtime_id) if isinstance(cache, dict) else None
        if isinstance(cached, dict):
            cached_ts = cached.get("ts")
            if isinstance(cached_ts, (int, float)) and (now_ts - float(cached_ts)) < sv.PI_COMMANDS_CACHE_TTL_SECONDS:
                if cached.get("thread_id") == s.thread_id and cached.get("session_path") == session_path_key:
                    commands = cached.get("commands")
                    if isinstance(commands, list):
                        return {"commands": list(commands)}
        sock = s.sock_path
        thread_id = s.thread_id
    try:
        resp = manager._sock_call(sock, {"cmd": "commands"}, timeout_s=2.0)
    except Exception:
        if not sv._pid_alive(s.broker_pid) and not sv._pid_alive(s.codex_pid):
            with manager._lock:
                manager._sessions.pop(runtime_id, None)
            manager._clear_deleted_session_state(runtime_id)
            sv._unlink_quiet(sock)
            sv._unlink_quiet(sock.with_suffix(".json"))
            raise KeyError("unknown session")
        raise ValueError("broker unavailable")
    if resp.get("error") == "unknown cmd":
        return {"commands": []}
    error = resp.get("error")
    if isinstance(error, str) and error:
        raise ValueError(error)
    payload = sv._sanitize_pi_commands_payload(resp)
    with manager._lock:
        cache = getattr(manager, "_pi_commands_cache", None)
        if isinstance(cache, dict):
            cache[runtime_id] = {
                "ts": time.time(),
                "thread_id": thread_id,
                "session_path": session_path_key,
                "commands": list(payload.get("commands", [])),
            }
    return payload



def submit_ui_response(manager, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    sv = _sv()
    runtime_id = manager._runtime_session_id_for_identifier(session_id)
    if runtime_id is None:
        raise KeyError("unknown session")
    manager.refresh_session_meta(runtime_id, strict=False)
    with manager._lock:
        s = manager._sessions.get(runtime_id)
        if not s:
            raise KeyError("unknown session")
        if s.backend != "pi":
            raise ValueError("ui interactions are only supported for pi sessions")
        sock = s.sock_path
    forward: dict[str, Any] = {"cmd": "ui_response"}
    for key in ("id", "value", "confirmed", "cancelled"):
        if key in payload:
            forward[key] = payload[key]
    try:
        resp = manager._sock_call(sock, forward, timeout_s=3.0)
    except Exception:
        if not sv._pid_alive(s.broker_pid) and not sv._pid_alive(s.codex_pid):
            with manager._lock:
                manager._sessions.pop(runtime_id, None)
            manager._clear_deleted_session_state(runtime_id)
            sv._unlink_quiet(sock)
            sv._unlink_quiet(sock.with_suffix(".json"))
            raise KeyError("unknown session")
        raise ValueError("broker unavailable")
    if resp.get("error") == "unknown cmd":
        if sv._session_supports_live_pi_ui(s):
            raise ValueError("live ui responses are unavailable for this pi session")
        if payload.get("cancelled") is True:
            manager.inject_keys(runtime_id, "\\x1b")
            return {"ok": True, "legacy_fallback": True}
        text = sv._legacy_pi_ui_response_text(payload)
        if text is None:
            raise ValueError("ui response value required")
        manager.send(runtime_id, text)
        return {"ok": True, "legacy_fallback": True}
    error = resp.get("error")
    if isinstance(error, str) and error:
        raise ValueError(error)
    return resp
