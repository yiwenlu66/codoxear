from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

from .runtime_access import manager_runtime


def _runtime(manager: Any):
    return manager_runtime(manager)


def sock_call(
    manager: Any, sock_path: Path, req: dict[str, Any], timeout_s: float = 2.0
) -> dict[str, Any]:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout_s)
    try:
        s.connect(str(sock_path))
        s.sendall((json.dumps(req) + "\n").encode("utf-8"))
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
        line = buf.split(b"\n", 1)[0]
        if not line:
            return {"error": "empty response"}
        payload = json.loads(line.decode("utf-8"))
        return payload if isinstance(payload, dict) else {"error": "invalid response"}
    finally:
        s.close()


def kill_session_via_pids(manager: Any, session: Any) -> bool:
    sv = _runtime(manager)
    group_alive = sv._process_group_alive(int(session.codex_pid))
    broker_alive = sv._pid_alive(int(session.broker_pid))
    if not group_alive and not broker_alive:
        sv._unlink_quiet(session.sock_path)
        sv._unlink_quiet(session.sock_path.with_suffix(".json"))
        return True
    if group_alive and (
        not sv._terminate_process_group(int(session.codex_pid), wait_seconds=1.0)
    ):
        return False
    if sv._pid_alive(int(session.broker_pid)) and (
        not sv._terminate_process(int(session.broker_pid), wait_seconds=1.0)
    ):
        return False
    group_dead = not sv._process_group_alive(int(session.codex_pid))
    broker_dead = not sv._pid_alive(int(session.broker_pid))
    if group_dead and broker_dead:
        sv._unlink_quiet(session.sock_path)
        sv._unlink_quiet(session.sock_path.with_suffix(".json"))
        return True
    return False


def kill_session(manager: Any, session_id: str) -> bool:
    runtime_id = manager._runtime_session_id_for_identifier(session_id)
    if runtime_id is None:
        return False
    with manager._lock:
        session = manager._sessions.get(runtime_id)
    if not session:
        return False
    try:
        resp = manager._sock_call(session.sock_path, {"cmd": "shutdown"}, timeout_s=1.0)
    except Exception:
        return manager._kill_session_via_pids(session)
    if resp.get("ok") is True:
        return True
    return manager._kill_session_via_pids(session)


def get_state(manager: Any, session_id: str) -> dict[str, Any]:
    sv = _runtime(manager)
    runtime_id = manager._runtime_session_id_for_identifier(session_id)
    if runtime_id is None:
        raise KeyError("unknown session")
    with manager._lock:
        session = manager._sessions.get(runtime_id)
        if not session:
            raise KeyError("unknown session")
        sock = session.sock_path
    cached_state = {
        "busy": bool(session.busy),
        "queue_len": int(session.queue_len),
        "token": session.token,
    }
    try:
        resp = manager._sock_call(sock, {"cmd": "state"}, timeout_s=1.5)
        sv._validated_session_state(resp)
    except Exception:
        if not sv._pid_alive(session.broker_pid) and not sv._pid_alive(session.codex_pid):
            with manager._lock:
                manager._sessions.pop(runtime_id, None)
            manager._clear_deleted_session_state(runtime_id)
            sv._unlink_quiet(sock)
            sv._unlink_quiet(sock.with_suffix(".json"))
            raise KeyError("unknown session")
        return cached_state
    with manager._lock:
        session2 = manager._sessions.get(runtime_id)
        if session2:
            session2.busy = sv._state_busy_value(resp)
            session2.queue_len = sv._state_queue_len_value(resp)
            if "token" in resp:
                tok = resp.get("token")
                if isinstance(tok, dict):
                    session2.token = tok
            return resp
    return cached_state


def get_tail(manager: Any, session_id: str) -> str:
    sv = _runtime(manager)
    runtime_id = manager._runtime_session_id_for_identifier(session_id)
    if runtime_id is None:
        raise KeyError("unknown session")
    with manager._lock:
        session = manager._sessions.get(runtime_id)
        if not session:
            raise KeyError("unknown session")
        sock = session.sock_path
    try:
        resp = manager._sock_call(sock, {"cmd": "tail"}, timeout_s=1.5)
    except Exception:
        if not sv._pid_alive(session.broker_pid) and not sv._pid_alive(session.codex_pid):
            with manager._lock:
                manager._sessions.pop(runtime_id, None)
            sv._unlink_quiet(sock)
            sv._unlink_quiet(sock.with_suffix(".json"))
            raise KeyError("unknown session")
        raise
    if "tail" not in resp:
        raise ValueError("invalid broker tail response")
    tail = resp.get("tail")
    if not isinstance(tail, str):
        raise ValueError("invalid broker tail response")
    return tail


def inject_keys(manager: Any, session_id: str, seq: str) -> dict[str, Any]:
    sv = _runtime(manager)
    runtime_id = manager._runtime_session_id_for_identifier(session_id)
    if runtime_id is None:
        raise KeyError("unknown session")
    with manager._lock:
        session = manager._sessions.get(runtime_id)
        if not session:
            raise KeyError("unknown session")
        sock = session.sock_path
    try:
        resp = manager._sock_call(sock, {"cmd": "keys", "seq": seq}, timeout_s=2.0)
    except Exception:
        if not sv._pid_alive(session.broker_pid) and not sv._pid_alive(session.codex_pid):
            with manager._lock:
                manager._sessions.pop(runtime_id, None)
            sv._unlink_quiet(sock)
            sv._unlink_quiet(sock.with_suffix(".json"))
            raise KeyError("unknown session")
        raise
    err = resp.get("error")
    if isinstance(err, str) and err:
        raise ValueError(err)
    return resp
