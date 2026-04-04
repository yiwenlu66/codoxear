#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .pi_rpc import PiRpcClient
from .util import _send_socket_json_line as _send_socket_json_line
from .util import _socket_peer_disconnected as _socket_peer_disconnected
from .util import default_app_dir as _default_app_dir


APP_DIR = _default_app_dir()
SOCK_DIR = APP_DIR / "socks"
PI_SESSION_DIR = APP_DIR / "pi-sessions"
OWNER_TAG = os.environ.get("CODEX_WEB_OWNER", "")

_TERMINAL_TURN_EVENT_TYPES = {
    "thread_rolled_back",
    "turn_end",
    "turn.aborted",
    "turn.completed",
    "turn.failed",
}

_DIALOG_UI_METHODS = {"select", "confirm", "input", "editor"}


def _seq_bytes(raw: str) -> bytes:
    try:
        return raw.encode("utf-8").decode("unicode_escape").encode("utf-8")
    except Exception:
        return raw.encode("utf-8")


def _extract_turn_id(obj: dict[str, Any]) -> str | None:
    for key in ("turn_id", "current_turn_id", "active_turn_id"):
        value = obj.get(key)
        if isinstance(value, str) and value:
            return value
    payload = obj.get("payload")
    if isinstance(payload, dict):
        return _extract_turn_id(payload)
    return None


def _extract_session_id(obj: dict[str, Any]) -> str | None:
    for key in ("session_id", "sessionId"):
        value = obj.get(key)
        if isinstance(value, str) and value:
            return value
    payload = obj.get("payload")
    if isinstance(payload, dict):
        return _extract_session_id(payload)
    return None


def _extract_busy(obj: dict[str, Any]) -> bool:
    busy = obj.get("busy")
    if isinstance(busy, bool):
        return busy
    streaming = obj.get("isStreaming")
    if isinstance(streaming, bool):
        return streaming
    return False


def _extract_event_type(obj: dict[str, Any]) -> str:
    event_type = obj.get("type")
    if isinstance(event_type, str) and event_type:
        return event_type
    payload = obj.get("payload")
    if isinstance(payload, dict):
        payload_type = payload.get("type")
        if isinstance(payload_type, str):
            return payload_type
    return ""


def _event_output_text(event: dict[str, Any]) -> str:
    event_type = _extract_event_type(event)
    if event_type == "message.delta":
        delta = event.get("delta")
        return delta if isinstance(delta, str) else ""

    text = event.get("text")
    if isinstance(text, str) and text:
        suffix = "" if text.endswith("\n") else "\n"
        if event_type == "turn.started":
            return f"> {text}{suffix}"
        return text + suffix

    if event_type == "tool.started":
        tool_name = event.get("tool_name")
        if isinstance(tool_name, str) and tool_name:
            return f"\n[tool] {tool_name}\n"

    return ""


def _record_ui_request(st: "State", event: dict[str, Any]) -> None:
    request_id = event.get("id")
    if not isinstance(request_id, str) or not request_id:
        return
    method = event.get("method")
    if not isinstance(method, str) or method not in _DIALOG_UI_METHODS:
        return

    options = event.get("options")
    timeout_ms = next(
        (
            event.get(key)
            for key in ("timeout_ms", "timeoutMs", "timeout")
            if isinstance(event.get(key), int)
        ),
        None,
    )
    allow_freeform = True
    if "allow_freeform" in event or "allowFreeform" in event:
        allow_freeform = bool(event.get("allow_freeform") if "allow_freeform" in event else event.get("allowFreeform"))
    allow_multiple = False
    if "allow_multiple" in event or "allowMultiple" in event:
        allow_multiple = bool(event.get("allow_multiple") if "allow_multiple" in event else event.get("allowMultiple"))

    st.pending_ui_requests[request_id] = {
        "id": request_id,
        "method": method,
        "title": event.get("title"),
        "message": event.get("message"),
        "question": event.get("question") if isinstance(event.get("question"), str) else None,
        "context": event.get("context") if isinstance(event.get("context"), str) else None,
        "options": list(options) if isinstance(options, list) else [],
        "allow_freeform": allow_freeform,
        "allow_multiple": allow_multiple,
        "timeout_ms": timeout_ms,
        "status": "pending",
    }


def _ask_user_request_id_from_message(message: Any) -> str | None:
    if not isinstance(message, dict):
        return None
    if message.get("role") != "toolResult":
        return None
    if message.get("toolName") != "ask_user":
        return None
    tool_call_id = message.get("toolCallId")
    return tool_call_id if isinstance(tool_call_id, str) and tool_call_id else None


def _resolved_ui_request_ids(event: dict[str, Any]) -> set[str]:
    resolved_ids: set[str] = set()

    def _collect(container: Any) -> None:
        if not isinstance(container, dict):
            return
        direct = _ask_user_request_id_from_message(container)
        if direct:
            resolved_ids.add(direct)
        message = _ask_user_request_id_from_message(container.get("message"))
        if message:
            resolved_ids.add(message)
        tool_results = container.get("toolResults")
        if isinstance(tool_results, list):
            for item in tool_results:
                tool_call_id = _ask_user_request_id_from_message(item)
                if tool_call_id:
                    resolved_ids.add(tool_call_id)

    _collect(event)
    payload = event.get("payload")
    if isinstance(payload, dict):
        _collect(payload)
    return resolved_ids


@dataclass
class State:
    session_id: str | None
    codex_pid: int
    sock_path: Path
    session_path: Path
    start_ts: float
    rpc: PiRpcClient
    busy: bool = False
    output_tail: str = ""
    output_tail_max: int = 64 * 1024
    token: dict[str, Any] | None = None
    last_turn_id: str | None = None
    backend: str = "pi"
    # Monotonic timestamp of last successful prompt RPC.  Used to prevent
    # _sync_state_from_rpc from prematurely clearing the busy flag before
    # Pi has acknowledged the new turn.
    prompt_sent_at: float = 0.0
    pending_ui_requests: dict[str, dict[str, Any]] = field(default_factory=dict)


class PiBroker:
    def __init__(self, *, cwd: str, session_path: Path | None = None, rpc: PiRpcClient | None = None) -> None:
        self.cwd = cwd
        self.session_path = session_path
        self.rpc = rpc
        self.state: State | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def _write_meta(self) -> None:
        with self._lock:
            st = self.state
        if not st:
            return
        meta = {
            "session_id": st.session_id,
            "backend": "pi",
            "owner": OWNER_TAG if OWNER_TAG else None,
            "broker_pid": os.getpid(),
            "agent_pid": st.codex_pid,
            "codex_pid": st.codex_pid,
            "cwd": self.cwd,
            "start_ts": float(st.start_ts),
            "log_path": None,
            "session_path": str(st.session_path),
            "sock_path": str(st.sock_path),
        }
        SOCK_DIR.mkdir(parents=True, exist_ok=True)
        meta_path = st.sock_path.with_suffix(".json")
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        os.chmod(meta_path, 0o600)

    def _sync_state_from_rpc(self) -> None:
        with self._lock:
            st = self.state
        if not st:
            return
        rpc_state: dict[str, Any] | None = None
        try:
            rpc_state = st.rpc.get_state()
        except Exception:
            rpc_state = None
        rewrite_meta = False
        with self._lock:
            st2 = self.state
            if not st2:
                return
            if isinstance(rpc_state, dict):
                rpc_busy = _extract_busy(rpc_state)
                # After a prompt is sent, Pi may not yet reflect busy=true
                # in its state.  Keep the broker-side busy flag set for a
                # grace period so that the frontend doesn't see a premature
                # idle dip between "prompt accepted" and "turn started".
                if st2.busy and not rpc_busy and st2.prompt_sent_at > 0:
                    elapsed = time.monotonic() - st2.prompt_sent_at
                    if elapsed < 5.0:
                        rpc_busy = True
                    else:
                        st2.prompt_sent_at = 0.0
                if rpc_busy:
                    st2.busy = True
                else:
                    st2.busy = False
                    st2.prompt_sent_at = 0.0
            self._drain_rpc_output_locked(st2)
            state_turn_id = _extract_turn_id(rpc_state) if isinstance(rpc_state, dict) else None
            if state_turn_id:
                st2.last_turn_id = state_turn_id
            sid = _extract_session_id(rpc_state) if isinstance(rpc_state, dict) else None
            if isinstance(sid, str) and sid and sid != st2.session_id:
                st2.session_id = sid
                rewrite_meta = True
        if rewrite_meta:
            self._write_meta()

    def _drain_rpc_output_locked(self, st: State) -> None:
        stderr_lines: list[str] = []
        drain_stderr = getattr(st.rpc, "drain_stderr_lines", None)
        if callable(drain_stderr):
            try:
                stderr_lines = [line for line in drain_stderr() if isinstance(line, str) and line]
            except Exception:
                stderr_lines = []
        for line in stderr_lines:
            suffix = "" if line.endswith("\n") else "\n"
            st.output_tail = (st.output_tail + f"[stderr] {line}{suffix}")[-st.output_tail_max :]
        try:
            events = st.rpc.drain_events()
        except Exception:
            events = []
        for event in events:
            event_type = _extract_event_type(event)
            if event_type == "extension_ui_request":
                _record_ui_request(st, event)
            for request_id in _resolved_ui_request_ids(event):
                st.pending_ui_requests.pop(request_id, None)
            if event_type in _TERMINAL_TURN_EVENT_TYPES:
                st.pending_ui_requests.clear()
            output = _event_output_text(event)
            if output:
                st.output_tail = (st.output_tail + output)[-st.output_tail_max :]
            event_turn_id = _extract_turn_id(event)
            if not event_turn_id:
                continue
            if event_type in _TERMINAL_TURN_EVENT_TYPES:
                if st.last_turn_id == event_turn_id:
                    st.last_turn_id = None
                continue
            st.last_turn_id = event_turn_id

    def _sync_output_from_rpc(self) -> None:
        with self._lock:
            st = self.state
            if not st:
                return
            self._drain_rpc_output_locked(st)

    def _get_state_snapshot(self) -> State | None:
        with self._lock:
            return self.state

    def _bg_sync_loop(self) -> None:
        """Background thread: periodically sync RPC state so socket handlers never block."""
        while not self._stop.is_set():
            try:
                self._sync_state_from_rpc()
            except Exception:
                pass
            self._stop.wait(1.0)

    def _close(self) -> None:
        self._stop.set()
        with self._lock:
            st = self.state
        if st is not None:
            st.rpc.close()

    def _handle_conn(self, conn: socket.socket) -> None:
        f = None
        try:
            f = conn.makefile("rb")
            line = f.readline()
            if not line:
                return
            req = json.loads(line.decode("utf-8"))
            cmd = req.get("cmd")

            if cmd == "state":
                with self._lock:
                    st = self.state
                    resp = {"busy": bool(st.busy) if st else False, "queue_len": 0, "token": st.token if st else None}
                _send_socket_json_line(conn, resp)
                return

            if cmd == "tail":
                with self._lock:
                    st = self.state
                    if st is not None:
                        self._drain_rpc_output_locked(st)
                    resp = {"tail": st.output_tail if st else ""}
                _send_socket_json_line(conn, resp)
                return

            if cmd == "ui_state":
                with self._lock:
                    st = self.state
                    if st is not None:
                        self._drain_rpc_output_locked(st)
                    resp = {
                        "requests": [
                            request
                            for request in st.pending_ui_requests.values()
                            if request.get("status") == "pending"
                        ]
                        if st
                        else [],
                    }
                _send_socket_json_line(conn, resp)
                return

            if cmd == "ui_response":
                request_id = req.get("id")
                if not isinstance(request_id, str) or not request_id:
                    _send_socket_json_line(conn, {"error": "id required"})
                    return
                ui_response_kwargs: dict[str, Any] = {}
                if bool(req.get("cancelled")):
                    ui_response_kwargs["cancelled"] = True
                else:
                    confirmed = req.get("confirmed")
                    if isinstance(confirmed, bool):
                        ui_response_kwargs["confirmed"] = confirmed
                    else:
                        ui_response_kwargs["value"] = req.get("value")
                with self._lock:
                    st = self.state
                    if not st:
                        _send_socket_json_line(conn, {"error": "no state"})
                        return
                    pending = st.pending_ui_requests.get(request_id)
                    if pending is None:
                        _send_socket_json_line(conn, {"error": "unknown or expired request"})
                        return
                    if pending.get("status") != "pending":
                        _send_socket_json_line(conn, {"error": "request already resolved"})
                        return
                    pending["status"] = "resolved"
                    rpc = st.rpc
                try:
                    rpc.send_ui_response(request_id, **ui_response_kwargs)
                except Exception:
                    with self._lock:
                        st = self.state
                        current = st.pending_ui_requests.get(request_id) if st else None
                        if current is pending and current.get("status") == "resolved":
                            current["status"] = "pending"
                    raise
                _send_socket_json_line(conn, {"ok": True})
                return

            if cmd == "send":
                text = req.get("text")
                if not isinstance(text, str) or not text.strip():
                    _send_socket_json_line(conn, {"error": "text required"})
                    return
                st = self._get_state_snapshot()
                if not st:
                    _send_socket_json_line(conn, {"error": "no state"})
                    return
                result = st.rpc.prompt(text)
                if not isinstance(result, dict):
                    _send_socket_json_line(conn, {"error": "invalid prompt response"})
                    return
                error = result.get("error")
                if isinstance(error, str) and error:
                    _send_socket_json_line(conn, {"error": error})
                    return
                with self._lock:
                    if self.state is st:
                        st.last_turn_id = _extract_turn_id(result) or st.last_turn_id
                        st.busy = True
                        st.prompt_sent_at = time.monotonic()
                _send_socket_json_line(conn, {"queued": False, "queue_len": 0})
                return

            if cmd == "keys":
                seq_raw = req.get("seq")
                if not isinstance(seq_raw, str) or not seq_raw:
                    _send_socket_json_line(conn, {"error": "seq required"})
                    return
                seq = _seq_bytes(seq_raw)
                with self._lock:
                    st = self.state
                    if not st:
                        _send_socket_json_line(conn, {"error": "no state"})
                        return
                if seq != b"\x1b":
                    _send_socket_json_line(conn, {"error": f"unsupported key sequence: {seq_raw}"})
                    return
                st = self._get_state_snapshot()
                if not st:
                    _send_socket_json_line(conn, {"error": "no state"})
                    return
                turn_id = st.last_turn_id if isinstance(st.last_turn_id, str) and st.last_turn_id else None
                # Pi RPC aborts the active turn without requiring a turn id.
                st.rpc.abort(turn_id)
                with self._lock:
                    if self.state is st:
                        st.busy = False
                _send_socket_json_line(conn, {"ok": True, "queued": False, "n": len(seq)})
                return

            if cmd == "shutdown":
                _send_socket_json_line(conn, {"ok": True})
                self._close()
                return

            _send_socket_json_line(conn, {"error": "unknown cmd"})
        except Exception as exc:
            if _socket_peer_disconnected(exc):
                return
            try:
                error = str(exc).strip() or "exception"
                payload = {"error": error}
                if error == "exception":
                    payload["trace"] = traceback.format_exc()
                _send_socket_json_line(conn, payload)
            except Exception:
                pass
        finally:
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass
            try:
                conn.close()
            except Exception:
                pass

    def _sock_server(self) -> None:
        with self._lock:
            st = self.state
        if not st:
            return
        SOCK_DIR.mkdir(parents=True, exist_ok=True)
        if st.sock_path.exists():
            st.sock_path.unlink()
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(st.sock_path))
        os.chmod(st.sock_path, 0o600)
        srv.listen(20)
        srv.settimeout(0.5)
        while not self._stop.is_set():
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            except Exception:
                break
            threading.Thread(target=self._handle_conn, args=(conn,), daemon=True).start()
        try:
            srv.close()
        except Exception:
            pass

    def run(self) -> int:
        SOCK_DIR.mkdir(parents=True, exist_ok=True)
        PI_SESSION_DIR.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        session_path = self.session_path or (PI_SESSION_DIR / f"{token}.jsonl")
        sock_path = SOCK_DIR / f"{token}.sock"
        rpc = self.rpc or PiRpcClient(cwd=self.cwd, session_path=session_path)
        self.state = State(
            session_id=None,
            codex_pid=rpc.pid or os.getpid(),
            sock_path=sock_path,
            session_path=session_path,
            start_ts=time.time(),
            rpc=rpc,
        )
        self._write_meta()
        threading.Thread(target=self._bg_sync_loop, name="pi-bg-sync", daemon=True).start()
        self._sock_server()
        return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cwd", required=True)
    ap.add_argument("--session-file")
    ns = ap.parse_args()
    session_path = Path(ns.session_file).expanduser().resolve() if ns.session_file else None
    raise SystemExit(PiBroker(cwd=ns.cwd, session_path=session_path).run())


if __name__ == "__main__":
    main()
