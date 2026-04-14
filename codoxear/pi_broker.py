#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import tempfile
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


def _tail_delta(previous: str, current: str) -> str:
    if not current:
        return ""
    if not previous:
        return current
    if current.startswith(previous):
        return current[len(previous) :]

    max_overlap = min(len(previous), len(current))
    for overlap in range(max_overlap, 0, -1):
        if previous[-overlap:] == current[:overlap]:
            return current[overlap:]
    return current


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
    allow_freeform = method in {"select", "input", "editor"}
    if "allow_freeform" in event or "allowFreeform" in event:
        allow_freeform = bool(
            event.get("allow_freeform")
            if "allow_freeform" in event
            else event.get("allowFreeform")
        )
    allow_multiple = False
    if "allow_multiple" in event or "allowMultiple" in event:
        allow_multiple = bool(
            event.get("allow_multiple")
            if "allow_multiple" in event
            else event.get("allowMultiple")
        )

    st.pending_ui_requests[request_id] = {
        "id": request_id,
        "method": method,
        "title": event.get("title"),
        "message": event.get("message"),
        "question": event.get("question")
        if isinstance(event.get("question"), str)
        else None,
        "context": event.get("context")
        if isinstance(event.get("context"), str)
        else None,
        "options": list(options) if isinstance(options, list) else [],
        "allow_freeform": allow_freeform,
        "allow_multiple": allow_multiple,
        "timeout_ms": timeout_ms,
        "status": "pending",
    }


def _resume_session_id_from_agent_args(args: list[str]) -> str | None:
    for idx, token in enumerate(args):
        if token != "--session":
            continue
        if (idx + 1) >= len(args):
            return None
        raw = str(args[idx + 1] or "").strip()
        if (not raw) or raw.endswith(".jsonl"):
            return None
        return raw
    return None


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
    session_path: Path | None
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
    def __init__(
        self,
        *,
        cwd: str,
        session_path: Path | None = None,
        rpc: PiRpcClient | None = None,
        agent_args: list[str] | None = None,
        resume_session_id: str | None = None,
    ) -> None:
        self.cwd = cwd
        self.session_path = session_path
        self.rpc = rpc
        self.agent_args = list(agent_args or [])
        self.resume_session_id = (
            resume_session_id or _resume_session_id_from_agent_args(self.agent_args)
        )
        self.state: State | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._previous_sigint_handler: Any | None = None

    def _agent_args_manage_session(self) -> bool:
        return any(
            arg in {"--session", "--no-session", "--session-dir"}
            for arg in self.agent_args
        )

    def _write_meta(self) -> None:
        with self._lock:
            st = self.state
        if not st:
            return
        supports_web_control = "--no-session" not in self.agent_args
        meta = {
            "session_id": st.session_id,
            "backend": "pi",
            "transport": "pi-rpc",
            "owner": OWNER_TAG if OWNER_TAG else None,
            "supports_web_control": supports_web_control,
            "supports_live_ui": True,
            "ui_protocol_version": 1,
            "broker_pid": os.getpid(),
            "agent_pid": st.codex_pid,
            "codex_pid": st.codex_pid,
            "cwd": self.cwd,
            "start_ts": float(st.start_ts),
            "log_path": None,
            "sock_path": str(st.sock_path),
            "resume_session_id": self.resume_session_id,
            "spawn_nonce": (os.environ.get("CODEX_WEB_SPAWN_NONCE") or "").strip()
            or None,
        }
        if st.session_path is not None:
            meta["session_path"] = str(st.session_path)
        SOCK_DIR.mkdir(parents=True, exist_ok=True)
        meta_path = st.sock_path.with_suffix(".json")
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=str(meta_path.parent), delete=False
        ) as tmp:
            tmp.write(json.dumps(meta))
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, meta_path)
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
            state_turn_id = (
                _extract_turn_id(rpc_state) if isinstance(rpc_state, dict) else None
            )
            if state_turn_id:
                st2.last_turn_id = state_turn_id
            sid = (
                _extract_session_id(rpc_state) if isinstance(rpc_state, dict) else None
            )
            if isinstance(sid, str) and sid and sid != st2.session_id:
                st2.session_id = sid
                rewrite_meta = True
            if self.resume_session_id and (not st2.busy) and (not st2.last_turn_id):
                self.resume_session_id = None
                rewrite_meta = True
        if rewrite_meta:
            self._write_meta()

    def _drain_rpc_output_locked(self, st: State) -> None:
        stderr_lines: list[str] = []
        drain_stderr = getattr(st.rpc, "drain_stderr_lines", None)
        if callable(drain_stderr):
            try:
                raw_stderr_lines = drain_stderr()
                if isinstance(raw_stderr_lines, list):
                    stderr_lines = [
                        line
                        for line in raw_stderr_lines
                        if isinstance(line, str) and line
                    ]
                else:
                    stderr_lines = []
            except Exception:
                stderr_lines = []
        for line in stderr_lines:
            suffix = "" if line.endswith("\n") else "\n"
            st.output_tail = (st.output_tail + f"[stderr] {line}{suffix}")[
                -st.output_tail_max :
            ]
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
            event_turn_id = _extract_turn_id(event)
            terminal_event_matches_active_turn = (
                event_type in _TERMINAL_TURN_EVENT_TYPES
                and (
                    (bool(event_turn_id) and st.last_turn_id == event_turn_id)
                    or (
                        (not event_turn_id)
                        and (not st.last_turn_id)
                        and st.prompt_sent_at <= 0.0
                    )
                )
            )
            if terminal_event_matches_active_turn:
                st.pending_ui_requests.clear()
            output = _event_output_text(event)
            if output:
                st.output_tail = (st.output_tail + output)[-st.output_tail_max :]
            if not event_turn_id:
                if (
                    event_type in _TERMINAL_TURN_EVENT_TYPES
                    and (not st.last_turn_id)
                    and st.prompt_sent_at <= 0.0
                ):
                    st.busy = False
                    st.prompt_sent_at = 0.0
                    st.last_turn_id = None
                continue
            if event_type in _TERMINAL_TURN_EVENT_TYPES:
                if st.last_turn_id == event_turn_id:
                    st.busy = False
                    st.prompt_sent_at = 0.0
                if st.last_turn_id == event_turn_id:
                    st.last_turn_id = None
                continue
            st.busy = True
            st.prompt_sent_at = 0.0
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

    def _submit_terminal_prompt(self, text: str) -> dict[str, Any]:
        st = self._get_state_snapshot()
        if not st:
            raise RuntimeError("no state")
        streaming_behavior: str | None = None
        with self._lock:
            if self.state is st:
                if st.busy:
                    streaming_behavior = "steer"
                st.busy = True
                st.prompt_sent_at = time.monotonic()
        try:
            result = st.rpc.prompt(text, streaming_behavior=streaming_behavior)
        except Exception:
            with self._lock:
                if self.state is st:
                    st.busy = False
                    st.prompt_sent_at = 0.0
            raise
        if not isinstance(result, dict):
            with self._lock:
                if self.state is st:
                    st.busy = False
                    st.prompt_sent_at = 0.0
            raise RuntimeError("invalid prompt response")
        error = result.get("error")
        if isinstance(error, str) and error:
            with self._lock:
                if self.state is st:
                    st.busy = False
                    st.prompt_sent_at = 0.0
            raise RuntimeError(error)
        with self._lock:
            if self.state is st:
                st.last_turn_id = _extract_turn_id(result) or st.last_turn_id
                st.busy = True
                st.prompt_sent_at = time.monotonic()
        return result

    def _interrupt_terminal_turn(self) -> dict[str, Any]:
        st = self._get_state_snapshot()
        if not st:
            raise RuntimeError("no state")
        result = st.rpc.abort(None)
        with self._lock:
            if self.state is st:
                st.busy = False
                st.last_turn_id = None
                st.prompt_sent_at = 0.0
        return result

    def _stdin_loop(self) -> None:
        while not self._stop.is_set():
            try:
                line = sys.stdin.readline()
            except KeyboardInterrupt:
                try:
                    self._interrupt_terminal_turn()
                except Exception as exc:
                    sys.stderr.write(f"[pi-broker] {exc}\n")
                    sys.stderr.flush()
                continue
            except Exception:
                self._stop.set()
                return
            if line == "":
                self._stop.set()
                return
            text = line.rstrip("\r\n")
            if not text.strip():
                continue
            try:
                self._submit_terminal_prompt(text)
            except Exception as exc:
                sys.stderr.write(f"[pi-broker] {exc}\n")
                sys.stderr.flush()

    def _stdout_loop(self) -> None:
        last_tail = ""
        while not self._stop.is_set():
            try:
                self._sync_output_from_rpc()
                with self._lock:
                    st = self.state
                    tail = st.output_tail if st else ""
            except Exception:
                tail = ""
            if tail:
                chunk = _tail_delta(last_tail, tail)
                if chunk:
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                last_tail = tail
            self._stop.wait(0.1)

    def _delegate_sigint(self, frame: Any | None) -> None:
        previous = self._previous_sigint_handler
        if previous in (None, signal.SIG_IGN):
            return
        if previous is signal.SIG_DFL:
            signal.default_int_handler(signal.SIGINT, frame)
            return
        if callable(previous):
            previous(signal.SIGINT, frame)

    def _handle_sigint(self, _signum: int, _frame: Any | None) -> None:
        st = self._get_state_snapshot()
        if not st or not st.busy:
            self._delegate_sigint(_frame)
            return
        try:
            self._interrupt_terminal_turn()
        except Exception as exc:
            sys.stderr.write(f"[pi-broker] {exc}\n")
            sys.stderr.flush()
            self._delegate_sigint(_frame)

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
                    resp = {
                        "busy": bool(st.busy) if st else False,
                        "queue_len": 0,
                        "token": st.token if st else None,
                    }
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
                        requests = [
                            request
                            for request in st.pending_ui_requests.values()
                            if request.get("status") == "pending"
                        ]
                    else:
                        requests = []
                    resp = {"requests": requests}
                _send_socket_json_line(conn, resp)
                return

            if cmd == "commands":
                with self._lock:
                    st = self.state
                    if st is not None:
                        self._drain_rpc_output_locked(st)
                        rpc = st.rpc
                    else:
                        rpc = None
                if rpc is None:
                    _send_socket_json_line(conn, {"error": "no state"})
                    return
                commands = rpc.get_commands()
                _send_socket_json_line(conn, {"commands": commands})
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
                        _send_socket_json_line(
                            conn, {"error": "unknown or expired request"}
                        )
                        return
                    if pending.get("status") != "pending":
                        _send_socket_json_line(
                            conn, {"error": "request already resolved"}
                        )
                        return
                    pending["status"] = "resolved"
                    rpc = st.rpc
                try:
                    rpc.send_ui_response(request_id, **ui_response_kwargs)
                except Exception:
                    with self._lock:
                        st = self.state
                        current = st.pending_ui_requests.get(request_id) if st else None
                        if (
                            current is pending
                            and isinstance(current, dict)
                            and current.get("status") == "resolved"
                        ):
                            current["status"] = "pending"
                    raise
                _send_socket_json_line(conn, {"ok": True})
                return

            if cmd == "send":
                text = req.get("text")
                if not isinstance(text, str) or not text.strip():
                    _send_socket_json_line(conn, {"error": "text required"})
                    return
                self._submit_terminal_prompt(text)
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
                    _send_socket_json_line(
                        conn, {"error": f"unsupported key sequence: {seq_raw}"}
                    )
                    return
                self._interrupt_terminal_turn()
                _send_socket_json_line(
                    conn, {"ok": True, "queued": False, "n": len(seq)}
                )
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
            threading.Thread(
                target=self._handle_conn, args=(conn,), daemon=True
            ).start()
        try:
            srv.close()
        except Exception:
            pass

    def run(self, *, foreground: bool = True) -> int:
        SOCK_DIR.mkdir(parents=True, exist_ok=True)
        PI_SESSION_DIR.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        session_path = self.session_path
        if session_path is None and not self._agent_args_manage_session():
            session_path = PI_SESSION_DIR / f"{token}.jsonl"
        sock_path = SOCK_DIR / f"{token}.sock"
        rpc = self.rpc or PiRpcClient(
            cwd=self.cwd, session_path=session_path, agent_args=self.agent_args
        )
        rpc_pid = getattr(rpc, "pid", None)
        self.state = State(
            session_id=None,
            codex_pid=rpc_pid or os.getpid(),
            sock_path=sock_path,
            session_path=session_path,
            start_ts=time.time(),
            rpc=rpc,
        )
        self._write_meta()
        threading.Thread(
            target=self._bg_sync_loop, name="pi-bg-sync", daemon=True
        ).start()
        self._previous_sigint_handler = None
        if foreground and sys.stdin.isatty() and sys.stdout.isatty():
            self._previous_sigint_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._handle_sigint)
            threading.Thread(
                target=self._stdin_loop, name="pi-stdin", daemon=True
            ).start()
            threading.Thread(
                target=self._stdout_loop, name="pi-stdout", daemon=True
            ).start()
        sock_thread = threading.Thread(
            target=self._sock_server, name="pi-sock-server", daemon=True
        )
        sock_thread.start()
        proc = getattr(rpc, "_proc", None)
        exit_code = 0
        try:
            if proc is None or not hasattr(proc, "poll"):
                sock_thread.join()
            else:
                while not self._stop.is_set():
                    code = proc.poll()
                    if code is not None:
                        exit_code = int(code)
                        self._stop.set()
                        break
                    if not sock_thread.is_alive():
                        self._stop.set()
                        break
                    time.sleep(0.1)
        finally:
            self._stop.set()
            sock_thread.join(timeout=1.0)
            if self._previous_sigint_handler is not None:
                signal.signal(signal.SIGINT, self._previous_sigint_handler)
                self._previous_sigint_handler = None
            self._close()
        return exit_code


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cwd", required=True)
    ap.add_argument("--session-file")
    ap.add_argument("args", nargs=argparse.REMAINDER)
    ns = ap.parse_args()
    session_path = (
        Path(ns.session_file).expanduser().resolve() if ns.session_file else None
    )
    args = list(ns.args)
    if args and args[0] == "--":
        args = args[1:]
    raise SystemExit(
        PiBroker(cwd=ns.cwd, session_path=session_path, agent_args=args).run()
    )


if __name__ == "__main__":
    main()
