from __future__ import annotations

import socket
import traceback
from collections.abc import Callable
from typing import Any

from codoxear.broker_turn_state import State
from codoxear.broker_turn_state import _clear_pi_error_probe
from codoxear.broker_turn_state import _mark_explicit_interrupt_request
from codoxear.control_socket import handle_control_socket_connection as _handle_control_socket_connection
from codoxear.util import _send_socket_json_line
from codoxear.util import _socket_peer_disconnected


def _handle_broker_control_connection(
    conn: socket.socket,
    *,
    lock: Any,
    get_state: Callable[[], State | None],
    seq_bytes: Callable[[str], bytes],
    encode_enter: Callable[[], bytes],
    write_all: Callable[[int, bytes], None],
    inject: Callable[..., None],
    now: Callable[[], float],
    teardown_managed_process_group: Callable[[], None],
) -> None:
    def state_handler(_req: dict[str, Any]) -> tuple[dict[str, Any], Any]:
        with lock:
            st = get_state()
            if not st:
                return {"error": "no state"}, None
            return {
                "busy": st.busy,
                "queue_len": 0,
                "token": st.token,
                "interrupted_idle": (not st.busy) and st.last_interrupted_idle_ts > 0.0,
                "pi_thinking_command": bool(st.pi_thinking_command),
            }, None

    def tail_handler(_req: dict[str, Any]) -> tuple[dict[str, Any], Any]:
        with lock:
            st = get_state()
            return {"tail": st.output_tail if st else ""}, None

    def send_handler(req: dict[str, Any]) -> tuple[dict[str, Any], Any]:
        text = req.get("text")
        if not isinstance(text, str) or not text.strip():
            return {"error": "text required"}, None
        seq_raw = req.get("enter_seq")
        seq = seq_bytes(seq_raw) if isinstance(seq_raw, str) else encode_enter()
        sync_commit = bool(req.get("sync"))
        fd: int | None = None
        with lock:
            st = get_state()
            if not st:
                return {"error": "no state"}, None
            now_ts = now()
            prev_busy = st.busy
            prev_turn_open = st.turn_open
            prev_turn_has_completion_candidate = st.turn_has_completion_candidate
            prev_last_interrupt_hint_ts = st.last_interrupt_hint_ts
            prev_last_interrupt_request_ts = st.last_interrupt_request_ts
            prev_last_interrupted_idle_ts = st.last_interrupted_idle_ts
            prev_last_pi_error_probe_ts = st.last_pi_error_probe_ts
            prev_last_pi_retry_hint_ts = st.last_pi_retry_hint_ts
            prev_pi_retry_status_active = st.pi_retry_status_active
            prev_last_turn_activity_ts = st.last_turn_activity_ts
            _clear_pi_error_probe(st)
            st.pending_calls.clear()
            st.busy = True
            st.turn_open = True
            if not prev_busy and not prev_turn_open:
                st.turn_has_completion_candidate = False
                st.last_interrupt_hint_ts = 0.0
                st.last_interrupt_request_ts = 0.0
                st.last_interrupted_idle_ts = 0.0
            if now_ts > st.last_turn_activity_ts:
                st.last_turn_activity_ts = now_ts
            fd = st.pty_master_fd

        def restore_state_after_inject_failure() -> None:
            with lock:
                if get_state() is st:
                    st.busy = prev_busy
                    st.turn_open = prev_turn_open
                    st.turn_has_completion_candidate = prev_turn_has_completion_candidate
                    st.last_interrupt_hint_ts = prev_last_interrupt_hint_ts
                    st.last_interrupt_request_ts = prev_last_interrupt_request_ts
                    st.last_interrupted_idle_ts = prev_last_interrupted_idle_ts
                    st.last_pi_error_probe_ts = prev_last_pi_error_probe_ts
                    st.last_pi_retry_hint_ts = prev_last_pi_retry_hint_ts
                    st.pi_retry_status_active = prev_pi_retry_status_active
                    st.last_turn_activity_ts = prev_last_turn_activity_ts

        if sync_commit:
            if fd is None:
                restore_state_after_inject_failure()
                return {"error": "no pty", "commit_unknown": False}, None
            try:
                inject(fd, text=text, suffix=seq)
            except Exception as e:
                restore_state_after_inject_failure()
                return {"error": str(e), "commit_unknown": True}, None
            return {"queued": False, "queue_len": 0}, None

        def after_reply() -> None:
            if fd is None:
                restore_state_after_inject_failure()
                return
            try:
                inject(fd, text=text, suffix=seq)
            except Exception:
                restore_state_after_inject_failure()
                traceback.print_exc()

        return {"queued": False, "queue_len": 0}, after_reply

    def keys_handler(req: dict[str, Any]) -> tuple[dict[str, Any], Any]:
        seq_raw = req.get("seq")
        if not isinstance(seq_raw, str) or not seq_raw:
            return {"error": "seq required"}, None
        b = seq_bytes(seq_raw)
        mark_interrupt = req.get("interrupt") is True and b == b"\x1b"
        fd: int | None = None
        with lock:
            st = get_state()
            if not st:
                return {"error": "no state"}, None
            fd = st.pty_master_fd
            resp = {"ok": True, "queued": False, "n": len(b), "key_queue_len": len(st.key_queue)}
        wrote_keys = False
        if fd is not None:
            try:
                write_all(fd, b)
                wrote_keys = True
            except Exception as e:
                return {"error": str(e), "queued": False, "n": 0, "key_queue_len": 0, "commit_unknown": True}, None
        if mark_interrupt and wrote_keys:
            with lock:
                if get_state() is st:
                    _mark_explicit_interrupt_request(st, now())
        return resp, None

    def shutdown_handler(_req: dict[str, Any]) -> tuple[dict[str, Any], Any]:
        return {"ok": True}, teardown_managed_process_group

    _handle_control_socket_connection(
        conn,
        handlers={
            "state": state_handler,
            "tail": tail_handler,
            "send": send_handler,
            "keys": keys_handler,
            "shutdown": shutdown_handler,
        },
        send_json_line=_send_socket_json_line,
        socket_peer_disconnected=_socket_peer_disconnected,
    )
