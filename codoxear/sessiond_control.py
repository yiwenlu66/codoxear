from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class SessiondControlDeps:
    lock: Any
    state: Callable[[], Any | None]
    encode_enter: Callable[[], bytes]
    seq_bytes: Callable[[str], bytes]
    write_all: Callable[[int, bytes], None]
    inject: Callable[..., None]
    teardown_managed_process_group: Callable[[], None]
    handle_control_socket_connection: Callable[..., None]
    send_json_line: Callable[[Any, dict[str, Any]], None]
    socket_peer_disconnected: Callable[[BaseException], bool]
    print_exception: Callable[[], None]


def handle_sessiond_control_connection(conn: Any, *, deps: SessiondControlDeps) -> None:
    def state_handler(_req: dict[str, Any]) -> tuple[dict[str, Any], Any]:
        with deps.lock:
            st = deps.state()
            if not st:
                return {"error": "no state"}, None
            return {"busy": st.busy, "queue_len": 0}, None

    def tail_handler(_req: dict[str, Any]) -> tuple[dict[str, Any], Any]:
        with deps.lock:
            st = deps.state()
            return {"tail": st.output_tail if st else ""}, None

    def send_handler(req: dict[str, Any]) -> tuple[dict[str, Any], Any]:
        text = req.get("text")
        if not isinstance(text, str) or not text.strip():
            return {"error": "text required"}, None
        fd: int | None = None
        enter = deps.encode_enter()
        sync_commit = bool(req.get("sync"))
        with deps.lock:
            st = deps.state()
            if not st:
                return {"error": "no state"}, None
            prev_busy = st.busy
            st.busy = True
            fd = st.pty_master_fd

        def restore_state_after_inject_failure() -> None:
            with deps.lock:
                if deps.state() is st:
                    st.busy = prev_busy
                    st.busy = prev_busy

        if sync_commit:
            if fd is None:
                restore_state_after_inject_failure()
                return {"error": "no pty", "commit_unknown": False}, None
            try:
                deps.inject(fd, text=text, suffix=enter)
            except Exception as e:
                restore_state_after_inject_failure()
                return {"error": str(e), "commit_unknown": True}, None
            return {"queued": False, "queue_len": 0}, None

        def after_reply() -> None:
            if fd is None:
                restore_state_after_inject_failure()
                return
            try:
                deps.inject(fd, text=text, suffix=enter)
            except Exception:
                restore_state_after_inject_failure()
                deps.print_exception()

        return {"queued": False, "queue_len": 0}, after_reply

    def keys_handler(req: dict[str, Any]) -> tuple[dict[str, Any], Any]:
        seq = req.get("seq")
        if not isinstance(seq, str) or not seq:
            return {"error": "seq required"}, None
        b = deps.seq_bytes(seq)
        with deps.lock:
            st = deps.state()
            if not st:
                return {"error": "no state"}, None
            try:
                deps.write_all(st.pty_master_fd, b)
                return {"ok": True}, None
            except Exception as e:
                return {"error": str(e), "commit_unknown": True}, None

    def shutdown_handler(_req: dict[str, Any]) -> tuple[dict[str, Any], Any]:
        return {"ok": True}, deps.teardown_managed_process_group

    deps.handle_control_socket_connection(
        conn,
        handlers={
            "state": state_handler,
            "tail": tail_handler,
            "send": send_handler,
            "keys": keys_handler,
            "shutdown": shutdown_handler,
        },
        send_json_line=deps.send_json_line,
        socket_peer_disconnected=deps.socket_peer_disconnected,
    )
