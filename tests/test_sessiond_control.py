import threading
from pathlib import Path

from codoxear.sessiond_control import SessiondControlDeps
from codoxear.sessiond_control import handle_sessiond_control_connection
from codoxear.sessiond_state import State


def _state(*, busy: bool = False, fd: int = 9) -> State:
    return State(
        session_id="sid",
        codex_pid=123,
        log_path=Path("/tmp/sessiond.jsonl"),
        sock_path=Path("/tmp/sessiond.sock"),
        pty_master_fd=fd,
        start_ts=1.0,
        busy=busy,
        output_tail="tail",
    )


def _run_handler(command: str, request: dict | None = None, *, state: State | None = None, inject=None, write_all=None):
    captured: dict[str, object] = {}

    def fake_control_connection(_conn, *, handlers, **_kwargs):
        response, after = handlers[command](request or {})
        captured["response"] = response
        captured["after"] = after
        if after is not None:
            after()

    deps = SessiondControlDeps(
        lock=threading.Lock(),
        state=lambda: state,
        encode_enter=lambda: b"\r",
        seq_bytes=lambda raw: raw.encode("utf-8"),
        write_all=write_all or (lambda _fd, _data: None),
        inject=inject or (lambda *_args, **_kwargs: None),
        teardown_managed_process_group=lambda: captured.setdefault("shutdown", True),
        handle_control_socket_connection=fake_control_connection,
        send_json_line=lambda _conn, _obj: None,
        socket_peer_disconnected=lambda _exc: False,
        print_exception=lambda: captured.setdefault("printed", True),
    )
    handle_sessiond_control_connection(object(), deps=deps)
    return captured


def test_sessiond_control_state_tail_and_shutdown_handlers_are_executable() -> None:
    state = _state(busy=True)
    assert _run_handler("state", state=state)["response"] == {"busy": True, "queue_len": 0, "interrupted_idle": False}
    assert _run_handler("tail", state=state)["response"] == {"tail": "tail"}
    shutdown = _run_handler("shutdown", state=state)
    assert shutdown["response"] == {"ok": True}
    assert shutdown["shutdown"] is True


def test_sessiond_control_send_validates_text_and_restores_busy_on_sync_failure() -> None:
    state = _state(busy=False)
    assert _run_handler("send", {"text": "   "}, state=state)["response"] == {"error": "text required"}
    assert _run_handler("send", {"text": "run", "sync": True}, state=None)["response"] == {"error": "no state"}
    state.pty_master_fd = None  # type: ignore[assignment]
    assert _run_handler("send", {"text": "run", "sync": True}, state=state)["response"] == {"error": "no pty", "commit_unknown": False}
    assert state.busy is False

    state.pty_master_fd = 9
    response = _run_handler("send", {"text": "run", "sync": True}, state=state, inject=lambda *_a, **_k: (_ for _ in ()).throw(OSError("write failed")))["response"]
    assert response == {"error": "write failed", "commit_unknown": True}
    assert state.busy is False


def test_sessiond_control_async_send_ack_and_key_errors_are_executable() -> None:
    state = _state(busy=False)
    calls: list[tuple[int, str, bytes]] = []

    def inject(fd, *, text, suffix):
        calls.append((fd, text, suffix))

    response = _run_handler("send", {"text": "run"}, state=state, inject=inject)["response"]
    assert response == {"queued": False, "queue_len": 0}
    assert calls == [(9, "run", b"\r")]
    assert state.busy is True

    key_calls: list[tuple[int, bytes]] = []
    response = _run_handler("keys", {"seq": "x"}, state=state, write_all=lambda fd, data: key_calls.append((fd, data)))["response"]
    assert response == {"ok": True}
    assert key_calls == [(9, b"x")]
    response = _run_handler("keys", {"seq": "x"}, state=state, write_all=lambda _fd, _data: (_ for _ in ()).throw(OSError("key failed")))["response"]
    assert response == {"error": "key failed", "commit_unknown": True}
