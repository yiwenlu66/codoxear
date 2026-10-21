from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

import pytest

from codoxear.broker import BUSY_QUIET_SECONDS
from codoxear.broker_control import _handle_broker_control_connection
from codoxear.broker_log_watcher import _apply_log_objects_to_state
from codoxear.broker_turn_state import State
from codoxear.broker_turn_state import _close_turn_state
from codoxear.broker_turn_state import _mark_busy_state_idle
from codoxear.broker_turn_state import _should_clear_busy_state


def _state() -> State:
    return State(
        codex_pid=1,
        pty_master_fd=9,
        cwd="/tmp",
        start_ts=0.0,
        codex_home=Path("/tmp"),
        sessions_dir=Path("/tmp"),
    )


def _run_handler(
    state: State | None,
    command: str,
    request: dict[str, Any] | None = None,
    *,
    inject: Callable[..., None] | None = None,
    now: Callable[[], float] = lambda: 10.0,
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def handle_connection(_conn: Any, *, handlers: dict[str, Callable[..., Any]], **_kwargs: Any) -> None:
        response, after_reply = handlers[command](request or {})
        captured["response"] = response
        captured["after_reply"] = after_reply
        if after_reply is not None:
            after_reply()

    with patch("codoxear.broker_control._handle_control_socket_connection", handle_connection):
        _handle_broker_control_connection(
            object(),
            lock=threading.Lock(),
            get_state=lambda: state,
            seq_bytes=lambda value: value.encode("utf-8"),
            encode_enter=lambda: b"\r",
            write_all=lambda _fd, _data: None,
            inject=inject or (lambda _fd, **_kwargs: None),
            now=now,
            teardown_managed_process_group=lambda: None,
            update_codex_settings=None,
        )
    return captured


def _settle_latch(state: State, *, now_ts: float) -> None:
    assert _should_clear_busy_state(
        state,
        now_ts,
        busy_quiet_seconds=BUSY_QUIET_SECONDS,
        busy_interrupt_grace_seconds=BUSY_QUIET_SECONDS,
    )
    _mark_busy_state_idle(state, now_ts)


def test_idle_plain_prompt_without_turn_rows_publishes_undelivered_verdict() -> None:
    state = _state()

    result = _run_handler(state, "send", {"text": "recover this prompt", "send_id": "send-1"})

    assert result["response"] == {"queued": False, "queue_len": 0}
    assert state.send_latch_id == "send-1"
    _settle_latch(state, now_ts=10.5 + BUSY_QUIET_SECONDS)

    assert state.send_latch_id is None
    assert state.send_verdict == {"send_id": "send-1", "verdict": "undelivered", "ts": 10.5 + BUSY_QUIET_SECONDS}
    assert isinstance(state.send_verdict["ts"], float)


def test_reduced_user_row_prevents_undelivered_verdict() -> None:
    state = _state()
    _run_handler(state, "send", {"text": "backend received this", "send_id": "send-2"})

    _apply_log_objects_to_state(
        state,
        [{"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "backend received this"}]}}],
        now=lambda: 10.1,
    )

    assert not _should_clear_busy_state(
        state,
        10.1 + BUSY_QUIET_SECONDS + 1.0,
        busy_quiet_seconds=BUSY_QUIET_SECONDS,
        busy_interrupt_grace_seconds=BUSY_QUIET_SECONDS,
    )
    _close_turn_state(state)
    assert state.send_latch_id is None
    assert state.send_verdict is None


def test_slash_command_never_publishes_undelivered_verdict() -> None:
    state = _state()
    _run_handler(state, "send", {"text": "/model x", "send_id": "command-1"})

    assert state.send_latch_id is None
    _settle_latch(state, now_ts=10.5 + BUSY_QUIET_SECONDS)

    assert state.send_verdict is None


def test_busy_origin_send_never_publishes_undelivered_verdict() -> None:
    state = _state()
    state.busy = True
    state.turn_open = True
    state.last_turn_activity_ts = 5.0

    _run_handler(state, "send", {"text": "steer the running turn", "send_id": "steer-1"})

    assert state.send_latch_id is None
    _settle_latch(state, now_ts=10.5 + BUSY_QUIET_SECONDS)

    assert state.send_verdict is None


def test_verdict_survives_later_delivered_send_and_new_undelivered_replaces_it() -> None:
    state = _state()
    _run_handler(state, "send", {"text": "first swallowed", "send_id": "send-a"})
    _settle_latch(state, now_ts=10.5 + BUSY_QUIET_SECONDS)
    first_verdict = dict(state.send_verdict or {})

    _run_handler(state, "send", {"text": "second delivered", "send_id": "send-b"}, now=lambda: 20.0)
    _apply_log_objects_to_state(
        state,
        [{"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "second delivered"}]}}],
        now=lambda: 20.1,
    )
    _close_turn_state(state)

    assert state.send_verdict == first_verdict

    _run_handler(state, "send", {"text": "third swallowed", "send_id": "send-c"}, now=lambda: 30.0)
    _settle_latch(state, now_ts=30.5 + BUSY_QUIET_SECONDS)

    assert state.send_verdict == {"send_id": "send-c", "verdict": "undelivered", "ts": 30.5 + BUSY_QUIET_SECONDS}


@pytest.mark.parametrize("send_id", [None, 3, "contains space", "bad/slash", "x" * 121])
def test_invalid_send_ids_are_ignored(send_id: object) -> None:
    state = _state()
    _run_handler(state, "send", {"text": "swallowed", "send_id": send_id})

    assert state.send_latch_id is None
    _settle_latch(state, now_ts=10.5 + BUSY_QUIET_SECONDS)

    assert state.send_verdict is None


def test_state_response_copies_send_verdict() -> None:
    state = _state()
    state.send_verdict = {"send_id": "send-copy", "verdict": "undelivered", "ts": 7.0}

    response = _run_handler(state, "state")["response"]
    assert response["send_verdict"] == state.send_verdict

    response["send_verdict"]["send_id"] = "mutated"
    assert state.send_verdict == {"send_id": "send-copy", "verdict": "undelivered", "ts": 7.0}


def test_inject_failure_restores_previous_latch_and_verdict() -> None:
    state = _state()
    state.send_latch_id = "previous-latch"
    state.send_verdict = {"send_id": "previous-send", "verdict": "undelivered", "ts": 4.0}

    result = _run_handler(
        state,
        "send",
        {"text": "will not inject", "send_id": "new-send", "sync": True},
        inject=lambda _fd, **_kwargs: (_ for _ in ()).throw(OSError("write failed")),
    )

    assert result["response"] == {"error": "write failed", "commit_unknown": True}
    assert state.send_latch_id == "previous-latch"
    assert state.send_verdict == {"send_id": "previous-send", "verdict": "undelivered", "ts": 4.0}
