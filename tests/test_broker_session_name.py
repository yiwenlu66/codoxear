from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import patch

import pytest

from codoxear import broker as broker_module
from codoxear.broker import BUSY_QUIET_SECONDS
from codoxear.broker import Broker
from codoxear.broker_control import _handle_broker_control_connection
from codoxear.broker_log_watcher import _apply_log_objects_to_state
from codoxear.broker_turn_state import State
from codoxear.broker_turn_state import _mark_busy_state_idle
from codoxear.broker_turn_state import _should_clear_busy_state


def _state(*, backend: str = "pi", thread_id: str = "thread-1") -> State:
    state = State(
        codex_pid=111,
        pty_master_fd=9,
        cwd="/tmp",
        start_ts=0.0,
        codex_home=Path("/tmp"),
        sessions_dir=Path("/tmp"),
        agent_backend=backend,
    )
    state.session_id = thread_id
    state.sock_path = Path("/tmp/broker.sock")
    return state


def _run_name_handler(
    state: State | None,
    request: dict[str, Any],
    *,
    set_session_name: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def handle_connection(_conn: Any, *, handlers: dict[str, Callable[..., Any]], **_kwargs: Any) -> None:
        response, after_reply = handlers["name"](request)
        captured["response"] = response
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
            inject=lambda _fd, **_kwargs: None,
            now=lambda: 10.0,
            teardown_managed_process_group=lambda: None,
            set_session_name=set_session_name,
        )
    return captured["response"]


def _broker(state: State) -> Broker:
    instance = object.__new__(Broker)
    instance._lock = threading.Lock()
    instance.state = state
    instance.pi_active_session_marker_path = Path("/tmp/broker-1.json")
    instance._codex_app_server = None
    instance._codex_name_control = False
    instance.meta_writes = 0
    instance._write_meta = lambda: setattr(instance, "meta_writes", instance.meta_writes + 1)
    return instance


# --- control surface -------------------------------------------------------


def test_name_command_rejects_a_rename_for_a_stale_transcript() -> None:
    state = _state(thread_id="thread-2")
    calls: list[dict[str, Any]] = []

    def set_name(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"ok": True, "backend": "pi", "name": kwargs["name"]}

    response = _run_name_handler(state, {"name": "Alpha", "thread_id": "thread-1"}, set_session_name=set_name)
    assert response == {"ok": False, "reason": "stale-thread"}
    assert calls == []


def test_name_command_requires_a_string_name() -> None:
    assert _run_name_handler(_state(), {"thread_id": "thread-1"}, set_session_name=lambda **_: {}) == {"error": "name required"}


def test_name_command_without_a_backend_hook_is_unsupported() -> None:
    response = _run_name_handler(_state(), {"name": "Alpha"})
    assert response == {"ok": False, "unsupported": True, "reason": "unsupported"}


def test_name_command_forwards_to_the_backend_hook() -> None:
    calls: list[dict[str, Any]] = []

    def set_name(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"ok": True, "backend": "pi", "name": kwargs["name"]}

    response = _run_name_handler(_state(), {"name": "Alpha", "thread_id": "thread-1"}, set_session_name=set_name)
    assert response == {"ok": True, "backend": "pi", "name": "Alpha"}
    assert calls == [{"name": "Alpha", "thread_id": "thread-1"}]


# --- backend dispatch ------------------------------------------------------


def test_pi_rename_goes_through_the_bridge_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _state(backend="pi")
    state.session_name_control = True
    broker = _broker(state)
    seen: list[tuple[Path, str, int | None]] = []

    def fake_set(marker: Path, name: str, *, expected_pid: int | None = None, timeout_s: float = 0.0) -> dict[str, Any]:
        seen.append((marker, name, expected_pid))
        return {"ok": True, "name": name}

    monkeypatch.setattr(broker_module, "AGENT_BACKEND", "pi")
    monkeypatch.setattr(broker_module, "set_bridge_session_name", fake_set)

    response = broker._set_session_name(name="auth refactor", thread_id="thread-1")

    assert response == {"ok": True, "backend": "pi", "name": "auth refactor"}
    assert seen == [(Path("/tmp/broker-1.json"), "auth refactor", 111)]
    assert state.backend_session_name == "auth refactor"
    assert broker.meta_writes == 1


def test_pi_without_the_bridge_capability_reports_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _state(backend="pi")
    state.session_name_control = False
    broker = _broker(state)
    called: list[str] = []

    monkeypatch.setattr(broker_module, "AGENT_BACKEND", "pi")
    monkeypatch.setattr(
        broker_module,
        "set_bridge_session_name",
        lambda *_args, **_kwargs: called.append("called") or {"ok": True},
    )

    assert broker._set_session_name(name="Alpha", thread_id="thread-1") == {
        "ok": False,
        "backend": "pi",
        "unsupported": True,
        "reason": "unsupported",
    }
    assert called == []


def test_pi_bridge_failure_is_a_transport_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _state(backend="pi")
    state.session_name_control = True
    broker = _broker(state)

    monkeypatch.setattr(broker_module, "AGENT_BACKEND", "pi")
    monkeypatch.setattr(
        broker_module,
        "set_bridge_session_name",
        lambda *_args, **_kwargs: {"ok": False, "error": "bridge pid mismatch"},
    )

    response = broker._set_session_name(name="Alpha", thread_id="thread-1")
    assert response["ok"] is False
    assert response["reason"] == "transport"
    assert response["detail"] == "bridge pid mismatch"


def test_codex_rename_uses_the_typed_app_server_call(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _state(backend="codex")
    state.session_name_control = True
    broker = _broker(state)
    broker._codex_app_server = SimpleNamespace(socket_path=Path("/tmp/app-server.sock"))
    seen: list[dict[str, Any]] = []

    def fake_update(sock: Path, *, thread_id: str, name: str, **_kwargs: Any) -> dict[str, Any]:
        seen.append({"sock": sock, "thread_id": thread_id, "name": name})
        return {"ok": True, "name": name}

    monkeypatch.setattr(broker_module, "AGENT_BACKEND", "codex")
    monkeypatch.setattr(broker_module, "update_codex_thread_name", fake_update)

    assert broker._set_session_name(name="deploy", thread_id="thread-1") == {
        "ok": True,
        "backend": "codex",
        "name": "deploy",
    }
    assert seen == [{"sock": Path("/tmp/app-server.sock"), "thread_id": "thread-1", "name": "deploy"}]


def test_codex_refuses_to_project_a_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _state(backend="codex")
    state.session_name_control = True
    broker = _broker(state)
    broker._codex_app_server = SimpleNamespace(socket_path=Path("/tmp/app-server.sock"))

    monkeypatch.setattr(broker_module, "AGENT_BACKEND", "codex")

    assert broker._set_session_name(name="", thread_id="thread-1") == {
        "ok": False,
        "backend": "codex",
        "reason": "unsupported-clear",
    }


def test_cc_rename_is_injected_only_when_the_tui_is_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _state(backend="cc")
    state.session_name_control = True
    state.backend_ready_ts = 1.0
    broker = _broker(state)
    injections: list[dict[str, Any]] = []

    monkeypatch.setattr(broker_module, "AGENT_BACKEND", "cc")
    monkeypatch.setattr(broker_module, "_inject", lambda fd, **kwargs: injections.append({"fd": fd, **kwargs}))

    # Idle: the native command is typed immediately.
    assert broker._set_session_name(name="deploy", thread_id="thread-1") == {
        "ok": True,
        "backend": "cc",
        "accepted": True,
    }
    assert injections == [{"fd": 9, "text": "/rename deploy", "suffix": b"\r"}]

    # Busy: the write is held for the next idle settle instead of polluting a turn.
    state.busy = True
    assert broker._set_session_name(name="later", thread_id="thread-1") == {
        "ok": True,
        "backend": "cc",
        "pending": True,
    }
    assert state.pending_session_name == "later"
    assert len(injections) == 1


def test_cc_pending_rename_lands_on_the_next_idle_settle(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _state(backend="cc")
    state.session_name_control = True
    state.backend_ready_ts = 1.0
    state.pending_session_name = "later"
    state.pending_session_name_ts = 5.0
    state.busy = True
    broker = _broker(state)
    injections: list[dict[str, Any]] = []

    monkeypatch.setattr(broker_module, "AGENT_BACKEND", "cc")
    monkeypatch.setattr(broker_module, "_inject", lambda fd, **kwargs: injections.append({"fd": fd, **kwargs}))

    broker._apply_pending_session_name()
    assert injections == []
    assert state.pending_session_name == "later"

    state.busy = False
    broker._apply_pending_session_name()
    assert injections == [{"fd": 9, "text": "/rename later", "suffix": b"\r"}]
    assert state.pending_session_name is None


def test_cc_clear_reports_that_the_backend_cannot_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _state(backend="cc")
    state.session_name_control = True
    state.backend_ready_ts = 1.0
    broker = _broker(state)

    monkeypatch.setattr(broker_module, "AGENT_BACKEND", "cc")
    assert broker._set_session_name(name="", thread_id="thread-1") == {
        "ok": False,
        "backend": "cc",
        "reason": "unsupported-clear",
    }


# --- transcript observation ------------------------------------------------


def test_session_name_rows_are_observed_without_touching_turn_state() -> None:
    state = _state(backend="pi")
    changed = _apply_log_objects_to_state(state, [{"type": "session_info", "name": "auth refactor"}], now=lambda: 5.0)
    assert changed is True
    assert state.backend_session_name == "auth refactor"
    assert state.busy is False
    assert state.turn_open is False
    assert state.last_turn_activity_ts == 0.0

    assert _apply_log_objects_to_state(state, [{"type": "session_info", "name": "auth refactor"}], now=lambda: 6.0) is False


def test_a_rename_row_cannot_hold_the_send_latch() -> None:
    state = _state(backend="pi")
    state.busy = True
    state.turn_open = True
    state.send_latch_id = "send-1"
    state.send_latch_log_off = 0
    state.send_latch_activity_ts = 10.0
    state.last_turn_activity_ts = 10.0

    _apply_log_objects_to_state(state, [{"type": "session_info", "name": "renamed"}], now=lambda: 10.2)

    # The row is not turn activity, so the quiet window still settles the latch.
    assert _should_clear_busy_state(
        state,
        10.0 + BUSY_QUIET_SECONDS + 0.5,
        busy_quiet_seconds=BUSY_QUIET_SECONDS,
        busy_interrupt_grace_seconds=BUSY_QUIET_SECONDS,
    )
    _mark_busy_state_idle(state, 10.0 + BUSY_QUIET_SECONDS + 0.5)
    assert state.send_verdict == {"send_id": "send-1", "verdict": "undelivered", "ts": 10.0 + BUSY_QUIET_SECONDS + 0.5}


def test_cc_custom_title_rows_are_observed() -> None:
    state = _state(backend="cc")
    changed = _apply_log_objects_to_state(state, [{"type": "custom-title", "customTitle": "codexear"}], now=lambda: 1.0)
    assert changed is True
    assert state.backend_session_name == "codexear"
    assert state.busy is False
