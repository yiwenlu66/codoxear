from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .broker_log_watcher import _apply_log_objects_to_state
from .util import read_jsonl_from_offset as _read_jsonl_from_offset_impl


@dataclass
class State:
    session_id: str | None
    codex_pid: int
    log_path: Path
    sock_path: Path
    pty_master_fd: int
    start_ts: float
    busy: bool = False
    output_tail: str = ""
    output_tail_max: int = 64 * 1024
    log_off: int = 0
    token: dict[str, Any] | None = None
    turn_open: bool = False
    turn_has_completion_candidate: bool = False
    last_turn_activity_ts: float = 0.0
    last_interrupt_hint_ts: float = 0.0
    last_interrupt_request_ts: float = 0.0
    last_interrupted_idle_ts: float = 0.0
    pending_calls: set[Any] = field(default_factory=set)
    interrupt_hint_tail: str = ""
    interrupt_hint_tail_max: int = 4096


def _read_jsonl_from_offset(path: Path, offset: int, max_bytes: int = 256 * 1024) -> tuple[list[dict[str, Any]], int]:
    if not path.exists():
        return [], offset
    return _read_jsonl_from_offset_impl(path, offset, max_bytes=max_bytes)


def _probe_state(*, busy: bool = False, turn_open: bool = False) -> State:
    return State(
        session_id=None,
        codex_pid=-1,
        log_path=Path("/dev/null"),
        sock_path=Path("/dev/null"),
        pty_master_fd=-1,
        start_ts=0.0,
        busy=busy,
        turn_open=turn_open,
        last_turn_activity_ts=1.0 if busy or turn_open else 0.0,
    )


def _state_has_active_turn(st: State) -> bool:
    return st.busy or st.turn_open or bool(st.pending_calls)


def _state_snapshot(st: State) -> tuple[Any, ...]:
    return (
        st.busy,
        st.turn_open,
        st.turn_has_completion_candidate,
        st.last_turn_activity_ts,
        st.last_interrupt_hint_ts,
        st.last_interrupt_request_ts,
        st.last_interrupted_idle_ts,
        frozenset(st.pending_calls),
        st.token,
        st.interrupt_hint_tail,
    )


def _log_busy_signals(obj: dict[str, Any]) -> tuple[bool, bool]:
    inactive_probe = _probe_state()
    _apply_log_objects_to_state(inactive_probe, [obj], now=lambda: 1.0)
    user_signal = _state_has_active_turn(inactive_probe)

    active_probe = _probe_state(busy=True, turn_open=True)
    _apply_log_objects_to_state(active_probe, [obj], now=lambda: 1.0)
    turn_end_signal = not _state_has_active_turn(active_probe)
    return user_signal, turn_end_signal


def _busy_value_after_log_batch(objs: list[dict[str, Any]]) -> bool | None:
    if not objs:
        return None
    seen_busy_signal = False
    for obj in objs:
        user_signal, turn_end_signal = _log_busy_signals(obj)
        if user_signal or turn_end_signal:
            seen_busy_signal = True
    if not seen_busy_signal:
        return None
    probe = _probe_state()
    _apply_log_objects_to_state(probe, objs, now=lambda: 1.0)
    return _state_has_active_turn(probe)


def apply_log_batch_to_state(st: State, objs: list[dict[str, Any]], *, now_ts: float) -> bool:
    before = _state_snapshot(st)
    _apply_log_objects_to_state(st, objs, now=lambda: now_ts)
    return _state_snapshot(st) != before
