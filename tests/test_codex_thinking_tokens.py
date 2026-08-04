from __future__ import annotations

import json
import threading
from pathlib import Path

from codoxear.rollout_idle import _analyze_log_chunk
from codoxear.session_discovery import DiscoveryDeps
from codoxear.session_discovery import discover_sessions
from codoxear.session_log_runtime import SessionLogRuntimeCoordinator
from codoxear.session_model import Session
from codoxear.util import read_jsonl_from_offset


def _token_count(total: int) -> dict[str, object]:
    return {
        "type": "event_msg",
        "payload": {"type": "token_count", "info": {"total_token_usage": {"reasoning_output_tokens": total}}},
    }


def _user_message(text: str) -> dict[str, object]:
    return {"type": "event_msg", "payload": {"type": "user_message", "message": text}}


def _turn_context(model: str) -> dict[str, object]:
    return {"type": "turn_context", "payload": {"model": model}}


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _session(log_path: Path, *, meta_log_off: int = 0) -> Session:
    return Session(
        session_id="broker-session",
        thread_id="parent-thread",
        broker_pid=1,
        codex_pid=2,
        agent_backend="codex",
        owned=False,
        start_ts=1.0,
        cwd=str(log_path.parent),
        log_path=log_path,
        sock_path=log_path.with_suffix(".sock"),
        busy=True,
        meta_log_off=meta_log_off,
    )


def _runtime(session: Session) -> SessionLogRuntimeCoordinator:
    sessions = {session.session_id: session}
    return SessionLogRuntimeCoordinator(
        lock=threading.Lock(),
        sessions=lambda: sessions,
        analyze_log_chunk=_analyze_log_chunk,
        turn_context_run_settings=lambda _payload: (None, None),
        compute_idle_from_log=lambda _path: False,
        read_jsonl_from_offset=read_jsonl_from_offset,
        find_latest_token_update=lambda _path: None,
    )


def _discovery_deps(log_path: Path) -> DiscoveryDeps:
    return DiscoveryDeps(
        pid_alive=lambda pid: pid in {11, 12},
        proc_find_open_rollout_log=lambda *_args: log_path,
        read_session_meta_or_none=lambda *_args: {"id": "parent-thread"},
        coerce_main_thread_log=lambda thread_id, path: (thread_id, path),
        session_transport=lambda _meta: (None, None, None),
        session_run_settings=lambda _meta, _path, _backend: (None, None, "gpt-5.4", "high"),
        sock_call=lambda _sock, _request, _timeout: {"busy": True, "queue_len": 0, "interrupted_idle": False},
        broker_busy_queue_from_state=lambda state: (bool(state["busy"]), int(state["queue_len"])),
        broker_interrupted_idle_from_state=lambda state: bool(state["interrupted_idle"]),
        sock_error_definitely_stale=lambda _exc: False,
        token_update_finder=lambda _path: None,
    )


def test_long_codex_session_rebaselines_repeated_forks_without_inflation(tmp_path: Path) -> None:
    log_path = tmp_path / "rollout-parent.jsonl"
    rows: list[dict[str, object]] = [
        _user_message("completed earlier turn"),
        _token_count(100),
        {"type": "event_msg", "payload": {"type": "turn_complete"}},
        _user_message("current turn"),
    ]
    expected_tokens = 0
    prior = 100
    for model, totals in (
        ("gpt-5.4", (125, 164, 19, 48)),
        ("gpt-5.3-codex", (6, 11, 41, 3, 13)),
        ("gpt-5.4", (4, 38)),
    ):
        rows.append(_turn_context(model))
        for total in totals:
            rows.append(_token_count(total))
            if total >= prior:
                expected_tokens += total - prior
            prior = total
    _write_jsonl(log_path, rows)

    session = _session(log_path)
    _runtime(session).update_meta_counters()

    assert session.meta_thinking_tokens == expected_tokens == 172
    assert session.meta_codex_reasoning_total == 38


def test_subagent_reasoning_model_is_not_aggregated_into_parent_log(tmp_path: Path) -> None:
    parent_log = tmp_path / "rollout-parent.jsonl"
    child_log = tmp_path / "rollout-child.jsonl"
    _write_jsonl(
        parent_log,
        [
            _user_message("parent task"),
            _turn_context("gpt-5.4"),
            _token_count(20),
            _token_count(47),
        ],
    )
    _write_jsonl(
        child_log,
        [
            {"type": "session_meta", "payload": {"id": "child-thread", "source": {"subagent": {"thread_spawn": {"parent_thread_id": "parent-thread"}}}}},
            _user_message("child task"),
            _turn_context("gpt-5.3-codex"),
            _token_count(900),
        ],
    )

    session = _session(parent_log)
    _runtime(session).update_meta_counters()

    assert child_log.exists()
    assert (session.meta_thinking_tokens, session.meta_codex_reasoning_total) == (47, 47)


def test_busy_codex_reconnect_replays_cumulative_history_after_server_restart(tmp_path: Path) -> None:
    log_path = tmp_path / "rollout-parent.jsonl"
    _write_jsonl(
        log_path,
        [
            {"type": "session_meta", "payload": {"id": "parent-thread"}},
            _user_message("completed earlier turn"),
            _token_count(80),
            {"type": "event_msg", "payload": {"type": "turn_complete"}},
            _user_message("still running when server restarts"),
            _token_count(104),
            _token_count(123),
            _token_count(9),
            _token_count(22),
        ],
    )
    sock_dir = tmp_path / "socks"
    sock_dir.mkdir()
    sock = sock_dir / "broker-session.sock"
    sock.touch()
    sock.with_suffix(".json").write_text(
        json.dumps(
            {
                "session_id": "parent-thread",
                "agent_backend": "codex",
                "owner": "terminal",
                "broker_pid": 11,
                "codex_pid": 12,
                "cwd": str(tmp_path),
                "start_ts": 1.0,
                "log_path": str(log_path),
            }
        ),
        encoding="utf-8",
    )

    registration = discover_sessions(sock_dir, proc_root=tmp_path / "proc", hidden_sessions=set(), deps=_discovery_deps(log_path)).registrations[0]
    assert registration.meta_log_off == 0

    reconnected = _session(log_path, meta_log_off=registration.meta_log_off)
    _runtime(reconnected).update_meta_counters()

    assert (reconnected.meta_thinking_tokens, reconnected.meta_codex_reasoning_total) == (56, 22)
