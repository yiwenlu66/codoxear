from __future__ import annotations

import json
import threading
from pathlib import Path

from codoxear.session_model import Session
from codoxear.session_refresh import SessionRefreshCoordinator


def _sidecar(tmp_path: Path, **overrides: object) -> Path:
    sock = tmp_path / "broker-1.sock"
    sock.touch()
    meta_path = sock.with_suffix(".json")
    meta = {
        "session_id": "thread-1",
        "broker_pid": 4242,
        "codex_pid": 4243,
        "start_ts": 1.0,
        "cwd": str(tmp_path),
        "agent_backend": "pi",
        "log_path": None,
        "pi_thinking_command": True,
        "session_name_control": True,
        "backend_session_name": "terminal title",
        "slash_commands": [],
    }
    meta.update(overrides)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return sock


def _session(sock: Path, tmp_path: Path) -> Session:
    return Session(
        session_id="s1",
        thread_id="thread-1",
        broker_pid=4242,
        codex_pid=4243,
        agent_backend="pi",
        owned=True,
        start_ts=1.0,
        cwd=str(tmp_path),
        log_path=None,
        sock_path=sock,
    )


def _coordinator(
    sessions: dict[str, Session],
    *,
    reconciled: list[str],
    reconcile=None,
) -> SessionRefreshCoordinator:
    return SessionRefreshCoordinator(
        lock=threading.Lock(),
        sessions=lambda: sessions,
        prune_stale_socket_without_metadata=lambda _sid, _sock: None,
        log_invalid_sidecar_metadata=lambda _ctx, _sock, _exc: None,
        session_transport=lambda **_kw: (None, None, None),
        sock_call=lambda *_a, **_kw: {"busy": False, "queue_len": 0, "token": None},
        broker_tail_has_session_detach_marker=lambda _backend, _tail: False,
        pid_alive=lambda _pid: True,
        proc_find_open_rollout_log=lambda **_kw: None,
        proc_root=Path("/proc"),
        read_session_meta_or_none=lambda *_a, **_kw: None,
        coerce_main_thread_log=lambda **kw: (kw["thread_id"], kw["log_path"]),
        clean_optional_text=lambda value: value.strip() if isinstance(value, str) and value.strip() else None,
        session_run_settings=lambda **_kw: (None, None, None, None),
        normalize_requested_service_tier=lambda _value: None,
        reset_log_caches=lambda _session, _log_off: None,
        queue_len=lambda _sid: 0,
        maybe_drain_session_queue=lambda _sid: None,
        reconcile_session_name=reconcile or (lambda sid: reconciled.append(sid) or {"ok": True, "reason": "unchanged"}),
    )


def test_refresh_projects_the_session_name_capability_and_reconciles(tmp_path: Path) -> None:
    sock = _sidecar(tmp_path)
    session = _session(sock, tmp_path)
    reconciled: list[str] = []

    _coordinator({"s1": session}, reconciled=reconciled).refresh_session_meta("s1")

    assert session.session_name_control is True
    assert session.backend_session_name == "terminal title"
    # The bind/discovery path is the trigger that survives a broker restart.
    assert reconciled == ["s1"]


def test_refresh_reports_an_incapable_backend_truthfully(tmp_path: Path) -> None:
    sock = _sidecar(tmp_path, session_name_control=False, backend_session_name=None)
    session = _session(sock, tmp_path)
    reconciled: list[str] = []

    _coordinator({"s1": session}, reconciled=reconciled).refresh_session_meta("s1")

    assert session.session_name_control is False
    assert session.backend_session_name == ""


def test_a_failing_projection_never_breaks_the_sidecar_refresh(tmp_path: Path) -> None:
    sock = _sidecar(tmp_path)
    session = _session(sock, tmp_path)

    def boom(_sid: str) -> dict:
        raise RuntimeError("projection exploded")

    coordinator = _coordinator({"s1": session}, reconciled=[], reconcile=boom)
    coordinator.refresh_session_meta("s1")
    assert session.backend_session_name == "terminal title"
