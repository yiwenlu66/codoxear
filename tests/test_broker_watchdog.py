from __future__ import annotations

import os
import threading
from pathlib import Path

from codoxear.broker_watchdog import BrokerWatchdogCoordinator
from codoxear.process_runtime import pid_alive
from codoxear.session_listing import build_active_session_rows_snapshot
from codoxear.session_listing import build_public_session_row
from codoxear.session_model import Session
from codoxear.session_prune import SessionPruneCoordinator
from codoxear.session_store import SessionStore
from codoxear.session_store import SessionStorePaths


def _session(session_id: str, *, broker_pid: int, sock_path: Path) -> Session:
    return Session(
        session_id=session_id,
        thread_id=session_id,
        broker_pid=broker_pid,
        codex_pid=0,
        agent_backend="pi",
        owned=False,
        start_ts=100.0,
        cwd=str(sock_path.parent),
        log_path=None,
        sock_path=sock_path,
        sync_send_supported=True,
    )


def _store(tmp_path: Path) -> SessionStore:
    return SessionStore(
        paths=SessionStorePaths(
            aliases=tmp_path / "aliases.json",
            sidebar_meta=tmp_path / "sidebar.json",
            hidden_sessions=tmp_path / "hidden.json",
            files=tmp_path / "files.json",
            queues=tmp_path / "queues.json",
            pending_attachments=tmp_path / "pending.json",
            commit_unknown_sends=tmp_path / "commit.json",
            recent_cwds=tmp_path / "recent.json",
            unattended=tmp_path / "unattended.json",
        ),
        file_history_max=5,
        recent_cwd_max=5,
        unattended_default_idle_minutes=5,
        unattended_default_max_injections=10,
        clean_alias=lambda value: value if isinstance(value, str) else "",
        clean_priority_offset=lambda value: float(value or 0.0),
        clean_snooze_until=lambda value: float(value) if value not in (None, "", 0) else None,
        clean_dependency_session_id=lambda value: value.strip() if isinstance(value, str) and value.strip() else None,
        clean_recent_cwd=lambda value: value.strip() if isinstance(value, str) and value.strip() else None,
        clean_commit_unknown_send_record=lambda value: value if isinstance(value, dict) else None,
    )


def _listing_rows(sessions: dict[str, Session], store: SessionStore, *, now_ts: float) -> list[dict[str, object]]:
    return build_active_session_rows_snapshot(
        sessions=sessions.values(),
        queues={},
        unattended={},
        aliases={},
        store=store,
        now_ts=now_ts,
        unattended_default_idle_minutes=5,
        unattended_default_max_injections=10,
        clean_unattended_cooldown_minutes=lambda value: int(value),
        clean_unattended_remaining_injections=lambda value, **_kwargs: int(value),
        provider_choice_for_settings=lambda **_kwargs: "openai-api",
        resolve_session_cwd=lambda cwd: Path(cwd),
        priority_half_life_seconds=3600.0,
        priority_bucket_seconds=30.0,
        subagent_runs={},
    ).rows


def test_watchdog_keeps_live_broker_sidecar(tmp_path: Path) -> None:
    sock = tmp_path / "live.sock"
    meta = sock.with_suffix(".json")
    sock.touch()
    meta.write_text("{}", encoding="utf-8")
    session = _session("live", broker_pid=os.getpid(), sock_path=sock)
    sessions = {session.session_id: session}

    BrokerWatchdogCoordinator(
        lock=threading.Lock(),
        sessions=lambda: sessions,
        pid_alive=pid_alive,
        unlink_quiet=lambda path: path.unlink(missing_ok=True),
        now=lambda: 100.0,
        grace_seconds=60.0,
    ).sweep()

    assert session.lost is False
    assert sock.exists()
    assert meta.exists()


def test_prune_marks_dead_broker_as_lost_instead_of_removing_tombstone(tmp_path: Path) -> None:
    sock = tmp_path / "lost.sock"
    meta = sock.with_suffix(".json")
    sock.touch()
    meta.write_text("{}", encoding="utf-8")
    session = _session("lost", broker_pid=2_147_483_647, sock_path=sock)
    sessions = {session.session_id: session}
    SessionPruneCoordinator(
        lock=threading.Lock(),
        sessions=lambda: sessions,
        sock_call=lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError("stale socket")),
        broker_busy_queue_from_state=lambda _state: (False, 0),
        broker_interrupted_idle_from_state=lambda _state: False,
        sock_error_definitely_stale=lambda error: isinstance(error, FileNotFoundError),
        pid_alive=pid_alive,
        latest_launch_attempt=lambda _launch_id: None,
        submitted_user_messages=lambda _record: [],
        launch_failure_tail=lambda _record: "",
        which_tmux=lambda _name: None,
        tmux_pane_snapshot=lambda *_args, **_kwargs: {},
        clean_optional_text=lambda value: value if isinstance(value, str) else None,
        record_launch_attempt=lambda _record: None,
        clear_deleted_session_state=lambda _session_id: None,
        unlink_quiet=lambda path: path.unlink(missing_ok=True),
    ).prune_dead_sessions()

    assert sessions == {"lost": session}
    assert session.lost is True
    assert sock.exists()
    assert meta.exists()


def test_prune_keeps_watchdog_lost_session_as_a_tombstone(tmp_path: Path) -> None:
    session = _session("lost", broker_pid=2_147_483_647, sock_path=tmp_path / "lost.sock")
    session.lost = True
    sessions = {session.session_id: session}
    SessionPruneCoordinator(
        lock=threading.Lock(),
        sessions=lambda: sessions,
        sock_call=lambda *_args, **_kwargs: {},
        broker_busy_queue_from_state=lambda _state: (False, 0),
        broker_interrupted_idle_from_state=lambda _state: False,
        sock_error_definitely_stale=lambda _error: True,
        pid_alive=pid_alive,
        latest_launch_attempt=lambda _launch_id: None,
        submitted_user_messages=lambda _record: [],
        launch_failure_tail=lambda _record: "",
        which_tmux=lambda _name: None,
        tmux_pane_snapshot=lambda *_args, **_kwargs: {},
        clean_optional_text=lambda value: value if isinstance(value, str) else None,
        record_launch_attempt=lambda _record: None,
        clear_deleted_session_state=lambda _session_id: None,
        unlink_quiet=lambda path: path.unlink(missing_ok=True),
    ).prune_dead_sessions()

    assert sessions == {"lost": session}


def test_watchdog_projects_lost_tombstone_then_prunes_stale_sidecar_after_grace(tmp_path: Path) -> None:
    sock = tmp_path / "crashed.sock"
    meta = sock.with_suffix(".json")
    sock.touch()
    meta.write_text("{}", encoding="utf-8")
    session = _session("crashed", broker_pid=2_147_483_647, sock_path=sock)
    sessions = {session.session_id: session}
    clock = [100.0]
    watchdog = BrokerWatchdogCoordinator(
        lock=threading.Lock(),
        sessions=lambda: sessions,
        pid_alive=pid_alive,
        unlink_quiet=lambda path: path.unlink(missing_ok=True),
        now=lambda: clock[0],
        grace_seconds=60.0,
    )

    watchdog.sweep()

    assert session.lost is True
    assert session.lost_since == 100.0
    assert sock.exists()
    assert meta.exists()
    rows = _listing_rows(sessions, _store(tmp_path), now_ts=100.0)
    assert len(rows) == 1
    assert rows[0]["session_id"] == "crashed"
    assert rows[0]["lost"] is True
    assert rows[0]["state_busy"] is False
    public_row = build_public_session_row(rows[0], git_branch="main", busy=False)
    assert public_row["lost"] is True
    assert public_row["busy"] is False

    clock[0] = 159.0
    watchdog.sweep()
    assert sock.exists()
    assert meta.exists()

    clock[0] = 160.0
    watchdog.sweep()
    assert not sock.exists()
    assert not meta.exists()
    assert sessions["crashed"].lost is True
