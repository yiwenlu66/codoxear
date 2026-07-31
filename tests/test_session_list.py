from __future__ import annotations

from pathlib import Path
import threading
from typing import Any

from codoxear.session_list import SessionListCoordinator
from codoxear.session_model import Session
from codoxear.session_runtime import ListingRuntimeProbes
from codoxear.session_store import SessionStore, SessionStorePaths


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


def _session(tmp_path: Path) -> Session:
    return Session(
        session_id="s1",
        thread_id="t1",
        broker_pid=2,
        codex_pid=1,
        agent_backend="codex",
        owned=True,
        start_ts=5.0,
        cwd=str(tmp_path),
        log_path=None,
        sock_path=tmp_path / "s1.sock",
    )


def _probes() -> ListingRuntimeProbes:
    return ListingRuntimeProbes(
        last_conversation_ts_from_tail=lambda path: None,
        read_run_settings_from_log=lambda path, agent_backend: (None, None, None),
        log_size_or_none=lambda path: None,
        send_boundary_unresolved=lambda sid, path, size: False,
        idle_from_log_path=lambda sid, path: True,
        current_git_branch=lambda path: "main",
    )


def test_session_list_coordinator_owns_listing_prelude_dirty_saves_and_sort(tmp_path: Path) -> None:
    lock = threading.Lock()
    sessions = {"s1": _session(tmp_path)}
    queues: dict[str, list[dict[str, Any]]] = {}
    unattended: dict[str, dict[str, Any]] = {}
    aliases = {"s1": "Alias"}
    hidden: set[str] = set()
    commit_unknown: dict[str, dict[str, Any]] = {}
    store = _store(tmp_path)
    store.files = {"s1": ["legacy.py"]}
    store.sidebar_meta = {"s1": {"dependency_session_id": "missing"}}
    calls: list[str] = []

    coordinator = SessionListCoordinator(
        lock=lock,
        sessions=lambda: sessions,
        queues=lambda: queues,
        unattended=lambda: unattended,
        aliases=lambda: aliases,
        hidden_sessions=lambda: hidden,
        commit_unknown_sends=lambda: commit_unknown,
        store=store,
        discover_existing_if_stale=lambda: calls.append("discover"),
        prune_dead_sessions=lambda: calls.append("prune"),
        update_meta_counters=lambda: calls.append("meta"),
        save_files=lambda: calls.append("save_files"),
        save_sidebar_meta=lambda: calls.append("save_sidebar"),
        save_recent_cwds=lambda: calls.append("save_recent"),
        now=lambda: 10.0,
        runtime_probes=_probes(),
        include_launch_attempts=lambda: False,
        read_launch_attempts=lambda: [],
        launch_attempt_row=lambda record: None,
        clean_unattended_cooldown_minutes=lambda value: int(value),
        clean_unattended_remaining_injections=lambda value, *, allow_zero=False: int(value),
        provider_choice_for_settings=lambda **_kwargs: "provider-choice",
        resolve_session_cwd=lambda cwd: Path(cwd),
        unattended_default_idle_minutes=5,
        unattended_default_max_injections=10,
        priority_half_life_seconds=100.0,
        priority_bucket_seconds=10.0,
    )

    rows = coordinator.list_sessions()

    assert [row["session_id"] for row in rows] == ["s1"]
    assert rows[0]["alias"] == "Alias"
    assert rows[0]["files"] == ["legacy.py"]
    assert rows[0]["git_branch"] == "main"
    assert rows[0]["busy"] is False
    assert calls == ["discover", "prune", "meta", "save_files", "save_sidebar", "save_recent"]


def test_session_list_coordinator_adds_launch_and_orphan_recovery_rows(tmp_path: Path) -> None:
    lock = threading.Lock()
    sessions: dict[str, Session] = {}
    queues = {"queue-orphan": [{"id": "q", "text": "recover", "orphan_recovery": True, "created_ts": 7.0}]}
    unattended: dict[str, dict[str, Any]] = {}
    aliases: dict[str, str] = {}
    hidden: set[str] = set()
    commit_unknown = {"direct-orphan": {"text": "maybe", "created_ts": 6.0}}
    store = _store(tmp_path)

    coordinator = SessionListCoordinator(
        lock=lock,
        sessions=lambda: sessions,
        queues=lambda: queues,
        unattended=lambda: unattended,
        aliases=lambda: aliases,
        hidden_sessions=lambda: hidden,
        commit_unknown_sends=lambda: commit_unknown,
        store=store,
        discover_existing_if_stale=lambda: None,
        prune_dead_sessions=lambda: None,
        update_meta_counters=lambda: None,
        save_files=lambda: None,
        save_sidebar_meta=lambda: None,
        save_recent_cwds=lambda: None,
        now=lambda: 10.0,
        runtime_probes=_probes(),
        include_launch_attempts=lambda: True,
        read_launch_attempts=lambda: [{"id": "launch-record"}],
        launch_attempt_row=lambda record: {
            "session_id": "launch-row",
            "thread_id": "launch-row",
            "pid": 0,
            "broker_pid": 0,
            "agent_backend": "codex",
            "owned": True,
            "transport": None,
            "cwd": str(tmp_path),
            "start_ts": 9.0,
            "updated_ts": 9.0,
            "log_path": None,
            "queue_len": 0,
            "pending_attachment": False,
            "commit_unknown_send": False,
            "token": None,
            "thinking": 0,
            "tools": 0,
            "system": 0,
            "unattended_enabled": False,
            "unattended_cooldown_minutes": 5,
            "unattended_remaining_injections": 10,
            "alias": "Pending launch",
            "files": [],
            "model_provider": None,
            "preferred_auth_method": None,
            "provider_choice": "openai-api",
            "model": None,
            "reasoning_effort": None,
            "service_tier": None,
            "tmux_session": None,
            "tmux_window": None,
            "launch_id": "launch-record",
            "spawn_nonce": None,
            "priority_offset": 0.0,
            "snooze_until": None,
            "dependency_session_id": None,
            "time_priority": 1.0,
            "base_priority": 1.0,
            "final_priority": 1.0,
            "blocked": False,
            "snoozed": False,
            "busy": False,
            "git_branch": None,
            "transcript_state": "failed",
        },
        clean_unattended_cooldown_minutes=lambda value: int(value),
        clean_unattended_remaining_injections=lambda value, *, allow_zero=False: int(value),
        provider_choice_for_settings=lambda **_kwargs: "provider-choice",
        resolve_session_cwd=lambda cwd: Path(cwd),
        unattended_default_idle_minutes=5,
        unattended_default_max_injections=10,
        priority_half_life_seconds=100.0,
        priority_bucket_seconds=10.0,
    )

    rows = coordinator.list_sessions()

    by_id = {row["session_id"]: row for row in rows}
    assert set(by_id) == {"launch-row", "direct-orphan", "queue-orphan"}
    assert by_id["direct-orphan"]["orphan_recovery"] is True
    assert by_id["direct-orphan"]["commit_unknown_send"] is True
    assert by_id["queue-orphan"]["orphan_recovery"] is True
    assert by_id["queue-orphan"]["queue_len"] == 1
