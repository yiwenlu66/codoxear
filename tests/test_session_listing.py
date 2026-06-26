from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from codoxear.session_listing import ActiveSessionRowFacts, build_active_session_row, build_launch_attempt_rows, build_orphan_recovery_rows, build_public_session_row, listing_priority, sidebar_time_priority_from_elapsed_seconds, sort_session_rows


def test_session_listing_import_does_not_load_server() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", "import sys; import codoxear.session_listing; raise SystemExit('codoxear.server' in sys.modules)"],
        check=False,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout


def test_listing_priority_applies_bucket_offset_and_blocking() -> None:
    priority = listing_priority(
        now_ts=125.0,
        updated_ts=100.0,
        priority_offset=0.2,
        blocked=False,
        snoozed=False,
        half_life_seconds=100.0,
        bucket_seconds=10.0,
    )

    assert priority.time_priority == sidebar_time_priority_from_elapsed_seconds(25.0, half_life_seconds=100.0, bucket_seconds=10.0)
    assert priority.base_priority == min(1.0, priority.time_priority + 0.2)
    assert priority.final_priority == priority.base_priority

    blocked = listing_priority(
        now_ts=125.0,
        updated_ts=100.0,
        priority_offset=0.2,
        blocked=True,
        snoozed=False,
        half_life_seconds=100.0,
        bucket_seconds=10.0,
    )
    assert blocked.final_priority == 0.0


def test_build_active_session_row_projects_public_and_staging_fields() -> None:
    files = ["notes.md"]
    facts = ActiveSessionRowFacts(
        session_id="s1",
        thread_id="t1",
        pid=11,
        broker_pid=22,
        agent_backend="pi",
        owned=True,
        transport="pty",
        cwd="/repo",
        start_ts=1.0,
        updated_ts=2.0,
        log_path=Path("/tmp/session.jsonl"),
        log_exists=True,
        needs_run_settings=True,
        needs_history_scan=False,
        state_busy=True,
        interrupted_idle=False,
        broker_queue_len=3,
        last_send_boundary_active=True,
        last_send_log_path=Path("/tmp/session.jsonl"),
        last_send_log_size=44,
        queue_len=2,
        queue_recovery=True,
        pending_attachment=True,
        commit_unknown_send={"text": "maybe", "created_ts": 4.0},
        token={"total": 5},
        thinking=6,
        tools=7,
        system=8,
        unattended_enabled=True,
        unattended_cooldown_minutes=9,
        unattended_remaining_injections=10,
        alias="Alias",
        files=files,
        cwd_path=Path("/repo"),
        model_provider="openai",
        preferred_auth_method="api-key",
        provider_choice="openai-api",
        model="gpt-x",
        reasoning_effort="high",
        service_tier="flex",
        tmux_session="tmux-s",
        tmux_window="1",
        launch_id="launch",
        spawn_nonce="nonce",
        priority_offset=0.2,
        snooze_until=100.0,
        dependency_session_id="other",
        time_priority=0.5,
        base_priority=0.7,
        final_priority=0.0,
        blocked=True,
        snoozed=True,
    )

    row = build_active_session_row(facts)
    files.append("later.md")

    assert row["session_id"] == "s1"
    assert row["thread_id"] == "t1"
    assert row["pid"] == 11
    assert row["broker_pid"] == 22
    assert row["agent_backend"] == "pi"
    assert row["owned"] is True
    assert row["transport"] == "pty"
    assert row["cwd"] == "/repo"
    assert row["log_path"] == "/tmp/session.jsonl"
    assert row["_log_path_obj"] == Path("/tmp/session.jsonl")
    assert row["log_exists"] is True
    assert row["needs_run_settings"] is True
    assert row["needs_history_scan"] is False
    assert row["state_busy"] is True
    assert row["interrupted_idle"] is False
    assert row["broker_queue_len"] == 3
    assert row["last_send_boundary_active"] is True
    assert row["last_send_log_path"] == Path("/tmp/session.jsonl")
    assert row["last_send_log_size"] == 44
    assert row["queue_len"] == 2
    assert row["queue_recovery"] is True
    assert row["pending_attachment"] is True
    assert row["commit_unknown_send"] is True
    assert row["commit_unknown_send_text"] == "maybe"
    assert row["commit_unknown_send_ts"] == 4.0
    assert row["token"] == {"total": 5}
    assert row["thinking"] == 6
    assert row["tools"] == 7
    assert row["system"] == 8
    assert row["unattended_enabled"] is True
    assert row["unattended_cooldown_minutes"] == 9
    assert row["unattended_remaining_injections"] == 10
    assert row["alias"] == "Alias"
    assert row["files"] == ["notes.md"]
    assert row["_cwd_path_obj"] == Path("/repo")
    assert row["provider_choice"] == "openai-api"
    assert row["model"] == "gpt-x"
    assert row["reasoning_effort"] == "high"
    assert row["service_tier"] == "flex"
    assert row["tmux_session"] == "tmux-s"
    assert row["tmux_window"] == "1"
    assert row["launch_id"] == "launch"
    assert row["spawn_nonce"] == "nonce"
    assert row["priority_offset"] == 0.2
    assert row["snooze_until"] == 100.0
    assert row["dependency_session_id"] == "other"
    assert row["time_priority"] == 0.5
    assert row["base_priority"] == 0.7
    assert row["final_priority"] == 0.0
    assert row["blocked"] is True
    assert row["snoozed"] is True


def test_build_public_session_row_removes_staging_fields_and_adds_runtime_fields() -> None:
    staged = {
        "session_id": "s1",
        "start_ts": 1.0,
        "updated_ts": 2.0,
        "final_priority": 0.5,
        "_log_path_obj": Path("/tmp/log"),
        "_cwd_path_obj": Path("/tmp/repo"),
        "log_exists": True,
        "needs_run_settings": True,
        "needs_history_scan": False,
        "state_busy": True,
        "interrupted_idle": False,
        "broker_queue_len": 1,
        "last_send_boundary_active": True,
        "last_send_log_path": Path("/tmp/log"),
        "last_send_log_size": 1,
    }

    row = build_public_session_row(staged, git_branch="main", busy=True)

    assert row["session_id"] == "s1"
    assert row["git_branch"] == "main"
    assert row["busy"] is True
    assert "_log_path_obj" not in row
    assert "_cwd_path_obj" not in row
    assert "log_exists" not in row
    assert "needs_run_settings" not in row
    assert "needs_history_scan" not in row
    assert "state_busy" not in row
    assert "interrupted_idle" not in row
    assert "broker_queue_len" not in row
    assert "last_send_boundary_active" not in row
    assert "last_send_log_path" not in row
    assert "last_send_log_size" not in row
    assert staged["_log_path_obj"] == Path("/tmp/log")


def test_sort_session_rows_orders_by_priority_recency_start_and_id() -> None:
    rows = [
        {"session_id": "z", "final_priority": 0.2, "updated_ts": 20.0, "start_ts": 1.0},
        {"session_id": "b", "final_priority": 1.0, "updated_ts": 10.0, "start_ts": 1.0},
        {"session_id": "a", "final_priority": 1.0, "updated_ts": 10.0, "start_ts": 1.0},
        {"session_id": "later-start", "final_priority": 1.0, "updated_ts": 10.0, "start_ts": 2.0},
        {"session_id": "newer", "final_priority": 1.0, "updated_ts": 11.0, "start_ts": 1.0},
    ]

    sort_session_rows(rows)

    assert [row["session_id"] for row in rows] == ["newer", "later-start", "a", "b", "z"]


def test_build_launch_attempt_rows_filters_hidden_and_active_identity() -> None:
    records = [
        {"id": "hidden"},
        {"id": "active-launch"},
        {"id": "active-nonce"},
        {"id": "kept"},
        {"id": "ignored"},
    ]

    def row_from_record(record: dict[str, object]) -> dict[str, object] | None:
        sid = str(record["id"])
        if sid == "ignored":
            return None
        row: dict[str, object] = {"session_id": sid, "launch_id": sid, "spawn_nonce": sid}
        return row

    rows = build_launch_attempt_rows(
        records=records,
        hidden_failure_ids={"hidden"},
        active_launch_ids={"active-launch"},
        active_spawn_nonces={"active-nonce"},
        row_from_record=row_from_record,
    )

    assert rows == [{"session_id": "kept", "launch_id": "kept", "spawn_nonce": "kept"}]


def _rows(**kwargs):
    defaults = {
        "active_session_ids": set(),
        "commit_unknown_sends": {},
        "queues": {},
        "existing_session_ids": set(),
        "now_ts": 99.0,
        "unattended_default_idle_minutes": 5,
        "unattended_default_max_injections": 10,
    }
    defaults.update(kwargs)
    return build_orphan_recovery_rows(**defaults)


def test_build_orphan_recovery_rows_projects_direct_unknown_row() -> None:
    rows = _rows(commit_unknown_sends={"direct": {"text": "maybe direct", "created_ts": 10.0}})

    assert len(rows) == 1
    row = rows[0]
    assert row["session_id"] == "direct"
    assert row["thread_id"] == "direct"
    assert row["start_ts"] == 10.0
    assert row["updated_ts"] == 10.0
    assert row["queue_len"] == 0
    assert row["commit_unknown_send"] is True
    assert row["commit_unknown_send_text"] == "maybe direct"
    assert row["commit_unknown_send_ts"] == 10.0
    assert row["orphan_recovery"] is True
    assert row["transcript_state"] == "failed"
    assert row["alias"] == "Recovery needed"
    assert row["provider_choice"] == "openai-api"
    assert row["unattended_cooldown_minutes"] == 5
    assert row["unattended_remaining_injections"] == 10
    assert row["final_priority"] == 1.0
    assert row["busy"] is False


def test_build_orphan_recovery_rows_projects_queue_recovery_row_and_timestamp() -> None:
    rows = _rows(
        queues={
            "plain": [{"id": "p", "text": "plain", "created_ts": 50.0}],
            "queue": [
                {"id": "a", "text": "recover", "created_ts": 3.0, "commit_unknown": True, "commit_unknown_ts": 20.0},
                {"id": "b", "text": "later", "created_ts": 30.0},
            ],
        },
        unattended_default_idle_minutes=7,
        unattended_default_max_injections=2,
    )

    assert [row["session_id"] for row in rows] == ["queue"]
    assert rows[0]["queue_len"] == 2
    assert rows[0]["commit_unknown_send"] is False
    assert rows[0]["commit_unknown_send_text"] is None
    assert rows[0]["start_ts"] == 30.0
    assert rows[0]["updated_ts"] == 30.0
    assert rows[0]["unattended_cooldown_minutes"] == 7
    assert rows[0]["unattended_remaining_injections"] == 2


def test_build_orphan_recovery_rows_filters_active_and_existing_sessions() -> None:
    rows = _rows(
        active_session_ids={"active"},
        existing_session_ids={"already"},
        commit_unknown_sends={
            "active": {"text": "active", "created_ts": 1.0},
            "already": {"text": "already", "created_ts": 2.0},
            "kept": {"text": "kept", "created_ts": 3.0},
        },
        queues={"active": [{"orphan_recovery": True}], "already": [{"orphan_recovery": True}], "kept": []},
    )

    assert [row["session_id"] for row in rows] == ["kept"]


def test_build_orphan_recovery_rows_uses_now_when_timestamps_invalid() -> None:
    rows = _rows(
        commit_unknown_sends={"direct": {"text": 123, "created_ts": -1.0}},
        queues={"queue": [{"id": "q", "orphan_recovery": True, "commit_unknown_ts": float("nan"), "created_ts": 0.0}]},
        now_ts=42.0,
    )

    by_id = {row["session_id"]: row for row in rows}
    assert by_id["direct"]["start_ts"] == 42.0
    assert by_id["direct"]["commit_unknown_send_text"] is None
    assert by_id["direct"]["commit_unknown_send_ts"] == -1.0
    assert by_id["queue"]["start_ts"] == 42.0


def test_build_orphan_recovery_rows_orders_ids() -> None:
    rows = _rows(
        commit_unknown_sends={"b": {"created_ts": 1.0}, "a": {"created_ts": 1.0}},
        queues={"c": [{"orphan_recovery": True}]},
    )

    assert [row["session_id"] for row in rows] == ["a", "b", "c"]
