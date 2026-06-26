from __future__ import annotations

import io

from codoxear.launch_ledger import LaunchAttemptRecorder


def test_launch_attempt_recorder_records_state_transition_without_mutating_base() -> None:
    base = {
        "launch_id": "launch-1",
        "state": "starting",
        "cwd": "/tmp/work",
        "created_ts": 10.0,
        "updated_ts": 10.0,
    }
    seen: list[dict] = []
    recorder = LaunchAttemptRecorder(base, record_launch_attempt=lambda rec: seen.append(dict(rec)) or dict(rec), now=lambda: 12.5)

    rec = recorder.record("broker_spawned", transport="direct", broker_pid=123)

    assert rec["launch_id"] == "launch-1"
    assert rec["state"] == "broker_spawned"
    assert rec["updated_ts"] == 12.5
    assert rec["transport"] == "direct"
    assert rec["broker_pid"] == 123
    assert seen == [rec]
    assert base == {
        "launch_id": "launch-1",
        "state": "starting",
        "cwd": "/tmp/work",
        "created_ts": 10.0,
        "updated_ts": 10.0,
    }


def test_launch_attempt_recorder_failure_record_carries_stage_error_and_extra_fields() -> None:
    base = {"launch_id": "launch-2", "state": "starting", "cwd": "/tmp/work", "created_ts": 20.0}
    recorder = LaunchAttemptRecorder(base, record_launch_attempt=lambda rec: {**rec, "persisted": True}, now=lambda: 21.0)

    rec = recorder.failure_record("broker_metadata", TimeoutError("metadata timed out"), transport="tmux", tmux_window="work-abc123")

    assert rec["launch_id"] == "launch-2"
    assert rec["state"] == "failed"
    assert rec["stage"] == "broker_metadata"
    assert rec["error"] == "metadata timed out"
    assert rec["updated_ts"] == 21.0
    assert rec["transport"] == "tmux"
    assert rec["tmux_window"] == "work-abc123"
    assert rec["persisted"] is True


def test_launch_attempt_recorder_failure_record_survives_persistence_error() -> None:
    stderr = io.StringIO()

    def fail_to_record(_rec: dict) -> dict:
        raise RuntimeError("disk full")

    recorder = LaunchAttemptRecorder(
        {"launch_id": "launch-3", "state": "starting", "cwd": "/tmp/work", "created_ts": 30.0},
        record_launch_attempt=fail_to_record,
        now=lambda: 31.0,
        stderr=stderr,
    )

    rec = recorder.failure_record("broker_spawn", "spawn failed: boom", transport="direct")

    assert rec == {
        "launch_id": "launch-3",
        "state": "failed",
        "cwd": "/tmp/work",
        "created_ts": 30.0,
        "stage": "broker_spawn",
        "error": "spawn failed: boom",
        "updated_ts": 31.0,
        "transport": "direct",
    }
    assert "error: failed to write launch attempt record: RuntimeError: disk full" in stderr.getvalue()
