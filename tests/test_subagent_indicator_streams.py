from __future__ import annotations

import json
import os
from pathlib import Path

from codoxear.agent_backend import get_agent_backend
from codoxear.cc_subagents import handle_hook_event
from codoxear.rollout_chat_batch import _extract_chat_events
from codoxear.util import scan_active_cc_subagents
from codoxear.util import scan_active_codex_subagents
from codoxear.util import scan_active_pi_subagents


CC_SESSION_ID = "11111111-2222-3333-4444-555555555555"


def _write_pi_status(root: Path, *, run_id: str, parent_log: Path, pid: int) -> None:
    status_dir = root / run_id
    status_dir.mkdir(parents=True)
    (status_dir / "status.json").write_text(
        json.dumps(
            {
                "runId": run_id,
                "sessionId": str(parent_log),
                "state": "running",
                "startedAt": 10,
                "pid": pid,
                "agent": "executor",
            }
        ),
        encoding="utf-8",
    )


def test_pi_subagent_indicator_requires_live_worker_and_emits_progress_event(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "pi-runs"
    parent_log = tmp_path / "pi-parent.jsonl"
    parent_log.write_text("", encoding="utf-8")
    monkeypatch.setenv("CODEX_WEB_SUBAGENT_RUNS_ROOT", str(root))
    _write_pi_status(root, run_id="live", parent_log=parent_log, pid=os.getpid())
    _write_pi_status(root, run_id="stale", parent_log=parent_log, pid=999_999_999)

    runs = scan_active_pi_subagents(now_monotonic=100.0)
    assert [run["run_id"] for run in runs[str(parent_log)]] == ["live"]

    events, _meta, _flags, _diag = _extract_chat_events(
        [
            {
                "type": "custom_message",
                "customType": "subagent_control_notice",
                "id": "pi-progress-1",
                "timestamp": "2026-08-05T00:00:00Z",
                "content": "Subagent progress update: executor\nRun: live\nUPDATE: inspected the repository",
            }
        ]
    )
    assert events == [
        {
            "role": "assistant",
            "text": "Subagent progress update — executor (run live): inspected the repository",
            "message_class": "narration",
            "message_id": "pi-subagent:pi-progress-1",
            "ts": 1_785_888_000.0,
        }
    ]


def test_codex_child_header_drives_count_source_and_normalized_event(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    child = sessions_dir / "2026" / "08" / "05" / "rollout-child.jsonl"
    child.parent.mkdir(parents=True)
    header = {
        "type": "session_meta",
        "timestamp": "2026-08-05T00:00:00Z",
        "payload": {
            "id": "child-thread",
            "source": {"subagent": {"thread_spawn": {"parent_thread_id": "parent-thread"}}},
        },
    }
    child.write_text(json.dumps(header) + "\n", encoding="utf-8")

    runs = scan_active_codex_subagents(
        sessions_dirs=(sessions_dir,), now_monotonic=100.0, now_wall=100.0, writable_paths={child}
    )
    assert len(runs["parent-thread"]) == 1
    assert runs["parent-thread"][0]["event"]["message_id"] == "codex-subagent:child-thread"

    event = get_agent_backend("codex").chat_event_from_log_row(header)
    assert event == {
        "role": "assistant",
        "text": "Subagent started (thread child-th)",
        "message_class": "narration",
        "message_id": "codex-subagent:child-thread",
        "ts": 1_785_888_000.0,
    }


def test_cc_hook_record_drives_count_source_and_normalized_event(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "cc-runs"
    monkeypatch.setenv("CODEX_WEB_CC_SUBAGENT_RUNS_ROOT", str(root))
    env = {
        "CODEX_WEB_OWNER": "web",
        "CODEX_WEB_AGENT_BACKEND": "cc",
        "CODEX_WEB_CC_SUBAGENT_RUNS_ROOT": str(root),
        "CODEX_WEB_CC_SUBAGENT_BROKER_PID": str(os.getpid()),
    }
    assert handle_hook_event(
        {"hook_event_name": "SubagentStart", "session_id": CC_SESSION_ID, "agent_id": "agent-abc", "agent_type": "Explore"},
        environ=env,
    )

    runs = scan_active_cc_subagents(parent_broker_pids={CC_SESSION_ID: os.getpid()})
    assert len(runs[CC_SESSION_ID]) == 1
    assert runs[CC_SESSION_ID][0]["event"] == {
        "role": "assistant",
        "text": "Subagent started — Explore",
        "message_class": "narration",
        "message_id": "cc-subagent:agent-abc",
    }
