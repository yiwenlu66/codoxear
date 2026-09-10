from __future__ import annotations

import os
from pathlib import Path

from codoxear.cc_subagents import handle_hook_event
from codoxear.util import scan_active_cc_subagents


SESSION_ID = "11111111-2222-3333-4444-555555555555"


def _env(root: Path) -> dict[str, str]:
    return {
        "CODEX_WEB_OWNER": "web",
        "CODEX_WEB_AGENT_BACKEND": "cc",
        "CODEX_WEB_CC_SUBAGENT_RUNS_ROOT": str(root),
        "CODEX_WEB_CC_SUBAGENT_BROKER_PID": str(os.getpid()),
    }


def test_cc_hook_status_is_live_only_between_start_and_stop(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "cc-subagent-runs"
    env = _env(root)
    monkeypatch.setenv("CODEX_WEB_CC_SUBAGENT_RUNS_ROOT", str(root))

    assert handle_hook_event(
        {"hook_event_name": "SubagentStart", "session_id": SESSION_ID, "agent_id": "agent-abc", "agent_type": "Explore"},
        environ=env,
    )
    runs = scan_active_cc_subagents(parent_broker_pids={SESSION_ID: os.getpid()})
    assert [run["agent_id"] for run in runs[SESSION_ID]] == ["agent-abc"]
    assert runs[SESSION_ID][0]["detail"] == {"role": "Explore", "model": None, "tools": None, "tokens": None}
    assert runs[SESSION_ID][0]["event"]["message_id"] == "cc-subagent:agent-abc"

    assert handle_hook_event(
        {"hook_event_name": "SubagentStop", "session_id": SESSION_ID, "agent_id": "agent-abc"},
        environ=env,
    )
    assert scan_active_cc_subagents(parent_broker_pids={SESSION_ID: os.getpid()}) == {}


def test_cc_hook_records_never_project_without_matching_web_broker(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "cc-subagent-runs"
    env = _env(root)
    monkeypatch.setenv("CODEX_WEB_CC_SUBAGENT_RUNS_ROOT", str(root))

    assert handle_hook_event(
        {"hook_event_name": "SubagentStart", "session_id": SESSION_ID, "agent_id": "agent-abc"},
        environ=env,
    )
    assert scan_active_cc_subagents(parent_broker_pids={SESSION_ID: 999_999_999}) == {}
    assert not handle_hook_event(
        {"hook_event_name": "SubagentStart", "session_id": SESSION_ID, "agent_id": "agent-terminal"},
        environ={"CODEX_WEB_CC_SUBAGENT_RUNS_ROOT": str(root)},
    )
