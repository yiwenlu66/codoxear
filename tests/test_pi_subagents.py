from __future__ import annotations

import json
import os
from pathlib import Path

from codoxear import util


def _write_status(root: Path, run_id: str, payload: dict) -> None:
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "status.json").write_text(json.dumps(payload), encoding="utf-8")


def test_scan_active_pi_subagents_groups_active_runs_and_ignores_bad_statuses(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "runs"
    monkeypatch.setenv("CODEX_WEB_SUBAGENT_RUNS_ROOT", str(root))
    parent = "/tmp/parent.jsonl"
    _write_status(
        root,
        "running-run",
        {"runId": "running-run", "sessionId": parent, "state": "running", "startedAt": 100, "pid": os.getpid(), "steps": [{"agent": "executor"}]},
    )
    _write_status(
        root,
        "pending-run",
        {"runId": "pending-run", "sessionId": parent, "state": "pending", "startedAt": 101, "pid": os.getpid(), "agent": "critic"},
    )
    _write_status(root, "complete-run", {"runId": "complete-run", "sessionId": parent, "state": "complete", "startedAt": 102})
    malformed = root / "malformed"
    malformed.mkdir(parents=True)
    (malformed / "status.json").write_text("not json", encoding="utf-8")

    assert util.scan_active_pi_subagents(now_monotonic=100.0) == {
        parent: [
            {"run_id": "pending-run", "agent": "critic", "started_at": 101, "detail": {"role": "critic", "model": None, "tools": None, "tokens": None}},
            {"run_id": "running-run", "agent": "executor", "started_at": 100, "detail": {"role": "executor", "model": None, "tools": None, "tokens": None}},
        ]
    }


def test_scan_active_pi_subagents_projects_current_step_telemetry(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "runs"
    monkeypatch.setenv("CODEX_WEB_SUBAGENT_RUNS_ROOT", str(root))
    parent = "/tmp/parent.jsonl"
    _write_status(
        root,
        "rich-run",
        {
            "runId": "rich-run", "sessionId": parent, "state": "running", "startedAt": 100, "pid": os.getpid(), "currentStep": 1,
            "steps": [
                {"agent": "scout", "status": "complete", "model": "old-model", "toolCount": 1, "tokens": {"total": 2}},
                {"agent": "reviewer", "status": "running", "model": "provider/model", "toolCount": 3, "tokens": {"total": 4200}},
            ],
        },
    )

    run = util.scan_active_pi_subagents(now_monotonic=400.0)[parent][0]
    assert run["detail"] == {"role": "reviewer", "model": "provider/model", "tools": 3, "tokens": 4200}


def test_scan_active_pi_subagents_tracks_each_running_parallel_or_chain_step(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "runs"
    monkeypatch.setenv("CODEX_WEB_SUBAGENT_RUNS_ROOT", str(root))
    parent = "/tmp/parent.jsonl"
    status = {
        "lifecycleArtifactVersion": 3,
        "runId": "workflow",
        "sessionId": parent,
        "mode": "parallel",
        "state": "running",
        "startedAt": 100,
        "pid": os.getpid(),
        "currentStep": 1,
        "parallelGroups": [{"start": 0, "count": 2, "stepIndex": 0}],
        "steps": [
            {"agent": "reviewer", "status": "running", "startedAt": 101, "model": "provider/model-a", "toolCount": 2, "tokens": {"input": 1000, "output": 200, "total": 1200}},
            {"agent": "worker", "status": "running", "startedAt": 102, "model": "provider/model-b", "toolCount": 4, "tokens": {"input": 2000, "output": 400, "total": 2400}},
        ],
    }
    _write_status(root, "workflow", status)

    first = util.scan_active_pi_subagents(now_monotonic=500.0)[parent]
    assert [(run["run_id"], run["detail"]) for run in first] == [
        ("workflow:0", {"role": "reviewer", "model": "provider/model-a", "tools": 2, "tokens": 1200}),
        ("workflow:1", {"role": "worker", "model": "provider/model-b", "tools": 4, "tokens": 2400}),
    ]

    status["steps"][0]["toolCount"] = 3
    status["steps"][0]["tokens"] = {"input": 1100, "output": 200, "total": 1300}
    status["steps"][1]["toolCount"] = 5
    status["steps"][1]["tokens"] = {"input": 2100, "output": 400, "total": 2500}
    _write_status(root, "workflow", status)
    same_count_fresh = util.scan_active_pi_subagents(now_monotonic=502.1)[parent]
    assert [run["detail"]["tools"] for run in same_count_fresh] == [3, 5]
    assert [run["detail"]["tokens"] for run in same_count_fresh] == [1300, 2500]

    status["steps"][0]["status"] = "complete"
    _write_status(root, "workflow", status)
    remaining = util.scan_active_pi_subagents(now_monotonic=504.2)[parent]
    assert [(run["run_id"], run["detail"]["role"]) for run in remaining] == [("workflow:1", "worker")]

    status.update({
        "mode": "chain",
        "currentStep": 1,
        "parallelGroups": [],
        "steps": [
            {"agent": "scout", "status": "complete", "model": "old"},
            {"agent": "executor", "status": "running", "model": "current", "toolCount": 1, "tokens": {"total": 84}},
            {"agent": "critic", "status": "pending", "model": "future"},
        ],
    })
    _write_status(root, "workflow", status)
    chain = util.scan_active_pi_subagents(now_monotonic=506.3)[parent]
    assert [(run["run_id"], run["detail"]) for run in chain] == [
        ("workflow:1", {"role": "executor", "model": "current", "tools": 1, "tokens": 84})
    ]


def test_scan_active_pi_subagents_caches_for_two_seconds_then_refreshes(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "runs"
    monkeypatch.setenv("CODEX_WEB_SUBAGENT_RUNS_ROOT", str(root))
    parent = "/tmp/parent.jsonl"
    _write_status(root, "run", {"runId": "run", "sessionId": parent, "state": "running", "startedAt": 1, "pid": os.getpid(), "agent": "executor"})

    assert util.scan_active_pi_subagents(now_monotonic=200.0)[parent][0]["run_id"] == "run"
    _write_status(root, "run", {"runId": "run", "sessionId": parent, "state": "complete", "startedAt": 1, "pid": os.getpid(), "agent": "executor"})
    assert util.scan_active_pi_subagents(now_monotonic=201.9)[parent][0]["run_id"] == "run"
    assert util.scan_active_pi_subagents(now_monotonic=202.1) == {}


def test_scan_active_pi_subagents_returns_empty_for_missing_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CODEX_WEB_SUBAGENT_RUNS_ROOT", str(tmp_path / "missing"))
    assert util.scan_active_pi_subagents(now_monotonic=300.0) == {}


def test_scan_active_pi_subagents_counts_a_workflow_child_once_from_its_own_record(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "runs"
    monkeypatch.setenv("CODEX_WEB_SUBAGENT_RUNS_ROOT", str(root))
    parent = "/tmp/parent.jsonl"
    # Real lifecycle-v3 shape: a workflow root mirrors each child it launched as
    # a step naming the child's run id, and the child writes its own record with
    # its own pid and telemetry. Projecting both would report the fanout twice.
    _write_status(
        root,
        "workflow-a",
        {
            "lifecycleArtifactVersion": 3, "runId": "workflow-a", "sessionId": parent,
            "mode": "workflow", "state": "running", "startedAt": 100, "pid": os.getpid(),
            "steps": [
                {"agent": "executor", "status": "completed", "async": True, "runId": "child-done"},
                {"agent": "theorist", "status": "running", "async": True, "runId": "child-live"},
            ],
        },
    )
    _write_status(
        root,
        "child-done",
        {
            "runId": "child-done", "sessionId": parent, "mode": "single", "state": "complete",
            "startedAt": 101, "pid": os.getpid(), "steps": [{"agent": "executor", "status": "complete"}],
        },
    )
    _write_status(
        root,
        "child-live",
        {
            "runId": "child-live", "sessionId": parent, "mode": "single", "state": "running",
            "startedAt": 102, "pid": os.getpid(),
            "steps": [{"agent": "theorist", "status": "running", "model": "provider/model-b", "toolCount": 4, "tokens": {"total": 900}}],
        },
    )

    runs = util.scan_active_pi_subagents(now_monotonic=600.0)[parent]
    assert [(run["run_id"], run["detail"]) for run in runs] == [
        ("child-live:0", {"role": "theorist", "model": "provider/model-b", "tools": 4, "tokens": 900}),
    ]


def test_scan_active_pi_subagents_drops_a_step_mirror_of_a_settled_child(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "runs"
    monkeypatch.setenv("CODEX_WEB_SUBAGENT_RUNS_ROOT", str(root))
    parent = "/tmp/parent.jsonl"
    # The mirror lags its child: the child's own record is the authority for the
    # child's liveness, so a running step mirror of a settled child reports none.
    _write_status(
        root,
        "workflow-a",
        {
            "runId": "workflow-a", "sessionId": parent, "mode": "workflow", "state": "running",
            "startedAt": 100, "pid": os.getpid(),
            "steps": [{"agent": "theorist", "status": "running", "async": True, "runId": "child-live"}],
        },
    )
    _write_status(
        root,
        "child-live",
        {
            "runId": "child-live", "sessionId": parent, "mode": "single", "state": "failed",
            "startedAt": 101, "pid": os.getpid(), "steps": [{"agent": "theorist", "status": "failed"}],
        },
    )

    assert util.scan_active_pi_subagents(now_monotonic=700.0) == {}


def test_scan_active_pi_subagents_keeps_steps_without_their_own_run_record(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "runs"
    monkeypatch.setenv("CODEX_WEB_SUBAGENT_RUNS_ROOT", str(root))
    parent = "/tmp/parent.jsonl"
    # In-process host steps have no run record of their own, so the parent file is
    # the only evidence for them. A step whose child directories were reaped
    # keeps its mirror rather than dropping the child entirely.
    _write_status(
        root,
        "workflow-b",
        {
            "runId": "workflow-b", "sessionId": parent, "mode": "workflow", "state": "running",
            "startedAt": 100, "pid": os.getpid(),
            "steps": [
                {"agent": "host-step", "status": "running", "model": "provider/host"},
                {"agent": "reaped-child", "status": "running", "async": True, "runId": "absent-child"},
            ],
        },
    )

    runs = util.scan_active_pi_subagents(now_monotonic=800.0)[parent]
    assert [(run["run_id"], run["detail"]["role"]) for run in runs] == [
        ("workflow-b:0", "host-step"),
        ("workflow-b:1", "reaped-child"),
    ]


def test_scan_active_pi_subagents_projects_no_row_for_a_childless_workflow_root(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "runs"
    monkeypatch.setenv("CODEX_WEB_SUBAGENT_RUNS_ROOT", str(root))
    parent = "/tmp/parent.jsonl"
    # A workflow root orchestrates the parent turn's children; it is not a child,
    # so it never projects a run-level row of its own (no bare "Subagent" line
    # while the workflow has not launched anything yet).
    _write_status(
        root,
        "workflow-c",
        {
            "runId": "workflow-c", "sessionId": parent, "mode": "workflow", "state": "running",
            "startedAt": 100, "pid": os.getpid(), "steps": [],
        },
    )
    assert util.scan_active_pi_subagents(now_monotonic=900.0) == {}

    _write_status(
        root,
        "workflow-c",
        {
            "runId": "workflow-c", "sessionId": parent, "mode": "workflow", "state": "running",
            "startedAt": 100, "pid": os.getpid(), "steps": [{"agent": "executor"}],
        },
    )
    assert util.scan_active_pi_subagents(now_monotonic=902.1) == {}


def _write_session(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")


_ROUTED = "jev-auto-model:routed:v1"


def _jev_child_session(provider: str = "dexgem-responses", model_id: str = "gpt-6.1-sol", level: str = "high") -> list[dict]:
    """The head pi actually writes for a jev-routed child (observed shape)."""
    return [
        {"type": "session", "version": 3, "id": "child", "timestamp": "2026-10-24T07:14:41.122Z", "cwd": "/tmp"},
        {"type": "model_change", "provider": "jev-auto", "modelId": "auto-jev-subagent"},
        {"type": "thinking_level_change", "thinkingLevel": "high"},
        {"type": "model_change", "provider": provider, "modelId": model_id},
        {"type": "custom", "customType": _ROUTED, "data": {"placeholder": "jev-auto/auto-jev-subagent", "provider": provider, "modelId": model_id, "level": level}},
    ]


def test_scan_resolves_jev_placeholder_from_child_session_without_events_mirror(tmp_path: Path, monkeypatch) -> None:
    """A routed child's own session file is the authority when the mirror is absent."""
    root = tmp_path / "runs"
    monkeypatch.setenv("CODEX_WEB_SUBAGENT_RUNS_ROOT", str(root))
    parent = "/tmp/parent.jsonl"
    child_session = tmp_path / "sessions" / "child" / "run-0" / "session.jsonl"
    _write_session(child_session, _jev_child_session(provider="dexgem-messages", model_id="kimi-k3", level="medium"))
    _write_status(
        root,
        "routed-run",
        {
            "runId": "routed-run", "sessionId": parent, "state": "running", "startedAt": 100, "pid": os.getpid(),
            "steps": [{"agent": "executor", "status": "running", "model": "jev-auto/auto-jev-subagent:high", "sessionFile": str(child_session), "toolCount": 2, "tokens": {"total": 10}}],
        },
    )
    # No events.jsonl in the run dir at all: this is the case that showed the
    # placeholder in the UI. The marker's own level wins over the launch level.
    assert util.scan_active_pi_subagents(now_monotonic=100.0)[parent][0]["detail"] == {
        "role": "executor", "model": "dexgem-messages/kimi-k3:medium", "tools": 2, "tokens": 10,
    }


def test_scan_resolves_jev_placeholder_from_events_mirror_when_child_session_is_gone(tmp_path: Path, monkeypatch) -> None:
    """The events.jsonl mirror still resolves a child whose session file is unreadable."""
    root = tmp_path / "runs"
    monkeypatch.setenv("CODEX_WEB_SUBAGENT_RUNS_ROOT", str(root))
    parent = "/tmp/parent.jsonl"
    _write_status(
        root,
        "mirrored-run",
        {
            "runId": "mirrored-run", "sessionId": parent, "state": "running", "startedAt": 100, "pid": os.getpid(),
            "steps": [{"agent": "executor", "status": "running", "model": "jev-auto/auto-jev-subagent:high", "sessionFile": str(tmp_path / "missing.jsonl")}],
        },
    )
    mirror = root / "mirrored-run" / "events.jsonl"
    mirror.write_text(
        json.dumps({"type": "entry_appended", "entry": {"type": "custom", "customType": _ROUTED, "data": {"provider": "dexgem-responses", "modelId": "gpt-6.1-sol", "level": "xhigh"}}}) + "\n",
        encoding="utf-8",
    )

    assert util.scan_active_pi_subagents(now_monotonic=100.0)[parent][0]["detail"]["model"] == "dexgem-responses/gpt-6.1-sol:xhigh"


def test_scan_keeps_placeholder_when_nothing_records_the_routing(tmp_path: Path, monkeypatch) -> None:
    """Unrouted or unreadable children degrade to the recorded value, never to a guess."""
    root = tmp_path / "runs"
    monkeypatch.setenv("CODEX_WEB_SUBAGENT_RUNS_ROOT", str(root))
    parent = "/tmp/parent.jsonl"
    unresolved = tmp_path / "sessions" / "unresolved.jsonl"
    _write_session(unresolved, _jev_child_session()[:3])  # launch placeholder only: routing never recorded
    _write_session(tmp_path / "sessions" / "placeholders-only.jsonl", [
        {"type": "session"},
        {"type": "model_change", "provider": "jev-auto", "modelId": "auto-jev-subagent"},
        {"type": "custom", "customType": "something-else", "data": {"provider": "dexgem-responses", "modelId": "gpt-6.1-sol"}},
    ])
    _write_status(
        root,
        "a",
        {"runId": "a", "sessionId": parent, "state": "running", "startedAt": 100, "pid": os.getpid(),
         "steps": [{"agent": "executor", "status": "running", "model": "jev-auto/auto-jev-subagent:high", "sessionFile": str(unresolved)}]},
    )
    assert util._resolve_jev_routed_model(root / "a", "jev-auto/auto-jev-subagent:high", str(unresolved)) == "jev-auto/auto-jev-subagent:high"
    assert util._resolve_jev_routed_model(root / "a", "jev-auto/auto-jev-subagent:high", str(tmp_path / "sessions" / "placeholders-only.jsonl")) == "jev-auto/auto-jev-subagent:high"
    assert util._resolve_jev_routed_model(root / "a", "jev-auto/auto-jev-subagent:high", None) == "jev-auto/auto-jev-subagent:high"
    # A resolved model_change with no marker still names the real model, with the
    # thinking level taken from the child's own level change.
    model_change_only = tmp_path / "sessions" / "model-change-only.jsonl"
    _write_session(model_change_only, [
        {"type": "model_change", "provider": "jev-auto", "modelId": "auto-jev-subagent"},
        {"type": "thinking_level_change", "thinkingLevel": "low"},
        {"type": "model_change", "provider": "dexgem-completions", "modelId": "deepseek-flash"},
    ])
    assert util._resolve_jev_routed_model(root / "a", "jev-auto/auto-jev-subagent:high", str(model_change_only)) == "dexgem-completions/deepseek-flash:low"


def test_jev_marker_inside_child_output_is_not_read_as_routing(tmp_path: Path, monkeypatch) -> None:
    """The marker string appears in tool output; only a real routed entry resolves."""
    root = tmp_path / "runs"
    monkeypatch.setenv("CODEX_WEB_SUBAGENT_RUNS_ROOT", str(root))
    noisy = tmp_path / "sessions" / "noisy.jsonl"
    _write_session(noisy, [
        {"type": "model_change", "provider": "jev-auto", "modelId": "auto-jev-subagent"},
        {"type": "message_start", "message": {"role": "user", "content": [{"type": "text", "text": f"read the spec: customType {_ROUTED} provider dexgem-responses modelId gpt-6-luna"}]}},
        {"type": "tool_execution_end", "toolName": "bash", "result": {"content": [{"type": "text", "text": '{"type":"custom","customType":"jev-auto-model:routed:v1","data":{"provider":"fake","modelId":"fake-model"}}'}]}},
    ])
    assert util._resolve_jev_routed_model(root / "noisy", "jev-auto/auto-jev-subagent:high", str(noisy)) == "jev-auto/auto-jev-subagent:high"

    noisy_events = root / "noisy"
    noisy_events.mkdir(parents=True, exist_ok=True)
    (noisy_events / "events.jsonl").write_text(
        json.dumps({"type": "tool_execution_end", "result": {"content": [{"type": "text", "text": f"spec mentions {_ROUTED}"}]}}) + "\n",
        encoding="utf-8",
    )
    assert util._resolve_jev_routed_model(noisy_events, "jev-auto/auto-jev-subagent:high", None) == "jev-auto/auto-jev-subagent:high"


def test_jev_resolution_reads_only_the_child_session_head(tmp_path: Path) -> None:
    """Routing sits in the head; a marker only past the cap must not cost a full read."""
    huge = tmp_path / "huge.jsonl"
    _write_session(huge, [
        {"type": "model_change", "provider": "jev-auto", "modelId": "auto-jev-subagent"},
        {"type": "message_start", "message": {"role": "user", "content": [{"type": "text", "text": "x" * (util._JEV_SESSION_HEAD_BYTES + 4096)}]}},
        {"type": "custom", "customType": _ROUTED, "data": {"provider": "dexgem-responses", "modelId": "beyond-cap", "level": "high"}},
    ])
    assert util._resolve_jev_routed_model(tmp_path / "run", "jev-auto/auto-jev-subagent:high", str(huge)) == "jev-auto/auto-jev-subagent:high"
