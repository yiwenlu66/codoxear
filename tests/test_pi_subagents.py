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
