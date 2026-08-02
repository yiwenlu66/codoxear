from __future__ import annotations

import json
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
        {"runId": "running-run", "sessionId": parent, "state": "running", "startedAt": 100, "steps": [{"agent": "executor"}]},
    )
    _write_status(
        root,
        "pending-run",
        {"runId": "pending-run", "sessionId": parent, "state": "pending", "startedAt": 101, "agent": "critic"},
    )
    _write_status(root, "complete-run", {"runId": "complete-run", "sessionId": parent, "state": "complete", "startedAt": 102})
    malformed = root / "malformed"
    malformed.mkdir(parents=True)
    (malformed / "status.json").write_text("not json", encoding="utf-8")

    assert util.scan_active_pi_subagents(now_monotonic=100.0) == {
        parent: [
            {"run_id": "pending-run", "agent": "critic", "started_at": 101},
            {"run_id": "running-run", "agent": "executor", "started_at": 100},
        ]
    }


def test_scan_active_pi_subagents_caches_for_two_seconds_then_refreshes(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "runs"
    monkeypatch.setenv("CODEX_WEB_SUBAGENT_RUNS_ROOT", str(root))
    parent = "/tmp/parent.jsonl"
    _write_status(root, "run", {"runId": "run", "sessionId": parent, "state": "running", "startedAt": 1, "agent": "executor"})

    assert util.scan_active_pi_subagents(now_monotonic=200.0)[parent][0]["run_id"] == "run"
    _write_status(root, "run", {"runId": "run", "sessionId": parent, "state": "complete", "startedAt": 1, "agent": "executor"})
    assert util.scan_active_pi_subagents(now_monotonic=201.9)[parent][0]["run_id"] == "run"
    assert util.scan_active_pi_subagents(now_monotonic=202.1) == {}


def test_scan_active_pi_subagents_returns_empty_for_missing_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CODEX_WEB_SUBAGENT_RUNS_ROOT", str(tmp_path / "missing"))
    assert util.scan_active_pi_subagents(now_monotonic=300.0) == {}
