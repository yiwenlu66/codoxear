from __future__ import annotations

import json
import os
from pathlib import Path

from codoxear.util import scan_active_codex_subagents


def _write_rollout(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "session_meta", "payload": payload}) + "\n", encoding="utf-8")


def test_scan_active_codex_subagents_groups_live_child_headers_by_parent_thread(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "codex" / "sessions"
    child = sessions_dir / "2026" / "08" / "02" / "rollout-child.jsonl"
    _write_rollout(
        child,
        {
            "id": "child-thread",
            "source": {"subagent": {"thread_spawn": {"parent_thread_id": "parent-thread"}}},
        },
    )
    os.utime(child, (1.0, 1.0))

    active = scan_active_codex_subagents(
        sessions_dirs=(sessions_dir,),
        now_monotonic=100.0,
        now_wall=100.0,
        writable_paths={child},
    )

    assert [
        {key: run[key] for key in ("thread_id", "log_path", "updated_at")}
        for run in active["parent-thread"]
    ] == [
        {"thread_id": "child-thread", "log_path": str(child), "updated_at": 1.0}
    ]
    assert active["parent-thread"][0]["detail"] == {"role": "Subagent", "model": None, "tools": None, "tokens": None}
    assert active["parent-thread"][0]["event"]["message_id"] == "codex-subagent:child-thread"
    assert scan_active_codex_subagents(
        sessions_dirs=(sessions_dir,),
        now_monotonic=101.0,
        now_wall=100.0,
        writable_paths=set(),
    ) == {}


def test_scan_active_codex_subagents_uses_recent_mtime_but_ignores_retained_children(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "codex" / "sessions"
    recent = sessions_dir / "2026" / "08" / "02" / "rollout-recent.jsonl"
    stale = sessions_dir / "2026" / "08" / "01" / "rollout-stale.jsonl"
    _write_rollout(
        recent,
        {
            "id": "recent-child",
            "source": {"subagent": {"thread_spawn": {"parent_thread_id": "parent-thread"}}},
        },
    )
    _write_rollout(
        stale,
        {
            "id": "stale-child",
            "source": {"subagent": {"thread_spawn": {"parent_thread_id": "parent-thread"}}},
        },
    )
    os.utime(recent, (97.0, 97.0))
    os.utime(stale, (91.0, 91.0))

    active = scan_active_codex_subagents(
        sessions_dirs=(sessions_dir,),
        now_monotonic=200.0,
        now_wall=100.0,
        writable_paths=set(),
    )

    assert [run["thread_id"] for run in active["parent-thread"]] == ["recent-child"]


def test_scan_active_codex_subagents_excludes_completed_child_even_with_writable_log(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "codex" / "sessions"
    child = sessions_dir / "2026" / "08" / "02" / "rollout-complete.jsonl"
    _write_rollout(
        child,
        {
            "id": "complete-child",
            "source": {"subagent": {"thread_spawn": {"parent_thread_id": "parent-thread"}}},
        },
    )
    with child.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"type": "event_msg", "payload": {"type": "task_complete"}}) + "\n")

    assert scan_active_codex_subagents(
        sessions_dirs=(sessions_dir,),
        now_monotonic=300.0,
        now_wall=300.0,
        writable_paths={child},
    ) == {}
