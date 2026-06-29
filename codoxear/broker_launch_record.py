from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from codoxear.util import append_launch_attempt as _append_launch_attempt
from codoxear.util import read_launch_attempts as _read_launch_attempts
from codoxear.util import redacted_launch_attempt_persist_record as _redacted_launch_attempt_persist_record


def _record_broker_launch_attempt(
    record: dict[str, Any],
    *,
    owner_tag: str,
    launch_attempts_path: Path,
    stderr: Any,
) -> None:
    if owner_tag != "web":
        return
    try:
        launch_id = record.get("launch_id")
        if isinstance(launch_id, str) and launch_id and "submitted_user_messages" not in record:
            for prev in _read_launch_attempts(path=launch_attempts_path, max_records=100, max_age_s=24 * 3600):
                if prev.get("launch_id") != launch_id:
                    continue
                submitted = prev.get("submitted_user_messages")
                if isinstance(submitted, list) and submitted:
                    record = dict(record)
                    record["submitted_user_messages"] = submitted
                break
        rec = _append_launch_attempt(_redacted_launch_attempt_persist_record(record), path=launch_attempts_path)
        if rec.get("state") == "failed":
            stderr.write(
                "error: session launch failed: "
                f"{rec.get('launch_id')}: {rec.get('stage')}: {rec.get('error')}\n"
            )
            stderr.flush()
    except Exception as e:
        stderr.write(f"error: failed to write launch attempt record: {type(e).__name__}: {e}\n")
        stderr.flush()


def _broker_launch_record(
    *,
    stage: str,
    error: str,
    cwd: str,
    start_ts: float,
    agent_backend: str,
    model_provider: str,
    preferred_auth_method: str,
    model: str,
    reasoning_effort: str,
    service_tier: str,
    agent_pid: int | None = None,
    log_path: Path | None = None,
    exit_code: int | None = None,
) -> dict[str, Any]:
    return {
        "launch_id": (os.environ.get("CODEX_WEB_LAUNCH_ID") or "").strip() or None,
        "state": "failed",
        "stage": stage,
        "error": error,
        "agent_backend": agent_backend,
        "cwd": cwd,
        "created_ts": start_ts,
        "updated_ts": time.time(),
        "broker_pid": os.getpid(),
        "agent_pid": agent_pid,
        "exit_code": exit_code,
        "log_path": str(log_path) if log_path else None,
        "transport": (os.environ.get("CODEX_WEB_TRANSPORT") or "").strip() or None,
        "tmux_session": (os.environ.get("CODEX_WEB_TMUX_SESSION") or "").strip() or None,
        "tmux_window": (os.environ.get("CODEX_WEB_TMUX_WINDOW") or "").strip() or None,
        "spawn_nonce": (os.environ.get("CODEX_WEB_SPAWN_NONCE") or "").strip() or None,
        "model_provider": model_provider or None,
        "preferred_auth_method": preferred_auth_method or None,
        "model": model or None,
        "reasoning_effort": reasoning_effort or None,
        "service_tier": service_tier or None,
        "resume_session_id": (os.environ.get("CODEX_WEB_RESUME_SESSION_ID") or "").strip() or None,
    }
