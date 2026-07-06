from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from .rollout_idle import _compute_idle_from_log


POST_LOG_BOUND_BACKEND_STOPPED_TEXT = "The backend process stopped before completing this turn."


def clean_record_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    out = value.strip()
    return out or None


def post_log_bound_stage(stage: Any) -> bool:
    text = clean_record_text(stage)
    return bool(text and text.endswith("_after_log_bind"))


def record_is_post_log_bound_recovery(record: Mapping[str, Any] | None) -> bool:
    if not isinstance(record, Mapping):
        return False
    if record.get("state") != "failed":
        return False
    if not post_log_bound_stage(record.get("stage")):
        return False
    log_path = clean_record_text(record.get("log_path"))
    return bool(log_path)


def log_needs_post_log_bound_recovery(
    log_path: Path | str | None,
    *,
    compute_idle_from_log: Callable[[Path], bool | None] | None = None,
) -> bool:
    if log_path is None:
        return False
    path = Path(log_path)
    if not path.exists():
        return False
    compute = compute_idle_from_log or _compute_idle_from_log
    try:
        idle = compute(path)
    except Exception:
        # A dead backend with an unreadable bound log still needs a visible
        # lifecycle outcome; the recovery row points at the log without mutating it.
        return True
    return idle is False


def compose_post_log_bound_failure_record(
    *,
    session_id: str | None,
    thread_id: str | None,
    launch_id: str | None,
    stage: str,
    error: str,
    agent_backend: str | None,
    cwd: str | None,
    log_path: Path | str,
    created_ts: Any,
    broker_pid: int | None,
    agent_pid: int | None,
    transport: str | None = None,
    tmux_session: str | None = None,
    tmux_window: str | None = None,
    spawn_nonce: str | None = None,
    model_provider: str | None = None,
    preferred_auth_method: str | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
    service_tier: str | None = None,
    resume_session_id: str | None = None,
    agent_exit_status: int | None = None,
    broker_exit_status: int | None = None,
    pty_tail: str | None = None,
    tmux_pane_tail: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "launch_id": clean_record_text(launch_id),
        "session_id": clean_record_text(session_id),
        "thread_id": clean_record_text(thread_id) or clean_record_text(session_id),
        "state": "failed",
        "stage": stage,
        "error": error,
        "agent_backend": clean_record_text(agent_backend),
        "cwd": cwd,
        "created_ts": created_ts,
        "broker_pid": broker_pid if isinstance(broker_pid, int) else None,
        "agent_pid": agent_pid if isinstance(agent_pid, int) else None,
        "log_path": str(log_path),
        "transport": clean_record_text(transport),
        "tmux_session": clean_record_text(tmux_session),
        "tmux_window": clean_record_text(tmux_window),
        "spawn_nonce": clean_record_text(spawn_nonce),
        "model_provider": clean_record_text(model_provider),
        "preferred_auth_method": clean_record_text(preferred_auth_method),
        "model": clean_record_text(model),
        "reasoning_effort": clean_record_text(reasoning_effort),
        "service_tier": clean_record_text(service_tier),
        "resume_session_id": clean_record_text(resume_session_id),
    }
    if isinstance(agent_exit_status, int):
        record["agent_exit_status"] = agent_exit_status
    if isinstance(broker_exit_status, int):
        record["broker_exit_status"] = broker_exit_status
    if isinstance(pty_tail, str) and pty_tail:
        record["pty_tail"] = pty_tail[-4000:]
    if isinstance(tmux_pane_tail, str) and tmux_pane_tail:
        record["tmux_pane_tail"] = tmux_pane_tail[-4000:]
    if isinstance(extra, Mapping):
        record.update(dict(extra))
    return {key: value for key, value in record.items() if value is not None}
