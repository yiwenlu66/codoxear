from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Callable

from .agent_backend import normalize_agent_backend
from .launch_config import provider_choice_for_settings
from .util import append_launch_attempt
from .util import read_launch_attempts
from .util import redacted_launch_attempt_persist_record
from .util import redact_launch_failure_text


ProviderChoiceFunc = Callable[[str | None, str | None], str]


def clean_optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    out = value.strip()
    return out or None


def record_launch_attempt(record: dict[str, Any], *, path: Path, stderr: Any | None = None) -> dict[str, Any]:
    rec = append_launch_attempt(redacted_launch_attempt_persist_record(record), path=path)
    if rec.get("state") == "failed":
        stream = stderr if stderr is not None else sys.stderr
        stream.write(
            "error: session launch failed: "
            f"{rec.get('launch_id')}: {rec.get('stage')}: {rec.get('error')}\n"
        )
        stream.flush()
    return rec


def launch_attempt_id(record: dict[str, Any]) -> str:
    raw = clean_optional_text(record.get("launch_id"))
    if raw is None:
        updated_ts = record.get("updated_ts", record.get("created_ts", 0))
        try:
            millis = int(float(updated_ts) * 1000)
        except (TypeError, ValueError):
            millis = 0
        raw = f"launch-{millis}"
    return raw


def latest_launch_attempt(launch_id: str, *, path: Path) -> dict[str, Any] | None:
    needle = str(launch_id or "").strip()
    if not needle:
        return None
    for rec in read_launch_attempts(path=path, max_records=100, max_age_s=24 * 3600):
        if launch_attempt_id(rec) == needle:
            return rec
    return None


def submitted_user_messages(record: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(record, dict):
        return []
    raw = record.get("submitted_user_messages")
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        ts = item.get("ts")
        ts_out = float(ts) if isinstance(ts, (int, float)) else float(record.get("updated_ts", time.time()) or time.time())
        source = clean_optional_text(item.get("source")) or "send"
        out.append({"text": text, "ts": ts_out, "source": source})
    return out


def launch_failure_tail(record: dict[str, Any]) -> str:
    for key in ("pty_tail", "tmux_pane_tail"):
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            return val[-4000:]
    return ""


def launch_attempt_transcript_payload(record: dict[str, Any]) -> dict[str, Any]:
    launch_id = launch_attempt_id(record)
    ts = record.get("updated_ts", record.get("created_ts", time.time()))
    ts_f = float(ts) if isinstance(ts, (int, float)) else time.time()
    events: list[dict[str, Any]] = []
    for msg in submitted_user_messages(record):
        events.append({"role": "user", "text": msg["text"], "ts": msg["ts"]})
    if record.get("state") == "failed":
        stage = clean_optional_text(record.get("stage"))
        err = redact_launch_failure_text(record.get("error")) or "session launch failed"
        lines = ["Session launch failed before a transcript log was created."]
        if stage:
            lines.append(f"Stage: {stage}")
        lines.append(f"Error: {err}")
        agent_status = record.get("agent_exit_status", record.get("exit_code"))
        broker_status = record.get("broker_exit_status")
        if isinstance(agent_status, int):
            lines.append(f"Agent exit status: {agent_status}")
        if isinstance(broker_status, int):
            lines.append(f"Broker exit status: {broker_status}")
        tail = redact_launch_failure_text(launch_failure_tail(record))
        if tail:
            lines.extend(["", "Pre-log terminal tail:", tail])
        events.append({"role": "assistant", "text": "\n".join(lines), "ts": ts_f, "message_class": "error"})
    return {
        "transcript_state": "failed",
        "thread_id": launch_id or None,
        "log_path": None,
        "live_cursor": None,
        "history_cursor": None,
        "events": events,
        "has_older": False,
        "busy": False,
        "queue_len": 0,
        "token": None,
    }


def launch_attempt_transcript_for_session_id(
    session_id: str,
    *,
    path: Path,
    default_agent_backend: str,
    unattended_default_idle_minutes: int,
    unattended_default_max_injections: int,
) -> dict[str, Any] | None:
    rec = latest_launch_attempt(session_id, path=path)
    if rec is None or rec.get("state") != "failed":
        return None
    row = launch_attempt_row(
        rec,
        default_agent_backend=default_agent_backend,
        unattended_default_idle_minutes=unattended_default_idle_minutes,
        unattended_default_max_injections=unattended_default_max_injections,
    )
    if row is None or row.get("session_id") != session_id:
        return None
    return launch_attempt_transcript_payload(rec)


def launch_attempt_row(
    record: dict[str, Any],
    *,
    default_agent_backend: str,
    unattended_default_idle_minutes: int,
    unattended_default_max_injections: int,
    provider_choice_func: ProviderChoiceFunc | None = None,
) -> dict[str, Any] | None:
    launch_id = launch_attempt_id(record)
    state = clean_optional_text(record.get("state")) or "starting"
    if state in {"live", "log_bound", "broker_spawned", "broker_meta_bound"}:
        return None
    cwd = clean_optional_text(record.get("cwd")) or "?"
    start_ts_raw = record.get("created_ts", record.get("start_ts", record.get("updated_ts", time.time())))
    updated_ts_raw = record.get("updated_ts", start_ts_raw)
    try:
        start_ts = float(start_ts_raw)
    except (TypeError, ValueError):
        start_ts = time.time()
    try:
        updated_ts = float(updated_ts_raw)
    except (TypeError, ValueError):
        updated_ts = start_ts
    backend = normalize_agent_backend(record.get("agent_backend"), default=default_agent_backend)
    provider = clean_optional_text(record.get("model_provider"))
    preferred_auth = clean_optional_text(record.get("preferred_auth_method"))
    choose_provider = provider_choice_func or (lambda model_provider, preferred_auth_method: provider_choice_for_settings(model_provider=model_provider, preferred_auth_method=preferred_auth_method))
    failed = state == "failed"
    return {
        "session_id": launch_id,
        "thread_id": launch_id,
        "pid": None,
        "broker_pid": record.get("broker_pid") if isinstance(record.get("broker_pid"), int) else None,
        "agent_backend": backend,
        "owned": True,
        "transport": clean_optional_text(record.get("transport")),
        "cwd": cwd,
        "start_ts": start_ts,
        "updated_ts": updated_ts,
        "log_path": None,
        "state_busy": False,
        "queue_len": 0,
        "token": None,
        "thinking": 0,
        "tools": 0,
        "system": 0,
        "unattended_enabled": False,
        "unattended_cooldown_minutes": unattended_default_idle_minutes,
        "unattended_remaining_injections": unattended_default_max_injections,
        "alias": "",
        "files": [],
        "git_branch": "",
        "model_provider": provider,
        "preferred_auth_method": preferred_auth,
        "provider_choice": choose_provider(provider, preferred_auth),
        "model": clean_optional_text(record.get("model")),
        "reasoning_effort": clean_optional_text(record.get("reasoning_effort")),
        "service_tier": clean_optional_text(record.get("service_tier")),
        "tmux_session": clean_optional_text(record.get("tmux_session")),
        "tmux_window": clean_optional_text(record.get("tmux_window")),
        "priority_offset": 0.0,
        "snooze_until": None,
        "dependency_session_id": None,
        "time_priority": 1.0,
        "base_priority": 1.0,
        "final_priority": 1.0,
        "blocked": False,
        "snoozed": False,
        "busy": False,
        "spawn_nonce": clean_optional_text(record.get("spawn_nonce")),
        "launch_id": launch_id,
        "launch_state": state,
        "launch_error": redact_launch_failure_text(record.get("error")) or ("session launch failed" if failed else ""),
        "launch_stage": clean_optional_text(record.get("stage")),
        "submitted_user_message_count": len(submitted_user_messages(record)),
    }
