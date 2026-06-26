from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ListingPriority:
    time_priority: float
    base_priority: float
    final_priority: float


@dataclass(frozen=True)
class ActiveSessionRowFacts:
    session_id: str
    thread_id: str
    pid: int
    broker_pid: int
    agent_backend: str
    owned: bool
    transport: str | None
    cwd: str
    start_ts: float
    updated_ts: float
    log_path: Path | None
    log_exists: bool
    needs_run_settings: bool
    needs_history_scan: bool
    state_busy: bool
    interrupted_idle: bool
    broker_queue_len: int
    last_send_boundary_active: bool
    last_send_log_path: Path | None
    last_send_log_size: int | None
    queue_len: int
    queue_recovery: bool
    pending_attachment: bool
    commit_unknown_send: Mapping[str, Any] | None
    token: Any
    thinking: int
    tools: int
    system: int
    unattended_enabled: bool
    unattended_cooldown_minutes: int
    unattended_remaining_injections: int
    alias: str
    files: list[str]
    cwd_path: Path | None
    model_provider: str | None
    preferred_auth_method: str | None
    provider_choice: str
    model: str | None
    reasoning_effort: str | None
    service_tier: str | None
    tmux_session: str | None
    tmux_window: str | None
    launch_id: str | None
    spawn_nonce: str | None
    priority_offset: float
    snooze_until: float | None
    dependency_session_id: str | None
    time_priority: float
    base_priority: float
    final_priority: float
    blocked: bool
    snoozed: bool


def clip01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def priority_from_elapsed_seconds(elapsed_s: float, *, half_life_seconds: float) -> float:
    if elapsed_s <= 0:
        return 1.0
    return clip01(math.exp(-(math.log(2.0) / float(half_life_seconds)) * float(elapsed_s)))


def sidebar_priority_elapsed_seconds(elapsed_s: float, *, bucket_seconds: float) -> float:
    elapsed = max(0.0, float(elapsed_s))
    bucket = float(bucket_seconds)
    if bucket <= 0:
        return elapsed
    return math.floor(elapsed / bucket) * bucket


def sidebar_time_priority_from_elapsed_seconds(elapsed_s: float, *, half_life_seconds: float, bucket_seconds: float) -> float:
    return priority_from_elapsed_seconds(
        sidebar_priority_elapsed_seconds(elapsed_s, bucket_seconds=bucket_seconds),
        half_life_seconds=half_life_seconds,
    )


def listing_priority(
    *,
    now_ts: float,
    updated_ts: float,
    priority_offset: float,
    blocked: bool,
    snoozed: bool,
    half_life_seconds: float,
    bucket_seconds: float,
) -> ListingPriority:
    elapsed_s = max(0.0, now_ts - updated_ts)
    time_priority = sidebar_time_priority_from_elapsed_seconds(
        elapsed_s,
        half_life_seconds=half_life_seconds,
        bucket_seconds=bucket_seconds,
    )
    base_priority = clip01(time_priority + priority_offset)
    final_priority = 0.0 if (snoozed or blocked) else base_priority
    return ListingPriority(time_priority=time_priority, base_priority=base_priority, final_priority=final_priority)


def _queue_has_recovery_item(queue: Sequence[Any]) -> bool:
    return any(isinstance(item, dict) and (bool(item.get("commit_unknown")) or bool(item.get("orphan_recovery"))) for item in queue)


def _float_from_number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _commit_unknown_text(record: Mapping[str, Any] | None) -> str | None:
    if not isinstance(record, Mapping):
        return None
    text = record.get("text")
    return str(text) if isinstance(text, str) else None


def _commit_unknown_created_ts(record: Mapping[str, Any] | None) -> float | None:
    if not isinstance(record, Mapping):
        return None
    return _float_from_number(record.get("created_ts"))


def _orphan_recovery_ts(direct_record: Mapping[str, Any], queue: Sequence[Any], *, now_ts: float) -> float:
    ts_candidates: list[float] = []
    direct_created_ts = _float_from_number(direct_record.get("created_ts"))
    if direct_created_ts is not None:
        ts_candidates.append(direct_created_ts)
    for item in queue:
        if not isinstance(item, dict):
            continue
        commit_unknown_ts = _float_from_number(item.get("commit_unknown_ts"))
        created_ts = _float_from_number(item.get("created_ts"))
        if commit_unknown_ts is not None:
            ts_candidates.append(commit_unknown_ts)
        elif created_ts is not None:
            ts_candidates.append(created_ts)
    return max([t for t in ts_candidates if math.isfinite(t) and t > 0], default=now_ts)


_PRIVATE_LISTING_KEYS = (
    "_log_path_obj",
    "_cwd_path_obj",
    "log_exists",
    "needs_run_settings",
    "needs_history_scan",
    "state_busy",
    "interrupted_idle",
    "broker_queue_len",
    "last_send_boundary_active",
    "last_send_log_path",
    "last_send_log_size",
)


def build_active_session_row(facts: ActiveSessionRowFacts) -> dict[str, Any]:
    commit_unknown = facts.commit_unknown_send if isinstance(facts.commit_unknown_send, Mapping) else None
    return {
        "session_id": facts.session_id,
        "thread_id": facts.thread_id,
        "pid": facts.pid,
        "broker_pid": facts.broker_pid,
        "agent_backend": facts.agent_backend,
        "owned": facts.owned,
        "transport": facts.transport,
        "cwd": facts.cwd,
        "start_ts": facts.start_ts,
        "updated_ts": facts.updated_ts,
        "log_path": (str(facts.log_path) if facts.log_path is not None else None),
        "_log_path_obj": facts.log_path,
        "log_exists": facts.log_exists,
        "needs_run_settings": facts.needs_run_settings,
        "needs_history_scan": facts.needs_history_scan,
        "state_busy": facts.state_busy,
        "interrupted_idle": facts.interrupted_idle,
        "broker_queue_len": facts.broker_queue_len,
        "last_send_boundary_active": facts.last_send_boundary_active,
        "last_send_log_path": facts.last_send_log_path,
        "last_send_log_size": facts.last_send_log_size,
        "queue_len": facts.queue_len,
        "queue_recovery": facts.queue_recovery,
        "pending_attachment": facts.pending_attachment,
        "commit_unknown_send": bool(commit_unknown),
        "commit_unknown_send_text": _commit_unknown_text(commit_unknown),
        "commit_unknown_send_ts": _commit_unknown_created_ts(commit_unknown),
        "token": facts.token,
        "thinking": facts.thinking,
        "tools": facts.tools,
        "system": facts.system,
        "unattended_enabled": facts.unattended_enabled,
        "unattended_cooldown_minutes": facts.unattended_cooldown_minutes,
        "unattended_remaining_injections": facts.unattended_remaining_injections,
        "alias": facts.alias,
        "files": list(facts.files),
        "_cwd_path_obj": facts.cwd_path,
        "model_provider": facts.model_provider,
        "preferred_auth_method": facts.preferred_auth_method,
        "provider_choice": facts.provider_choice,
        "model": facts.model,
        "reasoning_effort": facts.reasoning_effort,
        "service_tier": facts.service_tier,
        "tmux_session": facts.tmux_session,
        "tmux_window": facts.tmux_window,
        "launch_id": facts.launch_id,
        "spawn_nonce": facts.spawn_nonce,
        "priority_offset": facts.priority_offset,
        "snooze_until": facts.snooze_until,
        "dependency_session_id": facts.dependency_session_id,
        "time_priority": facts.time_priority,
        "base_priority": facts.base_priority,
        "final_priority": facts.final_priority,
        "blocked": facts.blocked,
        "snoozed": facts.snoozed,
    }


def build_launch_attempt_rows(
    *,
    records: Iterable[dict[str, Any]],
    hidden_failure_ids: set[str],
    active_launch_ids: set[str],
    active_spawn_nonces: set[str],
    row_from_record: Callable[[dict[str, Any]], dict[str, Any] | None],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        row = row_from_record(record)
        if row is None:
            continue
        if row["session_id"] in hidden_failure_ids:
            continue
        launch_id = row.get("launch_id")
        if isinstance(launch_id, str) and launch_id and launch_id in active_launch_ids:
            continue
        nonce = row.get("spawn_nonce")
        if isinstance(nonce, str) and nonce and nonce in active_spawn_nonces:
            continue
        rows.append(row)
    return rows


def build_orphan_recovery_rows(
    *,
    active_session_ids: set[str],
    commit_unknown_sends: Mapping[Any, Any],
    queues: Mapping[Any, Any],
    existing_session_ids: set[str],
    now_ts: float,
    unattended_default_idle_minutes: int,
    unattended_default_max_injections: int,
) -> list[dict[str, Any]]:
    direct_unknowns = {
        str(sid): dict(record)
        for sid, record in commit_unknown_sends.items()
        if str(sid) not in active_session_ids and isinstance(record, dict)
    }
    orphan_queues = {
        str(sid): list(queue)
        for sid, queue in queues.items()
        if str(sid) not in active_session_ids
        and isinstance(queue, list)
        and (str(sid) in direct_unknowns or _queue_has_recovery_item(queue))
    }
    rows: list[dict[str, Any]] = []
    for sid in sorted(set(direct_unknowns) | set(orphan_queues)):
        if sid in existing_session_ids:
            continue
        direct_record = direct_unknowns.get(sid) or {}
        queue = orphan_queues.get(sid) or []
        ts = _orphan_recovery_ts(direct_record, queue, now_ts=now_ts)
        rows.append(
            {
                "session_id": sid,
                "thread_id": sid,
                "pid": 0,
                "broker_pid": 0,
                "agent_backend": "codex",
                "owned": False,
                "transport": None,
                "cwd": "recovery needed",
                "start_ts": ts,
                "updated_ts": ts,
                "log_path": None,
                "queue_len": len(queue),
                "pending_attachment": False,
                "commit_unknown_send": bool(direct_record),
                "commit_unknown_send_text": _commit_unknown_text(direct_record),
                "commit_unknown_send_ts": _commit_unknown_created_ts(direct_record),
                "token": None,
                "thinking": 0,
                "tools": 0,
                "system": 0,
                "unattended_enabled": False,
                "unattended_cooldown_minutes": unattended_default_idle_minutes,
                "unattended_remaining_injections": unattended_default_max_injections,
                "alias": "Recovery needed",
                "files": [],
                "model_provider": None,
                "preferred_auth_method": None,
                "provider_choice": "openai-api",
                "model": None,
                "reasoning_effort": None,
                "service_tier": None,
                "tmux_session": None,
                "tmux_window": None,
                "launch_id": None,
                "spawn_nonce": None,
                "priority_offset": 0.0,
                "snooze_until": None,
                "dependency_session_id": None,
                "time_priority": 1.0,
                "base_priority": 1.0,
                "final_priority": 1.0,
                "blocked": False,
                "snoozed": False,
                "busy": False,
                "git_branch": None,
                "orphan_recovery": True,
                "transcript_state": "failed",
            }
        )
    return rows


def build_public_session_row(staged_row: Mapping[str, Any], *, git_branch: str | None, busy: bool) -> dict[str, Any]:
    row = dict(staged_row)
    for key in _PRIVATE_LISTING_KEYS:
        row.pop(key, None)
    row["git_branch"] = git_branch
    row["busy"] = bool(busy)
    return row


def sort_session_rows(rows: list[dict[str, Any]]) -> None:
    rows.sort(
        key=lambda item: (
            -float(item.get("final_priority", 0.0)),
            -float(item.get("updated_ts", item.get("start_ts", 0.0))),
            -float(item.get("start_ts", 0.0)),
            str(item.get("session_id", "")),
        )
    )
