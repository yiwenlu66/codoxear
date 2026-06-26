from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping

from .session_listing import build_public_session_row
from .session_listing import listing_priority
from .session_model import Session
from .session_store import SessionStore


@dataclass(frozen=True)
class BrokerRuntimeState:
    busy: bool
    queue_len: int
    interrupted_idle: bool = False

    @property
    def allows_interrupted_idle_override(self) -> bool:
        return (not self.busy) and self.queue_len == 0 and self.interrupted_idle


@dataclass(frozen=True)
class HistoryBackfillUpdate:
    updated_ts: float


@dataclass(frozen=True)
class RunSettingsUpdate:
    model_provider: str | None
    preferred_auth_method: str | None
    model: str | None
    reasoning_effort: str | None


@dataclass(frozen=True)
class RuntimeStatus:
    broker: BrokerRuntimeState
    log_exists: bool
    log_idle: bool | None
    send_boundary_unresolved: bool
    busy: bool
    remote_ready: bool


@dataclass(frozen=True)
class ListingRuntimeProbes:
    last_conversation_ts_from_tail: Callable[[Path], float | None]
    read_run_settings_from_log: Callable[[Path, str], tuple[str | None, str | None, str | None]]
    log_size_or_none: Callable[[Path | None], int | None]
    send_boundary_unresolved: Callable[[str, Path | None, int | None], bool]
    idle_from_log_path: Callable[[str, Path], bool]
    current_git_branch: Callable[[Path], str | None]


@dataclass(frozen=True)
class ListingRuntimeRowsResult:
    rows: list[dict[str, Any]]
    recent_cwd_dirty: bool


def apply_history_backfill(
    session: Session | None,
    *,
    expected_log_path: Path,
    conversation_ts: float | None,
) -> HistoryBackfillUpdate | None:
    if session is None or session.log_path != expected_log_path or session.last_chat_history_scanned:
        return None
    session.last_chat_history_scanned = True
    if isinstance(conversation_ts, (int, float)):
        session.last_chat_ts = float(conversation_ts)
    updated_ts = float(session.last_chat_ts) if isinstance(session.last_chat_ts, (int, float)) else float(session.start_ts)
    return HistoryBackfillUpdate(updated_ts=updated_ts)


def apply_run_settings_backfill(
    session: Session | None,
    *,
    expected_log_path: Path,
    log_provider: str | None,
    log_model: str | None,
    log_effort: str | None,
) -> RunSettingsUpdate | None:
    if session is None or session.log_path != expected_log_path:
        return None
    if session.model_provider is None:
        session.model_provider = log_provider
    if session.model is None:
        session.model = log_model
    if session.reasoning_effort is None:
        session.reasoning_effort = log_effort
    return RunSettingsUpdate(
        model_provider=session.model_provider,
        preferred_auth_method=session.preferred_auth_method,
        model=session.model,
        reasoning_effort=session.reasoning_effort,
    )


def build_runtime_enriched_session_rows(
    *,
    staged_rows: list[dict[str, Any]],
    sessions: MutableMapping[str, Session],
    lock: Any,
    store: SessionStore,
    probes: ListingRuntimeProbes,
    now_ts: float,
    provider_choice_for_settings: Callable[[str | None, str | None], str],
    priority_half_life_seconds: float,
    priority_bucket_seconds: float,
) -> ListingRuntimeRowsResult:
    recent_cwd_dirty = False
    out: list[dict[str, Any]] = []
    for it in staged_rows:
        sid = str(it["session_id"])
        log_exists = bool(it.get("log_exists"))
        log_path_obj = it.get("_log_path_obj")
        if bool(it.get("needs_history_scan")) and isinstance(log_path_obj, Path):
            try:
                conv_ts = probes.last_conversation_ts_from_tail(log_path_obj)
            except FileNotFoundError:
                conv_ts = None
            with lock:
                s_cur = sessions.get(sid)
                history_update = apply_history_backfill(
                    s_cur,
                    expected_log_path=log_path_obj,
                    conversation_ts=conv_ts,
                )
                if history_update is not None and s_cur is not None:
                    updated_ts = history_update.updated_ts
                    it["updated_ts"] = updated_ts
                    recent_cwd_dirty = recent_cwd_dirty or store.note_recent_cwd(s_cur.cwd, updated_ts)
                    priority = listing_priority(
                        now_ts=now_ts,
                        updated_ts=updated_ts,
                        priority_offset=float(it.get("priority_offset", 0.0)),
                        blocked=bool(it.get("blocked")),
                        snoozed=bool(it.get("snoozed")),
                        half_life_seconds=priority_half_life_seconds,
                        bucket_seconds=priority_bucket_seconds,
                    )
                    it["time_priority"] = priority.time_priority
                    it["base_priority"] = priority.base_priority
                    it["final_priority"] = priority.final_priority
        if bool(it.get("needs_run_settings")) and isinstance(log_path_obj, Path):
            try:
                log_provider, log_model, log_effort = probes.read_run_settings_from_log(
                    log_path_obj,
                    str(it.get("agent_backend") or "codex"),
                )
            except (FileNotFoundError, ValueError):
                log_provider = log_model = log_effort = None
            with lock:
                s_cur = sessions.get(sid)
                run_settings_update = apply_run_settings_backfill(
                    s_cur,
                    expected_log_path=log_path_obj,
                    log_provider=log_provider,
                    log_model=log_model,
                    log_effort=log_effort,
                )
                if run_settings_update is not None:
                    it["model_provider"] = run_settings_update.model_provider
                    it["preferred_auth_method"] = run_settings_update.preferred_auth_method
                    it["model"] = run_settings_update.model
                    it["reasoning_effort"] = run_settings_update.reasoning_effort
                    it["provider_choice"] = provider_choice_for_settings(
                        run_settings_update.model_provider,
                        run_settings_update.preferred_auth_method,
                    )
        log_path_for_boundary = log_path_obj if isinstance(log_path_obj, Path) else None
        log_size = probes.log_size_or_none(log_path_for_boundary)
        boundary_unresolved = probes.send_boundary_unresolved(sid, log_path_for_boundary, log_size)
        broker_runtime = broker_runtime_state(
            {
                "busy": bool(it.get("state_busy")),
                "queue_len": int(it.get("broker_queue_len", 0)),
                "interrupted_idle": bool(it.get("interrupted_idle")),
            }
        )
        try:
            log_idle = (
                bool(probes.idle_from_log_path(sid, log_path_obj))
                if log_exists and isinstance(log_path_obj, Path) and not boundary_unresolved
                else None
            )
            busy_out = resolve_runtime_status(
                broker=broker_runtime,
                log_exists=log_exists and isinstance(log_path_obj, Path),
                log_idle=log_idle,
                send_boundary_unresolved=boundary_unresolved,
            ).busy
        except FileNotFoundError:
            busy_out = False
        cwd_path_obj = it.get("_cwd_path_obj")
        git_branch = probes.current_git_branch(cwd_path_obj) if isinstance(cwd_path_obj, Path) else None
        out.append(build_public_session_row(it, git_branch=git_branch, busy=bool(busy_out)))
    return ListingRuntimeRowsResult(rows=out, recent_cwd_dirty=recent_cwd_dirty)


def broker_runtime_state(state: Mapping[str, Any]) -> BrokerRuntimeState:
    if not isinstance(state, Mapping) or "busy" not in state or "queue_len" not in state:
        raise ValueError("invalid broker state response")
    busy_raw = state.get("busy")
    queue_len_raw = state.get("queue_len")
    if not isinstance(busy_raw, bool):
        raise ValueError("invalid broker state response")
    if isinstance(queue_len_raw, bool) or not isinstance(queue_len_raw, int) or queue_len_raw < 0:
        raise ValueError("invalid broker state response")
    interrupted_idle_raw = state.get("interrupted_idle", False)
    if not isinstance(interrupted_idle_raw, bool):
        raise ValueError("invalid broker state response")
    return BrokerRuntimeState(busy=busy_raw, queue_len=int(queue_len_raw), interrupted_idle=interrupted_idle_raw)


def broker_busy_queue(state: Mapping[str, Any]) -> tuple[bool, int]:
    broker = broker_runtime_state(state)
    return broker.busy, broker.queue_len


def broker_interrupted_idle(state: Mapping[str, Any]) -> bool:
    return broker_runtime_state(state).interrupted_idle


def broker_allows_interrupted_idle_override(state: Mapping[str, Any]) -> bool:
    return broker_runtime_state(state).allows_interrupted_idle_override


def resolve_runtime_status(
    *,
    broker: BrokerRuntimeState,
    log_exists: bool,
    log_idle: bool | None,
    send_boundary_unresolved: bool,
) -> RuntimeStatus:
    log_bound = bool(log_exists)
    boundary = bool(send_boundary_unresolved)
    if log_bound and (not boundary) and not isinstance(log_idle, bool):
        raise ValueError("log_idle is required for a bound transcript log")
    if boundary:
        busy = True
    elif not log_bound:
        busy = False
    else:
        busy = not (bool(log_idle) or broker.allows_interrupted_idle_override)

    if broker.queue_len > 0:
        remote_ready = False
    elif boundary:
        remote_ready = False
    elif log_bound and (log_idle is not True) and not broker.allows_interrupted_idle_override:
        remote_ready = False
    elif broker.busy and log_idle is not True:
        remote_ready = False
    else:
        remote_ready = True

    return RuntimeStatus(
        broker=broker,
        log_exists=log_bound,
        log_idle=log_idle if log_bound else None,
        send_boundary_unresolved=boundary,
        busy=busy,
        remote_ready=remote_ready,
    )


def select_runtime_token(
    *,
    broker_state: Mapping[str, Any],
    session_token: dict[str, Any] | None,
    token_update: dict[str, Any] | None,
    log_available: bool,
) -> dict[str, Any] | None:
    if "token" in broker_state:
        broker_token = broker_state.get("token")
        if not (isinstance(broker_token, dict) or broker_token is None):
            raise ValueError("invalid token from broker state response")
    if isinstance(token_update, dict):
        return token_update
    if isinstance(session_token, dict):
        return session_token
    if (not log_available) and "token" in broker_state and isinstance(broker_state.get("token"), dict):
        return broker_state.get("token")
    return None
