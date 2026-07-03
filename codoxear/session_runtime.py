from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping

from .agent_backend import normalize_agent_backend
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
class SessionRuntimeReadiness:
    status: RuntimeStatus
    local_queue_len: int
    direct_send: bool
    queue_promotion: bool
    unattended_injection: bool


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


def complete_jsonl_offset_before(path: Path, before: int) -> int:
    before = max(0, int(before))
    if before <= 0:
        return 0
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        end = max(0, min(before, size))
        if end <= 0:
            return 0
        f.seek(end - 1)
        if f.read(1) == b"\n":
            return end
        offset = end
        while offset > 0:
            read_size = min(64 * 1024, offset)
            offset -= read_size
            f.seek(offset)
            chunk = f.read(read_size)
            found = chunk.rfind(b"\n")
            if found >= 0:
                return offset + found + 1
    return 0


def last_parseable_json_object_offset_before(path: Path, before: int) -> int:
    before = complete_jsonl_offset_before(path, before)
    if before <= 0:
        return 0

    def prev_newline_before(f: Any, pos: int) -> int:
        search_end = max(0, int(pos))
        while search_end > 0:
            start = max(0, search_end - 64 * 1024)
            f.seek(start)
            chunk = f.read(search_end - start)
            found = chunk.rfind(b"\n")
            if found >= 0:
                return start + found
            search_end = start
        return -1

    with path.open("rb") as f:
        line_end = before
        while line_end > 0:
            prev_nl = prev_newline_before(f, line_end - 1)
            line_start = prev_nl + 1
            f.seek(line_start)
            raw = f.read(line_end - line_start)
            if raw.strip():
                try:
                    obj = json.loads(raw.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    obj = None
                if isinstance(obj, dict):
                    return line_end
            line_end = line_start
    return 0


def log_path_size_or_none(log_path: Path | None) -> int | None:
    if not isinstance(log_path, Path):
        return None
    try:
        return last_parseable_json_object_offset_before(log_path, int(log_path.stat().st_size))
    except OSError:
        return None


def confirmed_send_boundary_unresolved(
    *,
    active: bool,
    last_send_log_path: Path | None,
    last_send_log_size: int | None,
    log_path: Path | None,
    log_size: int | None,
) -> bool:
    if not active:
        return False
    if last_send_log_path is None and last_send_log_size is None:
        return log_path is None or log_size is None or log_size <= 0
    if last_send_log_path != log_path:
        return False
    if last_send_log_size is None:
        return log_size is None or log_size <= 0
    return log_size is None or log_size <= last_send_log_size


def clear_session_confirmed_send_boundary(session: Session) -> None:
    session.last_send_boundary_active = False
    session.last_send_log_path = None
    session.last_send_log_size = None


def session_confirmed_send_boundary_unresolved(session: Session | None, log_path: Path | None, log_size: int | None) -> bool:
    if session is None:
        return False
    return confirmed_send_boundary_unresolved(
        active=bool(session.last_send_boundary_active),
        last_send_log_path=session.last_send_log_path,
        last_send_log_size=session.last_send_log_size,
        log_path=log_path,
        log_size=log_size,
    )


def consume_session_confirmed_send_boundary(session: Session | None, log_path: Path | None, log_size: int | None) -> bool:
    unresolved = session_confirmed_send_boundary_unresolved(session, log_path, log_size)
    if session is not None and session.last_send_boundary_active and not unresolved:
        clear_session_confirmed_send_boundary(session)
    return unresolved


def session_allows_direct_send(session: Session, *, allow_pending_attachment: bool) -> bool:
    if session.commit_unknown_send:
        return False
    if session.pending_attachment and not allow_pending_attachment:
        return False
    return True


def session_allows_queue_promotion(session: Session) -> bool:
    if session.commit_unknown_send:
        return False
    if session.pending_attachment:
        return False
    return True


def reset_session_log_caches(session: Session, *, meta_log_off: int) -> None:
    session.meta_thinking = 0
    session.meta_tools = 0
    session.meta_system = 0
    session.last_chat_ts = None
    session.last_chat_history_scanned = False
    session.meta_log_off = int(meta_log_off)
    session.delivery_log_off = int(meta_log_off)
    session.idle_cache_log_off = -1
    session.idle_cache_value = None
    session.queue_idle_since = None
    session.queue_sending_item_id = None
    session.model_provider = None
    session.preferred_auth_method = None
    session.model = None
    session.reasoning_effort = None
    session.service_tier = None


def session_transport_from_meta(*, meta: dict[str, Any], clean_optional_text: Callable[[Any], str | None]) -> tuple[str | None, str | None, str | None]:
    transport = clean_optional_text(meta.get("transport"))
    tmux_session = clean_optional_text(meta.get("tmux_session"))
    tmux_window = clean_optional_text(meta.get("tmux_window"))
    if transport is None and (tmux_session is not None or tmux_window is not None):
        transport = "tmux"
    return transport, tmux_session, tmux_window


def session_run_settings_from_meta(
    *,
    meta: dict[str, Any],
    log_path: Path | None,
    agent_backend: str,
    clean_optional_text: Callable[[Any], str | None],
    normalize_requested_preferred_auth_method: Callable[[Any], str | None],
    display_reasoning_effort: Callable[[Any], str | None],
    display_pi_reasoning_effort: Callable[[Any], str | None],
    normalize_requested_cc_reasoning_effort: Callable[[Any], str | None],
    read_run_settings_from_log: Callable[..., tuple[str | None, str | None, str | None]],
) -> tuple[str | None, str | None, str | None, str | None]:
    backend_name = normalize_agent_backend(agent_backend)
    model_provider = clean_optional_text(meta.get("model_provider"))
    preferred_auth_method = normalize_requested_preferred_auth_method(meta.get("preferred_auth_method"))
    model = clean_optional_text(meta.get("model"))
    if backend_name == "codex":
        reasoning_effort = display_reasoning_effort(meta.get("reasoning_effort"))
    elif backend_name == "pi":
        reasoning_effort = display_pi_reasoning_effort(meta.get("reasoning_effort"))
    else:
        reasoning_effort = normalize_requested_cc_reasoning_effort(meta.get("reasoning_effort"))
    if log_path is not None and log_path.exists():
        log_provider, log_model, log_effort = read_run_settings_from_log(log_path, agent_backend=backend_name)
        if log_provider is not None:
            model_provider = log_provider
        if log_model is not None:
            model = log_model
        if log_effort is not None:
            reasoning_effort = log_effort
    return model_provider, preferred_auth_method, model, reasoning_effort


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
    if broker.queue_len > 0:
        return RuntimeStatus(
            broker=broker,
            log_exists=log_bound,
            log_idle=None,
            send_boundary_unresolved=boundary,
            busy=True,
            remote_ready=False,
        )
    if log_bound and (not boundary) and not isinstance(log_idle, bool):
        raise ValueError("log_idle is required for a bound transcript log")
    if boundary:
        busy = True
    elif not log_bound:
        busy = False
    else:
        busy = not (bool(log_idle) or broker.allows_interrupted_idle_override)

    if boundary:
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


def session_runtime_readiness(
    status: RuntimeStatus,
    *,
    local_queue_len: int = 0,
    direct_send_precondition: bool = True,
    queue_promotion_precondition: bool = True,
) -> SessionRuntimeReadiness:
    queue_len = max(0, int(local_queue_len))
    direct_send = bool(status.remote_ready and direct_send_precondition)
    queue_promotion = bool(status.remote_ready and queue_promotion_precondition)
    unattended_injection = bool(status.remote_ready and queue_len == 0)
    return SessionRuntimeReadiness(
        status=status,
        local_queue_len=queue_len,
        direct_send=direct_send,
        queue_promotion=queue_promotion,
        unattended_injection=unattended_injection,
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
