from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import time

from .session_runtime import broker_runtime_state
from .session_runtime import resolve_runtime_status
from .session_runtime import select_runtime_token


JsonResponse = Callable[[Any, int, dict[str, Any]], None]
RouteMatcher = Callable[..., str | None]


@dataclass(frozen=True)
class DiagnosticsRouteDeps:
    require_auth: Callable[[Any], bool]
    json_response: JsonResponse
    provider_choice_for_settings: Callable[..., str]
    read_run_settings_from_log: Callable[..., tuple[str | None, str | None, str | None]]
    resolve_session_cwd: Callable[[str], Path]
    current_git_branch: Callable[[Path], str | None]
    sidebar_time_priority_from_elapsed_seconds: Callable[[float], float]
    clip01: Callable[[float], float]
    time_fn: Callable[[], float] = time.time


def handle_diagnostics_get_route(
    handler: Any,
    *,
    path: str,
    manager: Any,
    deps: DiagnosticsRouteDeps,
    match_session_route: RouteMatcher,
) -> bool:
    session_id = match_session_route(path, "diagnostics")
    if session_id is None:
        return False
    if not deps.require_auth(handler):
        handler._unauthorized()
        return True
    manager.refresh_session_meta(session_id)
    s = manager.get_session(session_id)
    if not s:
        deps.json_response(handler, 404, {"error": "unknown session"})
        return True
    state = manager.get_state(session_id)
    broker_runtime = broker_runtime_state(state)
    log_available = s.log_path is not None and s.log_path.exists()
    log_size = manager._log_size_or_none(s.log_path)
    boundary_unresolved = manager._confirmed_send_boundary_unresolved_for_session(session_id, s.log_path, log_size)
    log_idle = manager.idle_from_log(session_id) if log_available and not boundary_unresolved else None
    runtime = resolve_runtime_status(
        broker=broker_runtime,
        log_exists=log_available,
        log_idle=log_idle,
        send_boundary_unresolved=boundary_unresolved,
    )
    token_val = select_runtime_token(
        broker_state=state,
        session_token=s.token,
        token_update=None,
        log_available=log_available,
    )
    model_provider = s.model_provider
    preferred_auth_method = s.preferred_auth_method
    model = s.model
    reasoning_effort = s.reasoning_effort
    service_tier = s.service_tier
    if (model_provider is None or model is None or reasoning_effort is None) and s.log_path is not None and s.log_path.exists():
        log_provider, log_model, log_effort = deps.read_run_settings_from_log(s.log_path, agent_backend=s.agent_backend)
        if model_provider is None:
            model_provider = log_provider
        if model is None:
            model = log_model
        if reasoning_effort is None:
            reasoning_effort = log_effort
    sidebar_meta = manager.sidebar_meta_get(session_id)
    try:
        cwd_path = deps.resolve_session_cwd(s.cwd)
        git_branch = deps.current_git_branch(cwd_path)
    except ValueError:
        git_branch = None
    updated_ts = float(s.last_chat_ts) if isinstance(s.last_chat_ts, (int, float)) else float(s.start_ts)
    elapsed_s = max(0.0, deps.time_fn() - updated_ts)
    time_priority = deps.sidebar_time_priority_from_elapsed_seconds(elapsed_s)
    base_priority = deps.clip01(time_priority + float(sidebar_meta["priority_offset"]))
    blocked = sidebar_meta["dependency_session_id"] is not None
    snoozed = sidebar_meta["snooze_until"] is not None and float(sidebar_meta["snooze_until"]) > deps.time_fn()
    final_priority = 0.0 if (snoozed or blocked) else base_priority
    deps.json_response(
        handler,
        200,
        {
            "session_id": s.session_id,
            "thread_id": s.thread_id,
            "agent_backend": s.agent_backend,
            "owned": bool(s.owned),
            "transport": s.transport,
            "cwd": s.cwd,
            "start_ts": float(s.start_ts),
            "updated_ts": updated_ts,
            "log_path": (str(s.log_path) if s.log_path is not None else None),
            "broker_pid": int(s.broker_pid),
            "codex_pid": int(s.codex_pid),
            "busy": bool(runtime.busy),
            "broker_busy": broker_runtime.busy,
            "queue_len": manager._queue_len(session_id),
            "token": token_val,
            "model_provider": model_provider,
            "preferred_auth_method": preferred_auth_method,
            "provider_choice": deps.provider_choice_for_settings(
                model_provider=model_provider,
                preferred_auth_method=preferred_auth_method,
            ),
            "model": model,
            "reasoning_effort": reasoning_effort,
            "service_tier": service_tier,
            "tmux_session": s.tmux_session,
            "tmux_window": s.tmux_window,
            "git_branch": git_branch,
            "time_priority": time_priority,
            "base_priority": base_priority,
            "final_priority": final_priority,
            "priority_offset": sidebar_meta["priority_offset"],
            "snooze_until": sidebar_meta["snooze_until"],
            "dependency_session_id": sidebar_meta["dependency_session_id"],
        },
    )
    return True
