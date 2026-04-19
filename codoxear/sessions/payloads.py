from __future__ import annotations

from pathlib import Path
import time
from typing import Any


def _sv():
    from codoxear import server as sv

    return sv



def session_details_payload(manager, session_id: str) -> dict[str, Any]:
    sv = _sv()
    row = sv._listed_session_row(manager, session_id)
    if row is not None:
        return {"ok": True, "session": sv._normalize_session_cwd_row(row)}
    historical_row = sv._historical_session_row(session_id)
    if historical_row is not None:
        return {"ok": True, "session": sv._normalize_session_cwd_row(dict(historical_row))}
    raise KeyError("unknown session")



def session_context_usage_payload(s, token_val: dict[str, Any] | None) -> dict[str, Any] | None:
    sv = _sv()
    if s.backend != "pi":
        return None
    context_window = None
    model_provider, _preferred_auth_method, model, _reasoning_effort = sv._resolved_session_run_settings(s)
    if model_provider is not None and model is not None:
        context_window = sv._pi_model_context_window(model_provider, model)
    if (not isinstance(context_window, int) or context_window <= 0) and isinstance(token_val, dict):
        token_context_window = token_val.get("context_window")
        if isinstance(token_context_window, int) and token_context_window > 0:
            context_window = token_context_window
    if not isinstance(context_window, int) or context_window <= 0:
        return None
    used_tokens = 0
    if isinstance(token_val, dict):
        raw_used_tokens = token_val.get("tokens_in_context")
        if isinstance(raw_used_tokens, int) and raw_used_tokens > 0:
            used_tokens = raw_used_tokens
    used_tokens = min(max(used_tokens, 0), context_window)
    percent_used = int(round((used_tokens / context_window) * 100.0)) if context_window > 0 else 0
    return {
        "used_tokens": used_tokens,
        "total_tokens": context_window,
        "percent_used": percent_used,
    }



def session_diagnostics_payload(manager, session_id: str, s, state: dict[str, Any]) -> dict[str, Any]:
    sv = _sv()
    state = sv._validated_session_state(state)
    token_val: dict[str, Any] | None = None
    st_token = state.get("token")
    if isinstance(st_token, dict) or st_token is None:
        token_val = st_token if isinstance(st_token, dict) else (s.token if isinstance(s.token, dict) else None)
    model_provider, preferred_auth_method, model, reasoning_effort = sv._resolved_session_run_settings(s)
    service_tier = s.service_tier
    sidebar_meta = manager.sidebar_meta_get(session_id)
    cwd_path = sv._safe_expanduser(Path(s.cwd))
    if not cwd_path.is_absolute():
        cwd_path = cwd_path.resolve()
    git_branch = sv._current_git_branch(cwd_path)
    updated_ts = sv._display_updated_ts(s)
    elapsed_s = max(0.0, time.time() - updated_ts)
    time_priority = sv._priority_from_elapsed_seconds(elapsed_s)
    base_priority = sv._clip01(time_priority + float(sidebar_meta["priority_offset"]))
    blocked = sidebar_meta["dependency_session_id"] is not None
    snoozed = sidebar_meta["snooze_until"] is not None and float(sidebar_meta["snooze_until"]) > time.time()
    final_priority = 0.0 if (snoozed or blocked) else base_priority
    busy, broker_busy = sv._display_session_busy(manager, session_id, s, state)
    durable_session_id = sv._durable_session_id_for_live_session(s)
    return {
        "session_id": durable_session_id,
        "runtime_id": s.session_id,
        "thread_id": s.thread_id,
        "agent_backend": s.agent_backend,
        "backend": s.backend,
        "owned": bool(s.owned),
        "transport": s.transport,
        "cwd": s.cwd,
        "start_ts": float(s.start_ts),
        "updated_ts": updated_ts,
        "log_path": str(s.log_path) if s.log_path is not None else None,
        "session_file_path": sv._display_source_path(s),
        "broker_pid": int(s.broker_pid),
        "codex_pid": int(s.codex_pid),
        "busy": bool(busy),
        "broker_busy": broker_busy,
        "queue_len": manager._queue_len(session_id),
        "token": token_val,
        "context_usage": session_context_usage_payload(s, token_val),
        "model_provider": model_provider,
        "preferred_auth_method": preferred_auth_method,
        "provider_choice": sv._provider_choice_for_backend(
            backend=s.backend,
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
        "todo_snapshot": sv._todo_snapshot_payload_for_session(s),
    }



def session_workspace_payload(manager, session_id: str) -> dict[str, Any]:
    sv = _sv()
    manager.refresh_session_meta(session_id, strict=False)
    s = manager.get_session(session_id)
    if not s:
        historical_row = sv._historical_session_row(session_id)
        if historical_row is None:
            historical_row = sv._listed_session_row(manager, session_id)
        if historical_row is None:
            raise KeyError("unknown session")
        return {
            "ok": True,
            "session_id": str(historical_row.get("session_id") or session_id),
            "runtime_id": None,
            "diagnostics": None,
            "queue": {"items": []},
        }
    diagnostics = session_diagnostics_payload(manager, session_id, s, manager.get_state(session_id))
    return {
        "ok": True,
        "session_id": sv._durable_session_id_for_live_session(s),
        "runtime_id": s.session_id,
        "diagnostics": diagnostics,
        "queue": {"items": manager.queue_list(session_id)},
    }
