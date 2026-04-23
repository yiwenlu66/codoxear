from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..agent_backend import normalize_agent_backend
from ..page_state_sqlite import PageStateDB, SessionRef


def _clean_optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    out = value.strip()
    return out or None


def _parse_historical_session_id(session_id: str) -> tuple[str, str] | None:
    raw = str(session_id or "").strip()
    if not raw.startswith("history:"):
        return None
    _prefix, backend, resume_session_id = (
        raw.split(":", 2) if raw.count(":") >= 2 else ("", "", "")
    )
    backend_clean = normalize_agent_backend(backend, default="codex")
    resume_clean = _clean_optional_text(resume_session_id)
    if not resume_clean:
        return None
    return backend_clean, resume_clean


@dataclass(slots=True)
class SessionCatalogService:
    manager: Any

    def listed_session_row(self, session_id: str) -> dict[str, Any] | None:
        return listed_session_row(self.manager, session_id)

    def runtime_session_id_for_identifier(self, session_id: str) -> str | None:
        return runtime_session_id_for_identifier(self.manager, session_id)

    def durable_session_id_for_identifier(self, session_id: str) -> str | None:
        return durable_session_id_for_identifier(self.manager, session_id)

    def page_state_ref_for_session_id(self, session_id: str) -> SessionRef | None:
        return page_state_ref_for_session_id(self.manager, session_id)

    def discover_existing(
        self, *, force: bool = False, skip_invalid_sidecars: bool = False
    ) -> None:
        discover_existing(
            self.manager,
            force=force,
            skip_invalid_sidecars=skip_invalid_sidecars,
        )

    def refresh_session_state(
        self, session_id: str, sock_path: Path, timeout_s: float = 0.4
    ) -> tuple[bool, BaseException | None]:
        return refresh_session_state(
            self.manager,
            session_id,
            sock_path,
            timeout_s=timeout_s,
        )

    def prune_dead_sessions(self) -> None:
        prune_dead_sessions(self.manager)

    def list_sessions(self) -> list[dict[str, Any]]:
        return list_sessions(self.manager)

    def get_session(self, session_id: str) -> Any | None:
        return get_session(self.manager, session_id)

    def refresh_session_meta(self, session_id: str, *, strict: bool = True) -> None:
        refresh_session_meta(self.manager, session_id, strict=strict)


def service(manager: Any) -> SessionCatalogService:
    return SessionCatalogService(manager)


def runtime_session_id_for_identifier(manager: Any, session_id: str) -> str | None:
    target = _clean_optional_text(session_id)
    if target is None:
        return None
    with manager._lock:
        if target in manager._sessions:
            return target
        matches: list[tuple[float, str]] = []
        for runtime_id, session in manager._sessions.items():
            ref = manager._page_state_ref_for_session(session)
            if ref is not None and ref[1] == target:
                matches.append((float(session.start_ts or 0.0), runtime_id))
                continue
            thread_id = _clean_optional_text(session.thread_id)
            if thread_id == target:
                matches.append((float(session.start_ts or 0.0), runtime_id))
        if not matches:
            return None
        matches.sort(key=lambda item: (-item[0], item[1]))
        return matches[0][1]


def durable_session_id_for_identifier(manager: Any, session_id: str) -> str | None:
    runtime_id = runtime_session_id_for_identifier(manager, session_id)
    if runtime_id is not None:
        with manager._lock:
            session = manager._sessions.get(runtime_id)
        if session is not None:
            return manager._durable_session_id_for_session(session)
    target = _clean_optional_text(session_id)
    return target if target is not None else None


def page_state_ref_for_session_id(manager: Any, session_id: str) -> SessionRef | None:
    runtime_id = runtime_session_id_for_identifier(manager, session_id)
    if runtime_id is not None:
        with manager._lock:
            session = manager._sessions.get(runtime_id)
        if session is not None:
            return manager._page_state_ref_for_session(session)
    parsed = _parse_historical_session_id(session_id)
    if parsed is not None:
        return parsed
    target = _clean_optional_text(session_id)
    db = getattr(manager, "_page_state_db", None)
    if target is not None and isinstance(db, PageStateDB):
        matches = [ref for ref in db.known_session_refs() if ref[1] == target]
        if len(matches) == 1:
            return matches[0]
    return None


def get_session(manager: Any, session_id: str) -> Any | None:
    runtime_id = runtime_session_id_for_identifier(manager, session_id)
    if runtime_id is None:
        return None
    with manager._lock:
        return manager._sessions.get(runtime_id)


def listed_session_row(manager: Any, session_id: str) -> dict[str, Any] | None:
    for row in manager.list_sessions():
        if str(row.get("session_id") or "") == session_id:
            return dict(row)
    return None


def refresh_session_meta(manager: Any, session_id: str, *, strict: bool = True) -> None:
    sv = manager._runtime
    runtime_id = runtime_session_id_for_identifier(manager, session_id)
    if runtime_id is None:
        return
    with manager._lock:
        session = manager._sessions.get(runtime_id)
        if not session:
            return
        sock = session.sock_path
    try:
        meta_path = sock.with_suffix(".json")
        if not meta_path.exists():
            raise RuntimeError(f"missing metadata sidecar for socket {sock}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            raise ValueError(f"invalid metadata json for socket {sock}")

        thread_id = _clean_optional_text(meta.get("session_id")) or session.thread_id
        backend = normalize_agent_backend(
            meta.get("backend"),
            default=normalize_agent_backend(meta.get("agent_backend"), default=session.backend),
        )
        agent_backend = normalize_agent_backend(meta.get("agent_backend"), default=backend)
        owned = (meta.get("owner") == "web") if isinstance(meta.get("owner"), str) else session.owned
        transport, tmux_session, tmux_window = manager._session_transport(meta=meta)
        supports_live_ui = meta.get("supports_live_ui") if isinstance(meta.get("supports_live_ui"), bool) else None
        ui_protocol_version_raw = meta.get("ui_protocol_version")
        ui_protocol_version = ui_protocol_version_raw if type(ui_protocol_version_raw) is int else None
        log_path = sv._metadata_log_path(meta=meta, backend=backend, sock=sock)
        session_path_discovered = False
        if backend == "pi":
            preferred_session_path: Path | None = session.session_path
            if strict or ("session_path" in meta):
                preferred_session_path = sv._metadata_session_path(meta=meta, backend=backend, sock=sock)
            claimed: set[Path] | None = (
                manager._claimed_pi_session_paths(exclude_sid=session_id)
                if preferred_session_path is None
                else None
            )
            session_path, session_path_source = sv._resolve_pi_session_path(
                thread_id=thread_id,
                cwd=str(meta.get("cwd") or session.cwd),
                start_ts=float(meta.get("start_ts") or session.start_ts),
                preferred=preferred_session_path,
                exclude=claimed,
            )
            if session_path is not None and session_path_source in {"exact", "discovered"}:
                session_path_discovered = True
                sv._patch_metadata_session_path(
                    sock,
                    session_path,
                    force=preferred_session_path is not None and preferred_session_path != session_path,
                )
        else:
            session_path = sv._metadata_session_path(meta=meta, backend=backend, sock=sock)
        if log_path is not None and log_path.exists():
            thread_id, log_path = sv._coerce_main_thread_log(thread_id=thread_id, log_path=log_path)

        cwd_raw = meta.get("cwd")
        if not isinstance(cwd_raw, str) or (not cwd_raw.strip()):
            raise ValueError(f"invalid cwd in metadata for socket {sock}")
        cwd = cwd_raw

        start_ts_raw = meta.get("start_ts")
        start_ts = float(start_ts_raw) if isinstance(start_ts_raw, (int, float)) else session.start_ts
        resume_session_id = _clean_optional_text(meta.get("resume_session_id"))
        model_provider, preferred_auth_method, model, reasoning_effort = manager._session_run_settings(
            backend=backend,
            meta=meta,
            log_path=log_path,
        )
        service_tier = sv._normalize_requested_service_tier(meta.get("service_tier"))
    except Exception as exc:
        if strict:
            raise
        manager._quarantine_sidecar(sock, exc, log=False)
        return
    manager._clear_sidecar_quarantine(sock)

    pi_session_switched = False
    old_session_path: Path | None = None
    with manager._lock:
        current = manager._sessions.get(session_id)
        if (
            current
            and backend == "pi"
            and current.thread_id
            and thread_id
            and current.thread_id != thread_id
            and current.session_path is not None
        ):
            pi_session_switched = True
            old_session_path = current.session_path

    if pi_session_switched and old_session_path is not None:
        claimed = manager._claimed_pi_session_paths(exclude_sid=session_id)
        claimed.add(old_session_path)
        new_sp, new_sp_source = sv._resolve_pi_session_path(
            thread_id=thread_id,
            cwd=cwd,
            start_ts=start_ts,
            preferred=None,
            exclude=claimed,
        )
        if new_sp is not None and new_sp != old_session_path:
            session_path = new_sp
            if new_sp_source in {"exact", "discovered"}:
                session_path_discovered = True
            sv._patch_metadata_session_path(sock, new_sp, force=True)

    with manager._lock:
        current = manager._sessions.get(session_id)
        if not current:
            return
        if pi_session_switched:
            current.session_path = None
            current.pi_attention_scan_activity_ts = None
            manager._reset_log_caches(current, meta_log_off=0)
        current.thread_id = str(thread_id)
        current.agent_backend = agent_backend
        current.backend = backend
        current.cwd = str(cwd)
        current.owned = bool(owned)
        current.transport = transport
        current.supports_live_ui = supports_live_ui
        current.ui_protocol_version = ui_protocol_version
        manager._apply_session_source(current, log_path=log_path, session_path=session_path)
        current.model_provider = model_provider
        current.preferred_auth_method = preferred_auth_method
        current.model = model
        current.reasoning_effort = reasoning_effort
        current.service_tier = service_tier
        current.tmux_session = tmux_session
        current.tmux_window = tmux_window
        current.resume_session_id = resume_session_id
        current.pi_session_path_discovered = bool(current.pi_session_path_discovered or session_path_discovered)
    if manager._queue_len(session_id) > 0:
        manager._maybe_drain_session_queue(session_id)


def list_sessions(manager: Any) -> list[dict[str, Any]]:
    sv = manager._runtime
    recovered_catalog = {}
    db = getattr(manager, "_page_state_db", None)
    if isinstance(db, PageStateDB):
        recovered_catalog = db.load_sessions()
    if float(getattr(manager, "_last_discover_ts", 0.0) or 0.0) <= 0.0:
        manager._discover_existing_if_stale(force=True)
    manager._update_meta_counters()
    files_dirty = False
    sidebar_dirty = False
    now_ts = time.time()
    with manager._lock:
        items: list[dict[str, Any]] = []
        qmap = getattr(manager, "_queues", None)
        meta_map = getattr(manager, "_sidebar_meta", None)
        hidden_sessions = set(getattr(manager, "_hidden_sessions", set()))
        active_durable_ids = {
            (_clean_optional_text(v.thread_id) or _clean_optional_text(v.session_id) or "")
            for v in manager._sessions.values()
        }
        live_resume_keys: set[tuple[str, str]] = set()
        for s in manager._sessions.values():
            thread_id = str(s.thread_id or "").strip()
            if thread_id:
                live_resume_keys.add(
                    (
                        normalize_agent_backend(s.agent_backend, default=s.backend or "codex"),
                        thread_id,
                    )
                )
            cfg0 = manager._harness.get(s.session_id)
            h_enabled = bool(cfg0.get("enabled")) if isinstance(cfg0, dict) else False
            h_cooldown_minutes = (
                sv._clean_harness_cooldown_minutes(cfg0.get("cooldown_minutes"))
                if isinstance(cfg0, dict)
                else sv.HARNESS_DEFAULT_IDLE_MINUTES
            )
            h_remaining_injections = (
                sv._clean_harness_remaining_injections(cfg0.get("remaining_injections"), allow_zero=True)
                if isinstance(cfg0, dict)
                else sv.HARNESS_DEFAULT_MAX_INJECTIONS
            )
            log_exists = bool(s.log_path is not None and s.log_path.exists())
            if log_exists and s.log_path is not None and (
                s.model_provider is None or s.model is None or s.reasoning_effort is None
            ):
                try:
                    log_provider, log_model, log_effort = sv._read_run_settings_from_log(
                        s.log_path, agent_backend=s.agent_backend
                    )
                except (FileNotFoundError, ValueError):
                    log_provider = log_model = log_effort = None
                if s.model_provider is None:
                    s.model_provider = log_provider
                if s.model is None:
                    s.model = log_model
                if s.reasoning_effort is None:
                    s.reasoning_effort = log_effort
            if s.last_chat_ts is None and log_exists and s.log_path is not None and (not s.last_chat_history_scanned):
                conv_ts = sv._last_conversation_ts_from_tail(s.log_path)
                s.last_chat_history_scanned = True
                if isinstance(conv_ts, (int, float)):
                    s.last_chat_ts = float(conv_ts)
            if s.backend == "pi" and s.session_path is not None and s.session_path.exists():
                activity_ts = sv._session_file_activity_ts(s.session_path)
                scanned_activity_ts = s.pi_attention_scan_activity_ts
                should_refresh_attention = bool(
                    activity_ts is not None
                    and (
                        scanned_activity_ts is None
                        or float(activity_ts) > float(scanned_activity_ts)
                    )
                )
                if should_refresh_attention or (s.last_chat_ts is None and (not s.last_chat_history_scanned)):
                    conv_ts = sv._last_attention_ts_from_pi_tail(s.session_path)
                    s.last_chat_history_scanned = True
                    s.pi_attention_scan_activity_ts = activity_ts
                    if isinstance(conv_ts, (int, float)):
                        s.last_chat_ts = (
                            float(conv_ts)
                            if s.last_chat_ts is None
                            else max(float(s.last_chat_ts), float(conv_ts))
                        )
            updated_ts = sv._display_updated_ts(s)
            canonical_cwd = sv._canonical_session_cwd(s.cwd)
            cwd_recent = sv._clean_recent_cwd(canonical_cwd)
            recent_map = getattr(manager, "_recent_cwds", None)
            if cwd_recent is not None:
                if not isinstance(recent_map, dict):
                    manager._recent_cwds = {}
                    recent_map = manager._recent_cwds
                prev_recent_ts = recent_map.get(cwd_recent)
                if prev_recent_ts is None or prev_recent_ts < updated_ts:
                    recent_map[cwd_recent] = updated_ts
            ref = manager._page_state_ref_for_session(s)
            queue_len = 0
            if isinstance(qmap, dict):
                q0 = qmap.get(s.session_id)
                if not isinstance(q0, list) and ref is not None:
                    q0 = qmap.get(ref)
                if isinstance(q0, list):
                    queue_len = len(q0)
            meta0 = None
            if isinstance(meta_map, dict):
                meta0 = meta_map.get(s.session_id)
                if meta0 is None and ref is not None:
                    meta0 = meta_map.get(ref)
            if not isinstance(meta0, dict):
                meta0 = {}
            priority_offset = sv._clean_priority_offset(meta0.get("priority_offset"))
            snooze_until = sv._clean_snooze_until(meta0.get("snooze_until"))
            dependency_session_id = sv._clean_dependency_session_id(meta0.get("dependency_session_id"))
            active_durable_ids = {
                (_clean_optional_text(v.thread_id) or _clean_optional_text(v.session_id) or "")
                for v in manager._sessions.values()
            }
            if dependency_session_id == (_clean_optional_text(s.thread_id) or _clean_optional_text(s.session_id)) or (
                dependency_session_id is not None and dependency_session_id not in active_durable_ids
            ):
                dependency_session_id = None
                if isinstance(meta_map, dict) and isinstance(meta0, dict):
                    meta0.pop("dependency_session_id", None)
                    sidebar_dirty = True
            if snooze_until is not None and snooze_until <= now_ts:
                snooze_until = None
                if isinstance(meta_map, dict) and isinstance(meta0, dict):
                    meta0.pop("snooze_until", None)
                    sidebar_dirty = True
            elapsed_s = max(0.0, now_ts - updated_ts)
            time_priority = sv._priority_from_elapsed_seconds(elapsed_s)
            base_priority = sv._clip01(time_priority + priority_offset)
            blocked = dependency_session_id is not None
            snoozed = snooze_until is not None and snooze_until > now_ts
            final_priority = 0.0 if (snoozed or blocked) else base_priority
            cwd_path = sv._safe_expanduser(Path(canonical_cwd or s.cwd))
            if not cwd_path.is_absolute():
                cwd_path = cwd_path.resolve()
            git_branch = sv._current_git_branch(cwd_path)
            if s.title is None:
                try:
                    if s.backend == "pi" and s.session_path is not None and s.session_path.exists():
                        title = sv._pi_session_name_from_session_file(s.session_path)
                        if title:
                            s.title = title
                except Exception:
                    pass
            if s.first_user_message is None:
                try:
                    preview = ""
                    if s.backend == "pi" and s.session_path is not None and s.session_path.exists():
                        preview = sv._first_user_message_preview_from_pi_session(s.session_path)
                    elif log_exists and s.log_path is not None:
                        preview = sv._first_user_message_preview_from_log(s.log_path)
                    if preview:
                        s.first_user_message = preview
                except Exception:
                    pass
            durable_session_id = ref[1] if ref is not None else manager._durable_session_id_for_session(s)
            items.append(
                {
                    "session_id": durable_session_id,
                    "runtime_id": s.session_id,
                    "thread_id": s.thread_id,
                    "backend": s.backend,
                    "pid": s.codex_pid,
                    "broker_pid": s.broker_pid,
                    "agent_backend": s.agent_backend,
                    "owned": s.owned,
                    "transport": s.transport,
                    "cwd": canonical_cwd,
                    "start_ts": s.start_ts,
                    "updated_ts": updated_ts,
                    "log_path": str(s.log_path) if s.log_path is not None else None,
                    "log_exists": log_exists,
                    "state_busy": bool(s.busy),
                    "queue_len": int(queue_len),
                    "token": s.token,
                    "thinking": int(s.meta_thinking),
                    "tools": int(s.meta_tools),
                    "system": int(s.meta_system),
                    "harness_enabled": h_enabled,
                    "harness_cooldown_minutes": h_cooldown_minutes,
                    "harness_remaining_injections": h_remaining_injections,
                    "alias": (
                        manager._aliases.get(s.session_id)
                        if manager._aliases.get(s.session_id) is not None
                        else (manager._aliases.get(ref) if ref is not None else None)
                    ),
                    "title": s.title or "",
                    "first_user_message": s.first_user_message or "",
                    "files": (
                        list(manager._files.get(s.session_id, manager._files.get(ref, [])))
                        if ref is not None
                        else list(manager._files.get(s.session_id, []))
                    ),
                    "git_branch": git_branch,
                    "model_provider": s.model_provider,
                    "preferred_auth_method": s.preferred_auth_method,
                    "provider_choice": sv._provider_choice_for_backend(
                        backend=s.backend,
                        model_provider=s.model_provider,
                        preferred_auth_method=s.preferred_auth_method,
                    ),
                    "model": s.model,
                    "reasoning_effort": s.reasoning_effort,
                    "service_tier": s.service_tier,
                    "tmux_session": s.tmux_session,
                    "tmux_window": s.tmux_window,
                    "priority_offset": priority_offset,
                    "snooze_until": snooze_until,
                    "dependency_session_id": dependency_session_id,
                    "time_priority": time_priority,
                    "base_priority": base_priority,
                    "final_priority": final_priority,
                    "blocked": blocked,
                    "snoozed": snoozed,
                    "focused": bool(meta0.get("focused")),
                }
            )

        for ref, record in recovered_catalog.items():
            backend, durable_session_id = ref
            if backend != "pi":
                continue
            if (backend, durable_session_id) in live_resume_keys:
                continue
            session_row_id = durable_session_id if record.pending_startup else sv._historical_session_id(backend, durable_session_id)
            if hidden_sessions.intersection(
                manager._hidden_session_keys(
                    session_row_id,
                    durable_session_id,
                    durable_session_id,
                    backend,
                )
            ):
                continue
            meta0 = meta_map.get(ref) if isinstance(meta_map, dict) else None
            if not isinstance(meta0, dict):
                meta0 = {}
            priority_offset = sv._clean_priority_offset(meta0.get("priority_offset"))
            snooze_until = sv._clean_snooze_until(meta0.get("snooze_until"))
            dependency_session_id = sv._clean_dependency_session_id(meta0.get("dependency_session_id"))
            if dependency_session_id is not None and dependency_session_id not in active_durable_ids:
                dependency_session_id = None
                if isinstance(meta_map, dict):
                    meta0.pop("dependency_session_id", None)
                    sidebar_dirty = True
            if snooze_until is not None and snooze_until <= now_ts:
                snooze_until = None
                if isinstance(meta_map, dict):
                    meta0.pop("snooze_until", None)
                    sidebar_dirty = True
            updated_ts = float(record.updated_at or record.created_at or now_ts)
            elapsed_s = max(0.0, now_ts - updated_ts)
            time_priority = sv._priority_from_elapsed_seconds(elapsed_s)
            base_priority = sv._clip01(time_priority + priority_offset)
            blocked = dependency_session_id is not None
            snoozed = snooze_until is not None and snooze_until > now_ts
            final_priority = 0.0 if (snoozed or blocked) else base_priority
            alias = manager._aliases.get(ref) if isinstance(manager._aliases, dict) else None
            queue_rows = manager._queues.get(ref, []) if isinstance(manager._queues, dict) else []
            file_rows = manager._files.get(ref, []) if isinstance(manager._files, dict) else []
            cwd = record.cwd or ""
            history_cwd_path: Path | None = sv._safe_expanduser(Path(cwd)).resolve() if cwd else None
            git_branch = sv._current_git_branch(history_cwd_path) if history_cwd_path is not None else None
            items.append(
                {
                    "session_id": session_row_id,
                    "runtime_id": None,
                    "thread_id": durable_session_id,
                    "resume_session_id": durable_session_id,
                    "backend": backend,
                    "pid": None,
                    "broker_pid": None,
                    "agent_backend": backend,
                    "owned": False,
                    "transport": None,
                    "cwd": cwd,
                    "start_ts": float(record.created_at or updated_ts),
                    "updated_ts": updated_ts,
                    "busy": False,
                    "queue_len": len(queue_rows) if isinstance(queue_rows, list) else 0,
                    "token": None,
                    "thinking": 0,
                    "tools": 0,
                    "system": 0,
                    "harness_enabled": False,
                    "harness_cooldown_minutes": sv.HARNESS_DEFAULT_IDLE_MINUTES,
                    "harness_remaining_injections": sv.HARNESS_DEFAULT_MAX_INJECTIONS,
                    "alias": alias,
                    "title": record.title or "",
                    "first_user_message": record.first_user_message or "",
                    "focused": bool(meta0.get("focused")),
                    "files": list(file_rows) if isinstance(file_rows, list) else [],
                    "git_branch": git_branch,
                    "model_provider": None,
                    "preferred_auth_method": None,
                    "provider_choice": None,
                    "model": None,
                    "reasoning_effort": None,
                    "service_tier": None,
                    "tmux_session": None,
                    "tmux_window": None,
                    "priority_offset": priority_offset,
                    "snooze_until": snooze_until,
                    "dependency_session_id": dependency_session_id,
                    "time_priority": time_priority,
                    "base_priority": base_priority,
                    "final_priority": final_priority,
                    "blocked": blocked,
                    "snoozed": snoozed,
                    "historical": not record.pending_startup,
                    "pending_startup": bool(record.pending_startup),
                    "source_path": record.source_path,
                    "session_path": record.source_path,
                }
            )

        if bool(getattr(manager, "_include_historical_sessions", False)):
            for hist in sv._historical_sidebar_items(live_resume_keys=live_resume_keys, now_ts=now_ts):
                if hidden_sessions.intersection(
                    manager._hidden_session_keys(
                        hist.get("session_id"),
                        hist.get("thread_id"),
                        hist.get("resume_session_id"),
                        hist.get("agent_backend"),
                    )
                ):
                    continue
                items.append(hist)

    out: list[dict[str, Any]] = []
    for it in items:
        sid = str(it["session_id"])
        agent_backend = normalize_agent_backend(it.get("agent_backend"), default="codex")
        if it.get("historical"):
            out.append(sv._normalize_session_cwd_row(dict(it)))
            continue
        log_exists = bool(it.get("log_exists"))
        state_busy = bool(it.get("state_busy"))
        if not log_exists and it.get("backend") == "pi":
            s_obj = manager._sessions.get(sid)
            busy_out = sv._display_pi_busy(s_obj, broker_busy=state_busy) if s_obj is not None else state_busy
        elif not log_exists:
            busy_out = False
        else:
            idle_val = bool(manager.idle_from_log(sid))
            if agent_backend == "pi":
                busy_out = not idle_val
            else:
                busy_out = state_busy or (not idle_val)
        it2 = dict(it)
        it2.pop("log_exists", None)
        it2.pop("state_busy", None)
        it2["busy"] = bool(busy_out)
        out.append(sv._normalize_session_cwd_row(it2))
    for item in out:
        if item.get("busy") or int(item.get("queue_len", 0)) <= 0:
            continue
        manager._maybe_drain_session_queue(str(item["session_id"]))
    if files_dirty:
        manager._save_files()
    if sidebar_dirty:
        manager._save_sidebar_meta()
    out.sort(
        key=lambda item: (
            -float(item.get("final_priority", 0.0)),
            -float(item.get("updated_ts", item.get("start_ts", 0.0))),
            -float(item.get("start_ts", 0.0)),
            0 if normalize_agent_backend(item.get("agent_backend"), default="codex") == "pi" else 1,
            str(item.get("session_id", "")),
        )
    )
    deduped: list[dict[str, Any]] = []
    seen_row_keys: set[str] = set()
    for item in out:
        row_key = sv._session_row_dedupe_key(item)
        if row_key in seen_row_keys:
            continue
        seen_row_keys.add(row_key)
        deduped.append(item)
    return deduped


def discover_existing(
    manager: Any, *, force: bool = False, skip_invalid_sidecars: bool = False
) -> None:
    sv = manager._runtime
    if not force:
        now = time.time()
        with manager._lock:
            last = float(manager._last_discover_ts)
        if (now - last) < sv.DISCOVER_MIN_INTERVAL_SECONDS:
            return
    sv.SOCK_DIR.mkdir(parents=True, exist_ok=True)
    for sock in sorted(sv.SOCK_DIR.glob("*.sock")):
        if skip_invalid_sidecars and manager._sidecar_is_quarantined(sock):
            continue
        session_id = sock.stem
        try:
            meta_path = sock.with_suffix(".json")
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if not isinstance(meta, dict):
                raise ValueError(f"invalid metadata json for socket {sock}")

            thread_id = _clean_optional_text(meta.get("session_id")) or session_id
            backend = normalize_agent_backend(
                meta.get("backend"),
                default=normalize_agent_backend(meta.get("agent_backend"), default="codex"),
            )
            agent_backend = normalize_agent_backend(meta.get("agent_backend"), default=backend)
            codex_pid_raw = meta.get("codex_pid")
            broker_pid_raw = meta.get("broker_pid")
            if not isinstance(codex_pid_raw, int):
                raise ValueError(f"invalid codex_pid in metadata for socket {sock}")
            if not isinstance(broker_pid_raw, int):
                raise ValueError(f"invalid broker_pid in metadata for socket {sock}")
            codex_pid = int(codex_pid_raw)
            broker_pid = int(broker_pid_raw)
            owned = (meta.get("owner") == "web") if isinstance(meta.get("owner"), str) else False
            transport, tmux_session, tmux_window = manager._session_transport(meta=meta)
            supports_live_ui = meta.get("supports_live_ui") if isinstance(meta.get("supports_live_ui"), bool) else None
            ui_protocol_version_raw = meta.get("ui_protocol_version")
            ui_protocol_version = ui_protocol_version_raw if type(ui_protocol_version_raw) is int else None
            if backend == "pi" and transport is None and (owned or sv._supports_web_control(meta)):
                transport = "pi-rpc"
            if backend == "pi" and transport == "pi-rpc" and supports_live_ui is None:
                supports_live_ui = True
            if backend == "pi" and transport == "pi-rpc" and supports_live_ui is True and ui_protocol_version is None:
                ui_protocol_version = 1

            cwd_raw = meta.get("cwd")
            if not isinstance(cwd_raw, str) or (not cwd_raw.strip()):
                raise ValueError(f"invalid cwd in metadata for socket {sock}")
            cwd = cwd_raw

            start_ts_raw = meta.get("start_ts")
            if not isinstance(start_ts_raw, (int, float)):
                raise ValueError(f"invalid start_ts in metadata for socket {sock}")
            start_ts = float(start_ts_raw)

            session_path_discovered = False
            inferred_pi_session_path: Path | None = None
            if backend == "codex" and agent_backend == "codex":
                for key in ("session_path", "log_path"):
                    raw_path = meta.get(key)
                    if not isinstance(raw_path, str) or not raw_path.strip():
                        continue
                    candidate = Path(raw_path)
                    if sv.infer_agent_backend_from_log_path(candidate) != "pi":
                        continue
                    inferred_pi_session_path = candidate
                    break
                if inferred_pi_session_path is None and sv._pid_alive(codex_pid):
                    ignored_paths = manager._claimed_pi_session_paths(exclude_sid=session_id)
                    inferred_pi_session_path = sv._proc_find_open_rollout_log(
                        proc_root=sv.PROC_ROOT,
                        root_pid=codex_pid,
                        agent_backend="pi",
                        cwd=cwd,
                        ignored_paths=ignored_paths,
                    )
            if inferred_pi_session_path is not None:
                backend = "pi"
                agent_backend = "pi"
                session_path_discovered = True
                if transport is None and (owned or sv._supports_web_control(meta)):
                    transport = "pi-rpc"
                if supports_live_ui is None and transport == "pi-rpc":
                    supports_live_ui = True
                if ui_protocol_version is None and supports_live_ui is True:
                    ui_protocol_version = 1
                sv._patch_metadata_pi_binding(sock, inferred_pi_session_path)

            if backend == "pi":
                if transport != "pi-rpc":
                    continue
                if supports_live_ui is not True:
                    continue
                if not isinstance(ui_protocol_version, int) or ui_protocol_version < 1:
                    continue
                if (not owned) and (not sv._supports_web_control(meta)):
                    continue

            log_path = sv._metadata_log_path(meta=meta, backend=backend, sock=sock)
            if inferred_pi_session_path is not None:
                session_path = inferred_pi_session_path
            else:
                preferred_session_path: Path | None = None
                if backend == "pi":
                    try:
                        preferred_session_path = sv._metadata_session_path(meta=meta, backend=backend, sock=sock)
                    except ValueError as exc:
                        if "missing session_path" not in str(exc):
                            raise
                    claimed: set[Path] | None = (
                        manager._claimed_pi_session_paths(exclude_sid=session_id)
                        if preferred_session_path is None
                        else None
                    )
                    session_path, session_path_source = sv._resolve_pi_session_path(
                        thread_id=thread_id,
                        cwd=cwd,
                        start_ts=start_ts,
                        preferred=preferred_session_path,
                        exclude=claimed,
                    )
                    if session_path is not None and session_path_source in {"exact", "discovered"}:
                        session_path_discovered = True
                        sv._patch_metadata_session_path(
                            sock,
                            session_path,
                            force=preferred_session_path is not None and preferred_session_path != session_path,
                        )
                else:
                    session_path = sv._metadata_session_path(meta=meta, backend=backend, sock=sock)
            if log_path is not None and log_path.exists():
                thread_id, log_path = sv._coerce_main_thread_log(thread_id=thread_id, log_path=log_path)
            else:
                log_path = None
        except Exception as exc:
            if skip_invalid_sidecars:
                manager._quarantine_sidecar(sock, exc, log=False)
                continue
            raise
        manager._clear_sidecar_quarantine(sock)

        if (log_path is None) and (not sv._pid_alive(codex_pid)) and (not sv._pid_alive(broker_pid)):
            manager._unhide_session(session_id)
            sv._unlink_quiet(sock)
            sv._unlink_quiet(meta_path)
            continue
        resume_session_id = _clean_optional_text(meta.get("resume_session_id"))
        if manager._session_is_hidden(session_id, thread_id, resume_session_id, agent_backend):
            if (not sv._pid_alive(codex_pid)) and (not sv._pid_alive(broker_pid)):
                manager._unhide_session(session_id)
                sv._unlink_quiet(sock)
                sv._unlink_quiet(meta_path)
            continue

        try:
            model_provider, preferred_auth_method, model, reasoning_effort = manager._session_run_settings(
                backend=backend, meta=meta, log_path=log_path
            )
            service_tier = sv._normalize_requested_service_tier(meta.get("service_tier"))
        except Exception as exc:
            if skip_invalid_sidecars:
                manager._quarantine_sidecar(sock, exc, log=False)
                continue
            raise
        try:
            resp = manager._sock_call(sock, {"cmd": "state"}, timeout_s=0.5)
        except Exception as exc:
            if sv._probe_failure_safe_to_prune(broker_pid=broker_pid, codex_pid=codex_pid):
                sv._unlink_quiet(sock)
                sv._unlink_quiet(meta_path)
                continue
            if (not sv._sock_error_definitely_stale(exc)) and (not skip_invalid_sidecars):
                sv.sys.stderr.write(
                    f"error: discover: sock state call failed for {sock}: {type(exc).__name__}: {exc}\n"
                )
                sv.sys.stderr.flush()
            resp = {"busy": False, "queue_len": 0, "token": None}
        queue_len_raw = resp.get("queue_len") if isinstance(resp, dict) else None
        if (
            not isinstance(resp, dict)
            or not isinstance(resp.get("busy"), bool)
            or type(queue_len_raw) is not int
            or int(queue_len_raw) < 0
        ):
            state_error = ValueError(f"invalid broker state response for socket {sock}")
            if skip_invalid_sidecars:
                continue
            raise state_error

        meta_log_off = int(log_path.stat().st_size) if log_path is not None else 0
        queue_len = int(queue_len_raw) if type(queue_len_raw) is int and int(queue_len_raw) >= 0 else 0
        session = sv.Session(
            session_id=session_id,
            thread_id=thread_id,
            broker_pid=broker_pid,
            codex_pid=codex_pid,
            agent_backend=agent_backend,
            owned=owned,
            backend=backend,
            transport=transport,
            supports_live_ui=supports_live_ui,
            ui_protocol_version=ui_protocol_version,
            start_ts=float(start_ts),
            cwd=str(cwd),
            log_path=log_path,
            sock_path=sock,
            session_path=session_path,
            busy=sv._state_busy_value(resp),
            queue_len=queue_len,
            token=resp.get("token") if isinstance(resp.get("token"), (dict, type(None))) else None,
            meta_thinking=0,
            meta_tools=0,
            meta_system=0,
            meta_log_off=meta_log_off,
            model_provider=model_provider,
            preferred_auth_method=preferred_auth_method,
            model=model,
            reasoning_effort=reasoning_effort,
            service_tier=service_tier,
            tmux_session=tmux_session,
            tmux_window=tmux_window,
            resume_session_id=resume_session_id,
            pi_session_path_discovered=session_path_discovered,
        )
        with manager._lock:
            prev = manager._sessions.get(session_id)
            if not prev:
                manager._reset_log_caches(session, meta_log_off=meta_log_off)
                session.model_provider = model_provider
                session.preferred_auth_method = preferred_auth_method
                session.model = model
                session.reasoning_effort = reasoning_effort
                session.service_tier = service_tier
                manager._sessions[session_id] = session
            else:
                prev.sock_path = session.sock_path
                prev.thread_id = session.thread_id
                prev.backend = session.backend
                prev.broker_pid = session.broker_pid
                prev.codex_pid = session.codex_pid
                prev.agent_backend = session.agent_backend
                prev.owned = session.owned
                prev.transport = session.transport
                prev.supports_live_ui = session.supports_live_ui
                prev.ui_protocol_version = session.ui_protocol_version
                prev.start_ts = session.start_ts
                prev.cwd = session.cwd
                prev.busy = session.busy
                prev.queue_len = session.queue_len
                prev.token = session.token
                manager._apply_session_source(prev, log_path=session.log_path, session_path=session.session_path)
                prev.model_provider = model_provider
                prev.preferred_auth_method = preferred_auth_method
                prev.model = model
                prev.reasoning_effort = reasoning_effort
                prev.service_tier = service_tier
                prev.tmux_session = tmux_session
                prev.tmux_window = tmux_window
                prev.resume_session_id = resume_session_id
                prev.pi_session_path_discovered = session.pi_session_path_discovered or prev.pi_session_path_discovered
    with manager._lock:
        manager._last_discover_ts = time.time()


def refresh_session_state(
    manager: Any, session_id: str, sock_path: Path, timeout_s: float = 0.4
) -> tuple[bool, BaseException | None]:
    sv = manager._runtime
    try:
        resp = manager._sock_call(sock_path, {"cmd": "state"}, timeout_s=timeout_s)
        sv._validated_session_state(resp)
    except Exception as exc:
        return False, exc
    publish_sessions = False
    publish_live = False
    publish_workspace = False
    durable_session_id: str | None = None
    with manager._lock:
        session = manager._sessions.get(session_id)
        if session:
            next_busy = sv._state_busy_value(resp)
            next_queue_len = sv._state_queue_len_value(resp)
            next_token = resp.get("token") if isinstance(resp.get("token"), dict) else session.token
            durable_session_id = manager._durable_session_id_for_session(session)
            publish_sessions = session.busy != next_busy
            publish_live = publish_sessions or session.queue_len != next_queue_len or next_token != session.token
            publish_workspace = session.queue_len != next_queue_len
            session.busy = next_busy
            session.queue_len = next_queue_len
            if isinstance(resp.get("token"), dict):
                session.token = resp.get("token")
    if durable_session_id is not None:
        if publish_sessions:
            sv._publish_sessions_invalidate(reason="session_state_changed")
        if publish_live:
            sv._publish_session_live_invalidate(
                durable_session_id,
                runtime_id=session_id,
                reason="session_state_changed",
            )
        if publish_workspace:
            sv._publish_session_workspace_invalidate(
                durable_session_id,
                runtime_id=session_id,
                reason="session_state_changed",
            )
    return True, None


def prune_dead_sessions(manager: Any) -> None:
    sv = manager._runtime
    with manager._lock:
        items = list(manager._sessions.items())
    dead: list[tuple[str, Path]] = []
    for sid, session in items:
        if not session.sock_path.exists():
            dead.append((sid, session.sock_path))
            continue
        ok, _ = refresh_session_state(manager, sid, session.sock_path, timeout_s=0.4)
        if ok:
            continue
        if not sv._probe_failure_safe_to_prune(
            broker_pid=session.broker_pid, codex_pid=session.codex_pid
        ):
            continue
        dead.append((sid, session.sock_path))
    if not dead:
        return
    dead_events: list[tuple[str, str]] = []
    with manager._lock:
        for sid, _sock in dead:
            session = manager._sessions.pop(sid, None)
            if session is not None:
                dead_events.append((manager._durable_session_id_for_session(session), sid))
    for sid, sock in dead:
        manager._clear_deleted_session_state(sid)
        sv._unlink_quiet(sock)
        sv._unlink_quiet(sock.with_suffix(".json"))
    sv._publish_sessions_invalidate(reason="session_removed")
    for durable_session_id, runtime_id in dead_events:
        sv._publish_session_live_invalidate(
            durable_session_id,
            runtime_id=runtime_id,
            reason="session_removed",
        )
        sv._publish_session_workspace_invalidate(
            durable_session_id,
            runtime_id=runtime_id,
            reason="session_removed",
        )
