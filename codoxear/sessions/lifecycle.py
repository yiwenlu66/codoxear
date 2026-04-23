from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .runtime_access import manager_runtime


def _runtime(manager: Any):
    return manager_runtime(manager)


def catalog_record_for_ref(manager: Any, ref: Any):
    sv = _runtime(manager)
    backend, durable_session_id = ref
    source_path = sv._find_session_log_for_session_id(
        durable_session_id, agent_backend=backend
    )
    if source_path is None or (not source_path.exists()):
        return None
    if backend == "pi":
        row = sv._pi_resume_candidate_from_session_file(source_path)
        if not isinstance(row, dict):
            return None
        title = sv._clean_optional_text(row.get("title")) or ""
        first_user_message = sv._clean_optional_text(
            sv._first_user_message_preview_from_pi_session(source_path)
        )
    else:
        row = sv._resume_candidate_from_log(source_path, agent_backend=backend)
        if not isinstance(row, dict):
            return None
        title = sv._clean_optional_text(row.get("title")) or ""
        first_user_message = sv._clean_optional_text(
            sv._first_user_message_preview_from_log(source_path)
        )
    cwd = sv._clean_optional_text(row.get("cwd"))
    updated_ts = row.get("updated_ts")
    updated_at = (
        float(updated_ts)
        if isinstance(updated_ts, (int, float))
        else sv._safe_path_mtime(source_path)
    )
    return sv.DurableSessionRecord(
        backend=backend,
        session_id=durable_session_id,
        cwd=cwd,
        source_path=str(source_path),
        title=title,
        first_user_message=first_user_message,
        created_at=updated_at,
        updated_at=updated_at,
    )


def refresh_durable_session_catalog(manager: Any, *, force: bool = False) -> None:
    sv = _runtime(manager)
    db = getattr(manager, "_page_state_db", None)
    if not isinstance(db, sv.PageStateDB):
        return
    now = sv.time.time()
    last_refresh = float(getattr(manager, "_last_session_catalog_refresh_ts", 0.0) or 0.0)
    if (not force) and (now - last_refresh) < 5.0:
        return
    refs = set(db.known_session_refs())
    with manager._lock:
        for session in manager._sessions.values():
            ref = manager._page_state_ref_for_session(session)
            if ref is not None:
                refs.add(ref)
    existing = db.load_sessions()
    rows: dict[Any, Any] = {}
    for ref in sorted(refs):
        record = manager._catalog_record_for_ref(ref)
        if record is None:
            record = existing.get(ref)
        if record is not None:
            rows[ref] = record
    db.save_sessions(rows)
    manager._last_session_catalog_refresh_ts = now


def wait_for_live_session(
    manager: Any,
    durable_session_id: str,
    *,
    timeout_s: float = 8.0,
):
    sv = _runtime(manager)
    deadline = sv.time.time() + max(timeout_s, 0.1)
    while sv.time.time() < deadline:
        manager._discover_existing(force=True, skip_invalid_sidecars=True)
        runtime_id = manager._runtime_session_id_for_identifier(durable_session_id)
        if runtime_id is not None:
            with manager._lock:
                session = manager._sessions.get(runtime_id)
            if session is not None:
                return session
        sv.time.sleep(0.05)
    raise RuntimeError(
        f"spawned session is not yet discoverable: {durable_session_id}"
    )


def copy_session_ui_identity(
    manager: Any,
    *,
    source_session_id: str,
    target_session_id: str,
) -> str | None:
    sv = _runtime(manager)
    alias = sv._clean_optional_text(manager.alias_get(source_session_id))
    meta = manager.sidebar_meta_get(source_session_id)
    if alias is not None:
        alias = manager.alias_set(target_session_id, alias)
    manager.sidebar_meta_set(
        target_session_id,
        priority_offset=meta.get("priority_offset"),
        snooze_until=meta.get("snooze_until"),
        dependency_session_id=meta.get("dependency_session_id"),
    )
    if bool(meta.get("focused")):
        manager.focus_set(target_session_id, True)
    return alias


def capture_runtime_bound_restart_state(manager: Any, runtime_id: str, ref: Any) -> dict[str, Any]:
    with manager._lock:
        files_src = getattr(manager, "_files", None)
        queues_src = getattr(manager, "_queues", None)
        harness_src = getattr(manager, "_harness", None)
        harness_last_src = getattr(manager, "_harness_last_injected", None)
        files = (
            list(files_src.get(runtime_id, files_src.get(ref, [])))
            if isinstance(files_src, dict)
            else []
        )
        queue = (
            list(queues_src.get(runtime_id, queues_src.get(ref, [])))
            if isinstance(queues_src, dict)
            else []
        )
        harness = (
            dict(harness_src.get(runtime_id, {}))
            if isinstance(harness_src, dict)
            and isinstance(harness_src.get(runtime_id), dict)
            else None
        )
        harness_last = (
            harness_last_src.get(runtime_id)
            if isinstance(harness_last_src, dict)
            else None
        )
    return {
        "files": files,
        "queue": queue,
        "harness": harness,
        "harness_last_injected": harness_last,
    }


def stage_runtime_bound_restart_state(
    manager: Any, runtime_id: str, ref: Any, state: dict[str, Any]
) -> None:
    save_files = False
    save_queues = False
    save_harness = False
    with manager._lock:
        files_src = getattr(manager, "_files", None)
        if isinstance(files_src, dict):
            files = state.get("files")
            if isinstance(files, list):
                files_src[ref] = list(files)
            if runtime_id in files_src:
                files_src.pop(runtime_id, None)
            save_files = True
        queues_src = getattr(manager, "_queues", None)
        if isinstance(queues_src, dict):
            queue = state.get("queue")
            if isinstance(queue, list):
                queues_src[ref] = list(queue)
            if runtime_id in queues_src:
                queues_src.pop(runtime_id, None)
            save_queues = True
        harness_src = getattr(manager, "_harness", None)
        if isinstance(harness_src, dict):
            harness_src.pop(runtime_id, None)
            harness = state.get("harness")
            if isinstance(harness, dict):
                harness_src[ref[1]] = dict(harness)
            save_harness = True
        harness_last_src = getattr(manager, "_harness_last_injected", None)
        if isinstance(harness_last_src, dict):
            harness_last_src.pop(runtime_id, None)
        outbound_src = getattr(manager, "_outbound_requests", None)
        if isinstance(outbound_src, dict):
            outbound_src.pop(runtime_id, None)
        command_cache = getattr(manager, "_pi_commands_cache", None)
        if isinstance(command_cache, dict):
            command_cache.pop(runtime_id, None)
    if save_files:
        manager._save_files()
    if save_queues:
        manager._save_queues()
    if save_harness:
        manager._save_harness()


def restore_runtime_bound_restart_state(
    manager: Any, runtime_id: str, ref: Any, state: dict[str, Any]
) -> None:
    save_files = False
    save_queues = False
    save_harness = False
    with manager._lock:
        files_src = getattr(manager, "_files", None)
        if isinstance(files_src, dict):
            files = state.get("files")
            if isinstance(files, list):
                files_src[runtime_id] = list(files)
                files_src[ref] = list(files)
                save_files = True
        queues_src = getattr(manager, "_queues", None)
        if isinstance(queues_src, dict):
            queue = state.get("queue")
            if isinstance(queue, list):
                queues_src[runtime_id] = list(queue)
                queues_src[ref] = list(queue)
                save_queues = True
        harness_src = getattr(manager, "_harness", None)
        if isinstance(harness_src, dict):
            harness = state.get("harness")
            if isinstance(harness, dict):
                harness_src[runtime_id] = dict(harness)
                save_harness = True
            harness_src.pop(ref[1], None)
        harness_last_src = getattr(manager, "_harness_last_injected", None)
        if isinstance(harness_last_src, dict):
            harness_last = state.get("harness_last_injected")
            if isinstance(harness_last, (int, float)):
                harness_last_src[runtime_id] = float(harness_last)
    if save_files:
        manager._save_files()
    if save_queues:
        manager._save_queues()
    if save_harness:
        manager._save_harness()


def finalize_pending_pi_spawn(
    manager: Any,
    *,
    spawn_nonce: str,
    durable_session_id: str,
    cwd: str,
    session_path: Path,
    proc: Any = None,
    delete_on_failure: bool = True,
    restore_record_on_failure: Any = None,
) -> None:
    sv = _runtime(manager)
    ref = ("pi", durable_session_id)
    try:
        if proc is not None:
            sv._wait_or_raise(proc, label="pi broker", timeout_s=0.25)
            sv._start_proc_stderr_drain(proc)
        meta = sv._wait_for_spawned_broker_meta(spawn_nonce)
        live_session_id = sv._clean_optional_text(meta.get("session_id")) or durable_session_id
        if live_session_id != durable_session_id:
            raise RuntimeError(
                f"pi session id mismatch: expected {durable_session_id}, got {live_session_id}"
            )
        manager._discover_existing(force=True, skip_invalid_sidecars=True)
        manager._refresh_durable_session_catalog(force=True)
        db = getattr(manager, "_page_state_db", None)
        current = db.load_sessions().get(ref) if isinstance(db, sv.PageStateDB) else None
        manager._persist_durable_session_record(
            sv.DurableSessionRecord(
                backend="pi",
                session_id=durable_session_id,
                cwd=(current.cwd if current is not None else cwd),
                source_path=(current.source_path if current is not None else str(session_path)),
                title=current.title if current is not None else None,
                first_user_message=current.first_user_message if current is not None else None,
                created_at=(current.created_at if current is not None else sv._safe_path_mtime(session_path)),
                updated_at=(current.updated_at if current is not None else sv._safe_path_mtime(session_path)),
                pending_startup=False,
            )
        )
        sv._publish_sessions_invalidate(reason="session_created")
    except Exception:
        if delete_on_failure:
            manager._delete_durable_session_record(ref)
            manager._clear_deleted_session_state(durable_session_id)
            sv._publish_sessions_invalidate(reason="session_removed")
            return
        if restore_record_on_failure is not None:
            manager._persist_durable_session_record(restore_record_on_failure)
        else:
            manager._refresh_durable_session_catalog(force=True)
        sv._publish_sessions_invalidate(reason="session_created")


def reset_log_caches(manager: Any, session: Any, *, meta_log_off: int) -> None:
    session.meta_thinking = 0
    session.meta_tools = 0
    session.meta_system = 0
    session.last_chat_ts = None
    session.last_chat_history_scanned = False
    session.pi_idle_activity_ts = None
    session.pi_busy_activity_floor = None
    session.meta_log_off = int(meta_log_off)
    session.chat_index_events = []
    session.chat_index_scan_bytes = 0
    session.chat_index_scan_complete = False
    session.chat_index_log_off = int(meta_log_off)
    session.delivery_log_off = int(meta_log_off)
    session.idle_cache_log_off = -1
    session.idle_cache_value = None
    session.queue_idle_since = None
    session.model_provider = None
    session.preferred_auth_method = None
    session.model = None
    session.reasoning_effort = None
    session.service_tier = None
    session.first_user_message = None


def session_source_changed(
    manager: Any, session: Any, *, log_path: Path | None, session_path: Path | None
) -> bool:
    _runtime(manager)
    if session.log_path != log_path:
        return True
    if session.backend == "pi" and session.session_path != session_path:
        return True
    return False


def claimed_pi_session_paths(manager: Any, *, exclude_sid: str = "") -> set[Path]:
    with manager._lock:
        out: set[Path] = set()
        for session in manager._sessions.values():
            if (
                session.backend == "pi"
                and session.session_path is not None
                and session.session_id != exclude_sid
            ):
                out.add(session.session_path)
        return out


def apply_session_source(
    manager: Any, session: Any, *, log_path: Path | None, session_path: Path | None
) -> None:
    if session.backend == "pi" and session_path is None and session.session_path is not None:
        session_path = session.session_path
    source_changed = manager._session_source_changed(
        session,
        log_path=log_path,
        session_path=session_path,
    )
    session.log_path = log_path
    session.session_path = session_path
    if source_changed:
        log_off = (
            int(log_path.stat().st_size)
            if log_path is not None and log_path.exists()
            else 0
        )
        manager._reset_log_caches(session, meta_log_off=log_off)


def session_run_settings(
    manager: Any,
    *,
    meta: dict[str, Any],
    log_path: Path | None,
    backend: str | None = None,
    agent_backend: str | None = None,
) -> tuple[str | None, str | None, str | None, str | None]:
    sv = _runtime(manager)
    backend_name = sv.normalize_agent_backend(
        backend if backend is not None else agent_backend,
        default="codex",
    )
    model_provider = sv._clean_optional_text(meta.get("model_provider"))
    preferred_auth_method = sv._normalize_requested_preferred_auth_method(
        meta.get("preferred_auth_method")
    )
    model = sv._clean_optional_text(meta.get("model"))
    reasoning_effort = (
        sv._display_reasoning_effort(meta.get("reasoning_effort"))
        if backend_name == "codex"
        else sv._display_pi_reasoning_effort(meta.get("reasoning_effort"))
    )
    if log_path is not None and log_path.exists():
        log_provider, log_model, log_effort = sv._read_run_settings_from_log(
            log_path,
            agent_backend=backend_name,
        )
        if log_provider is not None:
            model_provider = log_provider
        if log_model is not None:
            model = log_model
        if log_effort is not None:
            reasoning_effort = log_effort
    return model_provider, preferred_auth_method, model, reasoning_effort
