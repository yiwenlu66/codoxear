from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .runtime_access import manager_runtime


def _runtime(manager: Any):
    return manager_runtime(manager)


@dataclass(slots=True)
class SessionMessageHistoryService:
    manager: Any

    def set_chat_index_snapshot(
        self,
        *,
        session_id: str,
        events: list[dict[str, Any]],
        token_update: dict[str, Any] | None,
        scan_bytes: int,
        scan_complete: bool,
        log_off: int,
    ) -> None:
        set_chat_index_snapshot(
            self.manager,
            session_id=session_id,
            events=events,
            token_update=token_update,
            scan_bytes=scan_bytes,
            scan_complete=scan_complete,
            log_off=log_off,
        )

    def append_chat_events(
        self,
        session_id: str,
        new_events: list[dict[str, Any]],
        *,
        new_off: int,
        latest_token: dict[str, Any] | None,
    ) -> None:
        append_chat_events(
            self.manager,
            session_id,
            new_events,
            new_off=new_off,
            latest_token=latest_token,
        )

    def attach_notification_texts(
        self, events: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return attach_notification_texts(self.manager, events)

    def update_pi_last_chat_ts(
        self,
        session_id: str,
        events: list[dict[str, Any]],
        *,
        session_path: Path | None,
    ) -> None:
        update_pi_last_chat_ts(
            self.manager,
            session_id,
            events,
            session_path=session_path,
        )

    def ensure_pi_chat_index(self, session_id: str, *, min_events: int, before: int):
        return ensure_pi_chat_index(
            self.manager,
            session_id,
            min_events=min_events,
            before=before,
        )

    def ensure_chat_index(self, session_id: str, *, min_events: int, before: int):
        return ensure_chat_index(
            self.manager,
            session_id,
            min_events=min_events,
            before=before,
        )

    def mark_log_delta(
        self, session_id: str, *, objs: list[dict[str, Any]], new_off: int
    ) -> None:
        mark_log_delta(self.manager, session_id, objs=objs, new_off=new_off)

    def idle_from_log(self, session_id: str) -> bool:
        return idle_from_log(self.manager, session_id)

    def get_messages_page(
        self,
        session_id: str,
        *,
        offset: int,
        init: bool,
        limit: int,
        before: int,
        view: str = "conversation",
    ) -> dict[str, Any]:
        return get_messages_page(
            self.manager,
            session_id,
            offset=offset,
            init=init,
            limit=limit,
            before=before,
            view=view,
        )


def service(manager: Any) -> SessionMessageHistoryService:
    return SessionMessageHistoryService(manager)


def set_chat_index_snapshot(
    manager: Any,
    *,
    session_id: str,
    events: list[dict[str, Any]],
    token_update: dict[str, Any] | None,
    scan_bytes: int,
    scan_complete: bool,
    log_off: int,
) -> None:
    def event_key(ev: dict[str, Any]) -> tuple[str, int, str] | None:
        role = ev.get("role")
        if role not in ("user", "assistant"):
            return None
        text = ev.get("text")
        if not isinstance(text, str):
            return None
        ts = ev.get("ts")
        if not isinstance(ts, (int, float)):
            return None
        return role, int(round(float(ts) * 1000.0)), text

    sv = _runtime(manager)
    with manager._lock:
        session = manager._sessions.get(session_id)
        if not session:
            return
        tail = list(events[-sv.CHAT_INDEX_MAX_EVENTS :])
        uniq_rev: list[dict[str, Any]] = []
        seen_exact: set[tuple[str, int, str]] = set()
        seen_assistant_stretch: set[str] = set()
        for ev in reversed(tail):
            key = event_key(ev)
            if key is not None and key in seen_exact:
                continue
            if key is not None:
                seen_exact.add(key)
            role = ev.get("role")
            if role == "user":
                seen_assistant_stretch.clear()
            elif role == "assistant":
                text = ev.get("text")
                if isinstance(text, str):
                    if text in seen_assistant_stretch:
                        continue
                    seen_assistant_stretch.add(text)
            uniq_rev.append(ev)
        session.chat_index_events = list(reversed(uniq_rev))
        session.chat_index_scan_bytes = int(scan_bytes)
        session.chat_index_scan_complete = bool(scan_complete) and (len(events) <= sv.CHAT_INDEX_MAX_EVENTS)
        session.chat_index_log_off = int(log_off)
        if token_update is not None:
            session.token = token_update


def append_chat_events(
    manager: Any,
    session_id: str,
    new_events: list[dict[str, Any]],
    *,
    new_off: int,
    latest_token: dict[str, Any] | None,
) -> None:
    def event_key(ev: dict[str, Any]) -> tuple[str, int, str] | None:
        role = ev.get("role")
        if role not in ("user", "assistant"):
            return None
        text = ev.get("text")
        if not isinstance(text, str):
            return None
        ts = ev.get("ts")
        if not isinstance(ts, (int, float)):
            return None
        return role, int(round(float(ts) * 1000.0)), text

    if not new_events and latest_token is None:
        with manager._lock:
            session = manager._sessions.get(session_id)
            if session:
                session.chat_index_log_off = int(new_off)
        return
    with manager._lock:
        session = manager._sessions.get(session_id)
        if not session:
            return
        if new_events:
            merged = list(session.chat_index_events)
            recent = merged[-256:] if len(merged) > 256 else merged
            seen_exact: set[tuple[str, int, str]] = set()
            for ev in recent:
                key = event_key(ev)
                if key is not None:
                    seen_exact.add(key)
            assistant_stretch: set[str] = set()
            for ev in reversed(merged):
                role = ev.get("role")
                if role == "user":
                    break
                if role == "assistant":
                    text = ev.get("text")
                    if isinstance(text, str):
                        assistant_stretch.add(text)
            for ev in new_events:
                key = event_key(ev)
                if key is not None and key in seen_exact:
                    continue
                if key is not None:
                    seen_exact.add(key)
                role = ev.get("role")
                if role == "user":
                    assistant_stretch.clear()
                elif role == "assistant":
                    text = ev.get("text")
                    if isinstance(text, str):
                        if text in assistant_stretch:
                            continue
                        assistant_stretch.add(text)
                merged.append(ev)
            sv = _runtime(manager)
            if len(merged) > sv.CHAT_INDEX_MAX_EVENTS:
                merged = merged[-sv.CHAT_INDEX_MAX_EVENTS :]
                session.chat_index_scan_complete = False
            session.chat_index_events = merged
        session.chat_index_log_off = int(new_off)
        if latest_token is not None:
            session.token = latest_token


def attach_notification_texts(manager: Any, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    voice_push = getattr(manager, "_voice_push", None)
    if voice_push is None:
        return list(events)
    out: list[dict[str, Any]] = []
    for ev in events:
        if not isinstance(ev, dict):
            out.append(ev)
            continue
        if ev.get("role") != "assistant" or ev.get("message_class") != "final_response":
            out.append(ev)
            continue
        message_id = ev.get("message_id")
        if not isinstance(message_id, str) or not message_id:
            out.append(ev)
            continue
        notification_text = voice_push.notification_text_for_message(message_id)
        if not notification_text:
            out.append(ev)
            continue
        ev2 = dict(ev)
        ev2["notification_text"] = notification_text
        out.append(ev2)
    return out


def update_pi_last_chat_ts(
    manager: Any,
    session_id: str,
    events: list[dict[str, Any]],
    *,
    session_path: Path | None,
) -> None:
    sv = _runtime(manager)
    if not events:
        return
    if not any(sv._is_attention_worthy_session_event(event) for event in events):
        return
    latest_chat_ts = sv._session_file_activity_ts(session_path)
    if latest_chat_ts is None:
        latest_chat_ts = sv._attention_updated_ts_from_events(events)
    if latest_chat_ts is None:
        return
    with manager._lock:
        session = manager._sessions.get(session_id)
        if not session:
            return
        session.last_chat_ts = latest_chat_ts if session.last_chat_ts is None else max(session.last_chat_ts, latest_chat_ts)


def ensure_pi_chat_index(manager: Any, session_id: str, *, min_events: int, before: int):
    with manager._lock:
        session = manager._sessions.get(session_id)
        if not session:
            return [], 0, False, 0, {"tool_names": [], "last_tool": None}
        session_path = session.session_path
        scan_bytes = int(session.chat_index_scan_bytes) if session.chat_index_scan_bytes > 0 else (256 * 1024)
        idx_off = int(session.chat_index_log_off)
    if session_path is None or (not session_path.exists()):
        return [], 0, False, 0, {"tool_names": [], "last_tool": None}

    size = int(session_path.stat().st_size)
    if size < idx_off:
        set_chat_index_snapshot(
            manager,
            session_id=session_id,
            events=[],
            token_update=None,
            scan_bytes=256 * 1024,
            scan_complete=False,
            log_off=0,
        )
        idx_off = 0

    with manager._lock:
        session = manager._sessions.get(session_id)
        ready = bool(session and (session.chat_index_events is not None))
        cached_count = len(session.chat_index_events) if session else 0
        scan_complete = bool(session.chat_index_scan_complete) if session else False
    target_events = max(0, int(min_events) + max(0, int(before)))
    seed_diag: dict[str, Any] = {"tool_names": [], "last_tool": None}
    if (not ready) or ((target_events > cached_count) and (not scan_complete)):
        events, token_update, new_off, used_scan, complete, diag = _runtime(manager)._pi_messages.read_pi_message_tail_snapshot(
            session_path,
            min_events=max(20, target_events),
            initial_scan_bytes=max(256 * 1024, scan_bytes),
            max_scan_bytes=64 * 1024 * 1024,
        )
        seed_diag = diag
        set_chat_index_snapshot(
            manager,
            session_id=session_id,
            events=events,
            token_update=token_update,
            scan_bytes=used_scan,
            scan_complete=complete,
            log_off=new_off,
        )
    with manager._lock:
        session = manager._sessions.get(session_id)
        if not session:
            return [], 0, False, 0, {"tool_names": [], "last_tool": None}
        session_path2 = session.session_path
        off2 = int(session.chat_index_log_off)
        prev_events = list(session.chat_index_events)
    if session_path2 is None or (not session_path2.exists()):
        return prev_events, off2, False, 0, {"tool_names": [], "last_tool": None}
    size2 = int(session_path2.stat().st_size)
    latest_diag = seed_diag
    if size2 > off2:
        events_delta, new_off, _meta_delta, _flags, latest_diag = _runtime(manager)._pi_messages.read_pi_message_delta(
            session_path2,
            offset=off2,
        )
        if events_delta:
            append_chat_events(
                manager,
                session_id,
                events_delta,
                new_off=new_off,
                latest_token=None,
            )
        elif new_off > off2:
            append_chat_events(manager, session_id, [], new_off=new_off, latest_token=None)
    with manager._lock:
        session = manager._sessions.get(session_id)
        if not session:
            return prev_events, off2, False, 0, latest_diag
        events2 = list(session.chat_index_events)
        off4 = int(session.chat_index_log_off)
        scan_complete3 = bool(session.chat_index_scan_complete)
    n = len(events2)
    b = max(0, int(before))
    end = max(0, n - b)
    start = max(0, end - max(20, int(min_events)))
    page = attach_notification_texts(manager, events2[start:end])
    has_older = start > 0 or ((not scan_complete3) and bool(page))
    next_before = b + len(page) if has_older else 0
    return page, off4, has_older, next_before, latest_diag


def ensure_chat_index(manager: Any, session_id: str, *, min_events: int, before: int):
    sv = _runtime(manager)
    with manager._lock:
        session = manager._sessions.get(session_id)
        if not session:
            return [], 0, False, 0, None
        log_path = session.log_path
        scan_bytes = int(session.chat_index_scan_bytes) if session.chat_index_scan_bytes > 0 else sv.CHAT_INIT_SEED_SCAN_BYTES
        idx_off = int(session.chat_index_log_off)
    if log_path is None or (not log_path.exists()):
        return [], 0, False, 0, None

    size = int(log_path.stat().st_size)
    if size < idx_off:
        idx_off = 0
        set_chat_index_snapshot(
            manager,
            session_id=session_id,
            events=[],
            token_update=None,
            scan_bytes=sv.CHAT_INIT_SEED_SCAN_BYTES,
            scan_complete=False,
            log_off=0,
        )

    with manager._lock:
        session = manager._sessions.get(session_id)
        ready = bool(session and (session.chat_index_events is not None))
        cached_count = len(session.chat_index_events) if session else 0
        scan_complete = bool(session.chat_index_scan_complete) if session else False

    target_events = max(0, int(min_events) + max(0, int(before)))
    if (not ready) or ((target_events > cached_count) and (not scan_complete)):
        events, token_update, used_scan, complete, log_size = sv._read_chat_tail_snapshot(
            log_path,
            min_events=max(20, target_events),
            initial_scan_bytes=max(sv.CHAT_INIT_SEED_SCAN_BYTES, scan_bytes),
            max_scan_bytes=sv.CHAT_INIT_MAX_SCAN_BYTES,
        )
        set_chat_index_snapshot(
            manager,
            session_id=session_id,
            events=events,
            token_update=token_update,
            scan_bytes=used_scan,
            scan_complete=complete,
            log_off=log_size,
        )

    with manager._lock:
        session = manager._sessions.get(session_id)
        if not session:
            return [], 0, False, 0, None
        log_path2 = session.log_path
        off2 = int(session.chat_index_log_off)
    if log_path2 is None or (not log_path2.exists()):
        return [], 0, False, 0, None

    size2 = int(log_path2.stat().st_size)
    if size2 > off2:
        delta = size2 - off2
        if delta >= sv.CHAT_INDEX_RESEED_THRESHOLD_BYTES:
            events, token_update, used_scan, complete, log_size = sv._read_chat_tail_snapshot(
                log_path2,
                min_events=max(20, target_events),
                initial_scan_bytes=max(sv.CHAT_INIT_SEED_SCAN_BYTES, scan_bytes),
                max_scan_bytes=sv.CHAT_INIT_MAX_SCAN_BYTES,
            )
            set_chat_index_snapshot(
                manager,
                session_id=session_id,
                events=events,
                token_update=token_update,
                scan_bytes=used_scan,
                scan_complete=complete,
                log_off=log_size,
            )
        else:
            cur = off2
            loops = 0
            latest_token: dict[str, Any] | None = None
            aggregated_events: list[dict[str, Any]] = []
            while cur < size2 and loops < 16:
                objs, new_off = sv._read_jsonl_from_offset(log_path2, cur, max_bytes=sv.CHAT_INDEX_INCREMENT_BYTES)
                if new_off <= cur:
                    break
                _th, _tools, _sys, _last_ts, token_update, new_events = sv._analyze_log_chunk(objs)
                if token_update is not None:
                    latest_token = token_update
                if new_events:
                    aggregated_events.extend(new_events)
                cur = new_off
                loops += 1
            append_chat_events(manager, session_id, aggregated_events, new_off=cur, latest_token=latest_token)

    with manager._lock:
        session = manager._sessions.get(session_id)
        if not session:
            return [], 0, False, 0, None
        events2 = list(session.chat_index_events)
        log_off2 = int(session.chat_index_log_off)
        scan_complete2 = bool(session.chat_index_scan_complete)
        token2 = session.token if isinstance(session.token, dict) or session.token is None else None

    n = len(events2)
    b = max(0, int(before))
    end = max(0, n - b)
    start = max(0, end - max(20, int(min_events)))
    page = attach_notification_texts(manager, events2[start:end])
    has_older = (start > 0) or ((not scan_complete2) and bool(page))
    next_before = b + len(page) if has_older else 0
    return page, log_off2, has_older, next_before, token2


def mark_log_delta(
    manager: Any, session_id: str, *, objs: list[dict[str, Any]], new_off: int
) -> None:
    sv = _runtime(manager)
    _th, _tools, _sys, last_ts, token_update, new_events = sv._analyze_log_chunk(objs)
    model = None
    reasoning_effort = None
    for obj in reversed(objs):
        if not isinstance(obj, dict) or obj.get("type") != "turn_context":
            continue
        model, reasoning_effort = sv._turn_context_run_settings(obj.get("payload"))
        break
    append_chat_events(manager, session_id, new_events, new_off=new_off, latest_token=token_update)
    durable_session_id: str | None = None
    with manager._lock:
        session = manager._sessions.get(session_id)
        if session:
            durable_session_id = manager._durable_session_id_for_session(session)
            if isinstance(last_ts, (int, float)):
                tsf = float(last_ts)
                session.last_chat_ts = tsf if session.last_chat_ts is None else max(session.last_chat_ts, tsf)
            if model is not None:
                session.model = model
            if reasoning_effort is not None:
                session.reasoning_effort = reasoning_effort
            session.idle_cache_log_off = -1
    if durable_session_id is not None and (new_events or token_update is not None or model is not None or reasoning_effort is not None):
        sv._publish_session_live_invalidate(durable_session_id, runtime_id=session_id, reason="log_delta")
        sv._publish_session_workspace_invalidate(durable_session_id, runtime_id=session_id, reason="log_delta")
        if any(sv._is_attention_worthy_session_event(event) for event in new_events):
            sv._publish_sessions_invalidate(reason="conversation_changed")


def idle_from_log(manager: Any, session_id: str) -> bool:
    sv = _runtime(manager)
    with manager._lock:
        session = manager._sessions.get(session_id)
        if not session:
            raise KeyError("unknown session")
        log_path = session.log_path
        cached_off = int(session.idle_cache_log_off)
        cached_idle = session.idle_cache_value
    if log_path is None or (not log_path.exists()):
        raise FileNotFoundError(f"missing rollout log for session {session_id}")
    size = int(log_path.stat().st_size)
    if (size >= 0) and (cached_off == size) and isinstance(cached_idle, bool):
        return bool(cached_idle)
    idle = sv._compute_idle_from_log(log_path)
    with manager._lock:
        session = manager._sessions.get(session_id)
        if session:
            session.idle_cache_log_off = size
            session.idle_cache_value = idle
    if idle is None:
        raise RuntimeError("unable to compute idle state from log")
    return bool(idle)


def get_messages_page(
    manager: Any,
    session_id: str,
    *,
    offset: int,
    init: bool,
    limit: int,
    before: int,
    view: str = "conversation",
) -> dict[str, Any]:
    sv = _runtime(manager)
    historical_row = sv._historical_session_row(session_id)
    if historical_row is not None:
        historical_backend = normalize_agent_backend(
            historical_row.get("agent_backend", historical_row.get("backend")),
            default="codex",
        )
        if historical_backend != "pi":
            raise KeyError("unknown session")
        session_path_raw = historical_row.get("session_path")
        session_path = Path(session_path_raw) if isinstance(session_path_raw, str) and session_path_raw else None
        if session_path is None or (not session_path.exists()):
            return {
                "thread_id": historical_row.get("resume_session_id"),
                "log_path": str(session_path) if session_path is not None else None,
                "offset": 0,
                "events": [],
                "meta_delta": {"thinking": 0, "tool": 0, "system": 0},
                "turn_start": False,
                "turn_end": False,
                "turn_aborted": False,
                "diag": {"pending_log": True},
                "busy": False,
                "queue_len": 0,
                "token": None,
                "has_older": False,
                "next_before": 0,
            }
        if init and offset == 0:
            historical_events, new_off, has_older, next_before, diag = sv._pi_messages.read_pi_message_page(
                session_path,
                limit=limit,
                before=before,
            )
            meta_delta = {"thinking": 0, "tool": 0, "system": 0}
            flags = {"turn_start": False, "turn_end": False, "turn_aborted": False}
        else:
            historical_events, new_off, meta_delta, flags, diag = sv._pi_messages.read_pi_message_delta(
                session_path,
                offset=offset,
            )
            has_older = False
            next_before = 0
        return {
            "thread_id": historical_row.get("resume_session_id"),
            "log_path": str(session_path),
            "offset": int(new_off),
            "events": historical_events,
            "meta_delta": meta_delta,
            "turn_start": bool(flags.get("turn_start")),
            "turn_end": bool(flags.get("turn_end")),
            "turn_aborted": bool(flags.get("turn_aborted")),
            "diag": diag,
            "busy": False,
            "queue_len": 0,
            "token": None,
            "has_older": bool(has_older),
            "next_before": int(next_before),
        }

    runtime_session_id = manager._runtime_session_id_for_identifier(session_id)
    if runtime_session_id is None:
        raise KeyError("unknown session")
    session_id = runtime_session_id
    manager.refresh_session_meta(session_id, strict=False)
    session = manager.get_session(session_id)
    if not session:
        raise KeyError("unknown session")

    state = manager.get_state(session_id)
    if not isinstance(state, dict):
        raise ValueError("invalid broker state response")
    if "busy" not in state:
        raise ValueError("missing busy from broker state response")
    if "queue_len" not in state:
        raise ValueError("missing queue_len from broker state response")
    state_token = state.get("token")
    if not (isinstance(state_token, dict) or (state_token is None)):
        raise ValueError("invalid token from broker state response")

    if session.backend == "pi":
        diag = {"tool_names": [], "last_tool": None}
        flags = {"turn_start": False, "turn_end": False, "turn_aborted": False}
        meta_delta = {"thinking": 0, "tool": 0, "system": 0}
        new_off = 0
        has_older = False
        next_before = 0
        if session.session_path is None and session.cwd:
            claimed = manager._claimed_pi_session_paths(exclude_sid=session_id)
            discovered, discovered_source = sv._resolve_pi_session_path(
                thread_id=session.thread_id,
                cwd=session.cwd,
                start_ts=session.start_ts,
                preferred=None,
                exclude=claimed,
            )
            if discovered is not None:
                session.session_path = discovered
                if discovered_source in {"exact", "discovered"}:
                    session.pi_session_path_discovered = True
                sv._patch_metadata_session_path(session.sock_path, discovered)
        if session.session_path is not None and session.session_path.exists():
            if init and offset == 0:
                events, new_off, has_older, next_before, diag = ensure_pi_chat_index(
                    manager,
                    session_id,
                    min_events=limit,
                    before=before,
                )
            else:
                events, new_off, meta_delta, flags, diag = sv._pi_messages.read_pi_message_delta(
                    session.session_path,
                    offset=offset,
                )
                if new_off > offset:
                    append_chat_events(manager, session_id, events, new_off=new_off, latest_token=None)
            update_pi_last_chat_ts(manager, session_id, events, session_path=session.session_path)
        if session.pi_session_path_discovered and session.session_path is not None and session.cwd and not events:
            sp_mtime = sv._safe_path_mtime(session.session_path)
            if sp_mtime is not None and (sv.time.time() - sp_mtime) > 2.0:
                old_sp = session.session_path
                claimed = manager._claimed_pi_session_paths(exclude_sid=session_id)
                claimed.add(old_sp)
                newer_sp, newer_sp_source = sv._resolve_pi_session_path(
                    thread_id=session.thread_id,
                    cwd=session.cwd,
                    start_ts=session.start_ts,
                    preferred=None,
                    exclude=claimed,
                )
                newer_sp_mtime = sv._safe_path_mtime(newer_sp) if newer_sp is not None else None
                if newer_sp is not None and newer_sp != old_sp and (
                    newer_sp_source == "exact" or (newer_sp_mtime is not None and newer_sp_mtime > sp_mtime)
                ):
                    session.session_path = newer_sp
                    sv._patch_metadata_session_path(session.sock_path, newer_sp, force=True)
                    manager._reset_log_caches(session, meta_log_off=0)
                    events, new_off, has_older, next_before, diag = ensure_pi_chat_index(
                        manager,
                        session_id,
                        min_events=limit,
                        before=0,
                    )
                    update_pi_last_chat_ts(manager, session_id, events, session_path=session.session_path)
        pi_busy = sv._display_pi_busy(session, broker_busy=sv._state_busy_value(state))
        return {
            "thread_id": session.thread_id,
            "log_path": str(session.session_path) if session.session_path is not None else None,
            "offset": int(new_off),
            "events": events,
            "meta_delta": meta_delta,
            "turn_start": bool(flags.get("turn_start")),
            "turn_end": bool(flags.get("turn_end")),
            "turn_aborted": bool(flags.get("turn_aborted")),
            "diag": diag,
            "busy": pi_busy,
            "queue_len": int(manager._queue_len(session_id)),
            "token": state_token,
            "has_older": bool(has_older),
            "next_before": int(next_before),
        }

    if session.log_path is None or (not session.log_path.exists()):
        return {
            "thread_id": session.thread_id,
            "log_path": None,
            "offset": 0,
            "events": [],
            "meta_delta": {"thinking": 0, "tool": 0, "system": 0},
            "turn_start": False,
            "turn_end": False,
            "turn_aborted": False,
            "diag": {"pending_log": True},
            "busy": False,
            "queue_len": int(manager._queue_len(session_id)),
            "token": state_token,
            "has_older": False,
            "next_before": 0,
        }

    if init and offset == 0:
        events, new_off, has_older, next_before, token_update = ensure_chat_index(
            manager,
            session_id,
            min_events=limit,
            before=before,
        )
        meta_delta = {"thinking": 0, "tool": 0, "system": 0}
        flags = {"turn_start": False, "turn_end": False, "turn_aborted": False}
        diag = {"tool_names": [], "last_tool": None}
    else:
        has_older = False
        next_before = 0
        objs, new_off = sv._read_jsonl_from_offset(session.log_path, offset)
        events, meta_delta, flags, diag = sv._extract_chat_events(objs)
        token_update = sv._extract_token_update(objs)
        mark_log_delta(manager, session_id, objs=objs, new_off=new_off)

    session2 = manager.get_session(session_id)
    if token_update is not None and session2 is not None:
        session2.token = token_update
    idle_val = idle_from_log(manager, session_id)
    busy_val = sv._state_busy_value(state) or (not bool(idle_val))
    token_val = state_token if state_token is not None else token_update if isinstance(token_update, dict) else None
    return {
        "thread_id": session.thread_id,
        "log_path": str(session.log_path),
        "offset": int(new_off),
        "events": events,
        "meta_delta": meta_delta,
        "turn_start": bool(flags.get("turn_start")),
        "turn_end": bool(flags.get("turn_end")),
        "turn_aborted": bool(flags.get("turn_aborted")),
        "diag": diag,
        "busy": bool(busy_val),
        "queue_len": int(manager._queue_len(session_id)),
        "token": token_val,
        "has_older": bool(has_older),
        "next_before": int(next_before),
    }
