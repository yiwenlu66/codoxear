from __future__ import annotations

import sys
import threading
import traceback
from pathlib import Path
from typing import Any

from .runtime_access import manager_runtime


def _runtime(manager: Any):
    return manager_runtime(manager)


def set_bridge_transport_state(
    manager: Any,
    runtime_id: str,
    *,
    state: str,
    error: str | None = None,
    checked_ts: float | None = None,
) -> None:
    sv = _runtime(manager)
    publish = False
    durable_session_id: str | None = None
    with manager._lock:
        session = manager._sessions.get(runtime_id)
        if session is None:
            return
        next_error = sv._clean_optional_text(error)
        publish = (
            session.bridge_transport_state != state
            or session.bridge_transport_error != next_error
        )
        session.bridge_transport_state = state
        session.bridge_transport_error = next_error
        session.bridge_transport_checked_ts = float(
            checked_ts if checked_ts is not None else sv.time.time()
        )
        durable_session_id = manager._durable_session_id_for_session(session)
    if publish and durable_session_id is not None:
        sv._publish_session_transport_invalidate(
            durable_session_id,
            runtime_id=runtime_id,
            reason="transport_state",
        )
        sv._publish_session_live_invalidate(
            durable_session_id,
            runtime_id=runtime_id,
            reason="transport_state",
        )


def probe_bridge_transport(
    manager: Any, session_id: str, *, force_rpc: bool = False
) -> tuple[str, str | None]:
    sv = _runtime(manager)
    runtime_id = manager._runtime_session_id_for_identifier(session_id)
    if runtime_id is None:
        return "dead", "unknown session"
    with manager._lock:
        session = manager._sessions.get(runtime_id)
    if session is None:
        return "dead", "unknown session"
    processes_dead = (not sv._pid_alive(session.broker_pid)) and (
        not sv._pid_alive(session.codex_pid)
    )
    now = sv.time.time()
    last_checked = float(session.bridge_transport_checked_ts or 0.0)
    if (
        not force_rpc
        and session.bridge_transport_state in {"alive", "degraded"}
        and (now - last_checked) < sv.BRIDGE_TRANSPORT_PROBE_STALE_SECONDS
    ):
        return session.bridge_transport_state, session.bridge_transport_error
    try:
        resp = manager._sock_call(
            session.sock_path,
            {"cmd": "state"},
            timeout_s=sv.BRIDGE_TRANSPORT_RPC_TIMEOUT_SECONDS,
        )
        if not isinstance(resp, dict):
            raise ValueError("invalid state probe response")
    except Exception as exc:
        if processes_dead:
            manager._set_bridge_transport_state(
                runtime_id,
                state="dead",
                error="broker exited",
                checked_ts=now,
            )
            return "dead", "broker exited"
        error = str(exc).strip() or type(exc).__name__
        manager._set_bridge_transport_state(
            runtime_id,
            state="degraded",
            error=error,
            checked_ts=now,
        )
        return "degraded", error
    manager._set_bridge_transport_state(
        runtime_id,
        state="alive",
        error=None,
        checked_ts=now,
    )
    return "alive", None


def enqueue_outbound_request(manager: Any, runtime_id: str, text: str):
    sv = _runtime(manager)
    with manager._lock:
        session = manager._sessions.get(runtime_id)
        if session is None:
            raise KeyError("unknown session")
        durable_session_id = manager._durable_session_id_for_session(session)
        requests_by_runtime = getattr(manager, "_outbound_requests", None)
        if not isinstance(requests_by_runtime, dict):
            manager._outbound_requests = {}
            requests_by_runtime = manager._outbound_requests
        request = sv.BridgeOutboundRequest(
            request_id=f"bridge-send-{sv.uuid.uuid4().hex}",
            runtime_id=runtime_id,
            durable_session_id=durable_session_id,
            text=str(text),
            created_ts=sv.time.time(),
        )
        requests_by_runtime.setdefault(runtime_id, []).append(request)
    queue_wakeup = getattr(manager, "_queue_wakeup", None)
    if isinstance(queue_wakeup, threading.Event):
        queue_wakeup.set()
    return request


def fail_outbound_request(manager: Any, request: Any, error: str) -> None:
    sv = _runtime(manager)
    event = {
        "type": "pi_event",
        "summary": "Bridge send failed",
        "text": f"Bridge could not deliver the queued prompt: {error}\n\nOriginal prompt:\n{request.text}",
        "is_error": True,
        "request_id": request.request_id,
        "request_state": "failed",
        "pending_text": request.text,
        "ts": sv.time.time(),
    }
    manager._append_bridge_event(request.durable_session_id, event)


def mark_outbound_request_buffered_for_compaction(manager: Any, request: Any) -> None:
    sv = _runtime(manager)
    if request.state == "buffered":
        return
    request.state = "buffered"
    event = {
        "type": "pi_event",
        "summary": "Bridge buffered prompt during compaction",
        "text": "Waiting for Pi compaction to finish before delivering this prompt.\n\nQueued prompt:\n"
        f"{request.text}",
        "request_id": request.request_id,
        "request_state": "buffered",
        "pending_text": request.text,
        "ts": sv.time.time(),
    }
    manager._append_bridge_event(request.durable_session_id, event)


def maybe_drain_outbound_request(manager: Any, runtime_id: str) -> bool:
    sv = _runtime(manager)
    with manager._lock:
        requests_by_runtime = getattr(manager, "_outbound_requests", None)
        session = manager._sessions.get(runtime_id)
        queue = (
            requests_by_runtime.get(runtime_id)
            if isinstance(requests_by_runtime, dict)
            else None
        )
        request = queue[0] if session is not None and isinstance(queue, list) and queue else None
    if session is None or request is None:
        return False
    if request.runtime_id != runtime_id:
        with manager._lock:
            queue2 = manager._outbound_requests.get(runtime_id)
            if isinstance(queue2, list) and queue2 and queue2[0] is request:
                queue2.pop(0)
        manager._fail_outbound_request(request, "stale runtime")
        return True
    state, transport_error = manager._probe_bridge_transport(runtime_id)
    if state == "dead":
        with manager._lock:
            queue2 = manager._outbound_requests.get(runtime_id)
            if isinstance(queue2, list) and queue2 and queue2[0] is request:
                queue2.pop(0)
                if not queue2:
                    manager._outbound_requests.pop(runtime_id, None)
        manager._fail_outbound_request(request, transport_error or "broker exited")
        return True
    try:
        st = manager.get_state(runtime_id)
    except Exception as exc:
        request.attempts += 1
        request.last_attempt_ts = sv.time.time()
        request.last_error = str(exc).strip() or type(exc).__name__
        if request.attempts >= sv.BRIDGE_OUTBOUND_FAILURE_MAX_ATTEMPTS or (
            sv.time.time() - request.created_ts
        ) >= sv.BRIDGE_OUTBOUND_FAILURE_MAX_AGE_SECONDS:
            with manager._lock:
                queue2 = manager._outbound_requests.get(runtime_id)
                if isinstance(queue2, list) and queue2 and queue2[0] is request:
                    queue2.pop(0)
                    if not queue2:
                        manager._outbound_requests.pop(runtime_id, None)
            manager._fail_outbound_request(request, request.last_error)
            return True
        return False
    if not isinstance(st, dict):
        return False
    if bool(st.get("isCompacting")):
        manager._mark_outbound_request_buffered_for_compaction(request)
        return False
    if bool(st.get("busy")) or int(st.get("queue_len") or 0) > 0:
        return False
    request.state = "sending"
    request.attempts += 1
    request.last_attempt_ts = sv.time.time()
    try:
        resp = manager._sock_call(
            session.sock_path,
            {"cmd": "send", "text": request.text},
            timeout_s=1.0,
        )
    except Exception as exc:
        request.last_error = str(exc).strip() or type(exc).__name__
        state2, transport_error2 = manager._probe_bridge_transport(
            runtime_id, force_rpc=True
        )
        if state2 == "dead" or request.attempts >= sv.BRIDGE_OUTBOUND_FAILURE_MAX_ATTEMPTS or (
            sv.time.time() - request.created_ts
        ) >= sv.BRIDGE_OUTBOUND_FAILURE_MAX_AGE_SECONDS:
            with manager._lock:
                queue2 = manager._outbound_requests.get(runtime_id)
                if isinstance(queue2, list) and queue2 and queue2[0] is request:
                    queue2.pop(0)
                    if not queue2:
                        manager._outbound_requests.pop(runtime_id, None)
            manager._fail_outbound_request(request, transport_error2 or request.last_error)
            return True
        return False
    error = resp.get("error") if isinstance(resp, dict) else None
    if isinstance(error, str) and error:
        with manager._lock:
            queue2 = manager._outbound_requests.get(runtime_id)
            if isinstance(queue2, list) and queue2 and queue2[0] is request:
                queue2.pop(0)
                if not queue2:
                    manager._outbound_requests.pop(runtime_id, None)
        manager._fail_outbound_request(request, error)
        return True
    with manager._lock:
        queue2 = manager._outbound_requests.get(runtime_id)
        if isinstance(queue2, list) and queue2 and queue2[0] is request:
            queue2.pop(0)
            if not queue2:
                manager._outbound_requests.pop(runtime_id, None)
        session2 = manager._sessions.get(runtime_id)
        if session2 is not None:
            if isinstance(resp, dict) and isinstance(resp.get("busy"), bool):
                session2.busy = sv._state_busy_value(resp)
            if isinstance(resp, dict):
                queue_len_raw = resp.get("queue_len")
                if type(queue_len_raw) is int and int(queue_len_raw) >= 0:
                    session2.queue_len = sv._state_queue_len_value(resp)
            session2.pi_idle_activity_ts = None
            if session2.backend == "pi":
                activity_ts = sv._touch_session_file(session2.session_path)
                session2.pi_busy_activity_floor = activity_ts if session2.busy else None
            else:
                session2.pi_busy_activity_floor = None
    return True


def session_display_name(manager: Any, session_id: str) -> str:
    sv = _runtime(manager)
    runtime_id = manager._runtime_session_id_for_identifier(session_id)
    if runtime_id is None:
        return "Session"
    with manager._lock:
        session = manager._sessions.get(runtime_id)
        if not session:
            return "Session"
        ref = manager._page_state_ref_for_session(session)
        alias = manager._aliases.get(ref) if ref is not None else None
        return sv._session_row_display_name(
            {
                "session_id": manager._durable_session_id_for_session(session),
                "alias": alias,
                "title": session.title or "",
                "first_user_message": session.first_user_message or "",
            }
        )


def observe_rollout_delta(
    manager: Any, session_id: str, *, objs: list[dict[str, Any]], new_off: int
) -> None:
    sv = _runtime(manager)
    voice_push = getattr(manager, "_voice_push", None)
    if voice_push is None:
        with manager._lock:
            session = manager._sessions.get(session_id)
            if session is not None:
                session.delivery_log_off = max(int(session.delivery_log_off), int(new_off))
        return
    with manager._lock:
        session0 = manager._sessions.get(session_id)
        resume_muted = bool(session0 and session0.resume_session_id)
    messages = sv._extract_delivery_messages(objs)
    if (not messages) or resume_muted:
        with manager._lock:
            session = manager._sessions.get(session_id)
            if session is not None:
                session.delivery_log_off = max(int(session.delivery_log_off), int(new_off))
        return
    session_name = manager._session_display_name(session_id)
    voice_push.observe_messages(
        session_id=session_id,
        session_display_name=session_name,
        messages=messages,
    )
    with manager._lock:
        session = manager._sessions.get(session_id)
        if session is not None:
            session.delivery_log_off = max(int(session.delivery_log_off), int(new_off))


def voice_push_scan_loop(manager: Any) -> None:
    sv = _runtime(manager)
    while not manager._stop.is_set():
        try:
            manager._voice_push_scan_sweep()
        except Exception as exc:
            sys.stderr.write(
                f"error: voice-push scan failed: {type(exc).__name__}: {exc}\n"
            )
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
        manager._stop.wait(sv.VOICE_PUSH_SWEEP_SECONDS)


def voice_push_scan_sweep(manager: Any) -> None:
    sv = _runtime(manager)
    manager._discover_existing_if_stale()
    manager._prune_dead_sessions()
    with manager._lock:
        session_ids = list(manager._sessions.keys())
    for sid in session_ids:
        try:
            manager.refresh_session_meta(sid)
        except Exception:
            continue
        with manager._lock:
            session = manager._sessions.get(sid)
            if session is None:
                continue
            log_path = session.log_path
            delivery_off = int(session.delivery_log_off)
        if log_path is None or (not log_path.exists()):
            continue
        try:
            size = int(log_path.stat().st_size)
        except FileNotFoundError:
            continue
        off = 0 if size < delivery_off else int(delivery_off)
        loops = 0
        while off < size and loops < 16:
            objs, new_off = sv._read_jsonl_from_offset(log_path, off, max_bytes=256 * 1024)
            if new_off <= off:
                break
            manager._observe_rollout_delta(sid, objs=objs, new_off=new_off)
            off = new_off
            loops += 1


def harness_loop(manager: Any) -> None:
    sv = _runtime(manager)
    while not manager._stop.is_set():
        try:
            manager._harness_sweep()
        except Exception as exc:
            sys.stderr.write(
                f"error: harness sweep failed: {type(exc).__name__}: {exc}\n"
            )
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
        manager._stop.wait(sv.HARNESS_SWEEP_SECONDS)


def harness_sweep(manager: Any) -> None:
    sv = _runtime(manager)
    now = sv.time.time()
    manager._discover_existing_if_stale()
    manager._prune_dead_sessions()
    with manager._lock:
        items: list[tuple[str, Any, dict[str, Any], float]] = []
        for sid, session in manager._sessions.items():
            cfg0 = manager._harness.get(sid)
            cfg = dict(cfg0) if isinstance(cfg0, dict) else {}
            last_inj = float(manager._harness_last_injected.get(sid, 0.0))
            items.append((sid, session, cfg, last_inj))
    for sid, session, cfg, last_inj in items:
        if not bool(cfg.get("enabled")):
            continue
        try:
            cooldown_minutes = sv._clean_harness_cooldown_minutes(
                cfg.get("cooldown_minutes")
            )
            cooldown_seconds = float(cooldown_minutes * 60)
            remaining_injections = sv._clean_harness_remaining_injections(
                cfg.get("remaining_injections"), allow_zero=True
            )
            if remaining_injections <= 0:
                with manager._lock:
                    cur0 = manager._harness.get(sid)
                    cur = dict(cur0) if isinstance(cur0, dict) else {}
                    cur["enabled"] = False
                    cur["remaining_injections"] = 0
                    manager._harness[sid] = cur
                    manager._harness_last_injected.pop(sid, None)
                manager._save_harness()
                continue
            request = cfg.get("request")
            if not isinstance(request, str):
                request = ""
            prompt = sv._render_harness_prompt(request)
            log_path = session.log_path
            if log_path is None or (not log_path.exists()):
                continue
            scope_key = (
                f"thread:{session.thread_id}"
                if session.thread_id
                else f"log:{str(log_path)}"
            )
            with manager._lock:
                scope_last = float(
                    manager._harness_last_injected_scope.get(scope_key, 0.0)
                )
            if (last_inj and (now - last_inj) < cooldown_seconds) or (
                scope_last and (now - scope_last) < cooldown_seconds
            ):
                continue
            st = manager.get_state(sid)
            if not isinstance(st, dict):
                raise ValueError("invalid broker state response")
            if "busy" not in st or "queue_len" not in st:
                raise ValueError("invalid broker state response")
            busy = sv._state_busy_value(st)
            ql = sv._state_queue_len_value(st)
            if busy or ql > 0 or manager._queue_len(sid) > 0:
                continue
            last = sv._last_chat_role_ts_from_tail(
                log_path, max_scan_bytes=sv.HARNESS_MAX_SCAN_BYTES
            )
            if not last:
                continue
            role, ts = last
            if role != "assistant":
                continue
            if (now - float(ts)) < cooldown_seconds:
                continue
            with manager._lock:
                scope_last = float(
                    manager._harness_last_injected_scope.get(scope_key, 0.0)
                )
            if scope_last and (now - scope_last) < cooldown_seconds:
                continue
            manager.send(sid, prompt)
            with manager._lock:
                manager._harness_last_injected[sid] = now
                manager._harness_last_injected_scope[scope_key] = now
                cur0 = manager._harness.get(sid)
                cur = dict(cur0) if isinstance(cur0, dict) else {}
                next_remaining = max(0, remaining_injections - 1)
                cur["remaining_injections"] = next_remaining
                if next_remaining <= 0:
                    cur["enabled"] = False
                    manager._harness_last_injected.pop(sid, None)
                manager._harness[sid] = cur
            manager._save_harness()
        except Exception as exc:
            if isinstance(exc, TimeoutError):
                continue
            sys.stderr.write(
                f"error: harness session {sid} skipped: {type(exc).__name__}: {exc}\n"
            )
            sys.stderr.flush()


def queue_loop(manager: Any) -> None:
    sv = _runtime(manager)
    while not manager._stop.is_set():
        try:
            manager._queue_sweep()
        except Exception:
            sys.stderr.write("error: queue sweep crashed; continuing\n")
            sys.stderr.flush()
        queue_wakeup = getattr(manager, "_queue_wakeup", None)
        if isinstance(queue_wakeup, threading.Event):
            queue_wakeup.wait(sv.QUEUE_SWEEP_SECONDS)
            queue_wakeup.clear()
        else:
            manager._stop.wait(sv.QUEUE_SWEEP_SECONDS)


def maybe_drain_session_queue(
    manager: Any, session_id: str, *, now_ts: float | None = None
) -> bool:
    sv = _runtime(manager)
    if now_ts is None:
        now_ts = sv.time.time()
    with manager._lock:
        session0 = manager._sessions.get(session_id)
        if not session0:
            return False
        queue = manager._queues.get(session_id)
        if not isinstance(queue, list) or not queue:
            session0.queue_idle_since = None
            return False
        text = queue[0]
        log_path = session0.log_path
    try:
        st = manager.get_state(session_id)
    except Exception:
        with manager._lock:
            session0 = manager._sessions.get(session_id)
            if session0:
                session0.queue_idle_since = None
        return False
    if not isinstance(st, dict) or "busy" not in st or "queue_len" not in st:
        with manager._lock:
            session0 = manager._sessions.get(session_id)
            if session0:
                session0.queue_idle_since = None
        return False
    queue_len_raw = st.get("queue_len")
    if not isinstance(queue_len_raw, int):
        with manager._lock:
            session0 = manager._sessions.get(session_id)
            if session0:
                session0.queue_idle_since = None
        return False
    if sv._state_busy_value(st) or int(queue_len_raw) > 0:
        with manager._lock:
            session0 = manager._sessions.get(session_id)
            if session0:
                session0.queue_idle_since = None
        return False
    try:
        if isinstance(log_path, Path) and log_path.exists() and (not manager.idle_from_log(session_id)):
            with manager._lock:
                session0 = manager._sessions.get(session_id)
                if session0:
                    session0.queue_idle_since = None
            return False
    except Exception:
        with manager._lock:
            session0 = manager._sessions.get(session_id)
            if session0:
                session0.queue_idle_since = None
        return False
    with manager._lock:
        session0 = manager._sessions.get(session_id)
        if not session0:
            return False
        idle_since = session0.queue_idle_since
        if idle_since is None:
            session0.queue_idle_since = float(now_ts)
            return False
        if (float(now_ts) - idle_since) < sv.QUEUE_IDLE_GRACE_SECONDS:
            return False
    try:
        manager.send(session_id, text)
    except Exception:
        with manager._lock:
            session0 = manager._sessions.get(session_id)
            if session0:
                session0.queue_idle_since = None
        return False
    ref = manager._page_state_ref_for_session_id(session_id)
    with manager._lock:
        queue = manager._queues.get(session_id)
        if not isinstance(queue, list) and ref is not None:
            queue = manager._queues.get(ref)
        session0 = manager._sessions.get(session_id)
        if session0:
            session0.queue_idle_since = None
        if isinstance(queue, list) and queue and queue[0] == text:
            queue.pop(0)
            if not queue:
                manager._queues.pop(session_id, None)
                if ref is not None:
                    manager._queues.pop(ref, None)
    manager._save_queues()
    return True


def queue_sweep(manager: Any) -> None:
    manager._discover_existing_if_stale()
    manager._prune_dead_sessions()
    with manager._lock:
        active_runtime_ids: list[str] = []
        active_outbound_runtime_ids: list[str] = []
        outbound_requests = getattr(manager, "_outbound_requests", None)
        for runtime_id, session in manager._sessions.items():
            ref = manager._page_state_ref_for_session(session)
            queue = manager._queues.get(runtime_id)
            if (not isinstance(queue, list)) and ref is not None:
                queue = manager._queues.get(ref)
            if isinstance(queue, list) and queue:
                active_runtime_ids.append(runtime_id)
            outbound = (
                outbound_requests.get(runtime_id)
                if isinstance(outbound_requests, dict)
                else None
            )
            if isinstance(outbound, list) and outbound:
                active_outbound_runtime_ids.append(runtime_id)
    for sid in active_outbound_runtime_ids:
        try:
            if manager._maybe_drain_outbound_request(sid):
                break
        except Exception:
            sys.stderr.write(
                f"error: outbound sweep failed for session {sid}; skipping\n"
            )
            sys.stderr.flush()
    for sid in active_runtime_ids:
        try:
            if manager._maybe_drain_session_queue(sid):
                break
        except Exception:
            sys.stderr.write(f"error: queue sweep failed for session {sid}; skipping\n")
            sys.stderr.flush()


def update_meta_counters(manager: Any) -> None:
    sv = _runtime(manager)
    with manager._lock:
        items = list(manager._sessions.items())
    for sid, session in items:
        log_path = session.log_path
        if (log_path is None or (not log_path.exists())) and session.backend == "pi":
            log_path = session.session_path
        if log_path is None or (not log_path.exists()):
            continue
        size = int(log_path.stat().st_size)
        off = int(session.meta_log_off)
        reset_last_chat = False
        if size < off:
            off = 0
            reset_last_chat = True
        total_th = 0
        total_tools = 0
        total_sys = 0
        latest_chat_ts: float | None = None
        latest_token: dict[str, Any] | None = None
        loops = 0
        while off < size and loops < 16:
            objs, new_off = sv._read_jsonl_from_offset(log_path, off, max_bytes=256 * 1024)
            if new_off <= off:
                break
            d_th, d_tools, d_sys, chunk_chat_ts, token_update, _chat_events = sv._analyze_log_chunk(objs)
            total_th += d_th
            total_tools += d_tools
            total_sys += d_sys
            if chunk_chat_ts is not None:
                latest_chat_ts = (
                    chunk_chat_ts
                    if latest_chat_ts is None
                    else max(latest_chat_ts, chunk_chat_ts)
                )
            if token_update is not None:
                latest_token = token_update
            off = new_off
            loops += 1
        if latest_token is None and session.token is None:
            latest_token = sv._rollout_log._find_latest_token_update(log_path)
        with manager._lock:
            session2 = manager._sessions.get(sid)
            if not session2:
                continue
            if reset_last_chat:
                session2.last_chat_ts = None
                session2.last_chat_history_scanned = False
            if latest_chat_ts is not None:
                session2.last_chat_ts = (
                    latest_chat_ts
                    if session2.last_chat_ts is None
                    else max(session2.last_chat_ts, latest_chat_ts)
                )
            if latest_token is not None:
                session2.token = latest_token
            if session2.busy:
                session2.meta_thinking += total_th
                session2.meta_tools += total_tools
                session2.meta_system += total_sys
            else:
                session2.meta_thinking = 0
                session2.meta_tools = 0
                session2.meta_system = 0
            session2.meta_log_off = off if off >= 0 else session2.meta_log_off
