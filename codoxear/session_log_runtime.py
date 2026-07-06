from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, MutableMapping

from .session_model import Session
from .session_runtime import suppress_session_interrupted_idle


@dataclass(frozen=True)
class SessionLogRuntimeCoordinator:
    lock: Any
    sessions: Callable[[], MutableMapping[str, Session]]
    analyze_log_chunk: Callable[[list[dict[str, Any]]], tuple[Any, Any, Any, Any, Any, Any]]
    turn_context_run_settings: Callable[[Any], tuple[str | None, str | None]]
    compute_idle_from_log: Callable[[Path], bool | None]
    read_jsonl_from_offset: Callable[..., tuple[list[dict[str, Any]], int]]
    find_latest_token_update: Callable[[Path], dict[str, Any] | None]

    def update_meta_counters(self) -> None:
        with self.lock:
            items = list(self.sessions().items())
        for sid, session in items:
            log_path = session.log_path
            if log_path is None or (not log_path.exists()):
                continue
            size = int(log_path.stat().st_size)
            offset = int(session.meta_log_off)
            reset_last_chat = False
            if size < offset:
                offset = 0
                reset_last_chat = True

            # Stale interrupted-idle guard: ``interrupted_idle_log_off`` is the
            # log byte offset captured when the broker last confirmed an
            # interrupt. Content at or beyond it arrived after the interrupt,
            # so it is post-interrupt activity that proves the turn resumed and
            # invalidates the stored interrupted-idle override. Advance the read
            # cursor past any pre-baseline bytes so the first chunk processed is
            # unambiguously post-interrupt; this keeps the interrupted turn's
            # own non-final tail (which keeps the override alive) separate from
            # genuine resumed activity. The skipped bytes only affect meta
            # counters, which are reset to zero below whenever the session is
            # not busy (the interrupted-idle case), so nothing is lost.
            interrupted_idle_active = bool(session.interrupted_idle)
            interrupted_idle_baseline = int(session.interrupted_idle_log_off) if interrupted_idle_active else 0
            clear_interrupted_idle = False
            post_baseline = False
            if interrupted_idle_active and 0 < interrupted_idle_baseline <= size:
                if offset < interrupted_idle_baseline:
                    offset = interrupted_idle_baseline
                post_baseline = True

            total_thinking = 0
            total_tools = 0
            total_system = 0
            latest_chat_ts: float | None = None
            latest_token: dict[str, Any] | None = None
            loops = 0
            while offset < size and loops < 16:
                objs, new_offset = self.read_jsonl_from_offset(log_path, offset, max_bytes=256 * 1024)
                if new_offset <= offset:
                    break
                delta_thinking, delta_tools, delta_system, chunk_chat_ts, token_update, chat_events = self.analyze_log_chunk(objs)
                total_thinking += delta_thinking
                total_tools += delta_tools
                total_system += delta_system
                if chunk_chat_ts is not None:
                    latest_chat_ts = chunk_chat_ts if latest_chat_ts is None else max(latest_chat_ts, chunk_chat_ts)
                if token_update is not None:
                    latest_token = token_update
                # Any user/assistant turn activity in a post-baseline chunk
                # proves the turn resumed after the interrupt. Visible
                # conversation rows surface as chat events; reasoning/tool-only
                # rows surface as thinking/tool counter deltas. Together they
                # cover every form of resumed turn activity (a lone
                # token_count after interrupt does not, and must not, clear).
                if post_baseline and (
                    any(isinstance(e, dict) and e.get("role") in ("user", "assistant") for e in chat_events)
                    or delta_thinking > 0
                    or delta_tools > 0
                ):
                    clear_interrupted_idle = True
                offset = new_offset
                loops += 1

            if latest_token is None and session.token is None:
                latest_token = self.find_latest_token_update(log_path)

            with self.lock:
                current = self.sessions().get(sid)
                if not current:
                    continue
                if reset_last_chat:
                    current.last_chat_ts = None
                    current.last_chat_history_scanned = False
                if latest_chat_ts is not None:
                    current.last_chat_ts = latest_chat_ts if current.last_chat_ts is None else max(current.last_chat_ts, latest_chat_ts)
                if latest_token is not None:
                    current.token = latest_token
                if current.busy:
                    current.meta_thinking += total_thinking
                    current.meta_tools += total_tools
                    current.meta_system += total_system
                else:
                    current.meta_thinking = 0
                    current.meta_tools = 0
                    current.meta_system = 0
                if clear_interrupted_idle and current.interrupted_idle:
                    suppress_session_interrupted_idle(current)
                current.meta_log_off = offset if offset >= 0 else current.meta_log_off

    def mark_log_delta(self, session_id: str, *, objs: list[dict[str, Any]], new_off: int) -> None:
        _thinking, _tools, _system, last_ts, _token_update, _chat_events = self.analyze_log_chunk(objs)
        model = None
        reasoning_effort = None
        for obj in reversed(objs):
            if not isinstance(obj, dict) or obj.get("type") != "turn_context":
                continue
            model, reasoning_effort = self.turn_context_run_settings(obj.get("payload"))
            break
        with self.lock:
            session = self.sessions().get(session_id)
            if session:
                if isinstance(last_ts, (int, float)):
                    tsf = float(last_ts)
                    session.last_chat_ts = tsf if session.last_chat_ts is None else max(session.last_chat_ts, tsf)
                if model is not None:
                    session.model = model
                if reasoning_effort is not None:
                    session.reasoning_effort = reasoning_effort
                session.idle_cache_log_off = -1

    def idle_from_log(self, session_id: str) -> bool:
        with self.lock:
            session = self.sessions().get(session_id)
            if not session:
                raise KeyError("unknown session")
            log_path = session.log_path
        if log_path is None:
            raise FileNotFoundError(f"missing rollout log for session {session_id}")
        return self.idle_from_log_path(session_id, log_path)

    def idle_from_log_path(self, session_id: str, log_path: Path) -> bool:
        with self.lock:
            session = self.sessions().get(session_id)
            cache_matches_path = bool(session and session.log_path == log_path)
            cached_off = int(session.idle_cache_log_off) if cache_matches_path and session else -1
            cached_idle = session.idle_cache_value if cache_matches_path and session else None
        if not log_path.exists():
            raise FileNotFoundError(f"missing rollout log for session {session_id}")
        size = int(log_path.stat().st_size)
        if cache_matches_path and (size >= 0) and (cached_off == size) and isinstance(cached_idle, bool):
            return bool(cached_idle)
        idle = self.compute_idle_from_log(log_path)
        with self.lock:
            current = self.sessions().get(session_id)
            if current and current.log_path == log_path:
                current.idle_cache_log_off = size
                current.idle_cache_value = idle
        if idle is None:
            raise RuntimeError("unable to compute idle state from log")
        return bool(idle)
