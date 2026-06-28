from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, MutableMapping

from .session_model import Session


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
                delta_thinking, delta_tools, delta_system, chunk_chat_ts, token_update, _chat_events = self.analyze_log_chunk(objs)
                total_thinking += delta_thinking
                total_tools += delta_tools
                total_system += delta_system
                if chunk_chat_ts is not None:
                    latest_chat_ts = chunk_chat_ts if latest_chat_ts is None else max(latest_chat_ts, chunk_chat_ts)
                if token_update is not None:
                    latest_token = token_update
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
