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
