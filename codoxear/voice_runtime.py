from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, MutableMapping

from .session_model import Session


@dataclass(frozen=True)
class VoiceRuntimeCoordinator:
    lock: Any
    sessions: Callable[[], MutableMapping[str, Session]]
    aliases: Callable[[], MutableMapping[str, Any]]
    voice_push: Callable[[], Any]
    discover_existing_if_stale: Callable[[], None]
    prune_dead_sessions: Callable[[], None]
    refresh_session_meta: Callable[[str], None]
    read_jsonl_from_offset: Callable[..., tuple[list[dict[str, Any]], int]]
    extract_delivery_messages: Callable[..., list[Any]]
    cc_pending_tool_ids_before: Callable[[Path, int], set[str]]

    def attach_notification_texts(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        voice_push = self.voice_push()
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

    def session_display_name(self, session_id: str) -> str:
        with self.lock:
            session = self.sessions().get(session_id)
            if not session:
                return "Session"
            alias = self.aliases().get(session_id)
            if isinstance(alias, str) and alias.strip():
                return alias.strip()
            cwd_name = Path(str(session.cwd)).name.strip()
            return cwd_name or "Session"

    def mark_delivery_offset(self, session_id: str, new_off: int) -> None:
        with self.lock:
            session = self.sessions().get(session_id)
            if session is not None:
                session.delivery_log_off = max(int(session.delivery_log_off), int(new_off))

    def observe_rollout_delta(self, session_id: str, *, log_path: Path | None = None, old_off: int = 0, objs: list[dict[str, Any]], new_off: int) -> None:
        voice_push = self.voice_push()
        if voice_push is None:
            self.mark_delivery_offset(session_id, new_off)
            return
        with self.lock:
            session = self.sessions().get(session_id)
            resume_muted = bool(session and session.resume_session_id)
        initial_cc_pending = self.cc_pending_tool_ids_before(log_path, old_off) if log_path is not None and old_off > 0 else set()
        messages = self.extract_delivery_messages(objs, initial_cc_pending_tool_ids=initial_cc_pending)
        if (not messages) or resume_muted:
            self.mark_delivery_offset(session_id, new_off)
            return
        session_name = self.session_display_name(session_id)
        voice_push.observe_messages(session_id=session_id, session_display_name=session_name, messages=messages)
        self.mark_delivery_offset(session_id, new_off)

    def scan_sweep(self) -> None:
        self.discover_existing_if_stale()
        self.prune_dead_sessions()
        with self.lock:
            session_ids = list(self.sessions().keys())
        for sid in session_ids:
            try:
                self.refresh_session_meta(sid)
            except Exception:
                continue
            with self.lock:
                session = self.sessions().get(sid)
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
                objs, new_off = self.read_jsonl_from_offset(log_path, off, max_bytes=256 * 1024)
                if new_off <= off:
                    break
                self.observe_rollout_delta(sid, log_path=log_path, old_off=off, objs=objs, new_off=new_off)
                off = new_off
                loops += 1
