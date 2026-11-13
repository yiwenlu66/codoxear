from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, MutableMapping

from .session_cleaners import clean_snoozed_indefinitely, sidebar_snoozed
from .session_listing import listing_priority, reorder_target_offset
from .session_model import Session


@dataclass(frozen=True)
class SessionUiStateCoordinator:
    lock: Any
    sessions: Callable[[], MutableMapping[str, Session]]
    aliases: Callable[[], MutableMapping[str, str]]
    set_aliases: Callable[[dict[str, str]], None]
    sidebar_meta: Callable[[], MutableMapping[str, dict[str, Any]]]
    set_sidebar_meta: Callable[[dict[str, dict[str, Any]]], None]
    hidden_sessions: Callable[[], set[str]]
    set_hidden_sessions: Callable[[set[str]], None]
    save_aliases: Callable[[], None]
    save_sidebar_meta: Callable[[], None]
    save_hidden_sessions: Callable[[], None]
    clean_alias: Callable[[Any], str]
    clean_priority_offset: Callable[[Any], float]
    clean_snooze_until: Callable[[Any], float | None]
    clean_dependency_session_id: Callable[[Any], str | None]

    def hide_session(self, session_id: str) -> None:
        with self.lock:
            hidden = self.hidden_sessions()
            if not isinstance(hidden, set):
                self.set_hidden_sessions(set())
                hidden = self.hidden_sessions()
            hidden.add(session_id)
        self.save_hidden_sessions()

    def unhide_session(self, session_id: str) -> None:
        changed = False
        with self.lock:
            hidden = self.hidden_sessions()
            if isinstance(hidden, set) and session_id in hidden:
                hidden.remove(session_id)
                changed = True
        if changed:
            self.save_hidden_sessions()

    def alias_set(self, session_id: str, name: str) -> str:
        """Set (or explicitly clear) one name.

        A clear keeps the key with an empty value so the difference between "the
        user removed this name" and "this key holds nothing known" survives on
        disk; the name projector may only erase a backend title for the former.
        """
        alias = self.clean_alias(name)
        with self.lock:
            if session_id not in self.sessions():
                raise KeyError("unknown session")
            self.aliases()[session_id] = alias if alias else ""
        self.save_aliases()
        return alias

    def alias_get(self, session_id: str) -> str:
        with self.lock:
            alias = self.aliases().get(session_id)
        return alias if isinstance(alias, str) else ""

    def alias_clear(self, session_id: str) -> None:
        with self.lock:
            if session_id not in self.aliases():
                return
            self.aliases()[session_id] = ""
        self.save_aliases()

    def alias_cleared(self, session_id: str) -> bool:
        """True only when the user explicitly removed this session's name."""
        with self.lock:
            return self.aliases().get(session_id) == ""

    def sidebar_meta_get(self, session_id: str) -> dict[str, Any]:
        with self.lock:
            if session_id not in self.sessions():
                raise KeyError("unknown session")
            entry = self.sidebar_meta().get(session_id)
        if not isinstance(entry, dict):
            return {"priority_offset": 0.0, "snooze_until": None, "snoozed_indefinitely": False, "dependency_session_id": None}
        return {
            "priority_offset": self.clean_priority_offset(entry.get("priority_offset")),
            "snooze_until": self.clean_snooze_until(entry.get("snooze_until")),
            "snoozed_indefinitely": clean_snoozed_indefinitely(entry.get("snoozed_indefinitely", False)),
            "dependency_session_id": self.clean_dependency_session_id(entry.get("dependency_session_id")),
        }

    def _validated_sidebar_entry(self, session_id: str, *, priority_offset: Any, snooze_until: Any, dependency_session_id: Any, snoozed_indefinitely: Any = False) -> tuple[dict[str, Any], dict[str, Any]]:
        offset = self.clean_priority_offset(priority_offset)
        snooze_until_clean = self.clean_snooze_until(snooze_until)
        indefinite = clean_snoozed_indefinitely(snoozed_indefinitely)
        if indefinite and snooze_until_clean is not None:
            raise ValueError("indefinite snooze cannot have snooze_until")
        dependency_clean = self.clean_dependency_session_id(dependency_session_id)
        if dependency_clean == session_id:
            raise ValueError("session cannot depend on itself")
        if dependency_clean is not None and dependency_clean not in self.sessions():
            raise ValueError("dependency session not found")
        entry: dict[str, Any] = {"priority_offset": offset}
        if indefinite:
            entry["snoozed_indefinitely"] = True
        elif snooze_until_clean is not None:
            entry["snooze_until"] = snooze_until_clean
        if dependency_clean is not None:
            entry["dependency_session_id"] = dependency_clean
        public = {"priority_offset": offset, "snooze_until": snooze_until_clean, "snoozed_indefinitely": indefinite, "dependency_session_id": dependency_clean}
        return entry, public

    def sidebar_meta_set(self, session_id: str, *, priority_offset: Any, snooze_until: Any, dependency_session_id: Any, snoozed_indefinitely: Any = False) -> dict[str, Any]:
        with self.lock:
            if session_id not in self.sessions():
                raise KeyError("unknown session")
            entry, public = self._validated_sidebar_entry(
                session_id,
                priority_offset=priority_offset,
                snooze_until=snooze_until,
                snoozed_indefinitely=snoozed_indefinitely,
                dependency_session_id=dependency_session_id,
            )
            meta_map = self.sidebar_meta()
            if not isinstance(meta_map, dict):
                self.set_sidebar_meta({})
                meta_map = self.sidebar_meta()
            meta_map[session_id] = entry
        self.save_sidebar_meta()
        return public

    def edit_session(self, session_id: str, *, name: str, priority_offset: Any, snooze_until: Any, dependency_session_id: Any, snoozed_indefinitely: Any = False) -> tuple[str, dict[str, Any]]:
        alias = self.clean_alias(name)
        with self.lock:
            if session_id not in self.sessions():
                raise KeyError("unknown session")
            entry, public = self._validated_sidebar_entry(
                session_id,
                priority_offset=priority_offset,
                snooze_until=snooze_until,
                snoozed_indefinitely=snoozed_indefinitely,
                dependency_session_id=dependency_session_id,
            )
            aliases = self.aliases()
            if not isinstance(aliases, dict):
                self.set_aliases({})
                aliases = self.aliases()
            if alias:
                aliases[session_id] = alias
            else:
                aliases.pop(session_id, None)
            meta_map = self.sidebar_meta()
            if not isinstance(meta_map, dict):
                self.set_sidebar_meta({})
                meta_map = self.sidebar_meta()
            meta_map[session_id] = entry
        self.save_aliases()
        self.save_sidebar_meta()
        return alias, public

    def snooze_session(self, session_id: str, *, indefinite: bool) -> dict[str, Any]:
        if not isinstance(indefinite, bool):
            raise ValueError("indefinite must be a boolean")
        with self.lock:
            if session_id not in self.sessions():
                raise KeyError("unknown session")
            meta_map = self.sidebar_meta()
            entry = dict(meta_map.get(session_id) or {})
            entry.pop("snooze_until", None)
            if indefinite:
                entry["snoozed_indefinitely"] = True
            else:
                entry.pop("snoozed_indefinitely", None)
            meta_map[session_id] = entry
        self.save_sidebar_meta()
        return {"snooze_until": None, "snoozed_indefinitely": indefinite}

    def reorder_session(
        self, session_id: str, *, above: str | None, below: str | None,
        now_ts: float, half_life_seconds: float, bucket_seconds: float,
    ) -> dict[str, Any]:
        with self.lock:
            sessions = self.sessions()
            if session_id not in sessions:
                raise KeyError("unknown session")
            if above is None and below is None:
                raise ValueError("above or below required")
            for neighbour in (above, below):
                if neighbour is not None and (not isinstance(neighbour, str) or neighbour not in sessions):
                    raise ValueError("unknown neighbour")
            meta_map = self.sidebar_meta()

            def priority(sid: str):
                session = sessions[sid]
                meta = meta_map.get(sid) or {}
                updated_ts = float(session.last_chat_ts) if isinstance(session.last_chat_ts, (int, float)) else float(session.start_ts)
                dependency = self.clean_dependency_session_id(meta.get("dependency_session_id"))
                return listing_priority(
                    now_ts=now_ts,
                    updated_ts=updated_ts,
                    priority_offset=self.clean_priority_offset(meta.get("priority_offset")),
                    blocked=dependency is not None and dependency != sid and dependency in sessions,
                    snoozed=sidebar_snoozed(
                        snoozed_indefinitely=clean_snoozed_indefinitely(meta.get("snoozed_indefinitely", False)),
                        snooze_until=self.clean_snooze_until(meta.get("snooze_until")),
                        now_ts=now_ts,
                    ),
                    half_life_seconds=half_life_seconds,
                    bucket_seconds=bucket_seconds,
                )

            offset, applied = reorder_target_offset(
                time_priority=priority(session_id).time_priority,
                above_priority=priority(above).final_priority if above is not None else None,
                below_priority=priority(below).final_priority if below is not None else None,
            )
            entry = dict(meta_map.get(session_id) or {})
            entry["priority_offset"] = self.clean_priority_offset(offset)
            meta_map[session_id] = entry
        self.save_sidebar_meta()
        return {"priority_offset": offset, "applied": applied}
