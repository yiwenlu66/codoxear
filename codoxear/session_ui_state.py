from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, MutableMapping

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
        alias = self.clean_alias(name)
        with self.lock:
            if session_id not in self.sessions():
                raise KeyError("unknown session")
            if alias:
                self.aliases()[session_id] = alias
            else:
                self.aliases().pop(session_id, None)
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
            self.aliases().pop(session_id, None)
        self.save_aliases()

    def sidebar_meta_get(self, session_id: str) -> dict[str, Any]:
        with self.lock:
            if session_id not in self.sessions():
                raise KeyError("unknown session")
            entry = self.sidebar_meta().get(session_id)
        if not isinstance(entry, dict):
            return {"priority_offset": 0.0, "snooze_until": None, "dependency_session_id": None}
        return {
            "priority_offset": self.clean_priority_offset(entry.get("priority_offset")),
            "snooze_until": self.clean_snooze_until(entry.get("snooze_until")),
            "dependency_session_id": self.clean_dependency_session_id(entry.get("dependency_session_id")),
        }

    def _validated_sidebar_entry(self, session_id: str, *, priority_offset: Any, snooze_until: Any, dependency_session_id: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        offset = self.clean_priority_offset(priority_offset)
        snooze_until_clean = self.clean_snooze_until(snooze_until)
        dependency_clean = self.clean_dependency_session_id(dependency_session_id)
        if dependency_clean == session_id:
            raise ValueError("session cannot depend on itself")
        if dependency_clean is not None and dependency_clean not in self.sessions():
            raise ValueError("dependency session not found")
        entry: dict[str, Any] = {"priority_offset": offset}
        if snooze_until_clean is not None:
            entry["snooze_until"] = snooze_until_clean
        if dependency_clean is not None:
            entry["dependency_session_id"] = dependency_clean
        public = {"priority_offset": offset, "snooze_until": snooze_until_clean, "dependency_session_id": dependency_clean}
        return entry, public

    def sidebar_meta_set(self, session_id: str, *, priority_offset: Any, snooze_until: Any, dependency_session_id: Any) -> dict[str, Any]:
        with self.lock:
            if session_id not in self.sessions():
                raise KeyError("unknown session")
            entry, public = self._validated_sidebar_entry(
                session_id,
                priority_offset=priority_offset,
                snooze_until=snooze_until,
                dependency_session_id=dependency_session_id,
            )
            meta_map = self.sidebar_meta()
            if not isinstance(meta_map, dict):
                self.set_sidebar_meta({})
                meta_map = self.sidebar_meta()
            meta_map[session_id] = entry
        self.save_sidebar_meta()
        return public

    def edit_session(self, session_id: str, *, name: str, priority_offset: Any, snooze_until: Any, dependency_session_id: Any) -> tuple[str, dict[str, Any]]:
        alias = self.clean_alias(name)
        with self.lock:
            if session_id not in self.sessions():
                raise KeyError("unknown session")
            entry, public = self._validated_sidebar_entry(
                session_id,
                priority_offset=priority_offset,
                snooze_until=snooze_until,
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
