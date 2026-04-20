from __future__ import annotations

from typing import Any

_SERVER = None


def bind_server_runtime(runtime) -> None:
    global _SERVER
    _SERVER = runtime



def _sv():
    if _SERVER is None:
        raise RuntimeError("server runtime not bound")
    return _SERVER


class SidebarStateFacade:
    def __init__(self, manager) -> None:
        self.manager = manager

    def persist_session_ui_state(self) -> None:
        db = getattr(self.manager, "_page_state_db", None)
        if db is None:
            return
        with self.manager._lock:
            aliases_src = dict(self.manager._aliases)
            sidebar_src = dict(self.manager._sidebar_meta)
            hidden_keys = set(self.manager._hidden_sessions)
        aliases = {}
        for key, value in aliases_src.items():
            ref = key if isinstance(key, tuple) and len(key) == 2 else self.manager._page_state_ref_for_session_id(str(key))
            if ref is None or not isinstance(value, str) or not value.strip():
                continue
            aliases[ref] = value
        sidebar_meta = {}
        for key, value in sidebar_src.items():
            ref = key if isinstance(key, tuple) and len(key) == 2 else self.manager._page_state_ref_for_session_id(str(key))
            if ref is None or not isinstance(value, dict):
                continue
            sidebar_meta[ref] = dict(value)
        db.save_session_ui_state(aliases, sidebar_meta, hidden_keys)

    def load_aliases(self) -> None:
        aliases, _sidebar_meta, _hidden_keys = self.manager._page_state_db.load_session_ui_state()
        with self.manager._lock:
            self.manager._aliases = aliases

    def load_sidebar_meta(self) -> None:
        _aliases, sidebar_meta, _hidden_keys = self.manager._page_state_db.load_session_ui_state()
        with self.manager._lock:
            self.manager._sidebar_meta = sidebar_meta

    def load_hidden_sessions(self) -> None:
        _aliases, _sidebar_meta, hidden_keys = self.manager._page_state_db.load_session_ui_state()
        with self.manager._lock:
            self.manager._hidden_sessions = hidden_keys

    def set_created_session_name(
        self,
        *,
        session_id: Any,
        runtime_id: Any = None,
        backend: Any = None,
        name: Any,
    ) -> str:
        sv = _sv()
        alias = sv._clean_alias(name)
        if not alias:
            return ""
        session_id_clean = sv._clean_optional_text(session_id)
        runtime_id_clean = sv._clean_optional_text(runtime_id)
        ref = None
        if session_id_clean is not None:
            ref = self.manager._page_state_ref_for_session_id(session_id_clean)
        if ref is None and runtime_id_clean is not None:
            ref = self.manager._page_state_ref_for_session_id(runtime_id_clean)
        backend_clean = sv.normalize_agent_backend(backend, default="codex") if backend is not None else None
        target = ref
        if target is None and session_id_clean is not None and backend_clean is not None:
            target = (backend_clean, session_id_clean)
        if target is None and runtime_id_clean is not None and backend_clean is not None:
            target = (backend_clean, runtime_id_clean)
        if target is None:
            raise KeyError("unknown session")
        with self.manager._lock:
            self.manager._aliases[target] = alias
            if session_id_clean is not None:
                self.manager._aliases.pop(session_id_clean, None)
            if runtime_id_clean is not None:
                self.manager._aliases.pop(runtime_id_clean, None)
        self.persist_session_ui_state()
        return alias

    def alias_set(self, session_id: str, name: str) -> str:
        sv = _sv()
        alias = sv._clean_alias(name)
        ref = self.manager._page_state_ref_for_session_id(session_id)
        runtime_id = self.manager._runtime_session_id_for_identifier(session_id)
        if ref is None:
            raise KeyError("unknown session")
        with self.manager._lock:
            if alias:
                self.manager._aliases[ref] = alias
            else:
                if runtime_id is not None:
                    self.manager._aliases.pop(runtime_id, None)
                self.manager._aliases.pop(ref, None)
        self.persist_session_ui_state()
        return alias

    def alias_get(self, session_id: str) -> str:
        ref = self.manager._page_state_ref_for_session_id(session_id)
        if ref is None:
            return ""
        with self.manager._lock:
            alias = self.manager._aliases.get(ref)
            if alias is None:
                alias = self.manager._aliases.get(session_id)
        return alias if isinstance(alias, str) else ""

    def alias_clear(self, session_id: str) -> None:
        ref = self.manager._page_state_ref_for_session_id(session_id)
        if ref is None:
            return
        with self.manager._lock:
            if ref not in self.manager._aliases and session_id not in self.manager._aliases:
                return
            self.manager._aliases.pop(ref, None)
            self.manager._aliases.pop(session_id, None)
        self.persist_session_ui_state()

    def sidebar_meta_get(self, session_id: str) -> dict[str, Any]:
        sv = _sv()
        ref = self.manager._page_state_ref_for_session_id(session_id)
        if ref is None:
            raise KeyError("unknown session")
        with self.manager._lock:
            entry = self.manager._sidebar_meta.get(ref)
            if entry is None:
                entry = self.manager._sidebar_meta.get(session_id)
        if not isinstance(entry, dict):
            return {
                "priority_offset": 0.0,
                "snooze_until": None,
                "dependency_session_id": None,
                "focused": False,
            }
        return {
            "priority_offset": sv._clean_priority_offset(entry.get("priority_offset")),
            "snooze_until": sv._clean_snooze_until(entry.get("snooze_until")),
            "dependency_session_id": sv._clean_dependency_session_id(entry.get("dependency_session_id")),
            "focused": bool(entry.get("focused")),
        }

    def sidebar_meta_set(
        self,
        session_id: str,
        *,
        priority_offset: Any,
        snooze_until: Any,
        dependency_session_id: Any,
    ) -> dict[str, Any]:
        sv = _sv()
        offset = sv._clean_priority_offset(priority_offset)
        snooze_until_clean = sv._clean_snooze_until(snooze_until)
        dependency_clean = sv._clean_dependency_session_id(dependency_session_id)
        ref = self.manager._page_state_ref_for_session_id(session_id)
        if ref is None:
            raise KeyError("unknown session")
        dep_ref = None
        if dependency_clean is not None:
            dep_ref = self.manager._page_state_ref_for_session_id(dependency_clean)
            if dep_ref is None:
                raise ValueError("dependency session not found")
            if dep_ref == ref:
                raise ValueError("session cannot depend on itself")
        with self.manager._lock:
            existing = self.manager._sidebar_meta.get(ref)
            entry = {"priority_offset": offset}
            if isinstance(existing, dict) and existing.get("focused"):
                entry["focused"] = True
            if snooze_until_clean is not None:
                entry["snooze_until"] = snooze_until_clean
            if dep_ref is not None:
                entry["dependency_session_id"] = dep_ref[1]
            self.manager._sidebar_meta[ref] = entry
        self.persist_session_ui_state()
        return {
            "priority_offset": offset,
            "snooze_until": snooze_until_clean,
            "dependency_session_id": dep_ref[1] if dep_ref is not None else None,
            "focused": bool(isinstance(existing, dict) and existing.get("focused")),
        }

    def focus_set(self, session_id: str, focused: Any) -> bool:
        sv = _sv()
        focused_clean = sv._clean_optional_bool(focused)
        if focused_clean is None:
            raise ValueError("focused must be a boolean")
        ref = self.manager._page_state_ref_for_session_id(session_id)
        if ref is None:
            raise KeyError("unknown session")
        with self.manager._lock:
            entry = self.manager._sidebar_meta.get(ref)
            next_entry = dict(entry) if isinstance(entry, dict) else {}
            if focused_clean:
                next_entry["focused"] = True
            else:
                next_entry.pop("focused", None)
            next_entry["priority_offset"] = sv._clean_priority_offset(next_entry.get("priority_offset"))
            next_entry["snooze_until"] = sv._clean_snooze_until(next_entry.get("snooze_until"))
            next_entry["dependency_session_id"] = sv._clean_dependency_session_id(next_entry.get("dependency_session_id"))
            if (
                not next_entry.get("focused")
                and not next_entry.get("priority_offset")
                and next_entry.get("snooze_until") is None
                and next_entry.get("dependency_session_id") is None
            ):
                self.manager._sidebar_meta.pop(ref, None)
            else:
                cleaned_entry = {"priority_offset": float(next_entry.get("priority_offset") or 0.0)}
                if next_entry.get("focused"):
                    cleaned_entry["focused"] = True
                if next_entry.get("snooze_until") is not None:
                    cleaned_entry["snooze_until"] = next_entry["snooze_until"]
                if next_entry.get("dependency_session_id") is not None:
                    cleaned_entry["dependency_session_id"] = next_entry["dependency_session_id"]
                self.manager._sidebar_meta[ref] = cleaned_entry
        self.persist_session_ui_state()
        return focused_clean

    def edit_session(
        self,
        session_id: str,
        *,
        name: str,
        priority_offset: Any,
        snooze_until: Any,
        dependency_session_id: Any,
    ) -> tuple[str, dict[str, Any]]:
        sv = _sv()
        alias = sv._clean_alias(name)
        offset = sv._clean_priority_offset(priority_offset)
        snooze_until_clean = sv._clean_snooze_until(snooze_until)
        dependency_clean = sv._clean_dependency_session_id(dependency_session_id)
        ref = self.manager._page_state_ref_for_session_id(session_id)
        runtime_id = self.manager._runtime_session_id_for_identifier(session_id)
        if ref is None:
            raise KeyError("unknown session")
        dep_ref = None
        if dependency_clean is not None:
            dep_ref = self.manager._page_state_ref_for_session_id(dependency_clean)
            if dep_ref is None:
                raise ValueError("dependency session not found")
            if dep_ref == ref:
                raise ValueError("session cannot depend on itself")
        with self.manager._lock:
            if alias:
                self.manager._aliases[ref] = alias
            else:
                if runtime_id is not None:
                    self.manager._aliases.pop(runtime_id, None)
                self.manager._aliases.pop(ref, None)
            existing = self.manager._sidebar_meta.get(ref)
            entry = {"priority_offset": offset}
            if isinstance(existing, dict) and existing.get("focused"):
                entry["focused"] = True
            if snooze_until_clean is not None:
                entry["snooze_until"] = snooze_until_clean
            if dep_ref is not None:
                entry["dependency_session_id"] = dep_ref[1]
            self.manager._sidebar_meta[ref] = entry
        self.persist_session_ui_state()
        return alias, {
            "priority_offset": offset,
            "snooze_until": snooze_until_clean,
            "dependency_session_id": dep_ref[1] if dep_ref is not None else None,
            "focused": bool(isinstance(existing, dict) and existing.get("focused")),
        }

    def hidden_session_keys(
        self,
        session_id: str | None,
        thread_id: str | None,
        resume_session_id: str | None,
        backend: str | None,
    ) -> set[str]:
        sv = _sv()
        keys: set[str] = set()
        session_clean = sv._clean_optional_text(session_id)
        thread_clean = sv._clean_optional_text(thread_id)
        resume_clean = sv._clean_optional_text(resume_session_id)
        backend_clean = sv.normalize_agent_backend(backend, default="codex")
        if session_clean:
            keys.add(session_clean)
            keys.add(f"session:{session_clean}")
        if thread_clean:
            keys.add(f"thread:{backend_clean}:{thread_clean}")
        if resume_clean:
            keys.add(f"resume:{backend_clean}:{resume_clean}")
            keys.add(sv._historical_session_id(backend_clean, resume_clean))
        return keys

    def session_is_hidden(
        self,
        session_id: str | None,
        thread_id: str | None,
        resume_session_id: str | None,
        backend: str | None,
    ) -> bool:
        with self.manager._lock:
            hidden = set(getattr(self.manager, "_hidden_sessions", set()))
        return bool(hidden and hidden.intersection(self.hidden_session_keys(session_id, thread_id, resume_session_id, backend)))

    def hide_session(self, session_id: str) -> None:
        with self.manager._lock:
            self.manager._hidden_sessions.add(session_id)
        self.persist_session_ui_state()

    def hide_session_identity_values(
        self,
        session_id: str | None,
        thread_id: str | None,
        resume_session_id: str | None,
        backend: str | None,
    ) -> None:
        hidden_keys = self.hidden_session_keys(session_id, thread_id, resume_session_id, backend)
        with self.manager._lock:
            self.manager._hidden_sessions.update(hidden_keys)
        self.persist_session_ui_state()

    def hide_session_identity(self, session) -> None:
        self.hide_session_identity_values(
            session.session_id,
            session.thread_id,
            session.resume_session_id,
            session.agent_backend or session.backend,
        )

    def unhide_session(self, session_id: str) -> None:
        changed = False
        with self.manager._lock:
            if session_id in self.manager._hidden_sessions:
                self.manager._hidden_sessions.remove(session_id)
                changed = True
        if changed:
            self.persist_session_ui_state()
