from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .queue_store import QueueStore
from .unattended import UnattendedStore
from .util import atomic_write_json
from .util import load_json_file


Cleaner1 = Callable[[Any], Any]
CommitUnknownCleaner = Callable[[Any], dict[str, Any] | None]


@dataclass(frozen=True)
class SessionStorePaths:
    aliases: Path
    sidebar_meta: Path
    hidden_sessions: Path
    files: Path
    queues: Path
    pending_attachments: Path
    commit_unknown_sends: Path
    recent_cwds: Path
    unattended: Path


class SessionStore:
    def __init__(
        self,
        *,
        paths: SessionStorePaths,
        file_history_max: int,
        recent_cwd_max: int,
        unattended_default_idle_minutes: int,
        unattended_default_max_injections: int,
        clean_alias: Cleaner1,
        clean_priority_offset: Cleaner1,
        clean_snooze_until: Cleaner1,
        clean_dependency_session_id: Cleaner1,
        clean_recent_cwd: Cleaner1,
        clean_commit_unknown_send_record: CommitUnknownCleaner,
    ) -> None:
        self.paths = paths
        self.file_history_max = int(file_history_max)
        self.recent_cwd_max = int(recent_cwd_max)
        self.clean_alias = clean_alias
        self.clean_priority_offset = clean_priority_offset
        self.clean_snooze_until = clean_snooze_until
        self.clean_dependency_session_id = clean_dependency_session_id
        self.clean_recent_cwd = clean_recent_cwd
        self.clean_commit_unknown_send_record = clean_commit_unknown_send_record
        self.unattended: dict[str, dict[str, Any]] = {}
        self.aliases: dict[str, str] = {}
        self.sidebar_meta: dict[str, dict[str, Any]] = {}
        self.hidden_sessions: set[str] = set()
        self.files: dict[str, list[str]] = {}
        self.queues: dict[str, list[dict[str, Any]]] = {}
        self.pending_attachment_ids: set[str] = set()
        self.commit_unknown_sends: dict[str, dict[str, Any]] = {}
        self.recent_cwds: dict[str, float] = {}
        self.unattended_store = UnattendedStore(
            path=paths.unattended,
            default_idle_minutes=unattended_default_idle_minutes,
            default_max_injections=unattended_default_max_injections,
        )
        self.queue_store = QueueStore(paths.queues)

    def load_unattended(self) -> dict[str, dict[str, Any]]:
        return self.unattended_store.load()

    def save_unattended(self, obj: dict[str, dict[str, Any]]) -> None:
        self.unattended_store.save(dict(obj))

    def load_aliases(self) -> dict[str, str]:
        obj = load_json_file(self.paths.aliases, default=None)
        if obj is None:
            return {}
        if not isinstance(obj, dict):
            raise ValueError("invalid session_aliases.json (expected object)")
        cleaned: dict[str, str] = {}
        for sid, value in obj.items():
            if not isinstance(sid, str) or not sid:
                continue
            if not isinstance(value, str):
                continue
            alias = self.clean_alias(value)
            if isinstance(alias, str) and alias:
                cleaned[sid] = alias
        return cleaned

    def save_aliases(self, obj: dict[str, str]) -> None:
        atomic_write_json(self.paths.aliases, dict(obj))

    def load_sidebar_meta(self) -> dict[str, dict[str, Any]]:
        obj = load_json_file(self.paths.sidebar_meta, default=None)
        if obj is None:
            return {}
        if not isinstance(obj, dict):
            raise ValueError("invalid session_sidebar.json (expected object)")
        cleaned: dict[str, dict[str, Any]] = {}
        for sid, value in obj.items():
            if not isinstance(sid, str) or not sid:
                continue
            if not isinstance(value, dict):
                continue
            offset = self.clean_priority_offset(value.get("priority_offset"))
            snooze_until = self.clean_snooze_until(value.get("snooze_until"))
            dependency_session_id = self.clean_dependency_session_id(value.get("dependency_session_id"))
            entry: dict[str, Any] = {"priority_offset": offset}
            if snooze_until is not None:
                entry["snooze_until"] = snooze_until
            if dependency_session_id is not None:
                entry["dependency_session_id"] = dependency_session_id
            cleaned[sid] = entry
        return cleaned

    def save_sidebar_meta(self, obj: dict[str, dict[str, Any]]) -> None:
        atomic_write_json(self.paths.sidebar_meta, dict(obj))

    def load_hidden_sessions(self) -> set[str]:
        obj = load_json_file(self.paths.hidden_sessions, default=None)
        if obj is None:
            return set()
        if not isinstance(obj, list):
            raise ValueError("invalid hidden_sessions.json (expected list)")
        return {sid.strip() for sid in obj if isinstance(sid, str) and sid.strip()}

    def save_hidden_sessions(self, sessions: set[str]) -> None:
        atomic_write_json(self.paths.hidden_sessions, sorted(sessions), sort_keys=True)

    def load_files(self) -> dict[str, list[str]]:
        obj = load_json_file(self.paths.files, default=None)
        if obj is None:
            return {}
        if not isinstance(obj, dict):
            raise ValueError("invalid session_files.json (expected object)")
        cleaned: dict[str, list[str]] = {}
        for sid, arr in obj.items():
            if not isinstance(sid, str) or not sid:
                continue
            if sid.startswith("cwd:"):
                continue
            key = sid if sid.startswith("sid:") else f"sid:{sid}"
            if not isinstance(arr, list):
                continue
            out: list[str] = []
            for value in arr:
                if not isinstance(value, str):
                    continue
                if value == "" or value in out:
                    continue
                out.append(value)
                if len(out) >= self.file_history_max:
                    break
            if out:
                cleaned[key] = out
        return cleaned

    def save_files(self, obj: dict[str, list[str]]) -> None:
        atomic_write_json(self.paths.files, dict(obj))

    def file_history_for_keys(self, key: str, legacy_keys: list[str]) -> tuple[list[str], bool]:
        cur = self.files.get(key)
        if isinstance(cur, list) and cur:
            return list(cur), False
        for legacy_key in legacy_keys:
            legacy = self.files.get(legacy_key)
            if isinstance(legacy, list) and legacy:
                out = list(legacy)
                if legacy_key != key:
                    self.files[key] = list(legacy)
                    self.files.pop(legacy_key, None)
                    return out, True
                return out, False
        return [], False

    def add_file_history_entry(self, key: str, legacy_keys: list[str], path: str) -> list[str]:
        cur = list(self.files.get(key, []))
        if not cur:
            for legacy_key in legacy_keys:
                legacy = self.files.get(legacy_key)
                if isinstance(legacy, list) and legacy:
                    cur = list(legacy)
                    if legacy_key != key:
                        self.files.pop(legacy_key, None)
                    break
        cur = [item for item in cur if item != path]
        cur.insert(0, path)
        if len(cur) > self.file_history_max:
            cur = cur[: self.file_history_max]
        self.files[key] = cur
        return list(cur)

    def clear_file_history_for_keys(self, key: str, legacy_keys: list[str], *, cwd: str = "") -> bool:
        keys_to_clear = list(legacy_keys)
        cwd_clean = str(cwd or "").strip()
        if cwd_clean:
            # `cwd:` buckets are legacy pre-session-scoping state. Do not migrate
            # them into active sessions because they leak history across sessions
            # with the same cwd, but do discard the matching legacy bucket when
            # the owning session/cwd is deleted.
            keys_to_clear.append(f"cwd:{cwd_clean}")
        dirty = False
        for legacy_key in keys_to_clear:
            if legacy_key in self.files:
                self.files.pop(legacy_key, None)
                dirty = True
        if key in self.files:
            self.files.pop(key, None)
            dirty = True
        return dirty

    def load_queues(self) -> dict[str, list[dict[str, Any]]]:
        return self.queue_store.load()

    def save_queues(self, obj: dict[str, list[dict[str, Any]]]) -> None:
        self.queue_store.save(dict(obj))

    def load_pending_attachments(self) -> set[str]:
        obj = load_json_file(self.paths.pending_attachments, default=None)
        if obj is None:
            return set()
        if not isinstance(obj, list):
            raise ValueError("invalid pending_attachments.json (expected array)")
        return {str(item).strip() for item in obj if isinstance(item, str) and str(item).strip()}

    def save_pending_attachments(self, ids: set[str]) -> None:
        atomic_write_json(self.paths.pending_attachments, sorted(str(item) for item in ids if str(item).strip()))

    def load_commit_unknown_sends(self) -> dict[str, dict[str, Any]]:
        obj = load_json_file(self.paths.commit_unknown_sends, default=None)
        if obj is None:
            return {}
        if not isinstance(obj, dict):
            raise ValueError("invalid commit_unknown_sends.json (expected object)")
        cleaned: dict[str, dict[str, Any]] = {}
        for sid, raw in obj.items():
            if not isinstance(sid, str) or not sid.strip():
                continue
            record = self.clean_commit_unknown_send_record(raw)
            if record is not None:
                cleaned[sid.strip()] = record
        return cleaned

    def save_commit_unknown_sends(self, source: dict[str, dict[str, Any]]) -> None:
        obj = {str(sid): dict(rec) for sid, rec in source.items() if str(sid).strip() and isinstance(rec, dict)}
        atomic_write_json(self.paths.commit_unknown_sends, obj)

    def load_recent_cwds(self) -> dict[str, float]:
        obj = load_json_file(self.paths.recent_cwds, default=None)
        if obj is None:
            return {}
        if not isinstance(obj, dict):
            raise ValueError("invalid recent_cwds.json (expected object)")
        cleaned: dict[str, float] = {}
        for raw_cwd, raw_ts in obj.items():
            cwd = self.clean_recent_cwd(raw_cwd)
            if cwd is None or isinstance(raw_ts, bool):
                continue
            try:
                ts = float(raw_ts)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(ts) or ts <= 0:
                continue
            prev = cleaned.get(cwd)
            if prev is None or ts > prev:
                cleaned[cwd] = ts
        return dict(sorted(cleaned.items(), key=lambda item: (-item[1], item[0]))[: self.recent_cwd_max])

    def save_recent_cwds(self, source: dict[str, float]) -> None:
        items = sorted(source.items(), key=lambda item: (-float(item[1]), item[0]))[: self.recent_cwd_max]
        atomic_write_json(self.paths.recent_cwds, {cwd: ts for cwd, ts in items})
