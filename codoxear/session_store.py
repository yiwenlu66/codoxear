from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .file_upload import remove_session_uploads
from .file_upload import remove_staged_attachment_file
from .file_upload import validate_staged_attachment_file_target
from .queue_store import QueueStore
from .unattended import UnattendedStore
from .util import atomic_write_json
from .util import load_json_file


Cleaner1 = Callable[[Any], Any]
CommitUnknownCleaner = Callable[[Any], dict[str, Any] | None]


@dataclass(frozen=True)
class SidebarSessionState:
    priority_offset: float
    snooze_until: float | None
    dependency_session_id: str | None
    dirty: bool


@dataclass(frozen=True)
class DeletedSessionStateChanges:
    aliases: bool = False
    sidebar_meta: bool = False
    hidden_sessions: bool = False
    unattended: bool = False
    files: bool = False
    queues: bool = False
    pending_attachments: bool = False
    staged_attachments: bool = False
    commit_unknown_sends: bool = False


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
    staged_attachments: Path | None = None
    uploads_root: Path | None = None


def _file_entry_path(entry: Any) -> str:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        path = entry.get("path")
        if isinstance(path, str):
            return path
    return ""


def _file_entry_api_path(entry: Any) -> str:
    if isinstance(entry, dict):
        api_path = entry.get("api_path")
        if isinstance(api_path, str):
            return api_path
    return ""


def _file_entry_identity(entry: Any) -> tuple[str, str]:
    return (_file_entry_path(entry), _file_entry_api_path(entry))


def _make_file_entry(path: str, api_path: str = "") -> Any:
    # Shape-preserving: legacy string entries (no token) stay as plain strings so
    # on-disk state and in-memory state remain byte-identical to the pre-token
    # era; only entries that carry a reversible api_path token use the dict form.
    if api_path:
        return {"path": path, "api_path": api_path}
    return path


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
        self.staged_attachments: dict[str, list[dict[str, Any]]] = {}
        self.commit_unknown_sends: dict[str, dict[str, Any]] = {}
        self.recent_cwds: dict[str, float] = {}
        self.unattended_store = UnattendedStore(
            path=paths.unattended,
            default_idle_minutes=unattended_default_idle_minutes,
            default_max_injections=unattended_default_max_injections,
        )
        self.queue_store = QueueStore(paths.queues)

    def reset_in_memory_state(self) -> None:
        self.unattended = {}
        self.aliases = {}
        self.sidebar_meta = {}
        self.hidden_sessions = set()
        self.files = {}
        self.queues = {}
        self.pending_attachment_ids = set()
        self.staged_attachments = {}
        self.commit_unknown_sends = {}
        self.recent_cwds = {}

    def load_persistent_state(self) -> None:
        self.unattended = self.load_unattended()
        self.aliases = self.load_aliases()
        self.sidebar_meta = self.load_sidebar_meta()
        self.hidden_sessions = self.load_hidden_sessions()
        self.files = self.load_files()
        self.queues = self.load_queues()
        self.staged_attachments = self.load_staged_attachments()
        self.pending_attachment_ids = self.load_pending_attachments() | set(self.staged_attachments.keys())
        self.commit_unknown_sends = self.load_commit_unknown_sends()
        self.recent_cwds = self.load_recent_cwds()

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

    def sidebar_state_for_session(self, session_id: str, *, active_session_ids: set[str], now_ts: float) -> SidebarSessionState:
        meta0 = self.sidebar_meta.get(session_id)
        if not isinstance(meta0, dict):
            meta0 = {}
        priority_offset = self.clean_priority_offset(meta0.get("priority_offset"))
        snooze_until = self.clean_snooze_until(meta0.get("snooze_until"))
        dependency_session_id = self.clean_dependency_session_id(meta0.get("dependency_session_id"))
        dirty = False
        if dependency_session_id == session_id or (dependency_session_id is not None and dependency_session_id not in active_session_ids):
            dependency_session_id = None
            meta0.pop("dependency_session_id", None)
            dirty = True
        if snooze_until is not None and snooze_until <= now_ts:
            snooze_until = None
            meta0.pop("snooze_until", None)
            dirty = True
        return SidebarSessionState(
            priority_offset=priority_offset,
            snooze_until=snooze_until,
            dependency_session_id=dependency_session_id,
            dirty=dirty,
        )

    def load_hidden_sessions(self) -> set[str]:
        obj = load_json_file(self.paths.hidden_sessions, default=None)
        if obj is None:
            return set()
        if not isinstance(obj, list):
            raise ValueError("invalid hidden_sessions.json (expected list)")
        return {sid.strip() for sid in obj if isinstance(sid, str) and sid.strip()}

    def save_hidden_sessions(self, sessions: set[str]) -> None:
        atomic_write_json(self.paths.hidden_sessions, sorted(sessions), sort_keys=True)

    def load_files(self) -> dict[str, list[Any]]:
        obj = load_json_file(self.paths.files, default=None)
        if obj is None:
            return {}
        if not isinstance(obj, dict):
            raise ValueError("invalid session_files.json (expected object)")
        cleaned: dict[str, list[Any]] = {}
        for sid, arr in obj.items():
            if not isinstance(sid, str) or not sid:
                continue
            if sid.startswith("cwd:"):
                continue
            key = sid if sid.startswith("sid:") else f"sid:{sid}"
            if not isinstance(arr, list):
                continue
            out: list[Any] = []
            seen: set[tuple[str, str]] = set()
            for value in arr:
                path = _file_entry_path(value)
                if path == "":
                    continue
                api_path = _file_entry_api_path(value)
                identity = (path, api_path)
                if identity in seen:
                    continue
                seen.add(identity)
                out.append(_make_file_entry(path, api_path))
                if len(out) >= self.file_history_max:
                    break
            if out:
                cleaned[key] = out
        return cleaned

    def save_files(self, obj: dict[str, list[Any]]) -> None:
        # Entries are already in their desired on-disk shape (plain string for
        # legacy/token-less entries, dict for tokenized ones), so write as-is.
        atomic_write_json(self.paths.files, dict(obj))

    def file_history_for_keys(self, key: str, legacy_keys: list[str]) -> tuple[list[Any], bool]:
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

    def add_file_history_entry(self, key: str, legacy_keys: list[str], path: str, api_path: str = "") -> list[Any]:
        cur = list(self.files.get(key, []))
        if not cur:
            for legacy_key in legacy_keys:
                legacy = self.files.get(legacy_key)
                if isinstance(legacy, list) and legacy:
                    cur = list(legacy)
                    if legacy_key != key:
                        self.files.pop(legacy_key, None)
                    break
        identity = (str(path), str(api_path or ""))
        cur = [item for item in cur if _file_entry_identity(item) != identity]
        cur.insert(0, _make_file_entry(str(path), str(api_path or "")))
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

    def _clean_staged_attachment_entry(self, raw: Any) -> dict[str, Any] | None:
        if not isinstance(raw, dict):
            return None
        entry_id = str(raw.get("id") or "").strip()
        path = str(raw.get("path") or "").strip()
        if not entry_id or not path:
            return None
        display_name = str(raw.get("display_name") or raw.get("filename") or Path(path).name or "file").strip() or "file"
        filename = str(raw.get("filename") or display_name).strip() or display_name
        try:
            size = int(raw.get("size"))
        except (TypeError, ValueError):
            return None
        try:
            created_ts = float(raw.get("created_ts"))
        except (TypeError, ValueError):
            return None
        if size < 0 or not math.isfinite(created_ts) or created_ts <= 0:
            return None
        return {
            "id": entry_id,
            "display_name": display_name[:256],
            "filename": filename[:256],
            "path": path,
            "size": size,
            "created_ts": created_ts,
        }

    def load_staged_attachments(self) -> dict[str, list[dict[str, Any]]]:
        if self.paths.staged_attachments is None:
            return {}
        obj = load_json_file(self.paths.staged_attachments, default=None)
        if obj is None:
            return {}
        if not isinstance(obj, dict):
            raise ValueError("invalid staged_attachments.json (expected object)")
        cleaned: dict[str, list[dict[str, Any]]] = {}
        for sid, arr in obj.items():
            if not isinstance(sid, str) or not sid.strip() or not isinstance(arr, list):
                continue
            out: list[dict[str, Any]] = []
            seen: set[str] = set()
            for raw in arr:
                entry = self._clean_staged_attachment_entry(raw)
                if entry is None or entry["id"] in seen:
                    continue
                seen.add(entry["id"])
                out.append(entry)
            if out:
                cleaned[sid.strip()] = out
        return cleaned

    def save_staged_attachments(self, source: dict[str, list[dict[str, Any]]]) -> None:
        obj: dict[str, list[dict[str, Any]]] = {}
        for sid, entries in source.items():
            if not isinstance(sid, str) or not sid.strip() or not isinstance(entries, list):
                continue
            cleaned = [entry for entry in (self._clean_staged_attachment_entry(raw) for raw in entries) if entry is not None]
            if cleaned:
                obj[sid.strip()] = cleaned
        if self.paths.staged_attachments is None:
            return
        atomic_write_json(self.paths.staged_attachments, obj)

    def staged_attachments_for_session(self, session_id: str) -> list[dict[str, Any]]:
        return [dict(entry) for entry in self.staged_attachments.get(session_id, [])]

    def add_staged_attachment(self, session_id: str, entry: dict[str, Any]) -> dict[str, Any]:
        clean = self._clean_staged_attachment_entry(entry)
        if clean is None:
            raise ValueError("invalid staged attachment")
        self.staged_attachments.setdefault(session_id, []).append(clean)
        return dict(clean)

    def remove_staged_attachment(self, session_id: str, attachment_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        entries = list(self.staged_attachments.get(session_id, []))
        kept: list[dict[str, Any]] = []
        removed: dict[str, Any] | None = None
        for entry in entries:
            if entry.get("id") == attachment_id and removed is None:
                removed = dict(entry)
            else:
                kept.append(entry)
        if removed is None:
            raise ValueError("unknown attachment")
        uploads_root = self.paths.uploads_root
        if uploads_root is not None:
            remove_staged_attachment_file(uploads_root, session_id, removed["path"])
        if kept:
            self.staged_attachments[session_id] = kept
        else:
            self.staged_attachments.pop(session_id, None)
        return removed, [dict(entry) for entry in kept]

    def clear_staged_attachments(self, session_id: str) -> list[dict[str, Any]]:
        removed = [dict(entry) for entry in self.staged_attachments.get(session_id, [])]
        uploads_root = self.paths.uploads_root
        if uploads_root is not None:
            for entry in removed:
                validate_staged_attachment_file_target(uploads_root, session_id, entry["path"])
            for entry in removed:
                remove_staged_attachment_file(uploads_root, session_id, entry["path"])
        self.staged_attachments.pop(session_id, None)
        return removed

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

    def note_recent_cwd(self, cwd_value: Any, updated_ts: float) -> bool:
        cwd = self.clean_recent_cwd(cwd_value)
        if cwd is None:
            return False
        prev_recent_ts = self.recent_cwds.get(cwd)
        if prev_recent_ts is None or prev_recent_ts < updated_ts:
            self.recent_cwds[cwd] = updated_ts
            return True
        return False

    def remember_recent_cwd(self, cwd_value: Any, *, ts: Any = None, now: Callable[[], float] = time.time) -> bool:
        if isinstance(ts, bool):
            ts_value = now()
        else:
            try:
                ts_value = float(ts) if ts is not None else now()
            except (TypeError, ValueError, OverflowError):
                ts_value = now()
        if not math.isfinite(ts_value) or ts_value <= 0:
            ts_value = now()
        changed = self.note_recent_cwd(cwd_value, ts_value)
        if not changed:
            return False
        if len(self.recent_cwds) > self.recent_cwd_max * 2:
            keep = dict(sorted(self.recent_cwds.items(), key=lambda item: (-float(item[1]), item[0]))[: self.recent_cwd_max])
            self.recent_cwds.clear()
            self.recent_cwds.update(keep)
        return True

    def list_recent_cwds(self, *, limit: int) -> list[str]:
        return [cwd for cwd, _ts in sorted(self.recent_cwds.items(), key=lambda item: (-float(item[1]), item[0]))[: max(0, int(limit))]]

    def clear_deleted_session_state(
        self,
        session_id: str,
        *,
        clear_recovery: bool = False,
        cwd: str = "",
    ) -> DeletedSessionStateChanges:
        aliases_changed = False
        sidebar_changed = False
        hidden_changed = False
        unattended_changed = False
        files_changed = False
        queues_changed = False
        pending_changed = False
        staged_changed = False
        unknown_changed = False

        if session_id in self.aliases:
            self.aliases.pop(session_id, None)
            aliases_changed = True

        if session_id in self.hidden_sessions:
            self.hidden_sessions.discard(session_id)
            hidden_changed = True

        if session_id in self.sidebar_meta:
            self.sidebar_meta.pop(session_id, None)
            sidebar_changed = True
        for entry in self.sidebar_meta.values():
            if not isinstance(entry, dict):
                continue
            if entry.get("dependency_session_id") != session_id:
                continue
            entry.pop("dependency_session_id", None)
            sidebar_changed = True

        if session_id in self.unattended:
            self.unattended.pop(session_id, None)
            unattended_changed = True

        if self.clear_file_history_for_keys(f"sid:{session_id}", [session_id], cwd=cwd):
            files_changed = True

        if session_id in self.pending_attachment_ids:
            self.pending_attachment_ids.discard(session_id)
            pending_changed = True

        if session_id in self.staged_attachments:
            self.staged_attachments.pop(session_id, None)
            staged_changed = True

        uploads_root = self.paths.uploads_root
        if uploads_root is not None:
            # Staged attachment bytes live under <uploads_root>/<session_id>/
            # (see file_upload.stage_uploaded_file). They can outlive the pending
            # flag, so remove the session-scoped directory unconditionally when
            # the store knows the upload root. An invalid session id fails loud
            # rather than widening the removal to the upload root.
            remove_session_uploads(uploads_root, session_id)

        has_direct_unknown = session_id in self.commit_unknown_sends
        queue = self.queues.get(session_id)
        if isinstance(queue, list):
            has_queued_recovery = self.queue_store.has_recovery_items(self.queues, session_id)
            if queue and has_direct_unknown:
                if self.queue_store.mark_orphan_recovery_items(self.queues, session_id):
                    queues_changed = True
                has_queued_recovery = True
            if clear_recovery or not has_queued_recovery:
                self.queues.pop(session_id, None)
                queues_changed = True

        if clear_recovery and has_direct_unknown:
            self.commit_unknown_sends.pop(session_id, None)
            unknown_changed = True

        return DeletedSessionStateChanges(
            aliases=aliases_changed,
            sidebar_meta=sidebar_changed,
            hidden_sessions=hidden_changed,
            unattended=unattended_changed,
            files=files_changed,
            queues=queues_changed,
            pending_attachments=pending_changed,
            staged_attachments=staged_changed,
            commit_unknown_sends=unknown_changed,
        )

    def prune_missing_commit_unknown_sends(
        self,
        *,
        active_session_ids: Iterable[str],
        now_ts: float,
        max_age_seconds: float,
    ) -> DeletedSessionStateChanges:
        active = {str(sid) for sid in active_session_ids}
        unknown_changed = False
        queues_changed = False
        for sid, record in list(self.commit_unknown_sends.items()):
            if sid in active:
                continue
            created_raw = record.get("created_ts") if isinstance(record, dict) else None
            try:
                created_ts = float(created_raw)
            except (TypeError, ValueError):
                created_ts = now_ts
            if math.isfinite(created_ts) and created_ts > 0 and (now_ts - created_ts) < max_age_seconds:
                continue
            if self.queue_store.mark_orphan_recovery_items(self.queues, str(sid)):
                queues_changed = True
            self.commit_unknown_sends.pop(sid, None)
            unknown_changed = True
        return DeletedSessionStateChanges(queues=queues_changed, commit_unknown_sends=unknown_changed)

    def save_deleted_session_state_changes(
        self,
        changes: DeletedSessionStateChanges,
        *,
        save_aliases: Callable[[], None],
        save_sidebar_meta: Callable[[], None],
        save_hidden_sessions: Callable[[], None],
        save_unattended: Callable[[], None],
        save_files: Callable[[], None],
        save_queues: Callable[[], None],
        save_pending_attachments: Callable[[], None],
        save_commit_unknown_sends: Callable[[], None],
        save_staged_attachments: Callable[[], None] = lambda: None,
    ) -> None:
        if changes.pending_attachments:
            save_pending_attachments()
        if changes.staged_attachments:
            save_staged_attachments()
        if changes.commit_unknown_sends:
            save_commit_unknown_sends()
        if changes.aliases:
            save_aliases()
        if changes.sidebar_meta:
            save_sidebar_meta()
        if changes.hidden_sessions:
            save_hidden_sessions()
        if changes.unattended:
            save_unattended()
        if changes.files:
            save_files()
        if changes.queues:
            save_queues()
