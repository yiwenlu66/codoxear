from __future__ import annotations

import copy
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any

from .runtime_access import manager_runtime


def _runtime(manager: Any):
    return manager_runtime(manager)


@dataclass(slots=True)
class PageStateService:
    manager: Any

    def load_harness(self) -> None:
        load_harness(self.manager)

    def save_harness(self) -> None:
        save_harness(self.manager)

    def clear_deleted_session_state(self, session_id: str) -> None:
        clear_deleted_session_state(self.manager, session_id)

    def load_files(self) -> None:
        load_files(self.manager)

    def save_files(self) -> None:
        save_files(self.manager)

    def load_queues(self) -> None:
        load_queues(self.manager)

    def save_queues(self) -> None:
        save_queues(self.manager)

    def load_recent_cwds(self) -> None:
        load_recent_cwds(self.manager)

    def save_recent_cwds(self) -> None:
        save_recent_cwds(self.manager)

    def load_cwd_groups(self) -> None:
        load_cwd_groups(self.manager)

    def save_cwd_groups(self) -> None:
        save_cwd_groups(self.manager)

    def cwd_groups_get(self) -> dict[str, dict[str, Any]]:
        return cwd_groups_get(self.manager)

    def prune_stale_workspace_dirs(self) -> None:
        prune_stale_workspace_dirs(self.manager)

    def known_cwd_group_keys(self) -> set[str]:
        return known_cwd_group_keys(self.manager)

    def cwd_group_set(
        self, cwd: str, label: str | None = None, collapsed: bool | None = None
    ) -> tuple[str, dict[str, Any]]:
        return cwd_group_set(self.manager, cwd, label=label, collapsed=collapsed)

    def remember_recent_cwd(self, cwd: Any, *, ts: Any = None) -> bool:
        return remember_recent_cwd(self.manager, cwd, ts=ts)

    def backfill_recent_cwds_from_logs(self) -> None:
        backfill_recent_cwds_from_logs(self.manager)

    def recent_cwds(self, *, limit: int) -> list[str]:
        return recent_cwds(self.manager, limit=limit)

    def queue_len(self, session_id: str) -> int:
        return queue_len(self.manager, session_id)

    def queue_list_local(self, session_id: str) -> list[str]:
        return queue_list_local(self.manager, session_id)

    def queue_enqueue_local(self, session_id: str, text: str) -> dict[str, Any]:
        return queue_enqueue_local(self.manager, session_id, text)

    def queue_delete_local(self, session_id: str, index: int) -> dict[str, Any]:
        return queue_delete_local(self.manager, session_id, index)

    def queue_update_local(
        self, session_id: str, index: int, text: str
    ) -> dict[str, Any]:
        return queue_update_local(self.manager, session_id, index, text)

    def files_key_for_session(
        self, session_id: str
    ) -> tuple[str, tuple[str, str], Any]:
        return files_key_for_session(self.manager, session_id)

    def files_get(self, session_id: str) -> list[str]:
        return files_get(self.manager, session_id)

    def files_add(self, session_id: str, path: str) -> list[str]:
        return files_add(self.manager, session_id, path)

    def files_clear(self, session_id: str) -> None:
        files_clear(self.manager, session_id)

    def harness_get(self, session_id: str) -> dict[str, Any]:
        return harness_get(self.manager, session_id)

    def harness_set(
        self,
        session_id: str,
        *,
        enabled: bool | None = None,
        request: str | None = None,
        cooldown_minutes: int | None = None,
        remaining_injections: int | None = None,
    ) -> dict[str, Any]:
        return harness_set(
            self.manager,
            session_id,
            enabled=enabled,
            request=request,
            cooldown_minutes=cooldown_minutes,
            remaining_injections=remaining_injections,
        )


def service(manager: Any) -> PageStateService:
    return PageStateService(manager)


def load_harness(manager: Any) -> None:
    sv = _runtime(manager)
    try:
        raw = sv.HARNESS_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("invalid harness.json (expected object)")
    cleaned: dict[str, dict[str, Any]] = {}
    for sid, v in obj.items():
        if not isinstance(sid, str) or not sid:
            continue
        if not isinstance(v, dict):
            continue
        enabled = bool(v.get("enabled")) if "enabled" in v else False
        if "text" in v:
            raise ValueError(
                f"invalid harness config for session {sid!r} (use 'request', not 'text')"
            )
        request = v.get("request")
        if request is None:
            request = ""
        if not isinstance(request, str):
            raise ValueError(f"invalid harness request for session {sid!r}")
        cooldown_minutes = sv._clean_harness_cooldown_minutes(v.get("cooldown_minutes"))
        remaining_injections = sv._clean_harness_remaining_injections(
            v.get("remaining_injections"), allow_zero=True
        )
        cleaned[sid] = {
            "enabled": enabled,
            "request": request,
            "cooldown_minutes": cooldown_minutes,
            "remaining_injections": remaining_injections,
        }
    with manager._lock:
        manager._harness = cleaned


def save_harness(manager: Any) -> None:
    sv = _runtime(manager)
    with manager._lock:
        obj = dict(manager._harness)
    os.makedirs(sv.APP_DIR, exist_ok=True)
    tmp = sv.HARNESS_PATH.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, sv.HARNESS_PATH)


def clear_deleted_session_state(manager: Any, session_id: str) -> None:
    changed_sidebar = False
    changed_harness = False
    changed_files = False
    changed_queues = False
    ref = manager._page_state_ref_for_session_id(session_id)
    with manager._lock:
        aliases = getattr(manager, "_aliases", None)
        if isinstance(aliases, dict):
            aliases.pop(session_id, None)
            if ref is not None:
                aliases.pop(ref, None)
        meta_map = getattr(manager, "_sidebar_meta", None)
        if isinstance(meta_map, dict):
            if session_id in meta_map:
                meta_map.pop(session_id, None)
                changed_sidebar = True
            if ref is not None and ref in meta_map:
                meta_map.pop(ref, None)
                changed_sidebar = True
        if isinstance(meta_map, dict) and ref is not None:
            for entry in meta_map.values():
                if not isinstance(entry, dict):
                    continue
                if entry.get("dependency_session_id") != ref[1]:
                    continue
                entry.pop("dependency_session_id", None)
                changed_sidebar = True
        harness = getattr(manager, "_harness", None)
        if isinstance(harness, dict) and session_id in harness:
            harness.pop(session_id, None)
            changed_harness = True
        files = getattr(manager, "_files", None)
        if isinstance(files, dict):
            for legacy_key in (session_id, f"sid:{session_id}"):
                if legacy_key in files:
                    files.pop(legacy_key, None)
                    changed_files = True
            if ref is not None and ref in files:
                files.pop(ref, None)
                changed_files = True
        queues = getattr(manager, "_queues", None)
        if isinstance(queues, dict):
            if session_id in queues:
                queues.pop(session_id, None)
                changed_queues = True
            if ref is not None and ref in queues:
                queues.pop(ref, None)
                changed_queues = True
        command_cache = getattr(manager, "_pi_commands_cache", None)
        if isinstance(command_cache, dict):
            command_cache.pop(session_id, None)
    manager._save_aliases()
    if changed_sidebar:
        manager._save_sidebar_meta()
    if changed_harness:
        manager._save_harness()
    if changed_files:
        manager._save_files()
    if changed_queues:
        manager._save_queues()


def load_files(manager: Any) -> None:
    sv = _runtime(manager)
    db = getattr(manager, "_page_state_db", None)
    if db is None:
        try:
            raw = sv.FILE_HISTORY_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError("invalid session_files.json (expected object)")
        cleaned: dict[Any, list[str]] = {}
        for sid, arr in obj.items():
            if not isinstance(sid, str) or not sid:
                continue
            if sid.startswith("cwd:"):
                continue
            key = sid if sid.startswith("sid:") else f"sid:{sid}"
            if not isinstance(arr, list):
                continue
            out: list[str] = []
            for v in arr:
                if not isinstance(v, str):
                    continue
                p = v.strip()
                if not p or p in out:
                    continue
                out.append(p)
                if len(out) >= sv.FILE_HISTORY_MAX:
                    break
            if out:
                cleaned[key] = out
        with manager._lock:
            manager._files = cleaned
        return
    with manager._lock:
        manager._files = db.load_files()


def save_files(manager: Any) -> None:
    sv = _runtime(manager)
    db = getattr(manager, "_page_state_db", None)
    if db is None:
        with manager._lock:
            obj = dict(manager._files)
        os.makedirs(sv.APP_DIR, exist_ok=True)
        tmp = sv.FILE_HISTORY_PATH.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, sv.FILE_HISTORY_PATH)
        return
    manager._persist_files()


def load_queues(manager: Any) -> None:
    sv = _runtime(manager)
    db = getattr(manager, "_page_state_db", None)
    if db is None:
        try:
            raw = sv.QUEUE_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError("invalid session_queues.json (expected object)")
        cleaned: dict[Any, list[str]] = {}
        for sid, arr in obj.items():
            if not isinstance(sid, str) or not sid:
                continue
            if not isinstance(arr, list):
                continue
            out: list[str] = []
            for v in arr:
                if not isinstance(v, str):
                    continue
                t = v.strip()
                if not t:
                    continue
                out.append(v)
            if out:
                cleaned[sid] = out
        with manager._lock:
            manager._queues = cleaned
        return
    with manager._lock:
        manager._queues = db.load_queues()


def save_queues(manager: Any) -> None:
    sv = _runtime(manager)
    db = getattr(manager, "_page_state_db", None)
    if db is None:
        with manager._lock:
            obj = dict(manager._queues)
        os.makedirs(sv.APP_DIR, exist_ok=True)
        tmp = sv.QUEUE_PATH.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, sv.QUEUE_PATH)
        return
    manager._persist_queues()


def load_recent_cwds(manager: Any) -> None:
    sv = _runtime(manager)
    db = getattr(manager, "_page_state_db", None)
    if db is None:
        try:
            raw = sv.RECENT_CWD_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError("invalid recent_cwds.json (expected object)")
        source_items = obj.items()
    else:
        source_items = db.load_recent_cwds().items()
    cleaned: dict[str, float] = {}
    for raw_cwd, raw_ts in source_items:
        cwd = sv._clean_recent_cwd(raw_cwd)
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
    top = sorted(cleaned.items(), key=lambda item: (-item[1], item[0]))[: sv.RECENT_CWD_MAX]
    with manager._lock:
        manager._recent_cwds = dict(top)


def save_recent_cwds(manager: Any) -> None:
    sv = _runtime(manager)
    db = getattr(manager, "_page_state_db", None)
    with manager._lock:
        items = sorted(
            getattr(manager, "_recent_cwds", {}).items(),
            key=lambda item: (-float(item[1]), item[0]),
        )[: sv.RECENT_CWD_MAX]
        manager._recent_cwds = dict(items)
    if db is not None:
        manager._persist_recent_cwds()
        return
    obj = {cwd: ts for cwd, ts in items}
    os.makedirs(sv.APP_DIR, exist_ok=True)
    tmp = sv.RECENT_CWD_PATH.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, sv.RECENT_CWD_PATH)


def load_cwd_groups(manager: Any) -> None:
    sv = _runtime(manager)
    db = getattr(manager, "_page_state_db", None)
    source_items: Any
    if db is None:
        try:
            raw = sv.CWD_GROUPS_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            with manager._lock:
                manager._cwd_groups = {}
            return
        try:
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                raise ValueError("invalid cwd_groups.json (expected object)")
            source_items = obj.items()
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            sv.LOG.warning("recovering malformed cwd_groups.json as empty state: %s", e)
            source_items = []
    else:
        source_items = db.load_cwd_groups().items()
    cleaned: dict[str, dict[str, Any]] = {}
    for cwd, v in source_items:
        try:
            normalized_cwd = sv._normalize_cwd_group_key(cwd)
        except ValueError:
            continue
        if not isinstance(v, dict):
            continue
        label = sv._clean_alias(v.get("label", ""))
        persisted_collapsed = v.get("collapsed", False)
        collapsed = persisted_collapsed if isinstance(persisted_collapsed, bool) else False
        if label or collapsed:
            cleaned[normalized_cwd] = {"label": label, "collapsed": collapsed}
    with manager._lock:
        manager._cwd_groups = cleaned


def save_cwd_groups(manager: Any) -> None:
    sv = _runtime(manager)
    db = getattr(manager, "_page_state_db", None)
    if db is not None:
        manager._persist_cwd_groups()
        return
    with manager._lock:
        obj = dict(manager._cwd_groups)
    os.makedirs(sv.APP_DIR, exist_ok=True)
    tmp = sv.CWD_GROUPS_PATH.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, sv.CWD_GROUPS_PATH)


def cwd_groups_get(manager: Any) -> dict[str, dict[str, Any]]:
    prune_stale_workspace_dirs(manager)
    with manager._lock:
        return copy.deepcopy(manager._cwd_groups)


def prune_stale_workspace_dirs(manager: Any) -> None:
    sv = _runtime(manager)
    if not bool(getattr(manager, "_prune_missing_workspace_dirs", False)):
        return
    active_cwds: set[str] = set()
    with manager._lock:
        sessions = list(getattr(manager, "_sessions", {}).values())
        recent_items = list(getattr(manager, "_recent_cwds", {}).keys())
        grouped_items = list(getattr(manager, "_cwd_groups", {}).keys())
    for session in sessions:
        try:
            active_cwds.add(sv._normalize_cwd_group_key(getattr(session, "cwd", None)))
        except ValueError:
            continue
    stale_recent = {
        cwd
        for cwd in recent_items
        if cwd not in active_cwds and sv._existing_workspace_dir(cwd) is None
    }
    stale_groups = {
        cwd
        for cwd in grouped_items
        if cwd not in active_cwds and sv._existing_workspace_dir(cwd) is None
    }
    save_recent = False
    save_groups = False
    if stale_recent or stale_groups:
        with manager._lock:
            recent_map = getattr(manager, "_recent_cwds", None)
            if isinstance(recent_map, dict):
                for cwd in stale_recent:
                    if recent_map.pop(cwd, None) is not None:
                        save_recent = True
            group_map = getattr(manager, "_cwd_groups", None)
            if isinstance(group_map, dict):
                for cwd in stale_groups:
                    if group_map.pop(cwd, None) is not None:
                        save_groups = True
    if save_recent:
        manager._save_recent_cwds()
    if save_groups:
        manager._save_cwd_groups()


def known_cwd_group_keys(manager: Any) -> set[str]:
    sv = _runtime(manager)
    prune_stale_workspace_dirs(manager)
    known: set[str] = set()
    with manager._lock:
        sessions = list(getattr(manager, "_sessions", {}).values())
        recent_items = list(getattr(manager, "_recent_cwds", {}).keys())
        grouped_items = list(getattr(manager, "_cwd_groups", {}).keys())
    for session in sessions:
        try:
            normalized = sv._normalize_cwd_group_key(getattr(session, "cwd", None))
        except ValueError:
            continue
        known.add(normalized)
    for cwd in recent_items:
        try:
            normalized = sv._normalize_cwd_group_key(cwd)
        except ValueError:
            continue
        known.add(normalized)
    for cwd in grouped_items:
        try:
            normalized = sv._normalize_cwd_group_key(cwd)
        except ValueError:
            continue
        known.add(normalized)
    return known


def cwd_group_set(
    manager: Any,
    cwd: str,
    label: str | None = None,
    collapsed: bool | None = None,
) -> tuple[str, dict[str, Any]]:
    sv = _runtime(manager)
    normalized_cwd = sv._normalize_cwd_group_key(cwd)
    if label is not None and not isinstance(label, str):
        raise ValueError("label must be a string")
    requested_label = sv._clean_alias(label) if label is not None else None
    requested_collapsed = collapsed
    if requested_collapsed is not None and not isinstance(requested_collapsed, bool):
        raise ValueError("collapsed must be a boolean")
    prune_stale_workspace_dirs(manager)
    with manager._lock:
        existing = manager._cwd_groups.get(
            normalized_cwd, {"label": "", "collapsed": False}
        )
    known_cwds = known_cwd_group_keys(manager)
    if normalized_cwd not in known_cwds:
        effective_label = requested_label if requested_label is not None else existing["label"]
        effective_collapsed = (
            requested_collapsed
            if requested_collapsed is not None
            else existing["collapsed"]
        )
        if not effective_label and not effective_collapsed:
            return normalized_cwd, {"label": "", "collapsed": False}
        raise ValueError("cwd is not a known session working directory")
    with manager._lock:
        existing = manager._cwd_groups.get(normalized_cwd, existing)
        new_label = requested_label if requested_label is not None else existing["label"]
        new_collapsed = (
            requested_collapsed
            if requested_collapsed is not None
            else existing["collapsed"]
        )
        entry = {"label": new_label, "collapsed": new_collapsed}
        if not new_label and not new_collapsed:
            manager._cwd_groups.pop(normalized_cwd, None)
        else:
            manager._cwd_groups[normalized_cwd] = entry
    manager._save_cwd_groups()
    return normalized_cwd, dict(entry)


def remember_recent_cwd(manager: Any, cwd: Any, *, ts: Any = None) -> bool:
    sv = _runtime(manager)
    cleaned = sv._clean_recent_cwd(cwd)
    if cleaned is None:
        return False
    if isinstance(ts, bool):
        ts_value = time.time()
    else:
        try:
            ts_value = float(ts) if ts is not None else time.time()
        except (TypeError, ValueError):
            ts_value = time.time()
    if not math.isfinite(ts_value) or ts_value <= 0:
        ts_value = time.time()
    with manager._lock:
        recent = getattr(manager, "_recent_cwds", None)
        if not isinstance(recent, dict):
            manager._recent_cwds = {}
            recent = manager._recent_cwds
        prev = recent.get(cleaned)
        if prev is not None and prev >= ts_value:
            return False
        recent[cleaned] = ts_value
        if len(recent) > sv.RECENT_CWD_MAX * 2:
            keep = dict(
                sorted(recent.items(), key=lambda item: (-float(item[1]), item[0]))[
                    : sv.RECENT_CWD_MAX
                ]
            )
            recent.clear()
            recent.update(keep)
    return True


def backfill_recent_cwds_from_logs(manager: Any) -> None:
    sv = _runtime(manager)
    changed = False
    seen: set[str] = set()
    for log_path in sv._iter_session_logs():
        try:
            row = sv._resume_candidate_from_log(log_path)
        except Exception:
            continue
        if not isinstance(row, dict):
            continue
        cwd = row.get("cwd")
        if not isinstance(cwd, str) or not cwd or cwd in seen:
            continue
        seen.add(cwd)
        if remember_recent_cwd(manager, cwd, ts=row.get("updated_ts")):
            changed = True
        if len(seen) >= sv.RECENT_CWD_MAX:
            break
    if changed:
        manager._save_recent_cwds()


def recent_cwds(manager: Any, *, limit: int) -> list[str]:
    prune_stale_workspace_dirs(manager)
    with manager._lock:
        items = sorted(
            getattr(manager, "_recent_cwds", {}).items(),
            key=lambda item: (-float(item[1]), item[0]),
        )
    return [cwd for cwd, _ts in items[: max(0, int(limit))]]


def queue_len(manager: Any, session_id: str) -> int:
    ref = manager._page_state_ref_for_session_id(session_id)
    runtime_id = manager._runtime_session_id_for_identifier(session_id)
    if ref is None:
        return 0
    with manager._lock:
        qmap = getattr(manager, "_queues", None)
        if not isinstance(qmap, dict):
            return 0
        q = qmap.get(ref)
        if not isinstance(q, list) and runtime_id is not None:
            q = qmap.get(runtime_id)
        if not isinstance(q, list):
            q = qmap.get(session_id)
        return int(len(q)) if isinstance(q, list) else 0


def queue_list_local(manager: Any, session_id: str) -> list[str]:
    ref = manager._page_state_ref_for_session_id(session_id)
    runtime_id = manager._runtime_session_id_for_identifier(session_id)
    if ref is None:
        return []
    with manager._lock:
        qmap = getattr(manager, "_queues", None)
        if not isinstance(qmap, dict):
            return []
        q = qmap.get(ref)
        if not isinstance(q, list) and runtime_id is not None:
            q = qmap.get(runtime_id)
        if not isinstance(q, list):
            q = qmap.get(session_id)
        if not isinstance(q, list) or not q:
            return []
        return list(q)


def queue_enqueue_local(manager: Any, session_id: str, text: str) -> dict[str, Any]:
    sv = _runtime(manager)
    t = str(text)
    if not t.strip():
        raise ValueError("text required")
    touched_ts: float | None = None
    ref = manager._page_state_ref_for_session_id(session_id)
    runtime_id = manager._runtime_session_id_for_identifier(session_id)
    durable_session_id = sv._clean_optional_text(session_id)
    with manager._lock:
        if runtime_id is not None:
            s0 = manager._sessions.get(runtime_id)
            if s0 is not None:
                durable_session_id = manager._durable_session_id_for_session(s0)
    if ref is None:
        raise KeyError("unknown session")
    with manager._lock:
        s = manager._sessions.get(runtime_id) if runtime_id is not None else None
        q = manager._queues.get(runtime_id) if runtime_id is not None else None
        if not isinstance(q, list):
            q = manager._queues.get(ref)
        if not isinstance(q, list):
            q = []
            if runtime_id is not None:
                manager._queues[runtime_id] = q
            else:
                manager._queues[ref] = q
        q.append(t)
        ql = len(q)
        if s is not None and s.backend == "pi":
            touched_ts = sv._touch_session_file(s.session_path)
            s.pi_idle_activity_ts = None
            s.pi_busy_activity_floor = touched_ts
    manager._save_queues()
    if durable_session_id is not None:
        sv._publish_session_workspace_invalidate(
            durable_session_id,
            runtime_id=runtime_id,
            reason="queue_changed",
        )
        sv._publish_session_live_invalidate(
            durable_session_id,
            runtime_id=runtime_id,
            reason="queue_changed",
        )
    return {"queued": True, "queue_len": int(ql)}


def queue_delete_local(manager: Any, session_id: str, index: int) -> dict[str, Any]:
    sv = _runtime(manager)
    ref = manager._page_state_ref_for_session_id(session_id)
    runtime_id = manager._runtime_session_id_for_identifier(session_id)
    if ref is None or runtime_id is None:
        raise KeyError("unknown session")
    with manager._lock:
        session = manager._sessions.get(runtime_id)
        if session is None:
            raise KeyError("unknown session")
        durable_session_id = manager._durable_session_id_for_session(session)
        q = manager._queues.get(runtime_id)
        if not isinstance(q, list):
            q = []
            manager._queues[runtime_id] = q
        if index < 0 or index >= len(q):
            raise ValueError("index out of range")
        q.pop(int(index))
        ql = len(q)
        if not q:
            manager._queues.pop(runtime_id, None)
            manager._queues.pop(ref, None)
    manager._save_queues()
    sv._publish_session_workspace_invalidate(
        durable_session_id,
        runtime_id=runtime_id,
        reason="queue_changed",
    )
    sv._publish_session_live_invalidate(
        durable_session_id,
        runtime_id=runtime_id,
        reason="queue_changed",
    )
    return {"ok": True, "queue_len": int(ql)}


def queue_update_local(
    manager: Any, session_id: str, index: int, text: str
) -> dict[str, Any]:
    sv = _runtime(manager)
    t = str(text)
    if not t.strip():
        raise ValueError("text required")
    ref = manager._page_state_ref_for_session_id(session_id)
    runtime_id = manager._runtime_session_id_for_identifier(session_id)
    if ref is None or runtime_id is None:
        raise KeyError("unknown session")
    with manager._lock:
        session = manager._sessions.get(runtime_id)
        if session is None:
            raise KeyError("unknown session")
        durable_session_id = manager._durable_session_id_for_session(session)
        q = manager._queues.get(runtime_id)
        if not isinstance(q, list):
            q = []
            manager._queues[runtime_id] = q
        if index < 0 or index >= len(q):
            raise ValueError("index out of range")
        q[int(index)] = t
        ql = len(q)
    manager._save_queues()
    sv._publish_session_workspace_invalidate(
        durable_session_id,
        runtime_id=runtime_id,
        reason="queue_changed",
    )
    return {"ok": True, "queue_len": int(ql)}


def files_key_for_session(manager: Any, session_id: str) -> tuple[str, Any, Any]:
    runtime_id = manager._runtime_session_id_for_identifier(session_id)
    if runtime_id is None:
        raise KeyError("unknown session")
    s = manager._sessions.get(runtime_id)
    if not s:
        raise KeyError("unknown session")
    ref = manager._page_state_ref_for_session(s)
    if ref is None:
        raise KeyError("unknown session")
    return runtime_id, ref, s


def files_get(manager: Any, session_id: str) -> list[str]:
    runtime_id, key, _s = files_key_for_session(manager, session_id)
    with manager._lock:
        arr = manager._files.get(runtime_id)
        if not isinstance(arr, list):
            arr = manager._files.get(key)
        if not isinstance(arr, list):
            arr = manager._files.get(session_id)
        return list(arr) if isinstance(arr, list) else []


def files_add(manager: Any, session_id: str, path: str) -> list[str]:
    sv = _runtime(manager)
    p = str(path).strip()
    if not p:
        return files_get(manager, session_id)
    runtime_id, key, _s = files_key_for_session(manager, session_id)
    with manager._lock:
        cur = list(
            manager._files.get(
                runtime_id,
                manager._files.get(key, manager._files.get(session_id, [])),
            )
        )
        cur = [x for x in cur if x != p]
        cur.insert(0, p)
        if len(cur) > sv.FILE_HISTORY_MAX:
            cur = cur[: sv.FILE_HISTORY_MAX]
        manager._files[runtime_id] = cur
    manager._save_files()
    return list(cur)


def files_clear(manager: Any, session_id: str) -> None:
    dirty = False
    runtime_id, key, _s = files_key_for_session(manager, session_id)
    with manager._lock:
        if runtime_id in manager._files:
            manager._files.pop(runtime_id, None)
            dirty = True
        if session_id in manager._files:
            manager._files.pop(session_id, None)
            dirty = True
        if key in manager._files:
            manager._files.pop(key, None)
            dirty = True
    if dirty:
        manager._save_files()


def harness_get(manager: Any, session_id: str) -> dict[str, Any]:
    sv = _runtime(manager)
    runtime_id = manager._runtime_session_id_for_identifier(session_id)
    if runtime_id is None:
        raise KeyError("unknown session")
    with manager._lock:
        s = manager._sessions.get(runtime_id)
        if not s:
            raise KeyError("unknown session")
        cfg0 = manager._harness.get(runtime_id)
        cfg = dict(cfg0) if isinstance(cfg0, dict) else {}
    enabled = bool(cfg.get("enabled"))
    request = cfg.get("request")
    if not isinstance(request, str):
        request = ""
    cooldown_minutes = sv._clean_harness_cooldown_minutes(cfg.get("cooldown_minutes"))
    remaining_injections = sv._clean_harness_remaining_injections(
        cfg.get("remaining_injections"), allow_zero=True
    )
    return {
        "enabled": enabled,
        "request": request,
        "cooldown_minutes": cooldown_minutes,
        "remaining_injections": remaining_injections,
    }


def harness_set(
    manager: Any,
    session_id: str,
    *,
    enabled: bool | None = None,
    request: str | None = None,
    cooldown_minutes: int | None = None,
    remaining_injections: int | None = None,
) -> dict[str, Any]:
    sv = _runtime(manager)
    runtime_id = manager._runtime_session_id_for_identifier(session_id)
    if runtime_id is None:
        raise KeyError("unknown session")
    with manager._lock:
        s = manager._sessions.get(runtime_id)
        if not s:
            raise KeyError("unknown session")
        cur0 = manager._harness.get(runtime_id)
        cur = dict(cur0) if isinstance(cur0, dict) else {}
        if enabled is not None:
            cur["enabled"] = bool(enabled)
        if request is not None:
            cur["request"] = str(request)
        if cooldown_minutes is not None:
            cur["cooldown_minutes"] = sv._clean_harness_cooldown_minutes(
                cooldown_minutes
            )
        if remaining_injections is not None:
            cur["remaining_injections"] = sv._clean_harness_remaining_injections(
                remaining_injections, allow_zero=True
            )
        cur["cooldown_minutes"] = sv._clean_harness_cooldown_minutes(
            cur.get("cooldown_minutes")
        )
        cur["remaining_injections"] = sv._clean_harness_remaining_injections(
            cur.get("remaining_injections"), allow_zero=True
        )
        manager._harness[runtime_id] = cur
        if enabled is not None and bool(enabled) is False:
            manager._harness_last_injected.pop(runtime_id, None)
    manager._save_harness()
    return harness_get(manager, session_id)
