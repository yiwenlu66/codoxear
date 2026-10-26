"""Persistent snapshot of live session launch configuration.

The snapshot is the durable record used to respawn sessions after broker
death (reboot, crash, server outage). It is a map keyed by **transcript
identity** — ``(agent_backend, thread_id)`` — because that is what a restore
resumes and what survives a broker's death. Each entry records the registry
session id it is currently known under (the broker sidecar's sock stem,
``broker-<pid>``, which the UI, queue store, and draft store key on) as an
*updatable attribute*: a resumed transcript gets a fresh registry id and the
entry follows it. The registry id alone is not an identity — pids are reused
after a reboot — so an entry also records the broker's ``start_ts``; the pair
``(session_id, start_ts)`` names one broker incarnation.

Upsert rules (``SessionSnapshotStore.upsert``), applied per incoming entry
``E`` against the stored entries:

(a) Same transcript key. The stored entry is updated in place. When the
    registry id differs the transcript was resumed by a new broker, and the
    result reports an ``IdentityRemap`` so the coordinator can move
    per-session state (queue, draft, alias, ...) from the old registry id to
    the new one. Any ``restoring`` mark clears: the live upsert is the proof
    the restore landed.
(b) Same broker incarnation (``session_id`` and ``start_ts``) under a
    different transcript key: the live session moved on (Pi ``/new``,
    ``/resume``), so the stale key is dropped. A different ``start_ts`` with
    the same registry id is pid reuse, and the dead entry is kept.
(c) Same backend, and ``E`` declares the stored entry's thread id as its
    resume target or shares its resolved log path, under a different thread
    id: the harness continued the transcript under a new id. The stored key
    is replaced and a remap is reported (old thread id → new thread id, old
    registry id → new one).

Write policy is upsert-only: discovery adds/updates entries for live
sessions and never removes them. Removal happens only through explicit
events (UI delete, dismiss, detected clean broker exit, a restore that finds
the transcript permanently gone) or the supersede rules above. As
belt-and-braces against a null live list wiping a meaningful snapshot, an
upsert write that would leave zero entries over a non-empty on-disk snapshot
is refused; only the explicit removal paths may persist an empty entry list.

A restore does not consume an entry. It stamps a ``restoring`` mark
(``{broker_pid, launched_at}``) that hides the entry from further restores
until the replacement's own upsert supersedes it by rule (a)/(c), or until
the mark expires (``RESTORING_TTL_SECONDS``): an expired mark means the
launch never bound the transcript and the entry is restorable again.
Expiry is evaluated on read, never by a timer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Iterable
import uuid


SNAPSHOT_VERSION = 1
SNAPSHOT_FILENAME = "session_snapshot.json"
RESTORING_TTL_SECONDS = 600.0

EntryKey = tuple[str, str]


class SnapshotCorrupt(ValueError):
    """The on-disk snapshot cannot be parsed into the documented shape."""


class SnapshotWriteRefused(RuntimeError):
    """An upsert write would have persisted an empty entry list over a non-empty snapshot."""


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    out = value.strip()
    return out or None


def _required_text(raw: dict[str, Any], key: str) -> str:
    out = _clean_text(raw.get(key))
    if out is None:
        raise SnapshotCorrupt(f"snapshot entry is missing {key}")
    return out


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _pid_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return int(value)


@dataclass(frozen=True)
class RestoringMark:
    """A restore launched a replacement for this entry and is waiting for it to bind."""

    broker_pid: int | None
    launched_at: float

    def to_dict(self) -> dict[str, Any]:
        return {"broker_pid": self.broker_pid, "launched_at": float(self.launched_at)}

    @classmethod
    def from_dict(cls, raw: Any) -> "RestoringMark":
        if not isinstance(raw, dict):
            raise SnapshotCorrupt("restoring mark must be an object")
        launched_at = _float_or_none(raw.get("launched_at"))
        if launched_at is None:
            raise SnapshotCorrupt("restoring mark is missing launched_at")
        return cls(broker_pid=_pid_or_none(raw.get("broker_pid")), launched_at=launched_at)

    def active(self, now: float, *, ttl_seconds: float = RESTORING_TTL_SECONDS) -> bool:
        return 0.0 <= (float(now) - float(self.launched_at)) < float(ttl_seconds)


@dataclass(frozen=True)
class SnapshotEntry:
    session_id: str
    thread_id: str
    agent_backend: str
    log_path: str
    cwd: str
    model_provider: str | None
    preferred_auth_method: str | None
    model: str | None
    reasoning_effort: str | None
    service_tier: str | None
    tmux: bool
    saved_at: float
    start_ts: float = 0.0
    resume_session_id: str | None = None
    restoring: RestoringMark | None = None

    @property
    def key(self) -> EntryKey:
        return (self.agent_backend, self.thread_id)

    def launch_fields(self) -> tuple[Any, ...]:
        """Everything that decides how the session is respawned or identified; ``saved_at``/``restoring`` excluded."""
        return (
            self.session_id,
            self.thread_id,
            self.agent_backend,
            self.log_path,
            self.cwd,
            self.model_provider,
            self.preferred_auth_method,
            self.model,
            self.reasoning_effort,
            self.service_tier,
            self.tmux,
            self.start_ts,
            self.resume_session_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "thread_id": self.thread_id,
            "agent_backend": self.agent_backend,
            "log_path": self.log_path,
            "cwd": self.cwd,
            "model_provider": self.model_provider,
            "preferred_auth_method": self.preferred_auth_method,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "service_tier": self.service_tier,
            "tmux": bool(self.tmux),
            "saved_at": float(self.saved_at),
            "start_ts": float(self.start_ts),
            "resume_session_id": self.resume_session_id,
            "restoring": self.restoring.to_dict() if self.restoring is not None else None,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "SnapshotEntry":
        if not isinstance(raw, dict):
            raise SnapshotCorrupt("snapshot entry must be an object")
        saved_at = _float_or_none(raw.get("saved_at"))
        start_ts = _float_or_none(raw.get("start_ts"))
        restoring_raw = raw.get("restoring")
        return cls(
            session_id=_required_text(raw, "session_id"),
            thread_id=_required_text(raw, "thread_id"),
            agent_backend=_required_text(raw, "agent_backend"),
            log_path=_required_text(raw, "log_path"),
            cwd=_required_text(raw, "cwd"),
            model_provider=_clean_text(raw.get("model_provider")),
            preferred_auth_method=_clean_text(raw.get("preferred_auth_method")),
            model=_clean_text(raw.get("model")),
            reasoning_effort=_clean_text(raw.get("reasoning_effort")),
            service_tier=_clean_text(raw.get("service_tier")),
            tmux=bool(raw.get("tmux")),
            saved_at=saved_at if saved_at is not None else 0.0,
            start_ts=start_ts if start_ts is not None else 0.0,
            resume_session_id=_clean_text(raw.get("resume_session_id")),
            restoring=RestoringMark.from_dict(restoring_raw) if restoring_raw is not None else None,
        )


@dataclass(frozen=True)
class IdentityRemap:
    """An upsert proved that ``new_session_id`` continues the transcript ``old_session_id`` held."""

    old_session_id: str
    new_session_id: str
    old_thread_id: str
    new_thread_id: str


@dataclass(frozen=True)
class UpsertResult:
    changed: list[EntryKey]
    remaps: list[IdentityRemap]


@dataclass(frozen=True)
class SnapshotState:
    saved_at: float | None
    entries: dict[EntryKey, SnapshotEntry]
    remap_intents: tuple[IdentityRemap, ...] = ()

    @classmethod
    def empty(cls) -> "SnapshotState":
        return cls(saved_at=None, entries={})

    def entries_matching(self, identifier: str) -> list[SnapshotEntry]:
        """Entries known under ``identifier`` as a registry session id or a thread id."""
        return [entry for entry in self.entries.values() if entry.session_id == identifier or entry.thread_id == identifier]


def snapshot_path(app_dir: Path) -> Path:
    return app_dir / SNAPSHOT_FILENAME


def _resolved_log(path: str) -> str:
    try:
        return str(Path(path).expanduser().resolve(strict=False))
    except OSError:
        return path


def _continues(stored: SnapshotEntry, incoming: SnapshotEntry) -> bool:
    """Rule (c): ``incoming`` continues ``stored``'s transcript under a different thread id."""
    if stored.agent_backend != incoming.agent_backend or stored.key == incoming.key:
        return False
    if incoming.resume_session_id is not None and incoming.resume_session_id == stored.thread_id:
        return True
    return _resolved_log(stored.log_path) == _resolved_log(incoming.log_path)


def _same_incarnation(stored: SnapshotEntry, incoming: SnapshotEntry) -> bool:
    """Rule (b): one broker process, identified by registry id plus start time."""
    return stored.key != incoming.key and stored.session_id == incoming.session_id and stored.start_ts == incoming.start_ts


def _parse_state(obj: Any) -> SnapshotState:
    if not isinstance(obj, dict):
        raise SnapshotCorrupt("snapshot must be a JSON object")
    version = obj.get("version")
    if version != SNAPSHOT_VERSION:
        raise SnapshotCorrupt(f"unsupported snapshot version {version!r}")
    raw_entries = obj.get("entries")
    if not isinstance(raw_entries, list):
        raise SnapshotCorrupt("snapshot entries must be a list")
    entries: dict[EntryKey, SnapshotEntry] = {}
    for raw in raw_entries:
        entry = SnapshotEntry.from_dict(raw)
        entries[entry.key] = entry
    intents = []
    raw_intents = obj.get("remap_intents", [])
    if not isinstance(raw_intents, list):
        raise SnapshotCorrupt("snapshot remap intents must be a list")
    for raw in raw_intents:
        if not isinstance(raw, dict):
            raise SnapshotCorrupt("snapshot remap intent must be an object")
        intents.append(IdentityRemap(**{key: _required_text(raw, key) for key in (
            "old_session_id", "new_session_id", "old_thread_id", "new_thread_id")}))
    return SnapshotState(saved_at=_float_or_none(obj.get("saved_at")), entries=entries, remap_intents=tuple(intents))


def _serialize_state(state: SnapshotState) -> dict[str, Any]:
    payload = {
        "version": SNAPSHOT_VERSION,
        "saved_at": state.saved_at,
        "entries": [entry.to_dict() for entry in state.entries.values()],
    }
    if state.remap_intents:
        payload["remap_intents"] = [dict(
            old_session_id=remap.old_session_id, new_session_id=remap.new_session_id,
            old_thread_id=remap.old_thread_id, new_thread_id=remap.new_thread_id,
        ) for remap in state.remap_intents]
    return payload


def _write_private_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    data = (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


class SessionSnapshotStore:
    """Read-modify-write store for the snapshot file. All mutations serialize on one lock.

    Reads are cached on the file's ``(mtime_ns, size)`` so frequent callers
    (the per-second queue sweep asks which ids to protect) re-parse only after
    a write.
    """

    def __init__(self, path: Path, *, now: Callable[[], float] | None = None) -> None:
        self.path = path
        self._now = now or time.time
        self._lock = threading.RLock()
        self._cache: tuple[tuple[int, int], SnapshotState] | None = None

    def read(self) -> SnapshotState:
        with self._lock:
            try:
                stat = self.path.stat()
            except FileNotFoundError:
                self._cache = None
                return SnapshotState.empty()
            stamp = (int(stat.st_mtime_ns), int(stat.st_size))
            if self._cache is not None and self._cache[0] == stamp:
                return self._cache[1]
            raw = self.path.read_text(encoding="utf-8")
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise SnapshotCorrupt(f"snapshot is not valid JSON: {exc}") from exc
            state = _parse_state(obj)
            self._cache = (stamp, state)
            return state

    def exists(self) -> bool:
        return self.path.exists()

    def _write(self, previous: SnapshotState, state: SnapshotState, *, allow_empty: bool) -> None:
        if not state.entries and previous.entries and not allow_empty:
            raise SnapshotWriteRefused("refusing to persist an empty session snapshot over a non-empty one")
        _write_private_json(self.path, _serialize_state(state))
        self._cache = None

    def upsert(self, entries: Iterable[SnapshotEntry]) -> UpsertResult:
        """Add or update entries per the module's rules (a)–(c); reports changed keys and identity remaps."""
        incoming = list(entries)
        with self._lock:
            previous = self.read()
            merged = dict(previous.entries)
            changed: list[EntryKey] = []
            remaps: list[IdentityRemap] = []
            for entry in incoming:
                current = merged.get(entry.key)
                continued = [key for key, stored in merged.items() if _continues(stored, entry)]
                moved_on = [key for key, stored in merged.items() if _same_incarnation(stored, entry)]
                for key in continued:
                    stored = merged.pop(key)
                    if stored.session_id != entry.session_id or stored.thread_id != entry.thread_id:
                        remaps.append(
                            IdentityRemap(
                                old_session_id=stored.session_id,
                                new_session_id=entry.session_id,
                                old_thread_id=stored.thread_id,
                                new_thread_id=entry.thread_id,
                            )
                        )
                for key in moved_on:
                    merged.pop(key, None)
                if current is not None and current.session_id != entry.session_id:
                    remaps.append(
                        IdentityRemap(
                            old_session_id=current.session_id,
                            new_session_id=entry.session_id,
                            old_thread_id=current.thread_id,
                            new_thread_id=entry.thread_id,
                        )
                    )
                unchanged = (
                    current is not None
                    and current.launch_fields() == entry.launch_fields()
                    and current.restoring is None
                    and not continued
                    and not moved_on
                )
                if unchanged:
                    continue
                merged[entry.key] = replace(entry, saved_at=self._now(), restoring=None)
                changed.append(entry.key)
            if not changed:
                return UpsertResult(changed=[], remaps=[])
            intents = tuple(dict.fromkeys((*previous.remap_intents, *remaps)))
            state = SnapshotState(saved_at=self._now(), entries=merged, remap_intents=intents)
            self._write(previous, state, allow_empty=False)
            return UpsertResult(changed=changed, remaps=remaps)

    def complete_remap(self, remap: IdentityRemap) -> None:
        """Acknowledge only after every per-session state writer succeeded."""
        with self._lock:
            previous = self.read()
            intents = tuple(item for item in previous.remap_intents if item != remap)
            if intents != previous.remap_intents:
                self._write(previous, replace(previous, remap_intents=intents), allow_empty=True)

    def remove(self, keys: Iterable[EntryKey]) -> list[EntryKey]:
        wanted = [key for key in keys if isinstance(key, tuple) and len(key) == 2]
        with self._lock:
            previous = self.read()
            removed = [key for key in wanted if key in previous.entries]
            if not removed:
                return []
            merged = {key: entry for key, entry in previous.entries.items() if key not in removed}
            self._write(previous, replace(previous, entries=merged), allow_empty=True)
            return removed

    def remove_matching(self, identifier: str) -> list[EntryKey]:
        """Remove every entry known under ``identifier`` (registry session id or thread id)."""
        with self._lock:
            return self.remove(entry.key for entry in self.read().entries_matching(identifier))

    def remove_incarnation(self, session_id: str, start_ts: float) -> list[EntryKey]:
        """Remove the entries of one broker incarnation (clean exit of that exact process)."""
        with self._lock:
            state = self.read()
            keys = [entry.key for entry in state.entries.values() if entry.session_id == session_id and entry.start_ts == float(start_ts)]
            return self.remove(keys)

    def mark_restoring(self, key: EntryKey, *, expected_session_id: str, mark: RestoringMark) -> bool:
        """Stamp a restoring mark; refused when the entry is gone or already superseded (its registry id moved)."""
        with self._lock:
            previous = self.read()
            current = previous.entries.get(key)
            if current is None or current.session_id != expected_session_id:
                return False
            merged = dict(previous.entries)
            merged[key] = replace(current, restoring=mark)
            self._write(previous, replace(previous, entries=merged), allow_empty=True)
            return True
