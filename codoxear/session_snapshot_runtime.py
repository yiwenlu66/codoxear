"""Auto-save triggers, liveness checks, and identity remap for the session snapshot.

Trigger model (all server-side, none in a request handler's critical path):

* ``observe_registrations`` — called after every discovery pass with the
  live registrations. A registration's launch-relevant signature (thread,
  backend, cwd, log, tmux transport, provider/auth/model/effort/tier, broker
  start time, resume target) is diffed against the last one seen for that
  registry id; a change marks the id dirty and arms one coalescing debounce
  timer. The flush re-reads each dirty session's sidecar and reconciles its
  *raw* launch settings through ``snapshot_run_settings`` (live bridge
  values > sidecar launch values > log replay) before upserting.
* ``session_sock_vanished`` — the prune coordinator's "registered session
  whose control socket file is gone" branch. A broker unlinks its socket and
  sidecar only in its own teardown after a zero exit status, so a vanished
  socket is the clean-exit signal; a crash, a signal death, or a SIGHUP
  leaves the files behind and produces a lost tombstone instead, which keeps
  the entry. Removal is scoped to that broker incarnation (registry id plus
  start time) so a reused pid cannot erase a dead session's entry.
* ``session_deleted`` — explicit UI/API delete. Removes the entries of the
  deleted session's transcript (bound thread or declared resume target) and
  of its broker incarnation.
* ``dismiss`` — explicit removal by registry session id or thread id
  (``DELETE /api/sessions/snapshot/<id>``), for entries that no longer have
  any row to delete from.

An upsert that supersedes an entry (store rules (a)/(c)) records durable
``remap_intents`` alongside the new identity. ``remap_session_identity`` moves
attachments, queue, draft, unattended config, alias, and sidebar metadata and
rewrites dependency links. Only successful persistence of every participant
acknowledges the intent. An unchanged subsequent upsert, including one from a
fresh process, replays unacknowledged intents; attachment transfers and the
other move operations are idempotent. A restore only stamps ``restoring``;
the replacement's own upsert proves continuity and creates the intent.

Flush robustness. A flush that cannot build an entry (torn sidecar read) or
cannot write the store re-marks those ids dirty and re-arms the timer at
``FLUSH_RETRY_MULTIPLIER`` × debounce, giving up (with a log line) after
``FLUSH_MAX_CONSECUTIVE_FAILURES``; ids left dirty ride along with the next
scheduled flush. Each entry is re-validated (still registered, not lost,
broker alive, same start time) immediately before the upsert. Residual
window: a removal that lands between that re-validation and the store write
is overwritten by the upsert, and the session reappears as restorable until
its next removal event.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
import threading
from typing import Any, Callable, Iterable, MutableMapping, Protocol

from .session_discovery import DiscoveryRegistration
from .session_model import Session
from .session_snapshot import EntryKey
from .session_snapshot import IdentityRemap
from .session_snapshot import RestoringMark
from .session_snapshot import SessionSnapshotStore
from .session_snapshot import SnapshotCorrupt
from .session_snapshot import SnapshotEntry
from .session_snapshot import SnapshotState
from .session_snapshot import SnapshotWriteRefused


FLUSH_RETRY_MULTIPLIER = 4.0
FLUSH_MAX_CONSECUTIVE_FAILURES = 3


class _TimerLike(Protocol):
    def start(self) -> None: ...

    def cancel(self) -> None: ...


def default_timer_factory(delay_seconds: float, fn: Callable[[], None]) -> _TimerLike:
    timer = threading.Timer(delay_seconds, fn)
    timer.daemon = True
    return timer


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    out = value.strip()
    return out or None


def _resolved(path: Path | str | None) -> str | None:
    if path is None:
        return None
    try:
        return str(Path(path).expanduser().resolve(strict=False))
    except OSError:
        return str(path)


def registration_signature(registration: DiscoveryRegistration) -> tuple[Any, ...]:
    return (
        registration.thread_id,
        registration.agent_backend,
        registration.cwd,
        str(registration.log_path) if registration.log_path is not None else None,
        registration.transport == "tmux",
        registration.model_provider,
        registration.preferred_auth_method,
        registration.model,
        registration.reasoning_effort,
        registration.service_tier,
        float(registration.start_ts),
        registration.resume_session_id,
    )


@dataclass(frozen=True)
class RemapResult:
    queues_moved: bool
    draft_moved: bool
    unattended_moved: bool
    alias_moved: bool = False
    sidebar_meta_moved: bool = False
    dependencies_rewritten: int = 0
    skipped_reused_id: bool = False


@dataclass
class SessionSnapshotCoordinator:
    store: SessionSnapshotStore
    lock: Any
    sessions: Callable[[], MutableMapping[str, Session]]
    session_store: Callable[[], Any]
    save_queues: Callable[[], None]
    save_drafts: Callable[[], None]
    save_unattended: Callable[[], None]
    save_aliases: Callable[[], None]
    save_sidebar_meta: Callable[[], None]
    read_sidecar_metadata: Callable[[Path], dict[str, Any]]
    snapshot_run_settings: Callable[[dict[str, Any], Path | None, str], tuple[str | None, str | None, str | None, str | None]]
    pid_alive: Callable[[int], bool]
    now: Callable[[], float]
    sock_dir: Path
    debounce_seconds: float = 2.0
    timer_factory: Callable[[float, Callable[[], None]], _TimerLike] = default_timer_factory
    stderr: Any = sys.stderr
    _state_lock: threading.Lock = field(init=False, default_factory=threading.Lock, repr=False)
    _signatures: dict[str, tuple[Any, ...]] = field(init=False, default_factory=dict, repr=False)
    _dirty: set[str] = field(init=False, default_factory=set, repr=False)
    _timer: _TimerLike | None = field(init=False, default=None, repr=False)
    _flush_failures: int = field(init=False, default=0, repr=False)
    _protect_warned: bool = field(init=False, default=False, repr=False)
    # Serializes restore runs: two concurrent restores must not both spawn one entry.
    restore_lock: threading.Lock = field(init=False, default_factory=threading.Lock, repr=False)
    _remap_lock: threading.Lock = field(init=False, default_factory=threading.Lock, repr=False)

    # ------------------------------------------------------------------ #
    # Discovery trigger
    # ------------------------------------------------------------------ #

    def observe_registrations(self, registrations: Iterable[DiscoveryRegistration]) -> None:
        with self._state_lock:
            for registration in registrations:
                if registration.log_path is None:
                    # Nothing to resume yet; the signature changes once the
                    # transcript binds, which marks the session dirty then.
                    continue
                signature = registration_signature(registration)
                if self._signatures.get(registration.session_id) != signature:
                    self._signatures[registration.session_id] = signature
                    self._dirty.add(registration.session_id)
            if self._dirty:
                self._schedule_flush_locked(self.debounce_seconds)

    def _schedule_flush_locked(self, delay_seconds: float) -> None:
        if self._timer is not None:
            return
        timer = self.timer_factory(delay_seconds, self._flush)
        self._timer = timer
        timer.start()

    def _flush(self) -> None:
        with self._state_lock:
            dirty = set(self._dirty)
            self._dirty.clear()
            self._timer = None
        if not dirty:
            return
        failed: set[str] = set()
        entries: list[SnapshotEntry] = []
        for session in self._live_sessions():
            if session.session_id not in dirty:
                continue
            try:
                entry = self.entry_for_session(session)
            except (OSError, ValueError) as exc:
                self._log(f"session snapshot: cannot build entry for {session.session_id}: {type(exc).__name__}: {exc}")
                failed.add(session.session_id)
                continue
            if entry is not None:
                entries.append(entry)
        entries = [entry for entry in entries if self._still_live(entry.session_id, entry.start_ts)]
        failed |= self._upsert(entries)
        self._after_flush(failed)

    def _after_flush(self, failed: set[str]) -> None:
        with self._state_lock:
            if not failed:
                self._flush_failures = 0
                return
            self._flush_failures += 1
            self._dirty |= failed
            if self._flush_failures > FLUSH_MAX_CONSECUTIVE_FAILURES:
                self._log(
                    f"session snapshot: giving up after {self._flush_failures - 1} consecutive flush failures; "
                    f"{sorted(failed)} stay dirty until the next scheduled flush"
                )
                return
            self._schedule_flush_locked(self.debounce_seconds * FLUSH_RETRY_MULTIPLIER)

    def _upsert(self, entries: list[SnapshotEntry]) -> set[str]:
        """Publish identities with durable remap intent, then replay transfers."""
        with self._remap_lock:
            try:
                self.store.upsert(entries)
            except (SnapshotCorrupt, SnapshotWriteRefused, OSError) as exc:
                self._log(f"session snapshot upsert failed: {type(exc).__name__}: {exc}")
                return {entry.session_id for entry in entries}
            return self._replay_remaps()

    def _replay_remaps(self) -> set[str]:
        failed = set()
        for remap in self.store.read().remap_intents:
            try:
                self.remap_session_identity(remap)
                self.store.complete_remap(remap)
            except (OSError, ValueError) as exc:
                self._log(f"session snapshot remap failed: {type(exc).__name__}: {exc}")
                failed.add(remap.new_session_id)
                # Preserve causal order if another intent continues this one.
                break
        return failed

    def _log(self, message: str) -> None:
        try:
            self.stderr.write(f"error: {message}\n")
            self.stderr.flush()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Entry construction
    # ------------------------------------------------------------------ #

    def _live_sessions(self) -> list[Session]:
        with self.lock:
            values = list(self.sessions().values())
        return [s for s in values if isinstance(s, Session) and not s.lost and self.pid_alive(int(s.broker_pid))]

    def _still_live(self, session_id: str, start_ts: float) -> bool:
        with self.lock:
            current = self.sessions().get(session_id)
        return (
            isinstance(current, Session)
            and not current.lost
            and float(current.start_ts) == float(start_ts)
            and self.pid_alive(int(current.broker_pid))
        )

    def _entries_for_sessions(self, sessions: Iterable[Session]) -> list[SnapshotEntry]:
        out: list[SnapshotEntry] = []
        for session in sessions:
            try:
                entry = self.entry_for_session(session)
            except (OSError, ValueError) as exc:
                self._log(f"session snapshot: cannot build entry for {session.session_id}: {type(exc).__name__}: {exc}")
                continue
            if entry is not None:
                out.append(entry)
        return out

    def entry_for_session(self, session: Session) -> SnapshotEntry | None:
        """Build the snapshot entry for a live session, or None when it is not restorable yet.

        A session without a bound log has nothing to resume; it becomes an
        entry once discovery binds its transcript. Raises ``OSError``/
        ``ValueError`` when the sidecar cannot be read, so a flush can retry.
        """
        if session.log_path is None:
            return None
        meta_path = session.sock_path.with_suffix(".json")
        meta = self.read_sidecar_metadata(meta_path)
        thread_id = _clean_text(meta.get("session_id")) or _clean_text(session.thread_id)
        if thread_id is None:
            return None
        backend = session.agent_backend
        model_provider, preferred_auth_method, model, reasoning_effort = self.snapshot_run_settings(meta, session.log_path, backend)
        return SnapshotEntry(
            session_id=session.session_id,
            thread_id=thread_id,
            agent_backend=backend,
            log_path=str(session.log_path),
            cwd=str(session.cwd),
            model_provider=model_provider,
            preferred_auth_method=preferred_auth_method,
            model=model,
            reasoning_effort=reasoning_effort,
            service_tier=_clean_text(meta.get("service_tier")) if backend == "codex" else None,
            tmux=session.transport == "tmux",
            saved_at=self.now(),
            start_ts=float(session.start_ts),
            resume_session_id=_clean_text(meta.get("resume_session_id")) or _clean_text(session.resume_session_id),
        )

    # ------------------------------------------------------------------ #
    # Manual save / read / status
    # ------------------------------------------------------------------ #

    def save_now(self) -> SnapshotState:
        """Force-upsert every live session immediately (manual save)."""
        with self._state_lock:
            timer = self._timer
            self._timer = None
            self._dirty.clear()
        if timer is not None:
            timer.cancel()
        entries = self._entries_for_sessions(self._live_sessions())
        with self._remap_lock:
            self.store.upsert(entries)
            failed = self._replay_remaps()
            if failed:
                self._after_flush(failed)
                raise OSError("session state transfer incomplete; remap retained for retry")
            return self.store.read()

    def read(self) -> SnapshotState:
        return self.store.read()

    def live_session_id_for_entry(self, entry: SnapshotEntry) -> str | None:
        """Registry id of a live session that continues the entry's transcript, else None.

        A registry session continues the entry when it is the same broker
        incarnation, has bound the entry's thread, declares that thread as
        its resume target (a Codex/Claude Code resume still binding its log),
        or holds the entry's log. Discovery can skip a live broker whose
        state call timed out, so the entry's own sidecar on disk is consulted
        too: a live broker pid there on the same transcript counts as live.
        """
        with self.lock:
            values = list(self.sessions().values())
        target_log = _resolved(entry.log_path)
        for session in values:
            if not isinstance(session, Session):
                continue
            continues = (
                (session.session_id == entry.session_id and float(session.start_ts) == float(entry.start_ts))
                or session.thread_id == entry.thread_id
                or session.resume_session_id == entry.thread_id
                or (session.log_path is not None and _resolved(session.log_path) == target_log)
            )
            if continues and (self.pid_alive(int(session.broker_pid)) or self.pid_alive(int(session.codex_pid))):
                return session.session_id
        meta_path = self.sock_dir / f"{entry.session_id}.json"
        try:
            meta = self.read_sidecar_metadata(meta_path) if meta_path.exists() else None
        except (OSError, ValueError):
            meta = None
        if isinstance(meta, dict):
            broker_pid = meta.get("broker_pid")
            same_transcript = (
                _clean_text(meta.get("session_id")) == entry.thread_id
                or _clean_text(meta.get("resume_session_id")) == entry.thread_id
                or _resolved(_clean_text(meta.get("log_path"))) == target_log
                or (isinstance(meta.get("start_ts"), (int, float)) and float(meta["start_ts"]) == float(entry.start_ts))
            )
            if same_transcript and isinstance(broker_pid, int) and not isinstance(broker_pid, bool) and self.pid_alive(broker_pid):
                return entry.session_id
        return None

    def restoring_active(self, entry: SnapshotEntry) -> bool:
        """The entry has an unexpired restoring mark whose launched broker (when known) is still alive."""
        mark = entry.restoring
        if mark is None or not mark.active(self.now()):
            return False
        return mark.broker_pid is None or self.pid_alive(int(mark.broker_pid))

    def public_annotation(self, entry: SnapshotEntry) -> dict[str, Any]:
        """``live``: the entry cannot be restored right now (running, or a restore is in flight)."""
        restoring = self.restoring_active(entry)
        live_id = self.live_session_id_for_entry(entry)
        return {
            "live": live_id is not None or restoring,
            "live_session_id": live_id,
            "restoring": entry.restoring.to_dict() if restoring and entry.restoring is not None else None,
        }

    def queue_protected_session_ids(self) -> set[str] | None:
        """Registry ids the queue sweep must not drop; None when the snapshot is unreadable (drop nothing)."""
        try:
            state = self.store.read()
        except SnapshotCorrupt as exc:
            if not self._protect_warned:
                self._log(f"session snapshot unreadable; queue sweep keeps every orphaned queue until it is repaired: {exc}")
                self._protect_warned = True
            return None
        self._protect_warned = False
        # A published snapshot may name only the destination while its state
        # still lives at a source/intermediate id. Keep every endpoint until
        # the complete transfer is acknowledged, including deferred intents.
        protected = {entry.session_id for entry in state.entries.values()}
        for remap in state.remap_intents:
            protected.update((remap.old_session_id, remap.new_session_id))
        return protected

    # ------------------------------------------------------------------ #
    # Removal triggers
    # ------------------------------------------------------------------ #

    def session_sock_vanished(self, session: Session) -> None:
        with self._state_lock:
            self._signatures.pop(session.session_id, None)
            self._dirty.discard(session.session_id)
        try:
            self.store.remove_incarnation(session.session_id, float(session.start_ts))
        except (SnapshotCorrupt, OSError) as exc:
            self._log(f"session snapshot removal on clean broker exit failed for {session.session_id}: {type(exc).__name__}: {exc}")

    def session_deleted(self, session_id: str, session: Session | None = None) -> None:
        with self._state_lock:
            self._signatures.pop(session_id, None)
            self._dirty.discard(session_id)
        try:
            state = self.store.read()
            keys: list[EntryKey] = []
            for entry in state.entries.values():
                if session is None:
                    matches = entry.session_id == session_id
                else:
                    matches = (
                        entry.thread_id == session.thread_id
                        or (session.resume_session_id is not None and entry.thread_id == session.resume_session_id)
                        or (entry.session_id == session_id and entry.start_ts == float(session.start_ts))
                    )
                if matches:
                    keys.append(entry.key)
            self.store.remove(keys)
        except (SnapshotCorrupt, OSError) as exc:
            self._log(f"session snapshot removal on session delete failed for {session_id}: {type(exc).__name__}: {exc}")

    def dismiss(self, identifier: str) -> bool:
        """Remove every entry known under a registry session id or thread id."""
        return bool(self.store.remove_matching(identifier))

    def drop_entry(self, entry: SnapshotEntry) -> bool:
        return bool(self.store.remove([entry.key]))

    # ------------------------------------------------------------------ #
    # Restore marking and identity remap
    # ------------------------------------------------------------------ #

    def session_id_for_broker_pid(self, broker_pid: int) -> str | None:
        with self.lock:
            values = list(self.sessions().values())
        for session in values:
            if int(session.broker_pid) == int(broker_pid):
                return session.session_id
        return None

    def mark_restoring(self, entry: SnapshotEntry, *, broker_pid: int | None) -> bool:
        """Stamp the entry restoring; a no-op when its replacement already superseded it."""
        return self.store.mark_restoring(
            entry.key,
            expected_session_id=entry.session_id,
            mark=RestoringMark(broker_pid=broker_pid, launched_at=self.now()),
        )

    def _require_remap_destination_locked(self, remap: IdentityRemap) -> None:
        """A registry id alone cannot authorize moving state or queued prompts."""
        if remap.new_agent_backend is None or remap.new_start_ts is None or remap.new_start_ts <= 0:
            raise ValueError("remap destination identity unavailable; transfer deferred")
        current = self.sessions().get(remap.new_session_id)
        if not (
            isinstance(current, Session)
            and not current.lost
            and current.agent_backend == remap.new_agent_backend
            and current.thread_id == remap.new_thread_id
            and float(current.start_ts) == remap.new_start_ts
            and self.pid_alive(int(current.broker_pid))
        ):
            raise ValueError("remap destination no longer matches intended transcript and incarnation; transfer deferred")

    def _old_id_reused_locked(self, remap: IdentityRemap) -> bool:
        """The old registry id currently names a live session on some other transcript (pid reuse)."""
        current = self.sessions().get(remap.old_session_id)
        if not isinstance(current, Session):
            return False
        if current.thread_id == remap.old_thread_id or current.resume_session_id == remap.old_thread_id:
            return False
        return self.pid_alive(int(current.broker_pid)) or self.pid_alive(int(current.codex_pid))

    def remap_session_identity(self, remap: IdentityRemap) -> RemapResult:
        """Move per-session state from the old registry id to the new one (move-only-if-absent, queues merge).

        Unattended config is normally keyed ``thread:<id>`` (resume stable):
        a legacy session-id key moves to the thread-scoped key the new
        session reads, and a thread-scoped key follows a thread id change.
        """
        if remap.old_session_id == remap.new_session_id and remap.old_thread_id == remap.new_thread_id:
            return RemapResult(queues_moved=False, draft_moved=False, unattended_moved=False)
        old, new = remap.old_session_id, remap.new_session_id
        queues_moved = draft_moved = unattended_moved = alias_moved = sidebar_moved = False
        dependencies = 0
        with self.lock:
            self._require_remap_destination_locked(remap)
            if self._old_id_reused_locked(remap):
                self._log(
                    f"session snapshot: not moving state from {remap.old_session_id} to {remap.new_session_id}; "
                    f"the old registry id now belongs to a live session on another transcript"
                )
                return RemapResult(queues_moved=False, draft_moved=False, unattended_moved=False, skipped_reused_id=True)
            store = self.session_store()
            if old != new:
                store.remap_attachment_resource(old, new)
                current = self.sessions().get(new)
                if current:
                    current.pending_attachment = new in store.pending_attachment_ids
                queues = store.queues
                old_queue = queues.pop(old, None)
                if isinstance(old_queue, list) and old_queue:
                    existing = queues.get(new)
                    queues[new] = list(old_queue) + (list(existing) if isinstance(existing, list) else [])
                    queues_moved = True
                drafts = store.drafts
                old_draft = drafts.pop(old, None)
                if isinstance(old_draft, dict) and new not in drafts:
                    drafts[new] = old_draft
                    draft_moved = True
                aliases = store.aliases
                old_alias = aliases.pop(old, None)
                if isinstance(old_alias, str) and old_alias and new not in aliases:
                    aliases[new] = old_alias
                    alias_moved = True
                sidebar_meta = store.sidebar_meta
                old_meta = sidebar_meta.pop(old, None)
                if isinstance(old_meta, dict) and new not in sidebar_meta:
                    sidebar_meta[new] = old_meta
                    sidebar_moved = True
                for meta in sidebar_meta.values():
                    if isinstance(meta, dict) and meta.get("dependency_session_id") == old:
                        meta["dependency_session_id"] = new
                        dependencies += 1
            unattended = store.unattended
            new_key = f"thread:{remap.new_thread_id}"
            for legacy_key in (old, f"thread:{remap.old_thread_id}"):
                if legacy_key == new_key:
                    continue
                legacy = unattended.pop(legacy_key, None)
                if isinstance(legacy, dict) and new_key not in unattended:
                    unattended[new_key] = legacy
                    unattended_moved = True
        # Replay may find in-memory state already moved by a failed save.
        # Re-save every participant before acknowledging the durable intent;
        # after restart each atomic store independently resumes from disk.
        if old != new:
            self.save_queues()
            self.save_drafts()
        self.save_unattended()
        if old != new:
            self.save_aliases()
            self.save_sidebar_meta()
        return RemapResult(
            queues_moved=queues_moved,
            draft_moved=draft_moved,
            unattended_moved=unattended_moved,
            alias_moved=alias_moved,
            sidebar_meta_moved=sidebar_moved,
            dependencies_rewritten=dependencies,
        )
