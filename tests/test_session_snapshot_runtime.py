"""Coordinator tests: debounced auto-save, remap-at-supersede, liveness, flush retries."""

from __future__ import annotations

import io
import json
import threading
from pathlib import Path
from types import SimpleNamespace

from codoxear.session_discovery import DiscoveryRegistration
from codoxear.session_model import Session
from codoxear.session_snapshot import RESTORING_TTL_SECONDS
from codoxear.session_snapshot import RestoringMark
from codoxear.session_snapshot import SessionSnapshotStore
from codoxear.session_snapshot import SnapshotEntry
from codoxear.session_snapshot import snapshot_path
from codoxear.session_snapshot_runtime import FLUSH_RETRY_MULTIPLIER
from codoxear.session_snapshot_runtime import SessionSnapshotCoordinator
from codoxear.sidecar_metadata import read_metadata


class _FakeTimer:
    def __init__(self, delay: float, fn) -> None:
        self.delay = delay
        self.fn = fn
        self.started = False
        self.cancelled = False

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
        self.cancelled = True

    def fire(self) -> None:
        self.fn()


class _Harness:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.sock_dir = tmp_path / "socks"
        self.sock_dir.mkdir(exist_ok=True)
        self.sessions: dict[str, Session] = {}
        self.alive: set[int] = set()
        self.timers: list[_FakeTimer] = []
        self.saved: list[str] = []
        self.clock = [50.0]
        # This harness exercises non-attachment stores. Attachment transfer and
        # durable revision behavior use the real SessionStore in
        # test_attachment_resource.py.
        self.session_store = SimpleNamespace(
            queues={}, drafts={}, unattended={}, aliases={}, sidebar_meta={},
            pending_attachment_ids=set(), remap_attachment_resource=lambda old, new: None,
        )
        self.stderr = io.StringIO()
        self.store = SessionSnapshotStore(snapshot_path(tmp_path), now=lambda: self.clock[0])
        self.lock = threading.Lock()

        def timer_factory(delay, fn):
            timer = _FakeTimer(delay, fn)
            self.timers.append(timer)
            return timer

        def snapshot_run_settings(meta, log_path, agent_backend):
            live = meta.get("live_run_settings") or {}
            return (
                live.get("model_provider") or meta.get("model_provider"),
                meta.get("preferred_auth_method"),
                live.get("model") or meta.get("model"),
                live.get("reasoning_effort") or meta.get("reasoning_effort"),
            )

        self.coordinator = SessionSnapshotCoordinator(
            store=self.store,
            lock=self.lock,
            sessions=lambda: self.sessions,
            session_store=lambda: self.session_store,
            save_queues=lambda: self.saved.append("queues"),
            save_drafts=lambda: self.saved.append("drafts"),
            save_unattended=lambda: self.saved.append("unattended"),
            save_aliases=lambda: self.saved.append("aliases"),
            save_sidebar_meta=lambda: self.saved.append("sidebar_meta"),
            read_sidecar_metadata=lambda meta_path: read_metadata(meta_path, sock=meta_path.with_suffix(".sock")),
            snapshot_run_settings=snapshot_run_settings,
            pid_alive=lambda pid: int(pid) in self.alive,
            now=lambda: self.clock[0],
            sock_dir=self.sock_dir,
            debounce_seconds=2.0,
            timer_factory=timer_factory,
            stderr=self.stderr,
        )

    def add_live_session(
        self,
        session_id: str,
        *,
        broker_pid: int,
        thread_id: str = "thread-1",
        start_ts: float = 1.0,
        log_path: Path | None = None,
        transport: str | None = "tmux",
        meta: dict | None = None,
    ) -> tuple[Session, DiscoveryRegistration]:
        if log_path is None:
            log_path = self.tmp_path / f"{thread_id}.jsonl"
            log_path.write_text("{}\n", encoding="utf-8")
        sock = self.sock_dir / f"{session_id}.sock"
        sock.touch()
        sidecar = {
            "session_id": thread_id,
            "broker_pid": broker_pid,
            "codex_pid": broker_pid + 1,
            "start_ts": start_ts,
            "cwd": "/repo",
            "log_path": str(log_path),
            "agent_backend": "pi",
            "model_provider": "dexgem",
            "model": "kimi",
            "reasoning_effort": "high",
            "service_tier": None,
            "transport": transport,
            "resume_session_id": None,
        }
        sidecar.update(meta or {})
        sock.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")
        session = Session(
            session_id=session_id,
            thread_id=sidecar["session_id"] if sidecar.get("session_id") else thread_id,
            broker_pid=broker_pid,
            codex_pid=broker_pid + 1,
            agent_backend="pi",
            owned=True,
            start_ts=start_ts,
            cwd="/repo",
            log_path=log_path,
            sock_path=sock,
            transport=transport,
            resume_session_id=sidecar.get("resume_session_id"),
        )
        self.sessions[session_id] = session
        self.alive.add(broker_pid)
        return session, self.registration(session, sidecar)

    @staticmethod
    def registration(session: Session, sidecar: dict) -> DiscoveryRegistration:
        live = sidecar.get("live_run_settings") or {}
        return DiscoveryRegistration(
            session_id=session.session_id,
            thread_id=session.thread_id,
            broker_pid=session.broker_pid,
            codex_pid=session.codex_pid,
            agent_backend=session.agent_backend,
            owned=True,
            transport=session.transport,
            start_ts=session.start_ts,
            cwd=session.cwd,
            log_path=session.log_path,
            sock_path=session.sock_path,
            busy=False,
            queue_len=0,
            token=None,
            meta_log_off=0,
            model_provider=live.get("model_provider") or sidecar.get("model_provider"),
            preferred_auth_method=None,
            model=live.get("model") or sidecar.get("model"),
            reasoning_effort=live.get("reasoning_effort") or sidecar.get("reasoning_effort"),
            service_tier=None,
            tmux_session="codoxear" if session.transport == "tmux" else None,
            tmux_window=None,
            launch_id=None,
            spawn_nonce=None,
            resume_session_id=sidecar.get("resume_session_id"),
            sync_send_supported=True,
            key_write_errors_supported=True,
            interrupted_idle=False,
        )

    def flush(self) -> None:
        self.timers[-1].fire()


def test_discovery_upserts_new_live_session_after_one_coalesced_debounce(tmp_path: Path) -> None:
    h = _Harness(tmp_path)
    _session, registration = h.add_live_session("broker-10", broker_pid=10)

    h.coordinator.observe_registrations([registration])
    h.coordinator.observe_registrations([registration])
    h.coordinator.observe_registrations([registration])

    assert len(h.timers) == 1
    assert h.timers[0].delay == 2.0
    assert h.timers[0].started is True
    assert not h.store.exists()

    h.flush()
    state = h.store.read()
    entry = state.entries[("pi", "thread-1")]
    assert entry.session_id == "broker-10"
    assert entry.agent_backend == "pi"
    assert entry.cwd == "/repo"
    assert entry.model == "kimi"
    assert entry.reasoning_effort == "high"
    assert entry.model_provider == "dexgem"
    assert entry.tmux is True
    assert entry.log_path == str(_session.log_path)
    assert entry.start_ts == 1.0

    # Unchanged registrations arm no further timer.
    h.coordinator.observe_registrations([registration])
    assert len(h.timers) == 1


def test_settings_change_upserts_raw_live_values(tmp_path: Path) -> None:
    h = _Harness(tmp_path)
    session, registration = h.add_live_session("broker-10", broker_pid=10)
    h.coordinator.observe_registrations([registration])
    h.flush()
    assert h.store.read().entries[("pi", "thread-1")].reasoning_effort == "high"

    sidecar = json.loads(session.sock_path.with_suffix(".json").read_text(encoding="utf-8"))
    sidecar["live_run_settings"] = {"model_provider": "dexgem", "model": "kimi", "reasoning_effort": "xhigh"}
    session.sock_path.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")
    changed = h.registration(session, sidecar)

    h.coordinator.observe_registrations([changed])
    assert len(h.timers) == 2
    h.flush()
    assert h.store.read().entries[("pi", "thread-1")].reasoning_effort == "xhigh"


def test_session_without_bound_log_is_not_snapshotted(tmp_path: Path) -> None:
    h = _Harness(tmp_path)
    session, registration = h.add_live_session("broker-10", broker_pid=10)
    session.log_path = None
    unbound = DiscoveryRegistration(**{**registration.__dict__, "log_path": None})

    h.coordinator.observe_registrations([unbound])
    assert h.timers == []
    assert h.coordinator.entry_for_session(session) is None


def test_crashed_session_absent_from_discovery_keeps_its_entry(tmp_path: Path) -> None:
    h = _Harness(tmp_path)
    _session, registration = h.add_live_session("broker-10", broker_pid=10)
    h.coordinator.observe_registrations([registration])
    h.flush()

    h.alive.discard(10)
    h.coordinator.observe_registrations([])
    assert ("pi", "thread-1") in h.store.read().entries
    assert len(h.timers) == 1


def test_clean_broker_exit_removes_only_that_incarnation(tmp_path: Path) -> None:
    """Pid reuse: the new broker-100's clean exit must not erase the dead broker-100's entry."""
    h = _Harness(tmp_path)
    old, reg_old = h.add_live_session("broker-100", broker_pid=100, thread_id="t-dead", start_ts=1.0)
    h.coordinator.observe_registrations([reg_old])
    h.flush()
    # The old broker dies; a reused pid registers an unrelated session.
    h.alive.discard(100)
    del h.sessions["broker-100"]
    new, reg_new = h.add_live_session("broker-100", broker_pid=100, thread_id="t-new", start_ts=2.0)
    h.coordinator.observe_registrations([reg_new])
    h.flush()
    assert set(h.store.read().entries) == {("pi", "t-dead"), ("pi", "t-new")}

    h.coordinator.session_sock_vanished(new)
    assert set(h.store.read().entries) == {("pi", "t-dead")}

    # The forgotten signature re-arms a flush for a same-id registration.
    h.coordinator.observe_registrations([reg_new])
    assert len(h.timers) == 3


def test_session_deleted_scopes_removal_to_transcript_and_incarnation(tmp_path: Path) -> None:
    h = _Harness(tmp_path)
    session, registration = h.add_live_session("broker-10", broker_pid=10, thread_id="t-1")
    other, reg_other = h.add_live_session("broker-100", broker_pid=100, thread_id="t-other", start_ts=9.0)
    h.coordinator.observe_registrations([registration, reg_other])
    h.flush()
    assert set(h.store.read().entries) == {("pi", "t-1"), ("pi", "t-other")}

    # Deleting a live session removes its transcript's entry.
    h.coordinator.session_deleted("broker-10", session)
    assert set(h.store.read().entries) == {("pi", "t-other")}

    # A pid-reused id with a different incarnation and transcript deletes nothing.
    reused = Session(
        session_id="broker-100",
        thread_id="t-unrelated",
        broker_pid=100,
        codex_pid=101,
        agent_backend="pi",
        owned=True,
        start_ts=2.0,
        cwd="/repo",
        log_path=other.log_path,
        sock_path=other.sock_path,
    )
    h.coordinator.session_deleted("broker-100", reused)
    assert set(h.store.read().entries) == {("pi", "t-other")}

    # A delete row without a registry session matches by registry id only.
    h.coordinator.session_deleted("broker-100", None)
    assert h.store.read().entries == {}


def test_session_deleted_also_removes_the_entry_of_a_resume_target(tmp_path: Path) -> None:
    """Deleting a session that was itself a resume of the entry's transcript removes that entry."""
    h = _Harness(tmp_path)
    session, registration = h.add_live_session("broker-10", broker_pid=10, thread_id="t-old")
    h.coordinator.observe_registrations([registration])
    h.flush()

    resumed = Session(
        session_id="broker-11",
        thread_id="t-new",
        broker_pid=11,
        codex_pid=12,
        agent_backend="pi",
        owned=True,
        start_ts=2.0,
        cwd="/repo",
        log_path=session.log_path,
        sock_path=session.sock_path,
        resume_session_id="t-old",
    )
    h.coordinator.session_deleted("broker-11", resumed)
    assert h.store.read().entries == {}


def test_save_now_writes_immediately_and_cancels_pending_timer(tmp_path: Path) -> None:
    h = _Harness(tmp_path)
    _session, registration = h.add_live_session("broker-10", broker_pid=10)
    h.coordinator.observe_registrations([registration])
    timer = h.timers[-1]

    state = h.coordinator.save_now()
    assert timer.cancelled is True
    assert set(state.entries) == {("pi", "thread-1")}
    assert set(h.store.read().entries) == {("pi", "thread-1")}
    timer.fire()
    assert set(h.store.read().entries) == {("pi", "thread-1")}


def test_save_now_skips_lost_and_dead_sessions(tmp_path: Path) -> None:
    h = _Harness(tmp_path)
    live, _ = h.add_live_session("broker-10", broker_pid=10, thread_id="t-live")
    lost, _ = h.add_live_session("broker-11", broker_pid=11, thread_id="t-lost")
    dead, _ = h.add_live_session("broker-12", broker_pid=12, thread_id="t-dead")
    lost.lost = True
    h.alive.discard(12)

    state = h.coordinator.save_now()
    assert set(state.entries) == {("pi", "t-live")}


def test_live_session_id_for_entry_matches_incarnation_thread_resume_target_and_log(tmp_path: Path) -> None:
    h = _Harness(tmp_path)
    session, _ = h.add_live_session("broker-10", broker_pid=10, thread_id="t-1")
    entry = h.coordinator.entry_for_session(session)
    assert entry is not None

    assert h.coordinator.live_session_id_for_entry(entry) == "broker-10"
    by_thread = SnapshotEntry(**{**entry.__dict__, "session_id": "broker-old", "log_path": "/elsewhere.jsonl"})
    assert h.coordinator.live_session_id_for_entry(by_thread) == "broker-10"
    by_log = SnapshotEntry(**{**entry.__dict__, "session_id": "broker-old", "thread_id": "other"})
    assert h.coordinator.live_session_id_for_entry(by_log) == "broker-10"

    # A resume still binding its log declares the entry's thread as its target.
    resuming = Session(
        session_id="broker-77",
        thread_id="broker-77",
        broker_pid=77,
        codex_pid=78,
        agent_backend="codex",
        owned=True,
        start_ts=3.0,
        cwd="/repo",
        log_path=None,
        sock_path=h.sock_dir / "broker-77.sock",
        resume_session_id="t-1",
    )
    h.sessions["broker-77"] = resuming
    h.alive.add(77)
    assert h.coordinator.live_session_id_for_entry(by_log) == "broker-10"  # broker-10 still wins
    h.alive.discard(10)
    h.alive.discard(11)
    assert h.coordinator.live_session_id_for_entry(by_thread) == "broker-77"
    h.alive.discard(77)
    h.alive.discard(78)
    assert h.coordinator.live_session_id_for_entry(entry) is None


def test_live_session_id_for_entry_does_not_match_a_pid_reused_incarnation(tmp_path: Path) -> None:
    h = _Harness(tmp_path)
    old, reg_old = h.add_live_session("broker-100", broker_pid=100, thread_id="t-dead", start_ts=1.0)
    entry = h.coordinator.entry_for_session(old)
    assert entry is not None
    h.alive.discard(100)
    h.alive.discard(101)
    del h.sessions["broker-100"]
    new, _reg_new = h.add_live_session("broker-100", broker_pid=100, thread_id="t-new", start_ts=2.0)
    assert new.session_id == entry.session_id

    assert h.coordinator.live_session_id_for_entry(entry) is None


def test_live_session_id_for_entry_falls_back_to_the_entry_sidecar(tmp_path: Path) -> None:
    """Discovery can skip a slow broker; a live broker pid in the entry's own sidecar counts as live."""
    h = _Harness(tmp_path)
    session, _ = h.add_live_session("broker-10", broker_pid=10, thread_id="t-1")
    entry = h.coordinator.entry_for_session(session)
    assert entry is not None

    # The broker is live but absent from the registry (discovery skipped it).
    del h.sessions["broker-10"]
    assert h.coordinator.live_session_id_for_entry(entry) == "broker-10"

    # Dead broker in the sidecar: not live.
    h.alive.discard(10)
    h.alive.discard(11)
    assert h.coordinator.live_session_id_for_entry(entry) is None


def _seed_stores(h: _Harness, old: str) -> None:
    h.session_store.queues[old] = [{"id": "q1", "text": "queued", "created_ts": 1.0}]
    h.session_store.drafts[old] = {"text": "draft text", "updated_ts": 2.0}
    h.session_store.unattended[old] = {"enabled": True, "request": "continue", "cooldown_minutes": 5, "remaining_injections": 3}
    h.session_store.aliases[old] = "main session"
    h.session_store.sidebar_meta[old] = {"priority": 1, "hidden": False}
    h.session_store.sidebar_meta["broker-other"] = {"dependency_session_id": old}


def test_supersede_remaps_every_per_session_store_to_the_new_registry_id(tmp_path: Path) -> None:
    """The remap fires at the supersede moment: the replacement's upsert moves all state."""
    h = _Harness(tmp_path)
    _session, registration = h.add_live_session("broker-10", broker_pid=10, thread_id="t-1")
    h.coordinator.observe_registrations([registration])
    h.flush()
    _seed_stores(h, "broker-10")
    h.session_store.queues["broker-30"] = [{"id": "q2", "text": "later", "created_ts": 3.0}]
    h.alive.discard(10)
    h.alive.discard(11)

    _new, replacement = h.add_live_session("broker-30", broker_pid=30, thread_id="t-1", start_ts=2.0)
    h.coordinator.observe_registrations([replacement])
    h.flush()

    state = h.store.read()
    assert set(state.entries) == {("pi", "t-1")}
    assert state.entries[("pi", "t-1")].session_id == "broker-30"
    assert [item["id"] for item in h.session_store.queues["broker-30"]] == ["q1", "q2"]
    assert "broker-10" not in h.session_store.queues
    assert h.session_store.drafts == {"broker-30": {"text": "draft text", "updated_ts": 2.0}}
    assert h.session_store.unattended == {"thread:t-1": {"enabled": True, "request": "continue", "cooldown_minutes": 5, "remaining_injections": 3}}
    assert h.session_store.aliases == {"broker-30": "main session"}
    assert h.session_store.sidebar_meta["broker-30"] == {"priority": 1, "hidden": False}
    assert h.session_store.sidebar_meta["broker-other"] == {"dependency_session_id": "broker-30"}
    assert h.saved == ["queues", "drafts", "unattended", "aliases", "sidebar_meta"]


def test_remap_at_supersede_covers_a_server_restart_between_launch_and_discovery(tmp_path: Path) -> None:
    """No in-memory remap state: a fresh coordinator remaps when the replacement's upsert lands."""
    h = _Harness(tmp_path)
    _session, registration = h.add_live_session("broker-10", broker_pid=10, thread_id="t-1")
    h.coordinator.observe_registrations([registration])
    h.flush()
    _seed_stores(h, "broker-10")
    h.alive.discard(10)
    h.alive.discard(11)

    # New server process: fresh coordinator over the same snapshot file and stores.
    h2 = _Harness(tmp_path)
    h2.session_store = h.session_store
    h2.coordinator.session_store = lambda: h.session_store
    _new, replacement = h2.add_live_session("broker-31", broker_pid=31, thread_id="t-1", start_ts=2.0)
    h2.coordinator.observe_registrations([replacement])
    h2.flush()

    assert set(h2.store.read().entries) == {("pi", "t-1")}
    assert set(h.session_store.queues) == {"broker-31"}
    assert set(h.session_store.drafts) == {"broker-31"}
    assert h.session_store.unattended == {"thread:t-1": {"enabled": True, "request": "continue", "cooldown_minutes": 5, "remaining_injections": 3}}
    assert h.session_store.aliases == {"broker-31": "main session"}


def test_remap_is_skipped_when_the_old_registry_id_now_names_a_live_other_transcript(tmp_path: Path) -> None:
    h = _Harness(tmp_path)
    old, reg_old = h.add_live_session("broker-100", broker_pid=100, thread_id="t-old", start_ts=1.0)
    h.coordinator.observe_registrations([reg_old])
    h.flush()
    _seed_stores(h, "broker-100")
    h.alive.discard(100)
    h.alive.discard(101)
    del h.sessions["broker-100"]

    # Pid reuse: broker-100 is a live unrelated session; broker-200 resumes t-old.
    h.add_live_session("broker-100", broker_pid=100, thread_id="t-other", start_ts=2.0)
    _new, replacement = h.add_live_session("broker-200", broker_pid=200, thread_id="t-old", start_ts=3.0)
    h.coordinator.observe_registrations([replacement])
    h.flush()

    # The entry followed the transcript, but state stays put: broker-100 is someone else's now.
    state = h.store.read()
    assert state.entries[("pi", "t-old")].session_id == "broker-200"
    assert set(h.session_store.queues) == {"broker-100"}
    assert h.session_store.aliases == {"broker-100": "main session"}
    assert h.session_store.sidebar_meta["broker-other"] == {"dependency_session_id": "broker-100"}
    assert h.saved == []
    assert "now belongs to a live session on another transcript" in h.stderr.getvalue()


def test_thread_changing_continuation_moves_unattended_config_to_the_new_thread_key(tmp_path: Path) -> None:
    h = _Harness(tmp_path)
    _session, registration = h.add_live_session("broker-10", broker_pid=10, thread_id="t-1")
    h.coordinator.observe_registrations([registration])
    h.flush()
    h.session_store.unattended["thread:t-1"] = {"enabled": True, "request": "go", "cooldown_minutes": 5, "remaining_injections": 1}
    h.alive.discard(10)
    h.alive.discard(11)

    _new, replacement = h.add_live_session("broker-40", broker_pid=40, thread_id="t-2", start_ts=2.0, meta={"resume_session_id": "t-1"})
    h.coordinator.observe_registrations([replacement])
    h.flush()

    state = h.store.read()
    assert set(state.entries) == {("pi", "t-2")}
    assert h.session_store.unattended == {"thread:t-2": {"enabled": True, "request": "go", "cooldown_minutes": 5, "remaining_injections": 1}}


def test_same_incarnation_moving_threads_drops_the_old_key_without_a_remap(tmp_path: Path) -> None:
    """Pi /new: one broker, new transcript — the old entry retires, state stays under the shared id."""
    h = _Harness(tmp_path)
    session, registration = h.add_live_session("broker-10", broker_pid=10, thread_id="t-old")
    h.coordinator.observe_registrations([registration])
    h.flush()
    h.session_store.queues["broker-10"] = [{"id": "q1", "text": "queued", "created_ts": 1.0}]

    # Pi /new: the same broker switches the transcript; its sidecar follows.
    new_log = h.tmp_path / "t-new.jsonl"
    new_log.write_text("{}\n", encoding="utf-8")
    sidecar = json.loads(session.sock_path.with_suffix(".json").read_text(encoding="utf-8"))
    sidecar.update({"session_id": "t-new", "log_path": str(new_log)})
    session.sock_path.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")
    session.thread_id = "t-new"
    session.log_path = new_log
    moved = h.registration(session, sidecar)
    h.coordinator.observe_registrations([moved])
    h.flush()

    assert set(h.store.read().entries) == {("pi", "t-new")}
    assert set(h.session_store.queues) == {"broker-10"}
    assert h.saved == []


def test_flush_retries_a_torn_sidecar_and_recovers_after_giving_up(tmp_path: Path) -> None:
    h = _Harness(tmp_path)
    session, registration = h.add_live_session("broker-10", broker_pid=10)
    sidecar_path = session.sock_path.with_suffix(".json")
    sidecar_path.write_text("{torn", encoding="utf-8")
    h.coordinator.observe_registrations([registration])

    h.flush()
    assert not h.store.exists()
    assert "cannot build entry" in h.stderr.getvalue()
    assert h.timers[-1].delay == 2.0 * FLUSH_RETRY_MULTIPLIER

    # Consecutive failures re-arm until the bound, then give up with a log line.
    for _ in range(3):
        timer_count = len(h.timers)
        h.flush()
    assert "giving up" in h.stderr.getvalue()
    assert len(h.timers) == timer_count  # no retry armed after the give-up
    assert not h.store.exists()

    # The id stayed dirty: once the sidecar is repaired, the next observation rides along.
    sidecar_path.write_text(
        json.dumps({"session_id": "thread-1", "broker_pid": 10, "cwd": "/repo", "log_path": str(session.log_path), "agent_backend": "pi", "model": "kimi"}),
        encoding="utf-8",
    )
    h.coordinator.observe_registrations([registration])
    assert h.timers[-1].delay == 2.0
    h.flush()
    assert set(h.store.read().entries) == {("pi", "thread-1")}


def test_flush_re_arms_when_the_upsert_fails(tmp_path: Path) -> None:
    h = _Harness(tmp_path)
    h.store.path.write_text("{broken", encoding="utf-8")
    _session, registration = h.add_live_session("broker-10", broker_pid=10)

    h.coordinator.observe_registrations([registration])
    h.flush()

    assert h.store.path.read_text(encoding="utf-8") == "{broken"
    assert "session snapshot upsert failed" in h.stderr.getvalue()
    assert h.timers[-1].delay == 2.0 * FLUSH_RETRY_MULTIPLIER


def test_session_dying_between_observe_and_flush_is_dropped_silently(tmp_path: Path) -> None:
    h = _Harness(tmp_path)
    _session, registration = h.add_live_session("broker-10", broker_pid=10)
    h.coordinator.observe_registrations([registration])

    h.alive.discard(10)
    h.flush()

    assert not h.store.exists()
    assert len(h.timers) == 1  # no failure re-arm
    assert h.stderr.getvalue() == ""


def test_queue_protected_session_ids_cover_snapshot_entries_and_fail_safe(tmp_path: Path) -> None:
    h = _Harness(tmp_path)
    _session, registration = h.add_live_session("broker-10", broker_pid=10, thread_id="t-1")
    h.coordinator.observe_registrations([registration])
    h.flush()

    assert h.coordinator.queue_protected_session_ids() == {"broker-10"}

    # An unreadable snapshot protects everything rather than dropping queues blindly.
    h.store.path.write_text("{broken", encoding="utf-8")
    assert h.coordinator.queue_protected_session_ids() is None
    assert "queue sweep keeps every orphaned queue" in h.stderr.getvalue()
    warning_count = h.stderr.getvalue().count("queue sweep keeps every orphaned queue")
    assert h.coordinator.queue_protected_session_ids() is None
    assert h.stderr.getvalue().count("queue sweep keeps every orphaned queue") == warning_count

    h.store.path.write_text(json.dumps({"version": 1, "saved_at": 1.0, "entries": []}), encoding="utf-8")
    assert h.coordinator.queue_protected_session_ids() == set()


def test_dismiss_removes_by_registry_id_or_thread_id(tmp_path: Path) -> None:
    h = _Harness(tmp_path)
    _a, reg_a = h.add_live_session("broker-10", broker_pid=10, thread_id="t-1")
    _b, reg_b = h.add_live_session("broker-11", broker_pid=11, thread_id="t-2")
    h.coordinator.observe_registrations([reg_a, reg_b])
    h.flush()

    assert h.coordinator.dismiss("t-1") is True
    assert set(h.store.read().entries) == {("pi", "t-2")}
    assert h.coordinator.dismiss("broker-11") is True
    assert h.store.read().entries == {}
    assert h.coordinator.dismiss("broker-11") is False


def test_restoring_mark_lifecycle_and_public_annotation(tmp_path: Path) -> None:
    h = _Harness(tmp_path)
    session, registration = h.add_live_session("broker-10", broker_pid=10, thread_id="t-1")
    h.coordinator.observe_registrations([registration])
    h.flush()
    entry = h.store.read().entries[("pi", "t-1")]
    # The entry's own broker is dead: only a restoring mark can make it non-restorable.
    h.alive.discard(10)
    h.alive.discard(11)
    assert h.coordinator.public_annotation(entry) == {"live": False, "live_session_id": None, "restoring": None}

    assert h.coordinator.mark_restoring(entry, broker_pid=99) is True
    h.alive.add(99)
    marked = h.store.read().entries[("pi", "t-1")]
    assert marked.restoring == RestoringMark(broker_pid=99, launched_at=50.0)
    annotation = h.coordinator.public_annotation(marked)
    assert annotation["live"] is True
    assert annotation["restoring"] == {"broker_pid": 99, "launched_at": 50.0}

    # A dead launched broker or an expired mark makes the entry restorable again.
    h.alive.discard(99)
    assert h.coordinator.restoring_active(h.store.read().entries[("pi", "t-1")]) is False
    h.alive.add(99)
    h.clock[0] = 50.0 + RESTORING_TTL_SECONDS
    expired = h.store.read().entries[("pi", "t-1")]
    assert h.coordinator.public_annotation(expired)["live"] is False

    # A superseded entry refuses the mark.
    stale = SnapshotEntry(**{**entry.__dict__, "session_id": "broker-old"})
    assert h.coordinator.mark_restoring(stale, broker_pid=1) is False
