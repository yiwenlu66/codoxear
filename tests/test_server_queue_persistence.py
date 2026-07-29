import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from codoxear.control_socket import ControlSocketCallError
from codoxear.queue_store import QueueStore
from codoxear.server import Session
from codoxear.server import SessionCommitUnknownError
from codoxear.server import SessionManager
from codoxear.server import SessionNotReadyError
from codoxear.server import _match_session_route
from codoxear.session_control import SessionControlCoordinator
from codoxear.session_errors import SessionInjectionError
from codoxear.session_pending_state import SessionPendingStateCoordinator
from codoxear.session_queue import SessionQueueCoordinator
from codoxear.session_store import SessionStore
from codoxear.session_store import SessionStorePaths


def _make_session(sid: str) -> Session:
    return Session(
        session_id=sid,
        thread_id="t1",
        broker_pid=1,
        codex_pid=1,
        agent_backend="codex",
        owned=False,
        start_ts=0.0,
        cwd="/tmp",
        log_path=None,
        sock_path=Path(f"/tmp/{sid}.sock"),
        sync_send_supported=True,
        key_write_errors_supported=True,
    )


def _queue_item(item_id: str, text: str) -> dict[str, object]:
    return {"id": item_id, "text": text, "created_ts": 1.0}


# ---------------------------------------------------------------------------
# Coordinator construction helpers.
#
# Every former ``codoxear.server`` module-global patch (``time.time``,
# ``_pid_alive``, ``QUEUE_PATH``) is replaced by constructing the relevant
# coordinator directly with the boundary injected as a constructor argument:
#
#   * ``now``      -> wall clock (time boundary) on queue/pending-state coords.
#   * ``pid_alive`` -> OS process liveness probe (process boundary) on the
#                      control coordinator.
#   * ``QueueStore(path)`` -> real filesystem path (replaces ``QUEUE_PATH``).
#
# No ``codoxear.server.*`` monkeypatching remains. No file under ``codoxear/``
# is modified.
# ---------------------------------------------------------------------------


def _session_store(td: Path) -> SessionStore:
    """Build a real SessionStore backed by temp-directory paths."""
    paths = SessionStorePaths(
        aliases=td / "aliases.json",
        sidebar_meta=td / "sidebar.json",
        hidden_sessions=td / "hidden.json",
        files=td / "files.json",
        queues=td / "queues.json",
        pending_attachments=td / "pending.json",
        commit_unknown_sends=td / "commit_unknown.json",
        recent_cwds=td / "cwds.json",
        unattended=td / "unattended.json",
    )
    return SessionStore(
        paths=paths,
        file_history_max=10,
        recent_cwd_max=10,
        unattended_default_idle_minutes=5,
        unattended_default_max_injections=10,
        clean_alias=lambda v: v,
        clean_priority_offset=lambda v: v,
        clean_snooze_until=lambda v: v,
        clean_dependency_session_id=lambda v: v,
        clean_recent_cwd=lambda v: v,
        clean_commit_unknown_send_record=lambda v: v,
    )


def _queue_coordinator(
    *,
    sessions: dict[str, Session],
    queues: dict[str, list[dict[str, Any]]],
    queue_dir: Path,
    now: float,
    remote_ready: Any = lambda _sid, _log_path: True,
    send: Any = lambda *_args, **_kwargs: {"queued": False, "queue_len": 0},
    recovery_items_locked: Any = None,
) -> tuple[SessionQueueCoordinator, list[dict[str, Any]]]:
    """Build a SessionQueueCoordinator with the wall clock (``now``) injected.

    Replaces former ``patch("codoxear.server.time.time", ...)`` calls: the
    commit-unknown timestamp is ``now`` passed here, not a module-global patch.
    """
    saves: list[dict[str, Any]] = []
    store = QueueStore(queue_dir / "session_queues.json")

    def save_queues() -> None:
        saves.append({sid: [dict(item) for item in items] for sid, items in queues.items()})

    coordinator = SessionQueueCoordinator(
        lock=threading.Lock(),
        sessions=lambda: sessions,
        queues=lambda: queues,
        queue_store=lambda: store,
        commit_unknown_sends=lambda: {},
        save_queues=save_queues,
        input_lock_for_session=lambda _sid: threading.RLock(),
        remote_ready=remote_ready,
        send=send,
        not_ready_error=SessionNotReadyError,
        retryable_send_errors=(SessionNotReadyError, SessionInjectionError),
        commit_unknown_error=SessionCommitUnknownError,
        queue_idle_grace_seconds=5.0,
        now=lambda: now,
        recovery_items_locked=recovery_items_locked,
    )
    return coordinator, saves


def _pending_state_coordinator(
    *,
    store: SessionStore,
    sessions: dict[str, Session],
    now: float,
    save_commit_unknown_sends: Any = lambda: None,
    save_queues: Any = lambda: None,
) -> SessionPendingStateCoordinator:
    """Build a SessionPendingStateCoordinator with the wall clock (``now``) injected.

    Replaces former ``patch("codoxear.server.time.time", ...)`` calls.
    """
    return SessionPendingStateCoordinator(
        lock=threading.Lock(),
        sessions=lambda: sessions,
        store=lambda: store,
        pending_attachment_ids=lambda: store.pending_attachment_ids,
        set_pending_attachment_ids=lambda ids: setattr(store, "pending_attachment_ids", ids),
        commit_unknown_sends=lambda: store.commit_unknown_sends,
        set_commit_unknown_sends=lambda d: store.commit_unknown_sends.update(d),
        mark_queue_orphan_recovery_locked=lambda sid: store.queue_store.mark_orphan_recovery_items(store.queues, sid),
        save_pending_attachments=lambda: None,
        save_commit_unknown_sends=save_commit_unknown_sends,
        save_queues=save_queues,
        now=lambda: now,
        commit_unknown_orphan_prune_seconds=7 * 24 * 3600,
    )


def _control_coordinator(
    *,
    sessions: dict[str, Session],
    sock_call: Any,
    pid_alive: Any,
    clear_deleted_session_state: Any = lambda _sid: None,
) -> SessionControlCoordinator:
    """Build a SessionControlCoordinator with the OS process liveness probe
    (``pid_alive``) injected.

    Replaces former ``patch("codoxear.server._pid_alive", ...)`` calls.
    """
    return SessionControlCoordinator(
        lock=threading.Lock(),
        sessions=lambda: sessions,
        sock_call=sock_call,
        pid_alive=pid_alive,
        unlink_quiet=lambda _path: None,
        clear_deleted_session_state=clear_deleted_session_state,
        broker_busy_queue=lambda _state: (False, 0),
        broker_interrupted_idle=lambda _state: False,
        control_socket_call_error=ControlSocketCallError,
        commit_unknown_error=SessionCommitUnknownError,
    )


class TestServerQueuePersistence(unittest.TestCase):
    def _mgr(self) -> SessionManager:
        mgr = SessionManager.__new__(SessionManager)
        mgr._lock = threading.Lock()
        mgr._sessions = {}
        mgr._queues = {}
        mgr._pending_attachment_ids = set()
        mgr._staged_attachments = {}
        mgr._commit_unknown_sends = {}
        mgr._save_queues = lambda: None
        mgr._save_pending_attachments = lambda: None
        mgr._save_staged_attachments = lambda: None
        mgr._save_commit_unknown_sends = lambda: None
        return mgr

    def test_pending_state_public_payloads_omit_internal_staged_paths(self) -> None:
        sid = "s1"
        with TemporaryDirectory() as td:
            store = _session_store(Path(td))
            sessions = {sid: _make_session(sid)}
            coord = _pending_state_coordinator(store=store, sessions=sessions, now=10.0)
            internal_path = Path("/home/tester/.local/share/codoxear/uploads/s1/1000_doc.txt")

            added = coord.add_staged_attachment(
                sid,
                display_name="doc.txt",
                filename="1000_doc.txt",
                path=internal_path,
                size=3,
                created_ts=10.0,
            )

            public_entry = {"id": added["attachment"]["id"], "display_name": "doc.txt", "filename": "1000_doc.txt", "size": 3, "created_ts": 10.0}
            self.assertEqual(added["attachment"], public_entry)
            self.assertEqual(added["attachments"], [public_entry])
            self.assertNotIn("path", added["attachment"])
            self.assertEqual(store.staged_attachments[sid][0]["path"], str(internal_path))

            listed = coord.list_staged_attachments(sid)
            self.assertEqual(listed["attachments"], [public_entry])
            self.assertNotIn(str(internal_path), str(listed))

            removed = coord.remove_staged_attachment(sid, public_entry["id"])
            self.assertEqual(removed["removed"], public_entry)
            self.assertNotIn("path", removed["removed"])


    def test_deleted_state_cleanup_preserves_recovery_markers_unless_explicit(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._aliases = {}
        mgr._sidebar_meta = {}
        mgr._unattended = {}
        mgr._files = {}
        mgr._input_locks = {}
        mgr._save_aliases = lambda: None
        mgr._save_sidebar_meta = lambda: None
        mgr._save_unattended = lambda: None
        mgr._save_files = lambda: None
        mgr._queues[sid] = [dict(_queue_item("q1", "maybe sent"), commit_unknown=True)]
        mgr._commit_unknown_sends[sid] = {"text": "maybe direct", "created_ts": 1.0}

        SessionManager._clear_deleted_session_state(mgr, sid)

        self.assertIn(sid, mgr._queues)
        self.assertIn(sid, mgr._commit_unknown_sends)

        SessionManager._clear_deleted_session_state(mgr, sid, clear_recovery=True)

        self.assertNotIn(sid, mgr._queues)
        self.assertNotIn(sid, mgr._commit_unknown_sends)

    def test_prune_missing_commit_unknown_sends_keeps_recent_orphans(self) -> None:
        # SessionPendingCoordinator.prune_missing_commit_unknown_sends with
        # the wall clock injected as ``now=100.0`` (replaces former
        # ``patch("codoxear.server.time.time", ...)``).
        sessions = {"live": _make_session("live")}
        commit_unknown_sends = {
            "live": {"text": "maybe live", "created_ts": 1.0},
            "recent_gone": {"text": "maybe recent", "created_ts": 90.0},
            "old_gone": {"text": "maybe old", "created_ts": 1.0},
        }
        saved: list[dict[str, object]] = []
        with TemporaryDirectory() as td:
            store = _session_store(Path(td))
            store.commit_unknown_sends = commit_unknown_sends
            store.queues = {}
            coord = _pending_state_coordinator(
                store=store,
                sessions=sessions,
                now=100.0,
                save_commit_unknown_sends=lambda: saved.append(dict(store.commit_unknown_sends)),
            )
            self.assertTrue(coord.prune_missing_commit_unknown_sends(max_age_seconds=50.0))

        self.assertEqual(set(commit_unknown_sends), {"live", "recent_gone"})
        self.assertEqual(saved, [{"live": {"text": "maybe live", "created_ts": 1.0}, "recent_gone": {"text": "maybe recent", "created_ts": 90.0}}])

    def test_match_session_route_requires_exact_suffix(self) -> None:
        self.assertEqual(_match_session_route("/api/sessions/s1/delete", "delete"), "s1")
        self.assertEqual(_match_session_route("/api/sessions/s%201/queue/delete", "queue", "delete"), "s%201")
        self.assertIsNone(_match_session_route("/api/sessions/s1/queue/delete", "delete"))
        for suffix in ["queue", "send", "unattended", "interrupt", "diagnostics", "edit", "rename", "inject_file", "inject_image"]:
            self.assertIsNone(_match_session_route(f"/api/sessions/s1/extra/{suffix}", suffix))
        for family, suffix in [("file", "read"), ("file", "search"), ("file", "list"), ("file", "blob"), ("file", "video_preview"), ("file", "download"), ("git", "changed_files"), ("git", "diff"), ("git", "file_versions")]:
            self.assertIsNone(_match_session_route(f"/api/sessions/s1/extra/{family}/{suffix}", family, suffix))
            self.assertEqual(_match_session_route(f"/api/sessions/s1/{family}/{suffix}", family, suffix), "s1")
        self.assertIsNone(_match_session_route("/api/sessions/s1/queue/extra/delete", "queue", "delete"))
        self.assertEqual(_match_session_route("/api/sessions/s1/queue/delete", "queue", "delete"), "s1")
        self.assertEqual(_match_session_route("/api/sessions/s1/diagnostics", "diagnostics"), "s1")
        self.assertEqual(_match_session_route("/api/sessions/s1/edit", "edit"), "s1")
        self.assertEqual(_match_session_route("/api/sessions/s1/rename", "rename"), "s1")
        self.assertEqual(_match_session_route("/api/sessions/s1/inject_file", "inject_file"), "s1")
        self.assertEqual(_match_session_route("/api/sessions/s1/commit_unknown_send/clear", "commit_unknown_send", "clear"), "s1")

    def test_attachment_staging_ready_allows_busy_broker_and_queued_work(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        # Staging writes only to disk, so it must not query broker or log idleness.
        mgr.get_state = lambda _sid: self.fail("staging should not query broker state")  # type: ignore[method-assign]
        mgr.idle_from_log = lambda _sid: self.fail("staging should not query log idleness")  # type: ignore[method-assign]

        self.assertTrue(SessionManager.attachment_staging_ready(mgr, sid))
        mgr._queues[sid] = [_queue_item("q1", "queued")]
        self.assertTrue(SessionManager.attachment_staging_ready(mgr, sid))
        mgr._sessions[sid].queue_sending_item_id = "q1"
        self.assertFalse(SessionManager.attachment_staging_ready(mgr, sid))
        mgr._sessions[sid].queue_sending_item_id = None
        self.assertTrue(SessionManager.attachment_staging_ready(mgr, sid))

    def test_unknown_direct_send_blocks_attachment_staging(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._sessions[sid].commit_unknown_send = {"text": "maybe sent", "created_ts": 1.0}
        mgr._commit_unknown_sends[sid] = {"text": "maybe sent", "created_ts": 1.0}
        mgr.get_state = lambda _sid: self.fail("unknown send should fail before broker readiness")  # type: ignore[method-assign]

        with self.assertRaisesRegex(SessionNotReadyError, "unknown send"):
            SessionManager.attachment_staging_ready(mgr, sid)

    def test_attachment_staging_ready_allows_log_busy_session(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        with TemporaryDirectory() as td:
            log_path = Path(td) / "rollout.jsonl"
            log_path.write_text("{}\n", encoding="utf-8")
            mgr._sessions[sid] = _make_session(sid)
            mgr._sessions[sid].log_path = log_path
            mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}  # type: ignore[method-assign]
            mgr.idle_from_log = lambda _sid: False  # type: ignore[method-assign]

            self.assertTrue(SessionManager.attachment_staging_ready(mgr, sid))

    def test_attachment_staging_ready_allows_local_queue(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._queues[sid] = [_queue_item("q1", "queued")]
        mgr.get_state = lambda _sid: self.fail("staging should not query broker state")  # type: ignore[method-assign]

        self.assertTrue(SessionManager.attachment_staging_ready(mgr, sid))

    def test_attachment_staging_does_not_query_state_for_log_path(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr.get_state = lambda _sid: self.fail("staging should not query broker state")  # type: ignore[method-assign]
        mgr.idle_from_log = lambda _sid: self.fail("staging should not query log idleness")  # type: ignore[method-assign]

        self.assertTrue(SessionManager.attachment_staging_ready(mgr, sid))

    def test_attachment_staging_refreshes_sidecar_without_log_idle_check(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        with TemporaryDirectory() as td:
            root = Path(td)
            session = _make_session(sid)
            session.sock_path = root / "s1.sock"
            old_idle_log = root / "old.jsonl"
            old_idle_log.write_text("{}\n", encoding="utf-8")
            new_busy_log = root / "new.jsonl"
            new_busy_log.write_text("{}\n", encoding="utf-8")
            session.log_path = old_idle_log
            mgr._sessions[sid] = session
            session.sock_path.with_suffix(".json").write_text("{}\n", encoding="utf-8")
            mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}  # type: ignore[method-assign]
            mgr.idle_from_log = lambda _sid: mgr._sessions[sid].log_path != new_busy_log  # type: ignore[method-assign]

            drain_flags: list[bool] = []

            def refresh(_sid: str, *, drain_queue: bool = True) -> None:
                drain_flags.append(drain_queue)
                mgr._sessions[sid].log_path = new_busy_log

            mgr.refresh_session_meta = refresh  # type: ignore[method-assign]

            self.assertTrue(SessionManager.attachment_staging_ready(mgr, sid))
            self.assertEqual(mgr._sessions[sid].log_path, new_busy_log)
            self.assertEqual(drain_flags, [False])

    def test_attachment_readiness_refresh_does_not_drain_queue(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        with TemporaryDirectory() as td:
            root = Path(td)
            session = _make_session(sid)
            session.sock_path = root / "s1.sock"
            mgr._sessions[sid] = session
            session.sock_path.with_suffix(".json").write_text("{}\n", encoding="utf-8")
            mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}  # type: ignore[method-assign]
            mgr._maybe_drain_session_queue = lambda _sid: self.fail("attachment readiness must not drain queue")  # type: ignore[attr-defined]

            def refresh(_sid: str, *, drain_queue: bool = True) -> None:
                if drain_queue:
                    mgr._maybe_drain_session_queue(_sid)  # type: ignore[attr-defined]
                mgr._queues[sid] = [_queue_item("q1", "queued")]

            mgr.refresh_session_meta = refresh  # type: ignore[method-assign]

            self.assertTrue(SessionManager.attachment_staging_ready(mgr, sid))

    def test_pending_attachment_blocks_queue_until_explicit_send(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._sessions[sid].pending_attachment = True
        mgr._pending_attachment_ids.add(sid)
        mgr._staged_attachments[sid] = [
            {"id": "a1", "display_name": "one.txt", "filename": "1_one.txt", "path": "/tmp/uploads/s1/1_one.txt", "size": 3, "created_ts": 1.0},
        ]
        mgr.clear_staged_attachments = lambda _sid, **_kwargs: (mgr._staged_attachments.pop(_sid, None), setattr(mgr._sessions[_sid], "pending_attachment", False), mgr._pending_attachment_ids.discard(_sid), {"ok": True, "attachments": [], "pending_attachment": False})[-1]  # type: ignore[method-assign]

        with self.assertRaisesRegex(SessionNotReadyError, "pending attachment"):
            SessionManager.enqueue(mgr, sid, "queued prompt")

        mgr._record_prelog_user_message = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
        mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}  # type: ignore[method-assign]
        seen: list[dict[str, object]] = []
        mgr._sock_call = lambda _sock, req, **_kwargs: seen.append(req) or {"queued": False, "queue_len": 0}  # type: ignore[method-assign]
        with self.assertRaisesRegex(SessionNotReadyError, "pending attachment"):
            SessionManager.send(mgr, sid, "stale direct prompt")
        self.assertTrue(mgr._sessions[sid].pending_attachment)

        self.assertEqual(SessionManager.send(mgr, sid, "intended prompt", allow_pending_attachment=True), {"queued": False, "queue_len": 0})
        self.assertEqual(seen, [{"cmd": "send", "text": "Attachment 1: /tmp/uploads/s1/1_one.txt\nintended prompt", "sync": True}])
        self.assertFalse(mgr._sessions[sid].pending_attachment)
        self.assertNotIn(sid, mgr._pending_attachment_ids)

    def test_pending_send_commit_error_preserves_pending_attachment(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._sessions[sid].pending_attachment = True
        mgr._pending_attachment_ids.add(sid)
        mgr._record_prelog_user_message = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
        mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}  # type: ignore[method-assign]
        seen: list[dict[str, object]] = []

        def sock_call(_sock: Path, req: dict[str, object], timeout_s: float | None = 0, **_kwargs: object) -> dict[str, str]:
            seen.append(req)
            return {"error": "write failed"}

        mgr._sock_call = sock_call  # type: ignore[method-assign]

        from codoxear.server import SessionInjectionError

        with self.assertRaisesRegex(SessionInjectionError, "write failed"):
            SessionManager.send(mgr, sid, "intended prompt", allow_pending_attachment=True)
        self.assertEqual(seen, [{"cmd": "send", "text": "intended prompt", "sync": True}])
        self.assertTrue(mgr._sessions[sid].pending_attachment)
        self.assertIn(sid, mgr._pending_attachment_ids)

    def test_normal_server_send_uses_sync_commit(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._record_prelog_user_message = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
        mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}  # type: ignore[method-assign]
        seen: list[tuple[dict[str, object], float | None]] = []

        def sock_call(_sock: Path, req: dict[str, object], timeout_s: float | None = 0, **_kwargs: object) -> dict[str, object]:
            seen.append((req, timeout_s))
            return {"queued": False, "queue_len": 0}

        mgr._sock_call = sock_call  # type: ignore[method-assign]

        self.assertEqual(SessionManager.send(mgr, sid, "normal prompt"), {"queued": False, "queue_len": 0})
        self.assertEqual(seen, [({"cmd": "send", "text": "normal prompt", "sync": True}, 30.0)])

    def test_staged_attachments_compose_at_confirmed_send_boundary_and_clear_on_success(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._sessions[sid].pending_attachment = True
        mgr._pending_attachment_ids.add(sid)
        mgr._staged_attachments[sid] = [
            {"id": "a1", "display_name": "one.txt", "filename": "1_one.txt", "path": "/tmp/uploads/s1/1_one.txt", "size": 3, "created_ts": 1.0},
            {"id": "a2", "display_name": "two.txt", "filename": "2_two.txt", "path": "/tmp/uploads/s1/2_two.txt", "size": 4, "created_ts": 2.0},
        ]
        recorded: list[str] = []
        mgr._record_prelog_user_message = lambda _session, text, **_kwargs: recorded.append(text)  # type: ignore[method-assign]
        cleanup_calls: list[tuple[str, bool]] = []

        def clear_staged(clear_sid: str, *, delete_files: bool = True) -> dict[str, object]:
            cleanup_calls.append((clear_sid, delete_files))
            mgr._staged_attachments.pop(clear_sid, None)
            mgr._sessions[clear_sid].pending_attachment = False
            mgr._pending_attachment_ids.discard(clear_sid)
            return {"ok": True, "attachments": [], "pending_attachment": False}

        mgr.clear_staged_attachments = clear_staged  # type: ignore[method-assign]
        mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}  # type: ignore[method-assign]
        seen: list[dict[str, object]] = []

        def sock_call(_sock: Path, req: dict[str, object], timeout_s: float | None = 0, **_kwargs: object) -> dict[str, object]:
            seen.append(req)
            return {"queued": False, "queue_len": 0}

        mgr._sock_call = sock_call  # type: ignore[method-assign]

        self.assertEqual(SessionManager.send(mgr, sid, "use these", allow_pending_attachment=True), {"queued": False, "queue_len": 0})
        committed_text = "Attachment 1: /tmp/uploads/s1/1_one.txt\nAttachment 2: /tmp/uploads/s1/2_two.txt\nuse these"
        self.assertEqual(seen, [{"cmd": "send", "text": committed_text, "sync": True}])
        self.assertEqual(recorded, [committed_text])
        self.assertEqual(cleanup_calls, [(sid, False)])
        self.assertEqual(mgr._staged_attachments, {})
        self.assertFalse(mgr._sessions[sid].pending_attachment)
        self.assertNotIn(sid, mgr._pending_attachment_ids)

    def test_prelog_valueerror_after_confirmed_send_is_success_with_visible_error(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._sessions[sid].pending_attachment = True
        mgr._pending_attachment_ids.add(sid)
        entry = {"id": "a1", "display_name": "one.txt", "filename": "1_one.txt", "path": "/tmp/uploads/s1/1_one.txt", "size": 3, "created_ts": 1.0}
        mgr._staged_attachments[sid] = [dict(entry)]
        recorded: list[str] = []

        def fail_prelog(_session: Session, text: str, **_kwargs: object) -> None:
            recorded.append(text)
            raise ValueError("launch ledger invalid")

        mgr._record_prelog_user_message = fail_prelog  # type: ignore[method-assign]
        cleanup_calls: list[str] = []

        def clear_staged(clear_sid: str, *, delete_files: bool = True) -> dict[str, object]:
            cleanup_calls.append(clear_sid)
            mgr._staged_attachments.pop(clear_sid, None)
            mgr._sessions[clear_sid].pending_attachment = False
            mgr._pending_attachment_ids.discard(clear_sid)
            return {"ok": True, "removed_count": 1, "attachments": [], "pending_attachment": False}

        mgr.clear_staged_attachments = clear_staged  # type: ignore[method-assign]
        mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}  # type: ignore[method-assign]
        seen: list[dict[str, object]] = []

        def sock_call(_sock: Path, req: dict[str, object], timeout_s: float | None = 0, **_kwargs: object) -> dict[str, object]:
            seen.append(req)
            return {"queued": False, "queue_len": 0, "busy": True}

        mgr._sock_call = sock_call  # type: ignore[method-assign]

        response = SessionManager.send(mgr, sid, "use this", allow_pending_attachment=True)

        committed_text = "Attachment 1: /tmp/uploads/s1/1_one.txt\nuse this"
        self.assertEqual(seen, [{"cmd": "send", "text": committed_text, "sync": True}])
        self.assertEqual(recorded, [committed_text])
        self.assertEqual(cleanup_calls, [sid])
        self.assertEqual(response, {"queued": False, "queue_len": 0, "busy": True, "send_state_cleanup_error": "prelog_user_message: launch ledger invalid"})
        self.assertEqual(mgr._staged_attachments, {})
        self.assertFalse(mgr._sessions[sid].pending_attachment)
        self.assertNotIn(sid, mgr._pending_attachment_ids)
        self.assertTrue(mgr._sessions[sid].busy)
        self.assertTrue(mgr._sessions[sid].last_send_boundary_active)

    def test_staged_attachment_cleanup_failure_after_confirmed_send_is_success_with_visible_error(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._sessions[sid].pending_attachment = True
        mgr._pending_attachment_ids.add(sid)
        entry = {"id": "a1", "display_name": "one.txt", "filename": "1_one.txt", "path": "/tmp/uploads/s1/1_one.txt", "size": 3, "created_ts": 1.0}
        mgr._staged_attachments[sid] = [dict(entry)]
        recorded: list[str] = []
        mgr._record_prelog_user_message = lambda _session, text, **_kwargs: recorded.append(text)  # type: ignore[method-assign]
        cleanup_calls: list[str] = []

        def fail_cleanup(cleanup_sid: str, *, delete_files: bool = True) -> dict[str, object]:
            cleanup_calls.append(cleanup_sid)
            raise ValueError("staged_path outside session uploads")

        mgr.clear_staged_attachments = fail_cleanup  # type: ignore[method-assign]
        mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}  # type: ignore[method-assign]
        seen: list[dict[str, object]] = []

        def sock_call(_sock: Path, req: dict[str, object], timeout_s: float | None = 0, **_kwargs: object) -> dict[str, object]:
            seen.append(req)
            return {"queued": False, "queue_len": 0}

        mgr._sock_call = sock_call  # type: ignore[method-assign]
        clear_unknown_calls: list[tuple[str, dict[str, Any] | None]] = []

        def record_clear_unknown(unknown_sid: str, record: dict[str, Any] | None) -> None:
            clear_unknown_calls.append((unknown_sid, record))
            SessionManager._set_commit_unknown_send(mgr, unknown_sid, record)

        mgr._set_commit_unknown_send = record_clear_unknown  # type: ignore[method-assign]

        response = SessionManager.send(mgr, sid, "use this", allow_pending_attachment=True)

        committed_text = "Attachment 1: /tmp/uploads/s1/1_one.txt\nuse this"
        self.assertEqual(seen, [{"cmd": "send", "text": committed_text, "sync": True}])
        self.assertEqual(recorded, [committed_text])
        self.assertEqual(cleanup_calls, [sid])
        self.assertEqual(response, {"queued": False, "queue_len": 0, "attachment_cleanup_error": "staged_path outside session uploads"})
        self.assertEqual(mgr._staged_attachments[sid], [entry])
        self.assertTrue(mgr._sessions[sid].pending_attachment)
        self.assertIn(sid, mgr._pending_attachment_ids)
        self.assertEqual(clear_unknown_calls, [(sid, None)])
        self.assertNotIn(sid, mgr._commit_unknown_sends)
        self.assertIsNone(mgr._sessions[sid].commit_unknown_send)

    def test_staged_attachment_oserror_after_confirmed_send_is_success_with_visible_error(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._sessions[sid].pending_attachment = True
        mgr._pending_attachment_ids.add(sid)
        entry = {"id": "a1", "display_name": "one.txt", "filename": "1_one.txt", "path": "/tmp/uploads/s1/1_one.txt", "size": 3, "created_ts": 1.0}
        mgr._staged_attachments[sid] = [dict(entry)]
        mgr._record_prelog_user_message = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
        cleanup_calls: list[str] = []

        def fail_cleanup(cleanup_sid: str, *, delete_files: bool = True) -> dict[str, object]:
            cleanup_calls.append(cleanup_sid)
            raise OSError("unlink failed")

        mgr.clear_staged_attachments = fail_cleanup  # type: ignore[method-assign]
        mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}  # type: ignore[method-assign]
        seen: list[dict[str, object]] = []

        def sock_call(_sock: Path, req: dict[str, object], timeout_s: float | None = 0, **_kwargs: object) -> dict[str, object]:
            seen.append(req)
            return {"queued": False, "queue_len": 0}

        mgr._sock_call = sock_call  # type: ignore[method-assign]

        response = SessionManager.send(mgr, sid, "use this", allow_pending_attachment=True)

        committed_text = "Attachment 1: /tmp/uploads/s1/1_one.txt\nuse this"
        self.assertEqual(seen, [{"cmd": "send", "text": committed_text, "sync": True}])
        self.assertEqual(cleanup_calls, [sid])
        self.assertEqual(response, {"queued": False, "queue_len": 0, "attachment_cleanup_error": "unlink failed"})
        self.assertEqual(mgr._staged_attachments[sid], [entry])
        self.assertTrue(mgr._sessions[sid].pending_attachment)
        self.assertIn(sid, mgr._pending_attachment_ids)
        self.assertNotIn(sid, mgr._commit_unknown_sends)
        self.assertIsNone(mgr._sessions[sid].commit_unknown_send)

    def test_staged_attachment_keyerror_after_confirmed_send_is_success_with_visible_error(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._sessions[sid].pending_attachment = True
        mgr._pending_attachment_ids.add(sid)
        entry = {"id": "a1", "display_name": "one.txt", "filename": "1_one.txt", "path": "/tmp/uploads/s1/1_one.txt", "size": 3, "created_ts": 1.0}
        mgr._staged_attachments[sid] = [dict(entry)]
        mgr._record_prelog_user_message = lambda *_args, **_kwargs: None  # type: ignore[method-assign]

        def fail_cleanup(_cleanup_sid: str, *, delete_files: bool = True) -> dict[str, object]:
            raise KeyError("unknown session")

        mgr.clear_staged_attachments = fail_cleanup  # type: ignore[method-assign]
        mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}  # type: ignore[method-assign]
        mgr._sock_call = lambda *_args, **_kwargs: {"queued": False, "queue_len": 0}  # type: ignore[method-assign]

        response = SessionManager.send(mgr, sid, "use this", allow_pending_attachment=True)

        self.assertEqual(response, {"queued": False, "queue_len": 0, "attachment_cleanup_error": "unknown session"})
        self.assertEqual(mgr._staged_attachments[sid], [entry])
        self.assertTrue(mgr._sessions[sid].pending_attachment)

    def test_pending_attachment_persistence_failure_after_confirmed_send_is_success_with_visible_error(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._record_prelog_user_message = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
        mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}  # type: ignore[method-assign]
        mgr._sock_call = lambda *_args, **_kwargs: {"queued": False, "queue_len": 0}  # type: ignore[method-assign]
        pending_clear_calls: list[tuple[str, bool]] = []

        def fail_pending_clear(clear_sid: str, value: bool) -> None:
            pending_clear_calls.append((clear_sid, value))
            SessionManager._set_pending_attachment(mgr, clear_sid, value)
            raise OSError("pending.json write failed")

        mgr._set_pending_attachment = fail_pending_clear  # type: ignore[method-assign]

        response = SessionManager.send(mgr, sid, "normal prompt")

        self.assertEqual(pending_clear_calls, [(sid, False)])
        self.assertEqual(response, {"queued": False, "queue_len": 0, "send_state_cleanup_error": "pending_attachment: pending.json write failed"})
        self.assertFalse(mgr._sessions[sid].pending_attachment)
        self.assertNotIn(sid, mgr._pending_attachment_ids)
        self.assertNotIn(sid, mgr._commit_unknown_sends)
        self.assertIsNone(mgr._sessions[sid].commit_unknown_send)

    def test_commit_unknown_clear_persistence_failure_after_confirmed_send_is_success_with_visible_error(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._record_prelog_user_message = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
        mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}  # type: ignore[method-assign]
        mgr._sock_call = lambda *_args, **_kwargs: {"queued": False, "queue_len": 0}  # type: ignore[method-assign]
        clear_unknown_calls: list[tuple[str, dict[str, Any] | None]] = []

        def fail_commit_unknown_clear(clear_sid: str, record: dict[str, Any] | None) -> None:
            clear_unknown_calls.append((clear_sid, record))
            SessionManager._set_commit_unknown_send(mgr, clear_sid, record)
            raise OSError("commit_unknown.json write failed")

        mgr._set_commit_unknown_send = fail_commit_unknown_clear  # type: ignore[method-assign]

        response = SessionManager.send(mgr, sid, "normal prompt")

        self.assertEqual(clear_unknown_calls, [(sid, None)])
        self.assertEqual(response, {"queued": False, "queue_len": 0, "send_state_cleanup_error": "commit_unknown_send: commit_unknown.json write failed"})
        self.assertNotIn(sid, mgr._commit_unknown_sends)
        self.assertIsNone(mgr._sessions[sid].commit_unknown_send)

    def test_staged_attachments_survive_commit_unknown_send(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._sessions[sid].pending_attachment = True
        mgr._pending_attachment_ids.add(sid)
        entry = {"id": "a1", "display_name": "one.txt", "filename": "1_one.txt", "path": "/tmp/uploads/s1/1_one.txt", "size": 3, "created_ts": 1.0}
        mgr._staged_attachments[sid] = [dict(entry)]
        mgr._record_prelog_user_message = lambda *_args, **_kwargs: self.fail("unknown commit should not be recorded as submitted")  # type: ignore[method-assign]
        mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}  # type: ignore[method-assign]
        mgr._sock_call = lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError("timed out"))  # type: ignore[method-assign]

        with self.assertRaisesRegex(SessionCommitUnknownError, "commit status unknown"):
            SessionManager.send(mgr, sid, "use this", allow_pending_attachment=True)
        self.assertEqual(mgr._staged_attachments[sid], [entry])
        self.assertTrue(mgr._sessions[sid].pending_attachment)
        self.assertIn(sid, mgr._pending_attachment_ids)
        self.assertEqual(mgr._commit_unknown_sends[sid]["text"], "Attachment 1: /tmp/uploads/s1/1_one.txt\nuse this")
        self.assertEqual(mgr._commit_unknown_sends[sid]["display_text"], "use this")

    def test_send_rejects_broker_without_sync_capability(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._sessions[sid].sync_send_supported = False
        mgr.get_state = lambda _sid: self.fail("unsupported broker should fail before readiness")  # type: ignore[method-assign]
        mgr._sock_call = lambda *_args, **_kwargs: self.fail("unsupported broker should not receive send")  # type: ignore[method-assign]

        with self.assertRaisesRegex(SessionNotReadyError, "broker must be restarted"):
            SessionManager.send(mgr, sid, "normal prompt")

    def test_send_timeout_preserves_pending_attachment_as_unknown(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._sessions[sid].pending_attachment = True
        mgr._pending_attachment_ids.add(sid)
        mgr._record_prelog_user_message = lambda *_args, **_kwargs: self.fail("unknown commit should not be recorded as submitted")  # type: ignore[method-assign]
        mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}  # type: ignore[method-assign]
        mgr._sock_call = lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError("timed out"))  # type: ignore[method-assign]

        with self.assertRaisesRegex(SessionCommitUnknownError, "commit status unknown"):
            SessionManager.send(mgr, sid, "intended prompt", allow_pending_attachment=True)
        self.assertTrue(mgr._sessions[sid].pending_attachment)
        self.assertIn(sid, mgr._pending_attachment_ids)

    def test_post_request_socket_failure_is_commit_unknown_even_if_pids_dead(self) -> None:
        # SessionControlCoordinator.call_confirmed_send with the OS process
        # liveness probe injected as ``pid_alive=lambda _: False`` (replaces
        # former ``patch("codoxear.server._pid_alive", ...)``).  Post-request
        # failure (request_sent=True) raises commit-unknown before the
        # dead-process pruning branch, so the session survives.
        sid = "s1"
        session = _make_session(sid)
        sessions = {sid: session}
        coord = _control_coordinator(
            sessions=sessions,
            sock_call=lambda *_args, **_kwargs: (_ for _ in ()).throw(ControlSocketCallError("reset", request_sent=True)),
            pid_alive=lambda _pid: False,
        )

        def raise_commit_unknown(message: str, cause: BaseException | None = None) -> None:
            if cause is not None:
                raise SessionCommitUnknownError(message) from cause
            raise SessionCommitUnknownError(message)

        with self.assertRaisesRegex(SessionCommitUnknownError, "response failed"):
            coord.call_confirmed_send(
                sid,
                session=session,
                sock=session.sock_path,
                text="normal prompt",
                timeout_s=30.0,
                raise_commit_unknown=raise_commit_unknown,
                not_ready_error=SessionNotReadyError,
                timeout_errors=(TimeoutError,),
            )
        self.assertIn(sid, sessions)

    def test_pre_request_socket_failure_can_prune_dead_session(self) -> None:
        # SessionControlCoordinator.call_confirmed_send with ``pid_alive``
        # injected (replaces former ``patch("codoxear.server._pid_alive", ...)`
        # ``).  Pre-request failure (request_sent=False) with dead processes
        # prunes the session and raises KeyError.
        sid = "s1"
        session = _make_session(sid)
        sessions = {sid: session}

        def clear_deleted(deleted_sid: str) -> None:
            sessions.pop(deleted_sid, None)

        coord = _control_coordinator(
            sessions=sessions,
            sock_call=lambda *_args, **_kwargs: (_ for _ in ()).throw(ControlSocketCallError("connect failed", request_sent=False)),
            pid_alive=lambda _pid: False,
            clear_deleted_session_state=clear_deleted,
        )

        def raise_commit_unknown(message: str, cause: BaseException | None = None) -> None:
            raise SessionCommitUnknownError(message)

        with self.assertRaises(KeyError):
            coord.call_confirmed_send(
                sid,
                session=session,
                sock=session.sock_path,
                text="normal prompt",
                timeout_s=30.0,
                raise_commit_unknown=raise_commit_unknown,
                not_ready_error=SessionNotReadyError,
                timeout_errors=(TimeoutError,),
            )
        self.assertNotIn(sid, sessions)

    def test_send_explicit_commit_unknown_overrides_success_fields(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._record_prelog_user_message = lambda *_args, **_kwargs: self.fail("explicit unknown should not be recorded as submitted")  # type: ignore[method-assign]
        mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}  # type: ignore[method-assign]
        mgr._sock_call = lambda *_args, **_kwargs: {"queued": False, "queue_len": 0, "commit_unknown": True}  # type: ignore[method-assign]

        with self.assertRaisesRegex(SessionCommitUnknownError, "marked commit unknown"):
            SessionManager.send(mgr, sid, "normal prompt")
        self.assertEqual(mgr._commit_unknown_sends[sid]["text"], "normal prompt")
        self.assertEqual(mgr._commit_unknown_sends[sid]["display_text"], "normal prompt")
        self.assertEqual(mgr._sessions[sid].commit_unknown_send["text"], "normal prompt")
        self.assertEqual(mgr._sessions[sid].commit_unknown_send["display_text"], "normal prompt")

    def test_commit_unknown_send_blocks_retry_queue_and_sweep_until_cleared(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._sessions[sid].commit_unknown_send = {"text": "maybe sent", "created_ts": 123.0}
        mgr._commit_unknown_sends[sid] = {"text": "maybe sent", "created_ts": 123.0}
        mgr.get_state = lambda _sid: self.fail("unknown send should fail before broker readiness")  # type: ignore[method-assign]
        mgr._sock_call = lambda *_args, **_kwargs: self.fail("unknown send should not reach broker")  # type: ignore[method-assign]

        with self.assertRaisesRegex(SessionNotReadyError, "unknown send"):
            SessionManager.send(mgr, sid, "retry")
        with self.assertRaisesRegex(SessionNotReadyError, "unknown send"):
            SessionManager.enqueue(mgr, sid, "queued")
        self.assertFalse(SessionManager._queue_remote_ready(mgr, sid, log_path=None))

        self.assertEqual(SessionManager.clear_commit_unknown_send(mgr, sid), {"ok": True, "commit_unknown_send": False})
        self.assertIsNone(mgr._sessions[sid].commit_unknown_send)
        self.assertNotIn(sid, mgr._commit_unknown_sends)

    def test_send_empty_response_is_commit_unknown(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._record_prelog_user_message = lambda *_args, **_kwargs: self.fail("empty response should not be recorded as submitted")  # type: ignore[method-assign]
        mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}  # type: ignore[method-assign]
        mgr._sock_call = lambda *_args, **_kwargs: {"error": "empty response"}  # type: ignore[method-assign]

        with self.assertRaisesRegex(SessionCommitUnknownError, "empty"):
            SessionManager.send(mgr, sid, "normal prompt")

    def test_malformed_parseable_send_responses_are_commit_unknown(self) -> None:
        for response in [
            {},
            None,
            {"queued": False, "queue_len": "notint"},
            {"queued": False, "queue_len": -1},
            {"queued": False, "queue_len": True},
            {"queued": False, "queue_len": 1.9},
            {"queued": False, "queue_len": "0"},
            {"queued": False, "queue_len": 0, "busy": "false"},
            {"queued": False, "queue_len": 0, "busy": 0},
        ]:
            with self.subTest(response=response):
                sid = "s1"
                mgr = self._mgr()
                mgr._sessions[sid] = _make_session(sid)
                mgr._record_prelog_user_message = lambda *_args, **_kwargs: self.fail("malformed response should not be recorded as submitted")  # type: ignore[method-assign]
                mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}  # type: ignore[method-assign]
                mgr._sock_call = lambda *_args, _response=response, **_kwargs: _response  # type: ignore[method-assign]

                with self.assertRaisesRegex(SessionCommitUnknownError, "commit status unknown"):
                    SessionManager.send(mgr, sid, "normal prompt")

    def test_attachment_rejects_broker_without_confirmed_send_capabilities(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._sessions[sid].sync_send_supported = False
        mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}  # type: ignore[method-assign]

        with self.assertRaisesRegex(SessionNotReadyError, "broker must be restarted"):
            SessionManager.attachment_staging_ready(mgr, sid)

    def test_attachment_staging_does_not_require_key_write_errors(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._sessions[sid].key_write_errors_supported = False
        mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}  # type: ignore[method-assign]

        self.assertTrue(SessionManager.attachment_staging_ready(mgr, sid))

    def test_queue_send_timeout_marks_head_commit_unknown(self) -> None:
        # SessionQueueCoordinator.promote_head_if_sendable with ``now=123.0``
        # injected (replaces former ``patch("codoxear.server.time.time", ...)``).
        sid = "s1"
        session = _make_session(sid)
        sessions = {sid: session}
        queues = {sid: [_queue_item("q1", "queued first")]}
        with TemporaryDirectory() as td:
            coord, _saves = _queue_coordinator(
                sessions=sessions,
                queues=queues,
                queue_dir=Path(td),
                now=123.0,
                send=lambda *_args, **_kwargs: (_ for _ in ()).throw(SessionCommitUnknownError("unknown")),
            )
            resp = coord.promote_head_if_sendable(sid, require_idle_grace=False, expected_item_id="q1")

        self.assertTrue(resp and resp.get("commit_unknown"))
        self.assertIsNone(session.queue_sending_item_id)
        self.assertTrue(queues[sid][0].get("commit_unknown"))
        self.assertEqual(queues[sid][0].get("commit_unknown_ts"), 123.0)

    def test_queue_generic_pre_dispatch_failure_clears_pre_dispatch_unknown_marker(self) -> None:
        # SessionQueueCoordinator.promote_head_if_sendable with ``now=111.0``
        # injected.  Generic (non-retryable, non-commit-unknown) send failure
        # clears the pre-dispatch commit-unknown marker.
        sid = "s1"
        session = _make_session(sid)
        sessions = {sid: session}
        queues = {sid: [_queue_item("q1", "queued first")]}
        with TemporaryDirectory() as td:
            coord, _saves = _queue_coordinator(
                sessions=sessions,
                queues=queues,
                queue_dir=Path(td),
                now=111.0,
                send=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("pre-send readiness exploded")),
            )
            self.assertIsNone(coord.promote_head_if_sendable(sid, require_idle_grace=False, expected_item_id="q1"))
        self.assertIsNone(session.queue_sending_item_id)
        self.assertFalse(queues[sid][0].get("commit_unknown"))

    def test_queue_known_send_failure_clears_pre_dispatch_unknown_marker(self) -> None:
        # SessionQueueCoordinator.promote_head_if_sendable with ``now=321.0``
        # injected.  SessionInjectionError is a retryable send error, so the
        # pre-dispatch commit-unknown marker is cleared.
        sid = "s1"
        session = _make_session(sid)
        sessions = {sid: session}
        queues = {sid: [_queue_item("q1", "queued first")]}
        with TemporaryDirectory() as td:
            coord, _saves = _queue_coordinator(
                sessions=sessions,
                queues=queues,
                queue_dir=Path(td),
                now=321.0,
                send=lambda *_args, **_kwargs: (_ for _ in ()).throw(SessionInjectionError("no pty")),
            )
            self.assertIsNone(coord.promote_head_if_sendable(sid, require_idle_grace=False, expected_item_id="q1"))
        self.assertIsNone(session.queue_sending_item_id)
        self.assertFalse(queues[sid][0].get("commit_unknown"))

    def test_queue_broker_declared_unknown_keeps_commit_unknown_marker(self) -> None:
        # SessionQueueCoordinator.promote_head_if_sendable with ``now=654.0``
        # injected.  SessionCommitUnknownError preserves the commit-unknown
        # marker with the injected timestamp.
        sid = "s1"
        session = _make_session(sid)
        sessions = {sid: session}
        queues = {sid: [_queue_item("q1", "queued first")]}
        with TemporaryDirectory() as td:
            coord, _saves = _queue_coordinator(
                sessions=sessions,
                queues=queues,
                queue_dir=Path(td),
                now=654.0,
                send=lambda *_args, **_kwargs: (_ for _ in ()).throw(SessionCommitUnknownError("partial write")),
            )
            resp = coord.promote_head_if_sendable(sid, require_idle_grace=False, expected_item_id="q1")
        self.assertTrue(resp and resp.get("commit_unknown"))
        self.assertTrue(queues[sid][0].get("commit_unknown"))
        self.assertEqual(queues[sid][0].get("commit_unknown_ts"), 654.0)

    def test_clear_pending_attachment_clears_persisted_flag(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._sessions[sid].pending_attachment = True
        mgr._pending_attachment_ids.add(sid)

        self.assertEqual(SessionManager.clear_pending_attachment(mgr, sid), {"ok": True, "pending_attachment": False})
        self.assertFalse(mgr._sessions[sid].pending_attachment)
        self.assertNotIn(sid, mgr._pending_attachment_ids)

    def test_queue_head_is_durably_unknown_before_dispatch(self) -> None:
        # SessionQueueCoordinator.promote_head_if_sendable with ``now=456.0``
        # injected.  The head is marked commit-unknown *before* send is called,
        # so the send callback observes the durable marker.
        sid = "s1"
        session = _make_session(sid)
        sessions = {sid: session}
        queues = {sid: [_queue_item("q1", "queued first")]}
        observed: list[bool] = []

        def send(_sid: str, _text: str, **_kwargs: object) -> dict[str, object]:
            observed.append(bool(queues[sid][0].get("commit_unknown")))
            return {"queued": False, "queue_len": 0}

        with TemporaryDirectory() as td:
            coord, _saves = _queue_coordinator(
                sessions=sessions,
                queues=queues,
                queue_dir=Path(td),
                now=456.0,
                send=send,
            )
            self.assertEqual(coord.promote_head_if_sendable(sid, require_idle_grace=False, expected_item_id="q1"), {"queued": False, "queue_len": 0})
        self.assertEqual(observed, [True])
        self.assertNotIn(sid, queues)

    def test_commit_unknown_queue_head_does_not_auto_promote(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._queues[sid] = [dict(_queue_item("q1", "maybe sent"), commit_unknown=True)]
        mgr.get_state = lambda _sid: self.fail("commit-unknown head should block before broker state")  # type: ignore[method-assign]

        self.assertIsNone(SessionManager._promote_queue_head_if_sendable(mgr, sid, require_idle_grace=False, expected_item_id="q1"))

    def test_recovery_tail_blocks_queue_auto_promote(self) -> None:
        sid = "s1"
        for flag in ["orphan_recovery", "commit_unknown"]:
            with self.subTest(flag=flag):
                mgr = self._mgr()
                mgr._sessions[sid] = _make_session(sid)
                mgr._queues[sid] = [_queue_item("q1", "normal first"), dict(_queue_item("q2", "recover tail"), **{flag: True})]
                state_calls = 0

                def get_state(_sid: str) -> dict[str, object]:
                    nonlocal state_calls
                    state_calls += 1
                    return {"busy": False, "queue_len": 0}

                mgr.get_state = get_state  # type: ignore[method-assign]

                self.assertIsNone(SessionManager._promote_queue_head_if_sendable(mgr, sid, require_idle_grace=False, expected_item_id="q1"))
                self.assertEqual(state_calls, 0)

    def test_active_recovery_queue_protects_unflagged_items(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._queues[sid] = [
            _queue_item("q1", "normal first"),
            dict(_queue_item("u", "maybe sent"), commit_unknown=True),
            _queue_item("q2", "normal tail"),
        ]

        listed = SessionManager.queue_list(mgr, sid)
        by_id = {item["id"]: item for item in listed}
        self.assertTrue(by_id["q1"]["orphan_recovery"])
        self.assertTrue(by_id["u"]["commit_unknown"])
        self.assertTrue(by_id["q2"]["orphan_recovery"])
        with self.assertRaisesRegex(ValueError, "preserved for recovery"):
            SessionManager.queue_update(mgr, sid, "q1", "edited")
        with self.assertRaisesRegex(ValueError, "preserved for recovery"):
            SessionManager.queue_move(mgr, sid, "q2", 0)
        with self.assertRaisesRegex(ValueError, "explicit confirmation"):
            SessionManager.queue_delete(mgr, sid, "q2")
        self.assertEqual(SessionManager.queue_delete(mgr, sid, "q2", allow_orphan_recovery=True), {"ok": True, "queue_len": 2})
        self.assertTrue(mgr._queues[sid][0]["orphan_recovery"])

    def test_commit_unknown_queue_delete_requires_explicit_confirmation(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._queues[sid] = [dict(_queue_item("q1", "maybe sent"), commit_unknown=True), _queue_item("q2", "later")]

        with self.assertRaisesRegex(ValueError, "explicit confirmation"):
            SessionManager.queue_delete(mgr, sid, "q1")
        with self.assertRaisesRegex(ValueError, "preserved for recovery"):
            SessionManager.queue_update(mgr, sid, "q1", "edited maybe sent")
        self.assertEqual([item["id"] for item in mgr._queues[sid]], ["q1", "q2"])

        self.assertEqual(SessionManager.queue_delete(mgr, sid, "q1", allow_commit_unknown=True), {"ok": True, "queue_len": 1})
        self.assertEqual([item["id"] for item in mgr._queues[sid]], ["q2"])

    def test_orphan_commit_unknown_queue_is_skipped_by_sweep_and_reviewable(self) -> None:
        mgr = self._mgr()
        mgr._sessions["live"] = _make_session("live")
        mgr._queues = {
            "orphan": [dict(_queue_item("u", "maybe sent"), commit_unknown=True)],
            "live": [_queue_item("q1", "live queued")],
        }
        mgr._discover_existing_if_stale = lambda: None  # type: ignore[method-assign]
        mgr._prune_dead_sessions = lambda: None  # type: ignore[method-assign]
        called: list[str] = []
        mgr._maybe_drain_session_queue = lambda sid: called.append(sid) or False  # type: ignore[method-assign]

        SessionManager._queue_sweep(mgr)

        self.assertEqual(called, ["live"])
        self.assertIn("orphan", mgr._queues)
        self.assertTrue(SessionManager.queue_list(mgr, "orphan")[0]["commit_unknown"])
        with self.assertRaisesRegex(ValueError, "explicit confirmation"):
            SessionManager.queue_delete(mgr, "orphan", "u")
        self.assertEqual(SessionManager.queue_delete(mgr, "orphan", "u", allow_commit_unknown=True), {"ok": True, "queue_len": 0})
        self.assertNotIn("orphan", mgr._queues)

    def test_orphan_recovery_queue_head_does_not_auto_promote(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._queues[sid] = [dict(_queue_item("n", "later unsent"), orphan_recovery=True)]
        mgr.get_state = lambda _sid: self.fail("orphan recovery head must block before broker state")  # type: ignore[method-assign]

        self.assertIsNone(SessionManager._promote_queue_head_if_sendable(mgr, sid, require_idle_grace=False, expected_item_id="n"))

    def test_delete_session_clears_orphan_recovery_rows(self) -> None:
        mgr = self._mgr()
        mgr._aliases = {}
        mgr._sidebar_meta = {}
        mgr._unattended = {}
        mgr._files = {}
        mgr._input_locks = {}
        mgr._save_aliases = lambda: None
        mgr._save_sidebar_meta = lambda: None
        mgr._save_unattended = lambda: None
        mgr._save_files = lambda: None
        mgr.kill_session = lambda _sid: self.fail("orphan delete should not kill a process")  # type: ignore[method-assign]
        mgr._commit_unknown_sends["direct"] = {"text": "maybe direct", "created_ts": 1.0}
        mgr._queues["queue"] = [dict(_queue_item("n", "later"), orphan_recovery=True)]

        self.assertTrue(SessionManager.delete_session(mgr, "direct"))
        self.assertTrue(SessionManager.delete_session(mgr, "queue"))

        self.assertNotIn("direct", mgr._commit_unknown_sends)
        self.assertNotIn("queue", mgr._queues)

    def test_orphan_queue_remains_reviewable_after_unknown_item_delete(self) -> None:
        mgr = self._mgr()
        mgr._discover_existing_if_stale = lambda: None  # type: ignore[method-assign]
        mgr._prune_dead_sessions = lambda: None  # type: ignore[method-assign]
        mgr._update_meta_counters = lambda: None  # type: ignore[method-assign]
        mgr._include_launch_attempts = False
        mgr._unattended = {}
        mgr._aliases = {}
        mgr._sidebar_meta = {}
        mgr._files = {}
        mgr._recent_cwds = {}
        mgr._save_files = lambda: None
        mgr._save_sidebar_meta = lambda: None
        mgr._save_recent_cwds = lambda: None
        mgr._queues["orphan"] = [dict(_queue_item("u", "maybe sent"), commit_unknown=True), _queue_item("n", "later unsent")]

        before = SessionManager.queue_list(mgr, "orphan")
        self.assertTrue(before[1]["orphan_recovery"])
        with self.assertRaisesRegex(ValueError, "explicit confirmation"):
            SessionManager.queue_delete(mgr, "orphan", "n")

        self.assertEqual(SessionManager.queue_delete(mgr, "orphan", "u", allow_commit_unknown=True), {"ok": True, "queue_len": 1})

        remaining = SessionManager.queue_list(mgr, "orphan")
        self.assertEqual([item["id"] for item in remaining], ["n"])
        self.assertTrue(remaining[0]["orphan_recovery"])
        with self.assertRaisesRegex(ValueError, "explicit confirmation"):
            SessionManager.queue_delete(mgr, "orphan", "n")
        with self.assertRaisesRegex(ValueError, "preserved for recovery"):
            SessionManager.queue_update(mgr, "orphan", "n", "changed")
        with self.assertRaisesRegex(ValueError, "preserved for recovery"):
            SessionManager.queue_move(mgr, "orphan", "n", 0)
        rows = SessionManager.list_sessions(mgr)
        by_id = {row["session_id"]: row for row in rows}
        self.assertTrue(by_id["orphan"]["orphan_recovery"])
        self.assertEqual(by_id["orphan"]["queue_len"], 1)
        self.assertEqual(SessionManager.queue_delete(mgr, "orphan", "n", allow_orphan_recovery=True), {"ok": True, "queue_len": 0})
        self.assertNotIn("orphan", mgr._queues)

    def test_active_orphan_recovery_queue_item_blocks_update_and_move(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._queues[sid] = [dict(_queue_item("r", "recover"), orphan_recovery=True), _queue_item("n", "normal")]

        with self.assertRaisesRegex(ValueError, "preserved for recovery"):
            SessionManager.queue_update(mgr, sid, "r", "changed")
        with self.assertRaisesRegex(ValueError, "preserved for recovery"):
            SessionManager.queue_move(mgr, sid, "r", 1)
        with self.assertRaisesRegex(ValueError, "preserved for recovery"):
            SessionManager.queue_move(mgr, sid, "n", 0)

        mgr._discover_existing_if_stale = lambda: None  # type: ignore[method-assign]
        mgr._prune_dead_sessions = lambda: None  # type: ignore[method-assign]
        mgr._update_meta_counters = lambda: None  # type: ignore[method-assign]
        mgr._include_launch_attempts = False
        mgr._unattended = {}
        mgr._aliases = {}
        mgr._sidebar_meta = {}
        mgr._files = {}
        mgr._recent_cwds = {}
        mgr._save_files = lambda: None
        mgr._save_sidebar_meta = lambda: None
        mgr._save_recent_cwds = lambda: None
        row = SessionManager.list_sessions(mgr)[0]
        self.assertFalse(row.get("orphan_recovery", False))
        self.assertTrue(row["queue_recovery"])
        self.assertEqual(row["queue_len"], 2)

    def test_active_commit_unknown_queue_item_is_queue_recovery(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._queues[sid] = [dict(_queue_item("u", "maybe sent"), commit_unknown=True)]
        mgr._discover_existing_if_stale = lambda: None  # type: ignore[method-assign]
        mgr._prune_dead_sessions = lambda: None  # type: ignore[method-assign]
        mgr._update_meta_counters = lambda: None  # type: ignore[method-assign]
        mgr._include_launch_attempts = False
        mgr._unattended = {}
        mgr._aliases = {}
        mgr._sidebar_meta = {}
        mgr._files = {}
        mgr._recent_cwds = {}
        mgr._save_files = lambda: None
        mgr._save_sidebar_meta = lambda: None
        mgr._save_recent_cwds = lambda: None

        row = SessionManager.list_sessions(mgr)[0]

        self.assertTrue(row["queue_recovery"])
        self.assertEqual(row["queue_len"], 1)

    def test_active_direct_unknown_with_queue_is_queue_recovery(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._sessions[sid].commit_unknown_send = {"text": "maybe direct", "created_ts": 1.0}
        mgr._commit_unknown_sends[sid] = {"text": "maybe direct", "created_ts": 1.0}
        mgr._queues[sid] = [_queue_item("q", "plain tail")]
        mgr._discover_existing_if_stale = lambda: None  # type: ignore[method-assign]
        mgr._prune_dead_sessions = lambda: None  # type: ignore[method-assign]
        mgr._update_meta_counters = lambda: None  # type: ignore[method-assign]
        mgr._include_launch_attempts = False
        mgr._unattended = {}
        mgr._aliases = {}
        mgr._sidebar_meta = {}
        mgr._files = {}
        mgr._recent_cwds = {}
        mgr._save_files = lambda: None
        mgr._save_sidebar_meta = lambda: None
        mgr._save_recent_cwds = lambda: None

        row = SessionManager.list_sessions(mgr)[0]
        listed = SessionManager.queue_list(mgr, sid)

        self.assertTrue(row["commit_unknown_send"])
        self.assertTrue(row["queue_recovery"])
        self.assertEqual(row["queue_len"], 1)
        self.assertTrue(listed[0]["orphan_recovery"])
        with self.assertRaisesRegex(ValueError, "preserved for recovery"):
            SessionManager.queue_update(mgr, sid, "q", "edited")
        with self.assertRaisesRegex(ValueError, "preserved for recovery"):
            SessionManager.queue_move(mgr, sid, "q", 0)
        with self.assertRaisesRegex(ValueError, "explicit confirmation"):
            SessionManager.queue_delete(mgr, sid, "q")

    def test_enqueue_rejects_active_recovery_queue_barriers(self) -> None:
        for flag in ["orphan_recovery", "commit_unknown"]:
            with self.subTest(flag=flag):
                sid = "s1"
                mgr = self._mgr()
                mgr._sessions[sid] = _make_session(sid)
                mgr._queues[sid] = [dict(_queue_item("r", "recover"), **{flag: True})]

                with self.assertRaisesRegex(SessionNotReadyError, "recovery queue"):
                    SessionManager.enqueue(mgr, sid, "new prompt")
                self.assertEqual(len(mgr._queues[sid]), 1)

    def test_enqueue_rechecks_recovery_barrier_at_append(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._queues[sid] = [_queue_item("q1", "normal head")]
        calls = 0
        real_check = SessionManager._queue_has_recovery_items_locked.__get__(mgr, SessionManager)

        def racing_check(session_id: str) -> bool:
            nonlocal calls
            calls += 1
            if calls == 1:
                mgr._queues[session_id][0]["commit_unknown"] = True
                return False
            return real_check(session_id)

        mgr._queue_has_recovery_items_locked = racing_check  # type: ignore[method-assign]

        with self.assertRaisesRegex(SessionNotReadyError, "recovery queue"):
            SessionManager.enqueue(mgr, sid, "new prompt")
        self.assertEqual(len(mgr._queues[sid]), 1)
        self.assertTrue(mgr._queues[sid][0]["commit_unknown"])

    def test_recovery_delete_marks_tail_even_before_session_prune(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._aliases = {}
        mgr._sidebar_meta = {}
        mgr._unattended = {}
        mgr._files = {}
        mgr._input_locks = {}
        mgr._save_aliases = lambda: None
        mgr._save_sidebar_meta = lambda: None
        mgr._save_unattended = lambda: None
        mgr._save_files = lambda: None
        mgr._queues[sid] = [dict(_queue_item("u", "maybe sent"), commit_unknown=True), _queue_item("n", "plain tail")]

        self.assertEqual(SessionManager.queue_delete(mgr, sid, "u", allow_commit_unknown=True), {"ok": True, "queue_len": 1})
        self.assertTrue(mgr._queues[sid][0]["orphan_recovery"])

        SessionManager._clear_deleted_session_state(mgr, sid)
        self.assertIn(sid, mgr._queues)
        self.assertTrue(mgr._queues[sid][0]["orphan_recovery"])

    def test_direct_unknown_preserves_plain_orphan_queue_tail(self) -> None:
        mgr = self._mgr()
        mgr._commit_unknown_sends["orphan"] = {"text": "maybe direct", "created_ts": 1.0}
        mgr._queues["orphan"] = [_queue_item("n", "plain tail")]
        mgr._discover_existing_if_stale = lambda: None  # type: ignore[method-assign]
        mgr._prune_dead_sessions = lambda: None  # type: ignore[method-assign]
        mgr._maybe_drain_session_queue = lambda sid: self.fail("orphan direct unknown queue must not drain")  # type: ignore[method-assign]

        SessionManager._queue_sweep(mgr)

        listed = SessionManager.queue_list(mgr, "orphan")
        self.assertTrue(listed[0]["orphan_recovery"])
        self.assertIn("orphan", mgr._queues)

    def test_orphan_queue_remains_reviewable_after_recovery_item_delete(self) -> None:
        mgr = self._mgr()
        mgr._queues["orphan"] = [dict(_queue_item("r", "recover"), orphan_recovery=True), _queue_item("n", "plain tail")]

        self.assertEqual(SessionManager.queue_delete(mgr, "orphan", "r", allow_orphan_recovery=True), {"ok": True, "queue_len": 1})

        remaining = SessionManager.queue_list(mgr, "orphan")
        self.assertEqual([item["id"] for item in remaining], ["n"])
        self.assertTrue(remaining[0]["orphan_recovery"])

    def test_clear_direct_unknown_preserves_plain_orphan_queue_tail(self) -> None:
        mgr = self._mgr()
        saved_queues: list[list[dict[str, object]]] = []
        mgr._save_queues = lambda: saved_queues.append([dict(item) for item in mgr._queues.get("orphan", [])])  # type: ignore[method-assign]
        mgr._commit_unknown_sends["orphan"] = {"text": "maybe direct", "created_ts": 1.0}
        mgr._queues["orphan"] = [_queue_item("n", "plain tail")]

        self.assertEqual(SessionManager.clear_commit_unknown_send(mgr, "orphan"), {"ok": True, "commit_unknown_send": False})
        dropped = mgr._queue_store_for_manager().drop_missing_sessions(mgr._queues, set())

        self.assertFalse(dropped)
        self.assertIn("orphan", mgr._queues)
        self.assertTrue(mgr._queues["orphan"][0]["orphan_recovery"])
        self.assertEqual(saved_queues[-1][0]["orphan_recovery"], True)

    def test_prune_old_direct_unknown_preserves_plain_orphan_queue_tail(self) -> None:
        # SessionPendingStateCoordinator.prune_missing_commit_unknown_sends
        # with ``now=10_000_000.0`` injected (replaces former
        # ``patch("codoxear.server.time.time", ...)``).  Pruning the stale
        # direct-unknown marks the orphan queue tail for recovery so it is not
        # dropped by drop_missing_sessions.
        saved_queues: list[list[dict[str, object]]] = []
        commit_unknown_sends = {"orphan": {"text": "maybe direct", "created_ts": 1.0}}
        queues = {"orphan": [_queue_item("n", "plain tail")]}
        with TemporaryDirectory() as td:
            store = _session_store(Path(td))
            store.commit_unknown_sends = commit_unknown_sends
            store.queues = queues
            coord = _pending_state_coordinator(
                store=store,
                sessions={},
                now=10_000_000.0,
                save_queues=lambda: saved_queues.append([dict(item) for item in queues.get("orphan", [])]),
            )
            self.assertTrue(coord.prune_missing_commit_unknown_sends(max_age_seconds=7 * 24 * 3600))
            dropped = store.queue_store.drop_missing_sessions(queues, set())

        self.assertFalse(dropped)
        self.assertNotIn("orphan", commit_unknown_sends)
        self.assertIn("orphan", queues)
        self.assertTrue(queues["orphan"][0]["orphan_recovery"])
        self.assertEqual(saved_queues[-1][0]["orphan_recovery"], True)

    def test_orphan_direct_unknown_can_be_cleared_without_active_session(self) -> None:
        mgr = self._mgr()
        mgr._commit_unknown_sends["orphan"] = {"text": "maybe direct", "created_ts": 1.0}

        self.assertEqual(SessionManager.clear_commit_unknown_send(mgr, "orphan"), {"ok": True, "commit_unknown_send": False})
        self.assertNotIn("orphan", mgr._commit_unknown_sends)

    def test_list_sessions_exposes_orphan_unknown_recovery_rows(self) -> None:
        mgr = self._mgr()
        mgr._discover_existing_if_stale = lambda: None  # type: ignore[method-assign]
        mgr._prune_dead_sessions = lambda: None  # type: ignore[method-assign]
        mgr._update_meta_counters = lambda: None  # type: ignore[method-assign]
        mgr._include_launch_attempts = False
        mgr._unattended = {}
        mgr._aliases = {}
        mgr._sidebar_meta = {}
        mgr._files = {}
        mgr._recent_cwds = {}
        mgr._save_files = lambda: None
        mgr._save_sidebar_meta = lambda: None
        mgr._save_recent_cwds = lambda: None
        mgr._commit_unknown_sends["direct-orphan"] = {"text": "maybe direct", "created_ts": 10.0}
        mgr._queues["queue-orphan"] = [dict(_queue_item("u", "maybe queued"), commit_unknown=True, commit_unknown_ts=20.0)]

        rows = SessionManager.list_sessions(mgr)
        by_id = {row["session_id"]: row for row in rows}

        self.assertTrue(by_id["direct-orphan"]["orphan_recovery"])
        self.assertEqual(by_id["direct-orphan"]["transcript_state"], "failed")
        self.assertTrue(by_id["direct-orphan"]["commit_unknown_send"])
        self.assertEqual(by_id["direct-orphan"]["commit_unknown_send_text"], "maybe direct")
        self.assertTrue(by_id["queue-orphan"]["orphan_recovery"])
        self.assertEqual(by_id["queue-orphan"]["queue_len"], 1)

    def test_send_rechecks_pending_attachment_after_waiting_for_input_lock(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._record_prelog_user_message = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
        mgr._sock_call = lambda *_args, **_kwargs: self.fail("send must not reach broker after pending appears")  # type: ignore[method-assign]
        input_lock = SessionManager._input_lock_for_session(mgr, sid)
        input_lock.acquire()
        try:
            mgr._sessions[sid].pending_attachment = True
            with self.assertRaisesRegex(SessionNotReadyError, "pending attachment"):
                input_lock.release()
                SessionManager.send(mgr, sid, "stale direct prompt")
                input_lock.acquire()
        finally:
            if input_lock.acquire(blocking=False):
                input_lock.release()

    def test_remote_readiness_skips_log_parse_when_broker_queue_is_nonempty(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        with TemporaryDirectory() as td:
            log_path = Path(td) / "pi.jsonl"
            log_path.write_text('{"type":"message","message":{"role":"assistant","content":[],"stopReason":"aborted"}}\n', encoding="utf-8")
            mgr._sessions[sid] = _make_session(sid)
            mgr._sessions[sid].log_path = log_path
            mgr.idle_from_log = lambda _sid: self.fail("broker queue should short-circuit before log idle parse")  # type: ignore[method-assign]

            self.assertFalse(SessionManager._remote_ready_from_state_and_log(mgr, sid, {"busy": False, "queue_len": 1}, log_path))

    def test_send_readiness_allows_stale_broker_busy_when_log_is_idle(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        with TemporaryDirectory() as td:
            log_path = Path(td) / "pi.jsonl"
            log_path.write_text('{"type":"message","message":{"role":"assistant","content":[],"stopReason":"aborted"}}\n', encoding="utf-8")
            session = _make_session(sid)
            session.log_path = log_path
            mgr._sessions[sid] = session
            mgr.get_state = lambda _sid: {"busy": True, "queue_len": 0}  # type: ignore[method-assign]
            mgr.idle_from_log = lambda _sid: True  # type: ignore[method-assign]

            self.assertTrue(SessionManager._send_remote_ready(mgr, sid))

    def test_send_readiness_allows_interrupted_idle_when_log_is_still_busy(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        with TemporaryDirectory() as td:
            log_path = Path(td) / "pi.jsonl"
            log_path.write_text('{"type":"message","message":{"role":"user","content":[{"type":"text","text":"run"}]}}\n', encoding="utf-8")
            session = _make_session(sid)
            session.log_path = log_path
            # Stored interrupted_idle=True mirrors what get_state records from the
            # broker's unsuppressed True; readiness authority is the stored flag.
            session.interrupted_idle = True
            mgr._sessions[sid] = session
            mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0, "interrupted_idle": True}  # type: ignore[method-assign]
            mgr.idle_from_log = lambda _sid: False  # type: ignore[method-assign]

            self.assertTrue(SessionManager._send_remote_ready(mgr, sid))

    def test_send_readiness_rejects_interrupted_idle_before_confirmed_send_advances(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        with TemporaryDirectory() as td:
            log_path = Path(td) / "pi.jsonl"
            log_path.write_text('{"type":"message","message":{"role":"user","content":[{"type":"text","text":"run"}]}}\n', encoding="utf-8")
            session = _make_session(sid)
            session.log_path = log_path
            # Stored interrupted_idle=True mirrors what get_state records from the
            # broker's unsuppressed True; readiness authority is the stored flag.
            session.interrupted_idle = True
            session.last_send_boundary_active = True
            session.last_send_log_path = log_path
            session.last_send_log_size = log_path.stat().st_size
            mgr._sessions[sid] = session
            mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0, "interrupted_idle": True}  # type: ignore[method-assign]
            mgr.idle_from_log = lambda _sid: False  # type: ignore[method-assign]

            self.assertFalse(SessionManager._send_remote_ready(mgr, sid))

            session.last_send_log_size -= 1
            self.assertTrue(SessionManager._send_remote_ready(mgr, sid))

    def test_send_readiness_rejects_stale_idle_log_before_confirmed_send_advances(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        with TemporaryDirectory() as td:
            log_path = Path(td) / "pi.jsonl"
            base_row = '{"type":"message","message":{"role":"assistant","content":[],"stopReason":"aborted"}}\n'
            log_path.write_text(base_row, encoding="utf-8")
            session = _make_session(sid)
            session.log_path = log_path
            session.last_send_boundary_active = True
            session.last_send_log_path = log_path
            session.last_send_log_size = mgr._log_size_or_none(log_path)
            mgr._sessions[sid] = session
            mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0, "interrupted_idle": True}  # type: ignore[method-assign]
            mgr.idle_from_log = lambda _sid: True  # type: ignore[method-assign]

            self.assertFalse(SessionManager._send_remote_ready(mgr, sid))

            for suffix in ("\n", "garbage\n", "[]\n", '{"type":"message","message":{"role":"assistant","content":['):
                log_path.write_text(base_row + suffix, encoding="utf-8")
                self.assertFalse(SessionManager._send_remote_ready(mgr, sid))
                self.assertTrue(session.last_send_boundary_active)

            log_path.write_text(base_row + base_row, encoding="utf-8")
            self.assertTrue(SessionManager._send_remote_ready(mgr, sid))

    def test_send_readiness_rejects_missing_log_before_confirmed_send_advances(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        with TemporaryDirectory() as td:
            log_path = Path(td) / "missing.jsonl"
            session = _make_session(sid)
            session.log_path = log_path
            session.last_send_boundary_active = True
            session.last_send_log_path = log_path
            session.last_send_log_size = 10
            mgr._sessions[sid] = session
            mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0, "interrupted_idle": True}  # type: ignore[method-assign]
            mgr.idle_from_log = lambda _sid: self.fail("missing log must not be parsed for readiness")  # type: ignore[method-assign]

            self.assertFalse(SessionManager._send_remote_ready(mgr, sid))

    def test_send_readiness_resolves_same_path_unknown_baseline_on_parseable_row(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        with TemporaryDirectory() as td:
            log_path = Path(td) / "missing-at-send.jsonl"
            session = _make_session(sid)
            session.log_path = log_path
            session.last_send_boundary_active = True
            session.last_send_log_path = log_path
            session.last_send_log_size = None
            mgr._sessions[sid] = session
            mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0, "interrupted_idle": True}  # type: ignore[method-assign]
            mgr.idle_from_log = lambda _sid: True  # type: ignore[method-assign]

            self.assertFalse(SessionManager._send_remote_ready(mgr, sid))
            self.assertTrue(session.last_send_boundary_active)

            log_path.write_text("\n", encoding="utf-8")
            self.assertFalse(SessionManager._send_remote_ready(mgr, sid))
            self.assertTrue(session.last_send_boundary_active)

            log_path.write_text('{"type":"message","message":{"role":"assistant","content":[],"stopReason":"aborted"}}\n', encoding="utf-8")
            self.assertTrue(SessionManager._send_remote_ready(mgr, sid))
            self.assertFalse(session.last_send_boundary_active)
            self.assertIsNone(session.last_send_log_path)
            self.assertIsNone(session.last_send_log_size)

    def test_send_readiness_rejects_no_log_confirmed_send_boundary(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        session = _make_session(sid)
        session.log_path = None
        session.last_send_boundary_active = True
        session.last_send_log_path = None
        session.last_send_log_size = None
        mgr._sessions[sid] = session
        mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0, "interrupted_idle": True}  # type: ignore[method-assign]
        mgr.idle_from_log = lambda _sid: self.fail("no-log boundary must not parse a log for readiness")  # type: ignore[method-assign]

        self.assertFalse(SessionManager._send_remote_ready(mgr, sid))

        session.last_send_boundary_active = False
        self.assertTrue(SessionManager._send_remote_ready(mgr, sid))

    def test_send_readiness_keeps_no_log_boundary_until_log_has_bytes(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        with TemporaryDirectory() as td:
            log_path = Path(td) / "pi.jsonl"
            log_path.write_text("", encoding="utf-8")
            session = _make_session(sid)
            session.log_path = log_path
            session.last_send_boundary_active = True
            session.last_send_log_path = None
            session.last_send_log_size = None
            mgr._sessions[sid] = session
            mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0, "interrupted_idle": True}  # type: ignore[method-assign]
            mgr.idle_from_log = lambda _sid: True  # type: ignore[method-assign]

            self.assertFalse(SessionManager._send_remote_ready(mgr, sid))

            for content in ("\n", "garbage\n", "[]\n", '{"type":"message","message":{"role":"assistant","content":['):
                log_path.write_text(content, encoding="utf-8")
                self.assertFalse(SessionManager._send_remote_ready(mgr, sid))
                self.assertTrue(session.last_send_boundary_active)

            log_path.write_text('{"type":"message","message":{"role":"assistant","content":[],"stopReason":"aborted"}}\n', encoding="utf-8")
            self.assertTrue(SessionManager._send_remote_ready(mgr, sid))
            self.assertFalse(session.last_send_boundary_active)
            self.assertIsNone(session.last_send_log_path)
            self.assertIsNone(session.last_send_log_size)

            session.log_path = None
            self.assertTrue(SessionManager._send_remote_ready(mgr, sid))

    def test_send_readiness_rejects_interrupted_idle_when_broker_is_busy(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        with TemporaryDirectory() as td:
            log_path = Path(td) / "pi.jsonl"
            log_path.write_text('{"type":"message","message":{"role":"user","content":[{"type":"text","text":"run"}]}}\n', encoding="utf-8")
            session = _make_session(sid)
            session.log_path = log_path
            mgr._sessions[sid] = session
            mgr.get_state = lambda _sid: {"busy": True, "queue_len": 0, "interrupted_idle": True}  # type: ignore[method-assign]
            mgr.idle_from_log = lambda _sid: False  # type: ignore[method-assign]

            self.assertFalse(SessionManager._send_remote_ready(mgr, sid))

    def test_attachment_staging_does_not_query_state_when_refresh_rebinds_log(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        with TemporaryDirectory() as td:
            root = Path(td)
            old_log = root / "old.jsonl"
            new_log = root / "new.jsonl"
            old_log.write_text('{"type":"message","message":{"role":"user","content":[{"type":"text","text":"old"}]}}\n', encoding="utf-8")
            new_log.write_text('{"type":"message","message":{"role":"user","content":[{"type":"text","text":"new"}]}}\n', encoding="utf-8")
            session = _make_session(sid)
            session.log_path = old_log
            mgr._sessions[sid] = session
            state_calls = []

            def get_state(_sid: str) -> dict[str, object]:
                state_calls.append(_sid)
                return {"busy": True, "queue_len": 0, "interrupted_idle": False}

            refresh_calls = []

            def refresh(_sid: str, *, drain_queue: bool = False) -> None:
                refresh_calls.append(_sid)
                mgr._sessions[sid].log_path = new_log
                mgr._sessions[sid].interrupted_idle = False

            mgr.get_state = get_state  # type: ignore[method-assign]
            mgr._refresh_session_meta_if_sidecar_exists = lambda _sid, drain_queue=False: refresh(_sid, drain_queue=drain_queue)  # type: ignore[method-assign]
            mgr.idle_from_log = lambda _sid: False  # type: ignore[method-assign]

            self.assertTrue(SessionManager.attachment_staging_ready(mgr, sid))
            self.assertEqual(state_calls, [])
            self.assertEqual(refresh_calls, [sid])

    def test_send_readiness_requeries_state_when_refresh_rebinds_log(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        with TemporaryDirectory() as td:
            root = Path(td)
            old_log = root / "old.jsonl"
            new_log = root / "new.jsonl"
            old_log.write_text('{"type":"message","message":{"role":"user","content":[{"type":"text","text":"old"}]}}\n', encoding="utf-8")
            new_log.write_text('{"type":"message","message":{"role":"user","content":[{"type":"text","text":"new"}]}}\n', encoding="utf-8")
            session = _make_session(sid)
            session.log_path = old_log
            mgr._sessions[sid] = session
            state_calls = []

            def get_state(_sid: str) -> dict[str, object]:
                state_calls.append(_sid)
                if len(state_calls) == 1:
                    return {"busy": False, "queue_len": 0, "interrupted_idle": True}
                return {"busy": True, "queue_len": 0, "interrupted_idle": False}

            refresh_calls = []

            def refresh(_sid: str, *, drain_queue: bool = False) -> None:
                refresh_calls.append(_sid)
                if len(refresh_calls) == 2:
                    mgr._sessions[sid].log_path = new_log
                    mgr._sessions[sid].interrupted_idle = False

            mgr.get_state = get_state  # type: ignore[method-assign]
            mgr.refresh_session_meta = refresh  # type: ignore[method-assign]
            mgr._refresh_session_meta_if_sidecar_exists = lambda _sid, drain_queue=False: refresh(_sid, drain_queue=drain_queue)  # type: ignore[method-assign]
            mgr.idle_from_log = lambda _sid: False  # type: ignore[method-assign]

            self.assertFalse(SessionManager._send_remote_ready(mgr, sid))
            self.assertEqual(len(state_calls), 2)

    def test_confirmed_send_records_log_size_after_readiness(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        with TemporaryDirectory() as td:
            log_path = Path(td) / "pi.jsonl"
            log_path.write_text('{"type":"message","message":{"role":"assistant","content":[],"stopReason":"aborted"}}\n', encoding="utf-8")
            session = _make_session(sid)
            session.log_path = log_path
            session.busy = False
            session.interrupted_idle = True
            mgr._sessions[sid] = session
            mgr._sock_call = lambda *_args, **_kwargs: {"queued": False, "queue_len": 0}  # type: ignore[method-assign]

            def ready(_sid: str, *, allow_pending_attachment: bool = False) -> bool:
                log_path.write_text(log_path.read_text(encoding="utf-8") + '{"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"prior done"}]}}\n', encoding="utf-8")
                return True

            mgr._send_remote_ready = ready  # type: ignore[method-assign]
            SessionManager.send(mgr, sid, "next")

            self.assertTrue(mgr._sessions[sid].last_send_boundary_active)
            self.assertEqual(mgr._sessions[sid].last_send_log_path, log_path)
            self.assertEqual(mgr._sessions[sid].last_send_log_size, log_path.stat().st_size)
            self.assertTrue(mgr._sessions[sid].busy)
            self.assertFalse(mgr._sessions[sid].interrupted_idle)

    def test_send_readiness_rejects_broker_busy_until_log_advances_after_send(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        with TemporaryDirectory() as td:
            log_path = Path(td) / "pi.jsonl"
            log_path.write_text('{"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"done"}]}}\n', encoding="utf-8")
            session = _make_session(sid)
            session.log_path = log_path
            session.last_send_boundary_active = True
            session.last_send_log_path = log_path
            session.last_send_log_size = log_path.stat().st_size
            mgr._sessions[sid] = session
            mgr.get_state = lambda _sid: {"busy": True, "queue_len": 0}  # type: ignore[method-assign]
            mgr.idle_from_log = lambda _sid: True  # type: ignore[method-assign]

            self.assertFalse(SessionManager._send_remote_ready(mgr, sid))
            log_path.write_text(log_path.read_text(encoding="utf-8") + '{"type":"message","message":{"role":"assistant","content":[],"stopReason":"aborted"}}\n', encoding="utf-8")
            self.assertTrue(SessionManager._send_remote_ready(mgr, sid))

    def test_send_readiness_ignores_last_send_size_from_different_log(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        with TemporaryDirectory() as td:
            root = Path(td)
            old_log = root / "old.jsonl"
            old_log.write_text("x" * 1000, encoding="utf-8")
            new_log = root / "new.jsonl"
            new_log.write_text('{"type":"message","message":{"role":"assistant","content":[],"stopReason":"aborted"}}\n', encoding="utf-8")
            session = _make_session(sid)
            session.log_path = new_log
            session.last_send_boundary_active = True
            session.last_send_log_path = old_log
            session.last_send_log_size = old_log.stat().st_size
            mgr._sessions[sid] = session
            mgr.get_state = lambda _sid: {"busy": True, "queue_len": 0}  # type: ignore[method-assign]
            mgr.idle_from_log = lambda _sid: True  # type: ignore[method-assign]

            self.assertTrue(SessionManager._send_remote_ready(mgr, sid))

    def test_queue_readiness_refreshes_sidecar_before_stale_busy_override(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        with TemporaryDirectory() as td:
            root = Path(td)
            log_path = root / "pi.jsonl"
            log_path.write_text('{"type":"message","message":{"role":"assistant","content":[],"stopReason":"aborted"}}\n', encoding="utf-8")
            session = _make_session(sid)
            session.sock_path = root / "s1.sock"
            session.log_path = None
            mgr._sessions[sid] = session
            session.sock_path.with_suffix(".json").write_text("{}\n", encoding="utf-8")
            mgr.get_state = lambda _sid: {"busy": True, "queue_len": 0}  # type: ignore[method-assign]
            mgr.idle_from_log = lambda _sid: True  # type: ignore[method-assign]

            def refresh(_sid: str, *, drain_queue: bool = True) -> None:
                self.assertFalse(drain_queue)
                mgr._sessions[sid].log_path = log_path

            mgr.refresh_session_meta = refresh  # type: ignore[method-assign]

            self.assertTrue(SessionManager._queue_remote_ready(mgr, sid, log_path=None))

    def test_queue_readiness_refreshes_sidecar_after_state_probe(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        with TemporaryDirectory() as td:
            root = Path(td)
            old_log = root / "old.jsonl"
            old_log.write_text('{"type":"message","message":{"role":"assistant","content":[],"stopReason":"aborted"}}\n', encoding="utf-8")
            new_log = root / "new.jsonl"
            new_log.write_text('{"type":"message","message":{"role":"user","content":[{"type":"text","text":"active"}]}}\n', encoding="utf-8")
            session = _make_session(sid)
            session.sock_path = root / "s1.sock"
            session.log_path = old_log
            mgr._sessions[sid] = session
            session.sock_path.with_suffix(".json").write_text("{}\n", encoding="utf-8")
            target = {"log": old_log}

            def refresh(_sid: str, *, drain_queue: bool = True) -> None:
                self.assertFalse(drain_queue)
                mgr._sessions[sid].log_path = target["log"]

            def get_state(_sid: str) -> dict[str, object]:
                target["log"] = new_log
                return {"busy": False, "queue_len": 0}

            mgr.refresh_session_meta = refresh  # type: ignore[method-assign]
            mgr.get_state = get_state  # type: ignore[method-assign]
            mgr.idle_from_log = lambda _sid: mgr._sessions[sid].log_path != new_log  # type: ignore[method-assign]

            self.assertFalse(SessionManager._queue_remote_ready(mgr, sid, log_path=old_log))
            self.assertEqual(mgr._sessions[sid].log_path, new_log)

    def test_send_rejects_remote_busy_before_socket_send(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        with TemporaryDirectory() as td:
            log_path = Path(td) / "rollout.jsonl"
            log_path.write_text("{}\n", encoding="utf-8")
            mgr._sessions[sid] = _make_session(sid)
            mgr._sessions[sid].log_path = log_path
            mgr.get_state = lambda _sid: {"busy": True, "queue_len": 0}  # type: ignore[method-assign]
            mgr.idle_from_log = lambda _sid: False  # type: ignore[method-assign]
            mgr._record_prelog_user_message = lambda *_args, **_kwargs: self.fail("busy send should fail before local echo record")  # type: ignore[method-assign]
            mgr._sock_call = lambda *_args, **_kwargs: self.fail("busy send should fail before broker send")  # type: ignore[method-assign]

            with self.assertRaisesRegex(SessionNotReadyError, "session is busy"):
                SessionManager.send(mgr, sid, "stale direct prompt")

    def test_readiness_rejects_malformed_broker_state(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        malformed = [
            {"busy": False, "queue_len": False},
            {"busy": False, "queue_len": -1},
            {"busy": False, "queue_len": 0.5},
            {"busy": False, "queue_len": "0"},
            {"busy": 0, "queue_len": 0},
        ]
        for state in malformed:
            with self.subTest(state=state):
                mgr._sock_call = lambda *_args, state=state, **_kwargs: state  # type: ignore[method-assign]
                with self.assertRaisesRegex(ValueError, "invalid broker state response"):
                    SessionManager._send_remote_ready(mgr, sid)

    def test_attachment_readiness_allows_stale_broker_busy_when_log_is_idle(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        with TemporaryDirectory() as td:
            log_path = Path(td) / "pi.jsonl"
            log_path.write_text('{"type":"message","message":{"role":"assistant","content":[],"stopReason":"aborted"}}\n', encoding="utf-8")
            session = _make_session(sid)
            session.log_path = log_path
            mgr._sessions[sid] = session
            mgr.get_state = lambda _sid: {"busy": True, "queue_len": 0}  # type: ignore[method-assign]
            mgr.idle_from_log = lambda _sid: True  # type: ignore[method-assign]

            self.assertTrue(SessionManager.attachment_staging_ready(mgr, sid))

    def test_direct_send_rejects_when_local_queue_exists(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._queues[sid] = [_queue_item("q1", "queued first")]
        mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}  # type: ignore[method-assign]
        mgr._sock_call = lambda *_args, **_kwargs: self.fail("direct send should not overtake local queue")  # type: ignore[method-assign]

        with self.assertRaisesRegex(SessionNotReadyError, "queued prompts"):
            SessionManager.send(mgr, sid, "direct second")

    def test_send_readiness_refreshes_sidecar_before_log_idle_check(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        with TemporaryDirectory() as td:
            root = Path(td)
            session = _make_session(sid)
            session.sock_path = root / "s1.sock"
            old_idle_log = root / "old.jsonl"
            old_idle_log.write_text("{}\n", encoding="utf-8")
            new_busy_log = root / "new.jsonl"
            new_busy_log.write_text("{}\n", encoding="utf-8")
            session.log_path = old_idle_log
            mgr._sessions[sid] = session
            session.sock_path.with_suffix(".json").write_text("{}\n", encoding="utf-8")
            mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}  # type: ignore[method-assign]
            mgr.idle_from_log = lambda _sid: mgr._sessions[sid].log_path != new_busy_log  # type: ignore[method-assign]
            mgr._sock_call = lambda *_args, **_kwargs: self.fail("send should not bypass refreshed busy log")  # type: ignore[method-assign]

            def refresh(_sid: str, *, drain_queue: bool = True) -> None:
                self.assertFalse(drain_queue)
                mgr._sessions[sid].log_path = new_busy_log

            mgr.refresh_session_meta = refresh  # type: ignore[method-assign]

            with self.assertRaisesRegex(SessionNotReadyError, "session is busy"):
                SessionManager.send(mgr, sid, "stale log send")

    def test_pending_attachment_stops_queue_promotion(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._sessions[sid].pending_attachment = True
        mgr.get_state = lambda _sid: self.fail("pending attachment should fail before broker state")  # type: ignore[method-assign]

        self.assertFalse(SessionManager._queue_remote_ready(mgr, sid, log_path=None))

    def test_enqueue_sends_immediately_when_idle(self) -> None:
        mgr = self._mgr()
        sid = "s1"
        mgr._sessions[sid] = _make_session(sid)
        sent: list[tuple[str, str]] = []
        mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}  # type: ignore[method-assign]
        mgr.send = lambda _sid, text, **_kwargs: sent.append((_sid, text)) or {"queued": False, "queue_len": 0}  # type: ignore[method-assign]

        resp = SessionManager.enqueue(mgr, sid, "hello now")

        self.assertFalse(resp.get("queued"))
        self.assertEqual(resp.get("queue_len"), 0)
        self.assertEqual(sent, [(sid, "hello now")])
        self.assertEqual(mgr._queues.get(sid, []), [])

    def test_enqueue_surfaces_immediate_commit_unknown(self) -> None:
        # SessionQueueCoordinator.enqueue with ``now=777.0`` injected
        # (replaces former ``patch("codoxear.server.time.time", ...)``).
        # When the session is idle and the queue is empty, enqueue promotes the
        # head immediately; a commit-unknown send error surfaces the marker.
        sid = "s1"
        sessions = {sid: _make_session(sid)}
        queues: dict[str, list[dict[str, Any]]] = {}
        with TemporaryDirectory() as td:
            coord, _saves = _queue_coordinator(
                sessions=sessions,
                queues=queues,
                queue_dir=Path(td),
                now=777.0,
                send=lambda *_args, **_kwargs: (_ for _ in ()).throw(SessionCommitUnknownError("unknown")),
            )
            resp = coord.enqueue(sid, "maybe sent")

        self.assertTrue(resp.get("queued"))
        self.assertTrue(resp.get("commit_unknown"))
        self.assertTrue(resp.get("item", {}).get("commit_unknown"))
        self.assertTrue(queues[sid][0].get("commit_unknown"))

    def test_enqueue_explicit_commit_unknown_response_preserves_queue_item(self) -> None:
        # SessionQueueCoordinator.enqueue with ``now=778.0`` injected
        # (replaces former ``patch("codoxear.server.time.time", ...)``).
        # The send dependency raises SessionCommitUnknownError (equivalent to
        # the broker declaring commit_unknown on the sock-call response, which
        # the full send coordinator parses and re-raises — tested separately
        # in the send-spec tests).
        sid = "s1"
        sessions = {sid: _make_session(sid)}
        queues: dict[str, list[dict[str, Any]]] = {}
        with TemporaryDirectory() as td:
            coord, _saves = _queue_coordinator(
                sessions=sessions,
                queues=queues,
                queue_dir=Path(td),
                now=778.0,
                send=lambda *_args, **_kwargs: (_ for _ in ()).throw(SessionCommitUnknownError("commit unknown")),
            )
            resp = coord.enqueue(sid, "maybe sent")

        self.assertTrue(resp.get("commit_unknown"))
        self.assertIn(sid, queues)
        self.assertTrue(queues[sid][0].get("commit_unknown"))

    def test_enqueue_rejects_broker_without_sync_capability_before_append(self) -> None:
        mgr = self._mgr()
        sid = "s1"
        mgr._sessions[sid] = _make_session(sid)
        mgr._sessions[sid].sync_send_supported = False

        with self.assertRaisesRegex(SessionNotReadyError, "broker must be restarted"):
            SessionManager.enqueue(mgr, sid, "cannot drain")
        self.assertNotIn(sid, mgr._queues)

    def test_enqueue_persists_when_busy(self) -> None:
        mgr = self._mgr()
        sid = "s1"
        with TemporaryDirectory() as td:
            log_path = Path(td) / "rollout.jsonl"
            log_path.write_text("{}\n", encoding="utf-8")
            mgr._sessions[sid] = _make_session(sid)
            mgr._sessions[sid].log_path = log_path
            mgr.get_state = lambda _sid: {"busy": True, "queue_len": 0}  # type: ignore[method-assign]
            mgr.idle_from_log = lambda _sid: False  # active turn: direct send gated, enqueue promotes to queue
            sent: list[tuple[str, str]] = []
            mgr.send = lambda _sid, text, **_kwargs: sent.append((_sid, text)) or {"queued": False, "queue_len": 0}  # type: ignore[method-assign]

            resp = SessionManager.enqueue(mgr, sid, "hello queued")

            self.assertTrue(resp.get("queued"))
            self.assertEqual(resp.get("queue_len"), 1)
            self.assertEqual(sent, [])
            items = mgr._queues.get(sid)
            self.assertIsInstance(items, list)
            self.assertEqual(len(items or []), 1)
            self.assertEqual((items or [])[0]["text"], "hello queued")

    def test_queue_update_delete_move_use_ids(self) -> None:
        mgr = self._mgr()
        sid = "s1"
        mgr._sessions[sid] = _make_session(sid)
        mgr._queues[sid] = [
            _queue_item("a", "first"),
            _queue_item("b", "second"),
            _queue_item("c", "third"),
        ]

        update = SessionManager.queue_update(mgr, sid, "b", "second edited")
        self.assertTrue(update["ok"])
        self.assertEqual(mgr._queues[sid][1]["text"], "second edited")

        move = SessionManager.queue_move(mgr, sid, "c", 1)
        self.assertTrue(move["ok"])
        self.assertEqual([item["id"] for item in mgr._queues[sid]], ["a", "c", "b"])

        delete = SessionManager.queue_delete(mgr, sid, "a")
        self.assertTrue(delete["ok"])
        self.assertEqual([item["id"] for item in mgr._queues[sid]], ["c", "b"])

    def test_queue_list_marks_sending_item(self) -> None:
        mgr = self._mgr()
        sid = "s1"
        session = _make_session(sid)
        session.queue_sending_item_id = "b"
        mgr._sessions[sid] = session
        mgr._queues[sid] = [_queue_item("a", "first"), _queue_item("b", "second")]

        items = SessionManager.queue_list(mgr, sid)

        self.assertEqual([item["id"] for item in items], ["a", "b"])
        self.assertFalse(items[0]["sending"])
        self.assertTrue(items[1]["sending"])

    def test_load_queues_migrates_legacy_string_entries(self) -> None:
        # QueueStore.load directly with a real temp file (replaces former
        # ``patch("codoxear.server.QUEUE_PATH", ...)``).
        with TemporaryDirectory() as td:
            queue_path = Path(td) / "session_queues.json"
            queue_path.write_text(json.dumps({"s1": ["one", "two"]}), encoding="utf-8")
            queues = QueueStore(queue_path).load()

        items = queues["s1"]
        self.assertEqual([item["text"] for item in items], ["one", "two"])
        self.assertTrue(all(isinstance(item["id"], str) and item["id"] for item in items))


if __name__ == "__main__":
    unittest.main()
