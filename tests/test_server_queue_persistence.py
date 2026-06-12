import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from codoxear.server import ControlSocketCallError
from codoxear.server import Session
from codoxear.server import SessionCommitUnknownError
from codoxear.server import SessionManager
from codoxear.server import SessionNotReadyError
from codoxear.server import _match_session_route


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


class TestServerQueuePersistence(unittest.TestCase):
    def _mgr(self) -> SessionManager:
        mgr = SessionManager.__new__(SessionManager)
        mgr._lock = threading.Lock()
        mgr._sessions = {}
        mgr._queues = {}
        mgr._pending_attachment_ids = set()
        mgr._commit_unknown_sends = {}
        mgr._save_queues = lambda: None
        mgr._save_pending_attachments = lambda: None
        mgr._save_commit_unknown_sends = lambda: None
        return mgr

    def test_prune_missing_commit_unknown_sends_keeps_recent_orphans(self) -> None:
        mgr = self._mgr()
        mgr._sessions["live"] = _make_session("live")
        mgr._commit_unknown_sends = {
            "live": {"text": "maybe live", "created_ts": 1.0},
            "recent_gone": {"text": "maybe recent", "created_ts": 90.0},
            "old_gone": {"text": "maybe old", "created_ts": 1.0},
        }
        saved = []
        mgr._save_commit_unknown_sends = lambda: saved.append(dict(mgr._commit_unknown_sends))  # type: ignore[method-assign]

        with patch("codoxear.server.time.time", return_value=100.0):
            self.assertTrue(SessionManager._prune_missing_commit_unknown_sends(mgr, max_age_seconds=50.0))

        self.assertEqual(set(mgr._commit_unknown_sends), {"live", "recent_gone"})
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

    def test_attachment_injection_ready_requires_idle_broker_and_empty_local_queue(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        states = iter([
            {"busy": True, "queue_len": 0},
            {"busy": False, "queue_len": 1},
            {"busy": False, "queue_len": 0},
        ])
        mgr.get_state = lambda _sid: next(states)  # type: ignore[method-assign]

        self.assertFalse(SessionManager.attachment_injection_ready(mgr, sid))
        self.assertFalse(SessionManager.attachment_injection_ready(mgr, sid))
        mgr._queues[sid] = [_queue_item("q1", "queued")]
        self.assertFalse(SessionManager.attachment_injection_ready(mgr, sid))
        mgr._queues.clear()
        mgr._sessions[sid].queue_sending_item_id = "q1"
        self.assertFalse(SessionManager.attachment_injection_ready(mgr, sid))
        mgr._sessions[sid].queue_sending_item_id = None
        self.assertTrue(SessionManager.attachment_injection_ready(mgr, sid))

    def test_unknown_direct_send_blocks_attachment_injection(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._sessions[sid].commit_unknown_send = {"text": "maybe sent", "created_ts": 1.0}
        mgr._commit_unknown_sends[sid] = {"text": "maybe sent", "created_ts": 1.0}
        mgr.get_state = lambda _sid: self.fail("unknown send should fail before broker readiness")  # type: ignore[method-assign]
        mgr.inject_keys = lambda *_args, **_kwargs: self.fail("unknown send should not inject attachment keys")  # type: ignore[method-assign]

        with self.assertRaisesRegex(SessionNotReadyError, "unknown send"):
            SessionManager.attachment_injection_ready(mgr, sid)
        with self.assertRaisesRegex(SessionNotReadyError, "unknown send"):
            SessionManager.inject_attachment_keys(mgr, sid, "ATTACH")

    def test_attachment_injection_ready_rejects_log_busy_session(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        with TemporaryDirectory() as td:
            log_path = Path(td) / "rollout.jsonl"
            log_path.write_text("{}\n", encoding="utf-8")
            mgr._sessions[sid] = _make_session(sid)
            mgr._sessions[sid].log_path = log_path
            mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}  # type: ignore[method-assign]
            mgr.idle_from_log = lambda _sid: False  # type: ignore[method-assign]

            self.assertFalse(SessionManager.attachment_injection_ready(mgr, sid))

    def test_attachment_readiness_rechecks_local_queue_after_broker_state(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)

        def get_state(_sid: str) -> dict[str, int | bool]:
            mgr._queues[sid] = [_queue_item("q1", "queued")]
            return {"busy": False, "queue_len": 0}

        mgr.get_state = get_state  # type: ignore[method-assign]

        self.assertFalse(SessionManager.attachment_injection_ready(mgr, sid))

    def test_attachment_readiness_uses_log_path_bound_during_state_refresh(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        with TemporaryDirectory() as td:
            log_path = Path(td) / "rollout.jsonl"
            log_path.write_text("{}\n", encoding="utf-8")

            def get_state(_sid: str) -> dict[str, int | bool]:
                mgr._sessions[sid].log_path = log_path
                return {"busy": False, "queue_len": 0}

            mgr.get_state = get_state  # type: ignore[method-assign]
            mgr.idle_from_log = lambda _sid: False  # type: ignore[method-assign]

            self.assertFalse(SessionManager.attachment_injection_ready(mgr, sid))

    def test_attachment_readiness_refreshes_sidecar_before_log_idle_check(self) -> None:
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

            self.assertFalse(SessionManager.attachment_injection_ready(mgr, sid))
            self.assertEqual(mgr._sessions[sid].log_path, new_busy_log)
            self.assertEqual(drain_flags, [False, False])

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

            self.assertFalse(SessionManager.attachment_injection_ready(mgr, sid))

    def test_inject_attachment_keys_rechecks_readiness_under_input_lock(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr.attachment_injection_ready = lambda _sid: False  # type: ignore[method-assign]
        mgr.inject_keys = lambda _sid, _seq: self.fail("inject_keys should not be called")  # type: ignore[method-assign]

        with self.assertRaisesRegex(SessionNotReadyError, "session is busy"):
            SessionManager.inject_attachment_keys(mgr, sid, "abc")

    def test_pending_attachment_blocks_queue_until_explicit_send(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr.attachment_injection_ready = lambda _sid: True  # type: ignore[method-assign]
        mgr.inject_keys = lambda _sid, _seq, **_kwargs: {"ok": True}  # type: ignore[method-assign]

        self.assertEqual(SessionManager.inject_attachment_keys(mgr, sid, "ATTACH"), {"ok": True})
        self.assertTrue(mgr._sessions[sid].pending_attachment)
        self.assertIn(sid, mgr._pending_attachment_ids)
        with self.assertRaisesRegex(SessionNotReadyError, "pending attachment"):
            SessionManager.enqueue(mgr, sid, "queued prompt")

        mgr._record_prelog_user_message = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
        mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}  # type: ignore[method-assign]
        mgr._sock_call = lambda *_args, **_kwargs: {"queued": False, "queue_len": 0}  # type: ignore[method-assign]
        with self.assertRaisesRegex(SessionNotReadyError, "pending attachment"):
            SessionManager.send(mgr, sid, "stale direct prompt")
        self.assertTrue(mgr._sessions[sid].pending_attachment)

        self.assertEqual(SessionManager.send(mgr, sid, "intended prompt", allow_pending_attachment=True), {"queued": False, "queue_len": 0})
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
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._record_prelog_user_message = lambda *_args, **_kwargs: self.fail("post-request failure should not be recorded as submitted")  # type: ignore[method-assign]
        mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}  # type: ignore[method-assign]
        mgr._sock_call = lambda *_args, **_kwargs: (_ for _ in ()).throw(ControlSocketCallError("reset", request_sent=True))  # type: ignore[method-assign]

        with patch("codoxear.server._pid_alive", return_value=False):
            with self.assertRaisesRegex(SessionCommitUnknownError, "response failed"):
                SessionManager.send(mgr, sid, "normal prompt")
        self.assertIn(sid, mgr._sessions)

    def test_pre_request_socket_failure_can_prune_dead_session(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}  # type: ignore[method-assign]
        mgr._sock_call = lambda *_args, **_kwargs: (_ for _ in ()).throw(ControlSocketCallError("connect failed", request_sent=False))  # type: ignore[method-assign]
        mgr._clear_deleted_session_state = lambda deleted_sid: mgr._sessions.pop(deleted_sid, None)  # type: ignore[method-assign]

        with patch("codoxear.server._pid_alive", return_value=False):
            with self.assertRaises(KeyError):
                SessionManager.send(mgr, sid, "normal prompt")
        self.assertNotIn(sid, mgr._sessions)

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
        self.assertEqual(mgr._sessions[sid].commit_unknown_send["text"], "normal prompt")

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
        for response in [{}, None, {"queued": False, "queue_len": "notint"}, {"queued": False, "queue_len": -1}, {"queued": False, "queue_len": True}, {"queued": False, "queue_len": 1.9}, {"queued": False, "queue_len": "0"}]:
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
        mgr.inject_keys = lambda *_args, **_kwargs: self.fail("unsupported broker should not receive attachment keys")  # type: ignore[method-assign]

        with self.assertRaisesRegex(SessionNotReadyError, "broker must be restarted"):
            SessionManager.attachment_injection_ready(mgr, sid)
        with self.assertRaisesRegex(SessionNotReadyError, "broker must be restarted"):
            SessionManager.inject_attachment_keys(mgr, sid, "ATTACH")

    def test_attachment_rejects_broker_without_key_error_capability(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._sessions[sid].key_write_errors_supported = False
        mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}  # type: ignore[method-assign]
        mgr.inject_keys = lambda *_args, **_kwargs: self.fail("unsupported broker should not receive attachment keys")  # type: ignore[method-assign]

        with self.assertRaisesRegex(SessionNotReadyError, "broker must be restarted"):
            SessionManager.attachment_injection_ready(mgr, sid)

    def test_queue_send_timeout_marks_head_commit_unknown(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._queues[sid] = [_queue_item("q1", "queued first")]
        mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}  # type: ignore[method-assign]
        mgr.send = lambda *_args, **_kwargs: (_ for _ in ()).throw(SessionCommitUnknownError("unknown"))  # type: ignore[method-assign]

        with patch("codoxear.server.time.time", return_value=123.0):
            resp = SessionManager._promote_queue_head_if_sendable(mgr, sid, require_idle_grace=False, expected_item_id="q1")

        self.assertTrue(resp and resp.get("commit_unknown"))
        self.assertIsNone(mgr._sessions[sid].queue_sending_item_id)
        self.assertTrue(mgr._queues[sid][0].get("commit_unknown"))
        self.assertEqual(mgr._queues[sid][0].get("commit_unknown_ts"), 123.0)

    def test_queue_generic_pre_dispatch_failure_clears_pre_dispatch_unknown_marker(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._queues[sid] = [_queue_item("q1", "queued first")]
        mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}  # type: ignore[method-assign]
        mgr.send = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("pre-send readiness exploded"))  # type: ignore[method-assign]

        with patch("codoxear.server.time.time", return_value=111.0):
            self.assertIsNone(SessionManager._promote_queue_head_if_sendable(mgr, sid, require_idle_grace=False, expected_item_id="q1"))
        self.assertIsNone(mgr._sessions[sid].queue_sending_item_id)
        self.assertFalse(mgr._queues[sid][0].get("commit_unknown"))

    def test_queue_known_send_failure_clears_pre_dispatch_unknown_marker(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._queues[sid] = [_queue_item("q1", "queued first")]
        mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}  # type: ignore[method-assign]

        from codoxear.server import SessionInjectionError

        mgr.send = lambda *_args, **_kwargs: (_ for _ in ()).throw(SessionInjectionError("no pty"))  # type: ignore[method-assign]

        with patch("codoxear.server.time.time", return_value=321.0):
            self.assertIsNone(SessionManager._promote_queue_head_if_sendable(mgr, sid, require_idle_grace=False, expected_item_id="q1"))
        self.assertIsNone(mgr._sessions[sid].queue_sending_item_id)
        self.assertFalse(mgr._queues[sid][0].get("commit_unknown"))

    def test_queue_broker_declared_unknown_keeps_commit_unknown_marker(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._queues[sid] = [_queue_item("q1", "queued first")]
        mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}  # type: ignore[method-assign]
        mgr.send = lambda *_args, **_kwargs: (_ for _ in ()).throw(SessionCommitUnknownError("partial write"))  # type: ignore[method-assign]

        with patch("codoxear.server.time.time", return_value=654.0):
            resp = SessionManager._promote_queue_head_if_sendable(mgr, sid, require_idle_grace=False, expected_item_id="q1")
        self.assertTrue(resp and resp.get("commit_unknown"))
        self.assertTrue(mgr._queues[sid][0].get("commit_unknown"))
        self.assertEqual(mgr._queues[sid][0].get("commit_unknown_ts"), 654.0)

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
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._queues[sid] = [_queue_item("q1", "queued first")]
        mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}  # type: ignore[method-assign]
        observed: list[bool] = []

        def send(_sid: str, _text: str, **_kwargs: object) -> dict[str, object]:
            observed.append(bool(mgr._queues[sid][0].get("commit_unknown")))
            return {"queued": False, "queue_len": 0}

        mgr.send = send  # type: ignore[method-assign]

        with patch("codoxear.server.time.time", return_value=456.0):
            self.assertEqual(SessionManager._promote_queue_head_if_sendable(mgr, sid, require_idle_grace=False, expected_item_id="q1"), {"queued": False, "queue_len": 0})
        self.assertEqual(observed, [True])
        self.assertNotIn(sid, mgr._queues)

    def test_commit_unknown_queue_head_does_not_auto_promote(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._queues[sid] = [dict(_queue_item("q1", "maybe sent"), commit_unknown=True)]
        mgr.get_state = lambda _sid: self.fail("commit-unknown head should block before broker state")  # type: ignore[method-assign]

        self.assertIsNone(SessionManager._promote_queue_head_if_sendable(mgr, sid, require_idle_grace=False, expected_item_id="q1"))

    def test_commit_unknown_queue_delete_requires_explicit_confirmation(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr._queues[sid] = [dict(_queue_item("q1", "maybe sent"), commit_unknown=True), _queue_item("q2", "later")]

        with self.assertRaisesRegex(ValueError, "explicit confirmation"):
            SessionManager.queue_delete(mgr, sid, "q1")
        with self.assertRaisesRegex(ValueError, "commit status is unknown"):
            SessionManager.queue_update(mgr, sid, "q1", "edited maybe sent")
        self.assertEqual([item["id"] for item in mgr._queues[sid]], ["q1", "q2"])

        self.assertEqual(SessionManager.queue_delete(mgr, sid, "q1", allow_commit_unknown=True), {"ok": True, "queue_len": 1})
        self.assertEqual([item["id"] for item in mgr._queues[sid]], ["q2"])

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

    def test_send_rejects_remote_busy_before_socket_send(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr.get_state = lambda _sid: {"busy": True, "queue_len": 0}  # type: ignore[method-assign]
        mgr._record_prelog_user_message = lambda *_args, **_kwargs: self.fail("busy send should fail before local echo record")  # type: ignore[method-assign]
        mgr._sock_call = lambda *_args, **_kwargs: self.fail("busy send should fail before broker send")  # type: ignore[method-assign]

        with self.assertRaisesRegex(SessionNotReadyError, "session is busy"):
            SessionManager.send(mgr, sid, "stale direct prompt")

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

    def test_attachment_injection_error_does_not_set_pending(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr.attachment_injection_ready = lambda _sid: True  # type: ignore[method-assign]
        mgr.inject_keys = lambda _sid, _seq, **_kwargs: {"error": "write failed"}  # type: ignore[method-assign]

        from codoxear.server import SessionInjectionError

        with self.assertRaisesRegex(SessionInjectionError, "write failed"):
            SessionManager.inject_attachment_keys(mgr, sid, "ATTACH")
        self.assertFalse(mgr._sessions[sid].pending_attachment)
        self.assertNotIn(sid, mgr._pending_attachment_ids)

    def test_attachment_explicit_commit_unknown_overrides_success_fields(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr.attachment_injection_ready = lambda _sid: True  # type: ignore[method-assign]
        mgr.inject_keys = lambda _sid, _seq, **_kwargs: {"ok": True, "commit_unknown": True}  # type: ignore[method-assign]

        with self.assertRaisesRegex(SessionCommitUnknownError, "marked commit unknown"):
            SessionManager.inject_attachment_keys(mgr, sid, "ATTACH")
        self.assertTrue(mgr._sessions[sid].pending_attachment)
        self.assertIn(sid, mgr._pending_attachment_ids)

    def test_attachment_empty_response_sets_pending_and_reports_unknown(self) -> None:
        sid = "s1"
        mgr = self._mgr()
        mgr._sessions[sid] = _make_session(sid)
        mgr.attachment_injection_ready = lambda _sid: True  # type: ignore[method-assign]
        mgr.inject_keys = lambda _sid, _seq, **_kwargs: {"error": "empty response"}  # type: ignore[method-assign]

        with self.assertRaisesRegex(SessionCommitUnknownError, "attachment commit status unknown"):
            SessionManager.inject_attachment_keys(mgr, sid, "ATTACH")
        self.assertTrue(mgr._sessions[sid].pending_attachment)
        self.assertIn(sid, mgr._pending_attachment_ids)

    def test_attachment_malformed_response_sets_pending_and_reports_unknown(self) -> None:
        for response in [None, {"ok": "false"}, {"ok": 1}, {"ok": {"x": 1}}]:
            with self.subTest(response=response):
                sid = "s1"
                mgr = self._mgr()
                mgr._sessions[sid] = _make_session(sid)
                mgr.attachment_injection_ready = lambda _sid: True  # type: ignore[method-assign]
                mgr.inject_keys = lambda _sid, _seq, _response=response, **_kwargs: _response  # type: ignore[method-assign]

                with self.assertRaisesRegex(SessionCommitUnknownError, "attachment commit status unknown"):
                    SessionManager.inject_attachment_keys(mgr, sid, "ATTACH")
                self.assertTrue(mgr._sessions[sid].pending_attachment)
                self.assertIn(sid, mgr._pending_attachment_ids)

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
        mgr = self._mgr()
        sid = "s1"
        mgr._sessions[sid] = _make_session(sid)
        mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}  # type: ignore[method-assign]
        mgr._record_prelog_user_message = lambda *_args, **_kwargs: self.fail("commit-unknown enqueue should not be recorded as submitted")  # type: ignore[method-assign]
        mgr.send = lambda *_args, **_kwargs: (_ for _ in ()).throw(SessionCommitUnknownError("unknown"))  # type: ignore[method-assign]

        with patch("codoxear.server.time.time", return_value=777.0):
            resp = SessionManager.enqueue(mgr, sid, "maybe sent")

        self.assertTrue(resp.get("queued"))
        self.assertTrue(resp.get("commit_unknown"))
        self.assertTrue(resp.get("item", {}).get("commit_unknown"))
        self.assertTrue(mgr._queues[sid][0].get("commit_unknown"))

    def test_enqueue_explicit_commit_unknown_response_preserves_queue_item(self) -> None:
        mgr = self._mgr()
        sid = "s1"
        mgr._sessions[sid] = _make_session(sid)
        mgr._record_prelog_user_message = lambda *_args, **_kwargs: self.fail("commit-unknown enqueue should not be recorded as submitted")  # type: ignore[method-assign]
        mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}  # type: ignore[method-assign]
        mgr._sock_call = lambda *_args, **_kwargs: {"queued": False, "queue_len": 0, "commit_unknown": True}  # type: ignore[method-assign]

        with patch("codoxear.server.time.time", return_value=778.0):
            resp = SessionManager.enqueue(mgr, sid, "maybe sent")

        self.assertTrue(resp.get("commit_unknown"))
        self.assertIn(sid, mgr._queues)
        self.assertTrue(mgr._queues[sid][0].get("commit_unknown"))

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
        mgr._sessions[sid] = _make_session(sid)
        mgr.get_state = lambda _sid: {"busy": True, "queue_len": 0}  # type: ignore[method-assign]
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
        with TemporaryDirectory() as td:
            queue_path = Path(td) / "session_queues.json"
            queue_path.write_text(json.dumps({"s1": ["one", "two"]}), encoding="utf-8")
            mgr = self._mgr()
            with patch("codoxear.server.QUEUE_PATH", queue_path):
                SessionManager._load_queues(mgr)

        items = mgr._queues["s1"]
        self.assertEqual([item["text"] for item in items], ["one", "two"])
        self.assertTrue(all(isinstance(item["id"], str) and item["id"] for item in items))


if __name__ == "__main__":
    unittest.main()
