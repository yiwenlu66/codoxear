import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from codoxear.server import Session
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
    )


def _queue_item(item_id: str, text: str) -> dict[str, object]:
    return {"id": item_id, "text": text, "created_ts": 1.0}


class TestServerQueuePersistence(unittest.TestCase):
    def _mgr(self) -> SessionManager:
        mgr = SessionManager.__new__(SessionManager)
        mgr._lock = threading.Lock()
        mgr._sessions = {}
        mgr._queues = {}
        mgr._save_queues = lambda: None
        return mgr

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
        mgr.inject_keys = lambda _sid, _seq: {"ok": True}  # type: ignore[method-assign]

        self.assertEqual(SessionManager.inject_attachment_keys(mgr, sid, "ATTACH"), {"ok": True})
        self.assertTrue(mgr._sessions[sid].pending_attachment)
        with self.assertRaisesRegex(SessionNotReadyError, "pending attachment"):
            SessionManager.enqueue(mgr, sid, "queued prompt")

        mgr._record_prelog_user_message = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
        mgr._sock_call = lambda *_args, **_kwargs: {"queued": False, "queue_len": 0}  # type: ignore[method-assign]
        with self.assertRaisesRegex(SessionNotReadyError, "pending attachment"):
            SessionManager.send(mgr, sid, "stale direct prompt")
        self.assertTrue(mgr._sessions[sid].pending_attachment)

        self.assertEqual(SessionManager.send(mgr, sid, "intended prompt", allow_pending_attachment=True), {"queued": False, "queue_len": 0})
        self.assertFalse(mgr._sessions[sid].pending_attachment)

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
        mgr.send = lambda _sid, text: sent.append((_sid, text)) or {"queued": False, "queue_len": 0}  # type: ignore[method-assign]

        resp = SessionManager.enqueue(mgr, sid, "hello now")

        self.assertFalse(resp.get("queued"))
        self.assertEqual(resp.get("queue_len"), 0)
        self.assertEqual(sent, [(sid, "hello now")])
        self.assertEqual(mgr._queues.get(sid, []), [])

    def test_enqueue_persists_when_busy(self) -> None:
        mgr = self._mgr()
        sid = "s1"
        mgr._sessions[sid] = _make_session(sid)
        mgr.get_state = lambda _sid: {"busy": True, "queue_len": 0}  # type: ignore[method-assign]
        sent: list[tuple[str, str]] = []
        mgr.send = lambda _sid, text: sent.append((_sid, text)) or {"queued": False, "queue_len": 0}  # type: ignore[method-assign]

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
