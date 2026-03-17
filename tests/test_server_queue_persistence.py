import threading
import unittest
from pathlib import Path

from codoxear.server import Session, SessionManager, _match_session_route


class TestServerQueuePersistence(unittest.TestCase):
    def test_match_session_route_requires_exact_suffix(self) -> None:
        self.assertEqual(_match_session_route("/api/sessions/s1/delete", "delete"), "s1")
        self.assertIsNone(_match_session_route("/api/sessions/s1/queue/delete", "delete"))
        self.assertEqual(_match_session_route("/api/sessions/s1/queue/delete", "queue", "delete"), "s1")

    def test_enqueue_persists_in_server_queue(self) -> None:
        mgr = SessionManager.__new__(SessionManager)
        mgr._lock = threading.Lock()
        mgr._sessions = {}
        mgr._queues = {}
        mgr._save_queues = lambda: None

        sid = "s1"
        mgr._sessions[sid] = Session(
            session_id=sid,
            thread_id="t1",
            broker_pid=1,
            codex_pid=1,
            owned=False,
            start_ts=0.0,
            cwd="/tmp",
            log_path=None,
            sock_path=Path("/tmp/s1.sock"),
        )

        resp = SessionManager.enqueue(mgr, sid, "hello queued")
        self.assertTrue(resp.get("queued"))
        self.assertEqual(resp.get("queue_len"), 1)
        self.assertEqual(mgr._queues.get(sid), ["hello queued"])
