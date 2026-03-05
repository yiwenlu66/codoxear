import threading
import unittest
from pathlib import Path

from codoxear.server import Session, SessionManager


class TestServerQueuePersistence(unittest.TestCase):
    def test_enqueue_persists_in_server_queue(self) -> None:
        mgr = SessionManager.__new__(SessionManager)
        mgr._lock = threading.Lock()
        mgr._sessions = {}
        mgr._queues = {}
        mgr._save_queues = lambda: None
        mgr._sock_call = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("enqueue must not call broker"))

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
        self.assertEqual(resp.get("backend"), "server_compat")
        self.assertEqual(resp.get("queue_len"), 1)
        self.assertEqual(resp.get("queue_len_total"), 1)
        self.assertEqual(mgr._queues.get(sid), ["hello queued"])

