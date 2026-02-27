import threading
import unittest
from pathlib import Path

from codoxear.server import Session
from codoxear.server import SessionManager


def _make_manager() -> SessionManager:
    mgr = SessionManager.__new__(SessionManager)
    mgr._lock = threading.Lock()
    mgr._sessions = {}
    return mgr


def _make_session(*, sid: str) -> Session:
    p = Path("/tmp") / f"{sid}.jsonl"
    return Session(
        session_id=sid,
        thread_id="thread-1",
        broker_pid=1,
        codex_pid=1,
        cli="codex",
        owned=False,
        start_ts=0.0,
        cwd="/tmp",
        log_path=p,
        sock_path=p.with_suffix(".sock"),
    )


class TestServerQueue(unittest.TestCase):
    def test_queue_get_updates_len(self) -> None:
        mgr = _make_manager()
        sess = _make_session(sid="sid")
        mgr._sessions["sid"] = sess

        seen = {}

        def sock_call(sock, req, timeout_s=0.0):
            seen["req"] = req
            return {"queue": ["a", " ", "b"]}

        mgr._sock_call = sock_call  # type: ignore[assignment]

        resp = mgr.queue_get("sid")
        self.assertEqual(resp["queue"], ["a", "b"])
        self.assertEqual(resp["queue_len"], 2)
        self.assertEqual(sess.queue_len, 2)
        self.assertEqual(seen["req"]["cmd"], "queue")
        self.assertEqual(seen["req"]["op"], "get")

    def test_queue_set_filters_empty(self) -> None:
        mgr = _make_manager()
        sess = _make_session(sid="sid")
        mgr._sessions["sid"] = sess

        seen = {}

        def sock_call(sock, req, timeout_s=0.0):
            seen["req"] = req
            return {"queue": list(req.get("queue") or [])}

        mgr._sock_call = sock_call  # type: ignore[assignment]

        resp = mgr.queue_set("sid", ["one", " ", "", "two"])
        self.assertEqual(resp["queue"], ["one", "two"])
        self.assertEqual(resp["queue_len"], 2)
        self.assertEqual(sess.queue_len, 2)
        self.assertEqual(seen["req"]["op"], "set")
        self.assertEqual(seen["req"]["queue"], ["one", "two"])

    def test_queue_push_passes_front(self) -> None:
        mgr = _make_manager()
        sess = _make_session(sid="sid")
        mgr._sessions["sid"] = sess

        seen = {}

        def sock_call(sock, req, timeout_s=0.0):
            seen["req"] = req
            return {"queue": ["a"]}

        mgr._sock_call = sock_call  # type: ignore[assignment]

        resp = mgr.queue_push("sid", "hello", front=True)
        self.assertEqual(resp["queue"], ["a"])
        self.assertEqual(resp["queue_len"], 1)
        self.assertEqual(seen["req"]["op"], "push")
        self.assertTrue(seen["req"]["front"])


if __name__ == "__main__":
    unittest.main()
