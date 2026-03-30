import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from codoxear.server import QUEUE_IDLE_GRACE_SECONDS, Session, SessionManager


class TestQueueSweepIdleGuard(unittest.TestCase):
    def _mgr(self) -> SessionManager:
        mgr = SessionManager.__new__(SessionManager)
        mgr._lock = threading.Lock()
        mgr._sessions = {}
        mgr._queues = {}
        mgr._save_queues = lambda: None
        mgr._discover_existing_if_stale = lambda: None
        mgr._prune_dead_sessions = lambda: None
        return mgr

    def test_queue_sweep_skips_when_log_not_idle(self) -> None:
        mgr = self._mgr()
        sid = "s1"
        lp = Path("/tmp/codoxear-test-rollout.jsonl")
        lp.write_text('{"type":"event_msg","payload":{"type":"user_message","message":"hi"},"timestamp":"2026-03-06T00:00:00Z"}\n', encoding="utf-8")
        self.addCleanup(lambda: lp.unlink(missing_ok=True))
        mgr._sessions[sid] = Session(
            session_id=sid,
            thread_id="t1",
            broker_pid=1,
            codex_pid=1,
            agent_backend="codex",
            owned=False,
            start_ts=0.0,
            cwd="/tmp",
            log_path=lp,
            sock_path=Path("/tmp/s1.sock"),
        )
        mgr._queues[sid] = ["queued"]
        mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}
        mgr.idle_from_log = lambda _sid: False
        sent = []
        mgr.send = lambda _sid, text: sent.append((_sid, text)) or {"queued": False, "queue_len": 0}

        SessionManager._queue_sweep(mgr)
        self.assertEqual(sent, [])
        self.assertEqual(mgr._queues[sid], ["queued"])

    def test_queue_sweep_injects_when_log_idle(self) -> None:
        mgr = self._mgr()
        sid = "s1"
        lp = Path("/tmp/codoxear-test-rollout2.jsonl")
        lp.write_text('{"type":"event_msg","payload":{"type":"task_complete"},"timestamp":"2026-03-06T00:00:00Z"}\n', encoding="utf-8")
        self.addCleanup(lambda: lp.unlink(missing_ok=True))
        mgr._sessions[sid] = Session(
            session_id=sid,
            thread_id="t1",
            broker_pid=1,
            codex_pid=1,
            agent_backend="codex",
            owned=False,
            start_ts=0.0,
            cwd="/tmp",
            log_path=lp,
            sock_path=Path("/tmp/s1.sock"),
        )
        mgr._queues[sid] = ["queued"]
        mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}
        mgr.idle_from_log = lambda _sid: True
        sent = []
        mgr.send = lambda _sid, text: sent.append((_sid, text)) or {"queued": False, "queue_len": 0}

        with patch("codoxear.server.time.time", return_value=100.0):
            SessionManager._queue_sweep(mgr)
        self.assertEqual(sent, [])
        self.assertEqual(mgr._sessions[sid].queue_idle_since, 100.0)

        with patch("codoxear.server.time.time", return_value=100.0 + QUEUE_IDLE_GRACE_SECONDS + 0.1):
            SessionManager._queue_sweep(mgr)
        self.assertEqual(sent, [(sid, "queued")])
        self.assertNotIn(sid, mgr._queues)

    def test_queue_sweep_requires_consecutive_idle_window(self) -> None:
        mgr = self._mgr()
        sid = "s1"
        lp = Path("/tmp/codoxear-test-rollout3.jsonl")
        lp.write_text('{"type":"event_msg","payload":{"type":"task_complete"},"timestamp":"2026-03-06T00:00:00Z"}\n', encoding="utf-8")
        self.addCleanup(lambda: lp.unlink(missing_ok=True))
        mgr._sessions[sid] = Session(
            session_id=sid,
            thread_id="t1",
            broker_pid=1,
            codex_pid=1,
            agent_backend="codex",
            owned=False,
            start_ts=0.0,
            cwd="/tmp",
            log_path=lp,
            sock_path=Path("/tmp/s1.sock"),
        )
        mgr._queues[sid] = ["queued"]
        sent = []
        mgr.send = lambda _sid, text: sent.append((_sid, text)) or {"queued": False, "queue_len": 0}

        mgr.idle_from_log = lambda _sid: True
        mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}
        with patch("codoxear.server.time.time", return_value=200.0):
            SessionManager._queue_sweep(mgr)
        self.assertEqual(sent, [])
        self.assertEqual(mgr._sessions[sid].queue_idle_since, 200.0)

        mgr.get_state = lambda _sid: {"busy": True, "queue_len": 0}
        with patch("codoxear.server.time.time", return_value=204.0):
            SessionManager._queue_sweep(mgr)
        self.assertIsNone(mgr._sessions[sid].queue_idle_since)

        mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}
        with patch("codoxear.server.time.time", return_value=205.0):
            SessionManager._queue_sweep(mgr)
        self.assertEqual(sent, [])
        self.assertEqual(mgr._sessions[sid].queue_idle_since, 205.0)

        with patch("codoxear.server.time.time", return_value=205.0 + QUEUE_IDLE_GRACE_SECONDS - 0.1):
            SessionManager._queue_sweep(mgr)
        self.assertEqual(sent, [])

        with patch("codoxear.server.time.time", return_value=205.0 + QUEUE_IDLE_GRACE_SECONDS + 0.1):
            SessionManager._queue_sweep(mgr)
        self.assertEqual(sent, [(sid, "queued")])
