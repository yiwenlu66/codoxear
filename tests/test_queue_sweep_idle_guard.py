import threading
import unittest
from pathlib import Path

from codoxear.server import Session, SessionManager


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

        SessionManager._queue_sweep(mgr)
        self.assertEqual(sent, [(sid, "queued")])
        self.assertNotIn(sid, mgr._queues)

