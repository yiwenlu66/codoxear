import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from codoxear.server import QUEUE_IDLE_GRACE_SECONDS, Session, SessionManager


def _queue_item(item_id: str, text: str) -> dict[str, object]:
    return {"id": item_id, "text": text, "created_ts": 1.0}


class TestQueueSweepIdleGuard(unittest.TestCase):
    def _mgr(self) -> SessionManager:
        mgr = SessionManager.__new__(SessionManager)
        mgr._lock = threading.Lock()
        mgr._sessions = {}
        mgr._queues = {}
        mgr._queue_sweep_cursor = 0
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
        mgr._queues[sid] = [_queue_item("q1", "queued")]
        mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}
        mgr.idle_from_log = lambda _sid: False
        sent = []
        mgr.send = lambda _sid, text, **_kwargs: sent.append((_sid, text)) or {"queued": False, "queue_len": 0}

        SessionManager._queue_sweep(mgr)
        self.assertEqual(sent, [])
        self.assertEqual([item["text"] for item in mgr._queues[sid]], ["queued"])

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
        mgr._queues[sid] = [_queue_item("q1", "queued")]
        mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}
        mgr.idle_from_log = lambda _sid: True
        sent = []
        mgr.send = lambda _sid, text, **_kwargs: sent.append((_sid, text)) or {"queued": False, "queue_len": 0}

        with patch("codoxear.server.time.time", return_value=100.0):
            SessionManager._queue_sweep(mgr)
        self.assertEqual(sent, [])
        self.assertEqual(mgr._sessions[sid].queue_idle_since, 100.0)

        with patch("codoxear.server.time.time", return_value=100.0 + QUEUE_IDLE_GRACE_SECONDS + 0.1):
            SessionManager._queue_sweep(mgr)
        self.assertEqual(sent, [(sid, "queued")])
        self.assertNotIn(sid, mgr._queues)

    def test_queue_sweep_keeps_idle_window_when_log_idle_overrides_stale_broker_busy(self) -> None:
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
        mgr._queues[sid] = [_queue_item("q1", "queued")]
        sent = []
        mgr.send = lambda _sid, text, **_kwargs: sent.append((_sid, text)) or {"queued": False, "queue_len": 0}

        mgr.idle_from_log = lambda _sid: True
        mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}
        with patch("codoxear.server.time.time", return_value=200.0):
            SessionManager._queue_sweep(mgr)
        self.assertEqual(sent, [])
        self.assertEqual(mgr._sessions[sid].queue_idle_since, 200.0)

        mgr.get_state = lambda _sid: {"busy": True, "queue_len": 0}
        with patch("codoxear.server.time.time", return_value=204.0):
            SessionManager._queue_sweep(mgr)
        self.assertEqual(sent, [])
        self.assertEqual(mgr._sessions[sid].queue_idle_since, 200.0)

        with patch("codoxear.server.time.time", return_value=200.0 + QUEUE_IDLE_GRACE_SECONDS - 0.1):
            SessionManager._queue_sweep(mgr)
        self.assertEqual(sent, [])

        with patch("codoxear.server.time.time", return_value=200.0 + QUEUE_IDLE_GRACE_SECONDS + 0.1):
            SessionManager._queue_sweep(mgr)
        self.assertEqual(sent, [(sid, "queued")])

    def test_queue_sweep_blocks_broker_busy_until_log_advances_after_send(self) -> None:
        mgr = self._mgr()
        sid = "s1"
        lp = Path("/tmp/codoxear-test-rollout3b.jsonl")
        lp.write_text('{"type":"event_msg","payload":{"type":"task_complete"},"timestamp":"2026-03-06T00:00:00Z"}\n', encoding="utf-8")
        self.addCleanup(lambda: lp.unlink(missing_ok=True))
        session = Session(
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
            last_send_boundary_active=True,
            last_send_log_path=lp,
            last_send_log_size=lp.stat().st_size,
        )
        mgr._sessions[sid] = session
        mgr._queues[sid] = [_queue_item("q1", "queued")]
        mgr.idle_from_log = lambda _sid: True
        mgr.get_state = lambda _sid: {"busy": True, "queue_len": 0}
        sent = []
        mgr.send = lambda _sid, text, **_kwargs: sent.append((_sid, text)) or {"queued": False, "queue_len": 0}

        with patch("codoxear.server.time.time", return_value=200.0):
            SessionManager._queue_sweep(mgr)
        self.assertEqual(sent, [])
        self.assertIsNone(mgr._sessions[sid].queue_idle_since)

        lp.write_text(lp.read_text(encoding="utf-8") + '{"type":"message","message":{"role":"assistant","content":[],"stopReason":"aborted"}}\n', encoding="utf-8")
        with patch("codoxear.server.time.time", return_value=201.0):
            SessionManager._queue_sweep(mgr)
        self.assertEqual(sent, [])
        self.assertEqual(mgr._sessions[sid].queue_idle_since, 201.0)

    def test_queue_sweep_pops_duplicate_texts_by_item_id(self) -> None:
        mgr = self._mgr()
        sid = "s1"
        lp = Path("/tmp/codoxear-test-rollout4.jsonl")
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
        mgr._queues[sid] = [_queue_item("q1", "dup"), _queue_item("q2", "dup")]
        mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}
        mgr.idle_from_log = lambda _sid: True
        sent = []
        mgr.send = lambda _sid, text, **_kwargs: sent.append((_sid, text)) or {"queued": False, "queue_len": 0}

        with patch("codoxear.server.time.time", return_value=300.0):
            SessionManager._queue_sweep(mgr)
        with patch("codoxear.server.time.time", return_value=300.0 + QUEUE_IDLE_GRACE_SECONDS + 0.1):
            SessionManager._queue_sweep(mgr)

        self.assertEqual(sent, [(sid, "dup")])
        self.assertEqual([item["id"] for item in mgr._queues[sid]], ["q2"])

    def test_queue_sweep_drains_multiple_ready_sessions_up_to_budget(self) -> None:
        mgr = self._mgr()
        sent = []
        for idx, sid in enumerate(["s1", "s2", "s3"], start=1):
            lp = Path(f"/tmp/codoxear-test-rollout-budget-{sid}.jsonl")
            lp.write_text('{"type":"event_msg","payload":{"type":"task_complete"},"timestamp":"2026-03-06T00:00:00Z"}\n', encoding="utf-8")
            self.addCleanup(lambda path=lp: path.unlink(missing_ok=True))
            mgr._sessions[sid] = Session(
                session_id=sid,
                thread_id=f"t{idx}",
                broker_pid=idx,
                codex_pid=idx,
                agent_backend="codex",
                owned=False,
                start_ts=0.0,
                cwd="/tmp",
                log_path=lp,
                sock_path=Path(f"/tmp/{sid}.sock"),
                queue_idle_since=10.0,
            )
            mgr._queues[sid] = [_queue_item(f"q{idx}", f"queued-{idx}")]
        mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0}
        mgr.idle_from_log = lambda _sid: True
        mgr.send = lambda _sid, text, **_kwargs: sent.append((_sid, text)) or {"queued": False, "queue_len": 0}

        with patch("codoxear.server.QUEUE_SWEEP_MAX_DRAINS", 2), patch("codoxear.server.time.time", return_value=10.0 + QUEUE_IDLE_GRACE_SECONDS + 0.1):
            SessionManager._queue_sweep(mgr)

        self.assertEqual(sent, [("s1", "queued-1"), ("s2", "queued-2")])
        self.assertNotIn("s1", mgr._queues)
        self.assertNotIn("s2", mgr._queues)
        self.assertEqual(mgr._queue_sweep_cursor, 2)
        self.assertEqual([item["text"] for item in mgr._queues["s3"]], ["queued-3"])

    def test_queue_sweep_attempt_budget_rotates_past_unready_prefix(self) -> None:
        mgr = self._mgr()
        for idx, sid in enumerate(["s1", "s2", "s3", "s4"], start=1):
            mgr._sessions[sid] = Session(
                session_id=sid,
                thread_id=f"t{idx}",
                broker_pid=idx,
                codex_pid=idx,
                agent_backend="codex",
                owned=False,
                start_ts=0.0,
                cwd="/tmp",
                log_path=None,
                sock_path=Path(f"/tmp/{sid}.sock"),
            )
            mgr._queues[sid] = [_queue_item(f"q{idx}", f"queued-{idx}")]
        attempts: list[str] = []

        def maybe_drain(session_id: str) -> bool:
            attempts.append(session_id)
            return session_id == "s3"

        mgr._maybe_drain_session_queue = maybe_drain  # type: ignore[method-assign]

        with patch("codoxear.server.QUEUE_SWEEP_MAX_DRAINS", 1), patch("codoxear.server.QUEUE_SWEEP_MAX_ATTEMPTS", 2):
            SessionManager._queue_sweep(mgr)

        self.assertEqual(attempts, ["s1", "s2"])
        self.assertEqual(mgr._queue_sweep_cursor, 2)

        attempts.clear()
        with patch("codoxear.server.QUEUE_SWEEP_MAX_DRAINS", 1), patch("codoxear.server.QUEUE_SWEEP_MAX_ATTEMPTS", 2):
            SessionManager._queue_sweep(mgr)

        self.assertEqual(attempts, ["s3"])
        self.assertEqual(mgr._queue_sweep_cursor, 3)


if __name__ == "__main__":
    unittest.main()
