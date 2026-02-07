import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from codoxear.server import Session
from codoxear.server import SessionManager


def _make_manager() -> SessionManager:
    mgr = SessionManager.__new__(SessionManager)
    mgr._lock = threading.Lock()
    mgr._sessions = {}
    mgr._harness = {}
    mgr._harness_last_injected = {}
    mgr._harness_last_injected_scope = {}
    mgr._discover_existing = lambda: None  # type: ignore[method-assign]
    mgr._prune_dead_sessions = lambda: None  # type: ignore[method-assign]
    return mgr


def _make_session(*, sid: str, thread_id: str, log_path: Path) -> Session:
    return Session(
        session_id=sid,
        thread_id=thread_id,
        broker_pid=1,
        codex_pid=1,
        owned=False,
        start_ts=0.0,
        cwd="/tmp",
        log_path=log_path,
        sock_path=log_path.with_suffix(".sock"),
    )


class TestHarnessSweep(unittest.TestCase):
    def test_dedupes_injection_for_same_thread(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout.jsonl"
            p.write_text("{}", encoding="utf-8")

            mgr = _make_manager()
            mgr._sessions["sid-a"] = _make_session(sid="sid-a", thread_id="thread-1", log_path=p)
            mgr._sessions["sid-b"] = _make_session(sid="sid-b", thread_id="thread-1", log_path=p)
            mgr._harness["sid-a"] = {"enabled": True, "text": "A"}
            mgr._harness["sid-b"] = {"enabled": True, "text": "B"}

            sent: list[tuple[str, str]] = []
            mgr.get_state = lambda sid: {"busy": False, "queue_len": 0}  # type: ignore[method-assign]
            mgr.send = lambda sid, text: (sent.append((sid, text)) or {"ok": True})  # type: ignore[method-assign]

            with patch("codoxear.server.time.time", return_value=1000.0), patch("codoxear.server.HARNESS_IDLE_SECONDS", 300), patch(
                "codoxear.server._last_chat_role_ts_from_tail", return_value=("assistant", 600.0)
            ):
                mgr._harness_sweep()

            self.assertEqual(sent, [("sid-a", "A")])
            self.assertIn("thread:thread-1", mgr._harness_last_injected_scope)

    def test_injects_once_per_distinct_thread(self) -> None:
        with TemporaryDirectory() as td:
            p1 = Path(td) / "rollout-a.jsonl"
            p2 = Path(td) / "rollout-b.jsonl"
            p1.write_text("{}", encoding="utf-8")
            p2.write_text("{}", encoding="utf-8")

            mgr = _make_manager()
            mgr._sessions["sid-a"] = _make_session(sid="sid-a", thread_id="thread-1", log_path=p1)
            mgr._sessions["sid-b"] = _make_session(sid="sid-b", thread_id="thread-2", log_path=p2)
            mgr._harness["sid-a"] = {"enabled": True, "text": "A"}
            mgr._harness["sid-b"] = {"enabled": True, "text": "B"}

            sent: list[tuple[str, str]] = []
            mgr.get_state = lambda sid: {"busy": False, "queue_len": 0}  # type: ignore[method-assign]
            mgr.send = lambda sid, text: (sent.append((sid, text)) or {"ok": True})  # type: ignore[method-assign]

            with patch("codoxear.server.time.time", return_value=1000.0), patch("codoxear.server.HARNESS_IDLE_SECONDS", 300), patch(
                "codoxear.server._last_chat_role_ts_from_tail", return_value=("assistant", 600.0)
            ):
                mgr._harness_sweep()

            self.assertEqual(sent, [("sid-a", "A"), ("sid-b", "B")])


if __name__ == "__main__":
    unittest.main()
