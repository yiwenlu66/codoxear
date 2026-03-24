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
    mgr._save_harness = lambda: None  # type: ignore[method-assign]
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
            mgr._harness["sid-a"] = {"enabled": True, "request": "A", "cooldown_minutes": 5, "remaining_injections": 10}
            mgr._harness["sid-b"] = {"enabled": True, "request": "B", "cooldown_minutes": 5, "remaining_injections": 10}

            sent: list[tuple[str, str]] = []
            mgr.get_state = lambda sid: {"busy": False, "queue_len": 0}  # type: ignore[method-assign]
            mgr.send = lambda sid, text: (sent.append((sid, text)) or {"ok": True})  # type: ignore[method-assign]

            with patch("codoxear.server.time.time", return_value=1000.0), patch(
                "codoxear.server._last_chat_role_ts_from_tail", return_value=("assistant", 600.0)
            ), patch("codoxear.server.HARNESS_PROMPT_PREFIX", "PFX"):
                mgr._harness_sweep()

            self.assertEqual(sent, [("sid-a", "PFX\n\n---\n\nAdditional request from user: A\n")])
            self.assertIn("thread:thread-1", mgr._harness_last_injected_scope)
            self.assertEqual(mgr._harness["sid-a"]["remaining_injections"], 9)

    def test_injects_once_per_distinct_thread(self) -> None:
        with TemporaryDirectory() as td:
            p1 = Path(td) / "rollout-a.jsonl"
            p2 = Path(td) / "rollout-b.jsonl"
            p1.write_text("{}", encoding="utf-8")
            p2.write_text("{}", encoding="utf-8")

            mgr = _make_manager()
            mgr._sessions["sid-a"] = _make_session(sid="sid-a", thread_id="thread-1", log_path=p1)
            mgr._sessions["sid-b"] = _make_session(sid="sid-b", thread_id="thread-2", log_path=p2)
            mgr._harness["sid-a"] = {"enabled": True, "request": "A", "cooldown_minutes": 5, "remaining_injections": 10}
            mgr._harness["sid-b"] = {"enabled": True, "request": "B", "cooldown_minutes": 5, "remaining_injections": 10}

            sent: list[tuple[str, str]] = []
            mgr.get_state = lambda sid: {"busy": False, "queue_len": 0}  # type: ignore[method-assign]
            mgr.send = lambda sid, text: (sent.append((sid, text)) or {"ok": True})  # type: ignore[method-assign]

            with patch("codoxear.server.time.time", return_value=1000.0), patch(
                "codoxear.server._last_chat_role_ts_from_tail", return_value=("assistant", 600.0)
            ), patch("codoxear.server.HARNESS_PROMPT_PREFIX", "PFX"):
                mgr._harness_sweep()

            self.assertEqual(
                sent,
                [
                    ("sid-a", "PFX\n\n---\n\nAdditional request from user: A\n"),
                    ("sid-b", "PFX\n\n---\n\nAdditional request from user: B\n"),
                ],
            )

    def test_session_timeout_does_not_kill_other_injections(self) -> None:
        with TemporaryDirectory() as td:
            p1 = Path(td) / "rollout-a.jsonl"
            p2 = Path(td) / "rollout-b.jsonl"
            p1.write_text("{}", encoding="utf-8")
            p2.write_text("{}", encoding="utf-8")

            mgr = _make_manager()
            mgr._sessions["sid-timeout"] = _make_session(sid="sid-timeout", thread_id="thread-timeout", log_path=p1)
            mgr._sessions["sid-ok"] = _make_session(sid="sid-ok", thread_id="thread-ok", log_path=p2)
            mgr._harness["sid-timeout"] = {"enabled": True, "request": "A", "cooldown_minutes": 5, "remaining_injections": 10}
            mgr._harness["sid-ok"] = {"enabled": True, "request": "B", "cooldown_minutes": 5, "remaining_injections": 10}

            sent: list[tuple[str, str]] = []

            def _state(sid: str) -> dict[str, int | bool]:
                if sid == "sid-timeout":
                    raise TimeoutError("timed out")
                return {"busy": False, "queue_len": 0}

            mgr.get_state = _state  # type: ignore[method-assign]
            mgr.send = lambda sid, text: (sent.append((sid, text)) or {"ok": True})  # type: ignore[method-assign]

            with patch("codoxear.server.time.time", return_value=1000.0), patch(
                "codoxear.server._last_chat_role_ts_from_tail", return_value=("assistant", 600.0)
            ), patch("codoxear.server.HARNESS_PROMPT_PREFIX", "PFX"):
                mgr._harness_sweep()

            self.assertEqual(sent, [("sid-ok", "PFX\n\n---\n\nAdditional request from user: B\n")])

    def test_uses_per_session_cooldown_minutes(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout.jsonl"
            p.write_text("{}", encoding="utf-8")

            mgr = _make_manager()
            mgr._sessions["sid-a"] = _make_session(sid="sid-a", thread_id="thread-1", log_path=p)
            mgr._harness["sid-a"] = {"enabled": True, "request": "A", "cooldown_minutes": 2, "remaining_injections": 10}
            mgr._harness_last_injected["sid-a"] = 950.0

            sent: list[tuple[str, str]] = []
            mgr.get_state = lambda sid: {"busy": False, "queue_len": 0}  # type: ignore[method-assign]
            mgr.send = lambda sid, text: (sent.append((sid, text)) or {"ok": True})  # type: ignore[method-assign]

            with patch("codoxear.server.time.time", return_value=1000.0), patch(
                "codoxear.server._last_chat_role_ts_from_tail", return_value=("assistant", 600.0)
            ), patch("codoxear.server.HARNESS_PROMPT_PREFIX", "PFX"):
                mgr._harness_sweep()

            self.assertEqual(sent, [])

    def test_disables_harness_after_last_injection(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout.jsonl"
            p.write_text("{}", encoding="utf-8")

            mgr = _make_manager()
            mgr._sessions["sid-a"] = _make_session(sid="sid-a", thread_id="thread-1", log_path=p)
            mgr._harness["sid-a"] = {"enabled": True, "request": "A", "cooldown_minutes": 5, "remaining_injections": 1}

            sent: list[tuple[str, str]] = []
            mgr.get_state = lambda sid: {"busy": False, "queue_len": 0}  # type: ignore[method-assign]
            mgr.send = lambda sid, text: (sent.append((sid, text)) or {"ok": True})  # type: ignore[method-assign]

            with patch("codoxear.server.time.time", return_value=1000.0), patch(
                "codoxear.server._last_chat_role_ts_from_tail", return_value=("assistant", 600.0)
            ), patch("codoxear.server.HARNESS_PROMPT_PREFIX", "PFX"):
                mgr._harness_sweep()

            self.assertEqual(sent, [("sid-a", "PFX\n\n---\n\nAdditional request from user: A\n")])
            self.assertEqual(mgr._harness["sid-a"]["remaining_injections"], 0)
            self.assertFalse(mgr._harness["sid-a"]["enabled"])


if __name__ == "__main__":
    unittest.main()
