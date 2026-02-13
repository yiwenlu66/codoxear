import json
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
    mgr._last_discover_ts = 0.0
    mgr._discover_existing = lambda force=False: None  # type: ignore[method-assign]
    mgr._prune_dead_sessions = lambda: None  # type: ignore[method-assign]
    return mgr


def _make_session(*, sid: str, log_path: Path) -> Session:
    return Session(
        session_id=sid,
        thread_id="thread-1",
        broker_pid=1,
        codex_pid=1,
        owned=False,
        start_ts=0.0,
        cwd="/tmp",
        log_path=log_path,
        sock_path=log_path.with_suffix(".sock"),
    )


class TestMessageIndex(unittest.TestCase):
    def test_init_pagination_returns_stable_windows(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout.jsonl"
            rows = []
            for i in range(200):
                rows.append(
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": f"a{i}"}],
                        },
                        "ts": float(i),
                    }
                )
            p.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")

            mgr = _make_manager()
            mgr._sessions["sid"] = _make_session(sid="sid", log_path=p)

            page1, off1, has_older1, next_before1, _tok1 = mgr._ensure_chat_index("sid", min_events=80, before=0)
            self.assertEqual(len(page1), 80)
            self.assertTrue(has_older1)
            self.assertEqual(next_before1, 80)
            self.assertEqual(page1[0].get("text"), "a120")
            self.assertEqual(page1[-1].get("text"), "a199")
            self.assertGreater(off1, 0)

            page2, _off2, has_older2, next_before2, _tok2 = mgr._ensure_chat_index("sid", min_events=80, before=next_before1)
            self.assertEqual(len(page2), 80)
            self.assertTrue(has_older2)
            self.assertEqual(next_before2, 160)
            self.assertEqual(page2[0].get("text"), "a40")
            self.assertEqual(page2[-1].get("text"), "a119")

            page3, _off3, has_older3, next_before3, _tok3 = mgr._ensure_chat_index("sid", min_events=80, before=next_before2)
            self.assertEqual(len(page3), 40)
            self.assertFalse(has_older3)
            self.assertEqual(next_before3, 0)
            self.assertEqual(page3[0].get("text"), "a0")
            self.assertEqual(page3[-1].get("text"), "a39")

    def test_idle_fallback_uses_log_size_cache(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout.jsonl"
            p.write_text(
                "\n".join(
                    [
                        json.dumps({"type": "event_msg", "payload": {"type": "user_message", "message": "hi"}}),
                        json.dumps({"type": "event_msg", "payload": {"type": "agent_reasoning"}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            mgr = _make_manager()
            mgr._sessions["sid"] = _make_session(sid="sid", log_path=p)

            with patch("codoxear.server._compute_idle_from_log", return_value=False) as compute_idle:
                self.assertFalse(mgr.idle_from_log("sid"))
                self.assertFalse(mgr.idle_from_log("sid"))
                self.assertEqual(compute_idle.call_count, 1)


if __name__ == "__main__":
    unittest.main()
