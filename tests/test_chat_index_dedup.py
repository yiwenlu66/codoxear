import threading
import unittest
from pathlib import Path

from codoxear.server import Session
from codoxear.server import SessionManager


class TestChatIndexDedup(unittest.TestCase):
    def test_append_chat_events_dedup(self) -> None:
        mgr = SessionManager.__new__(SessionManager)
        mgr._lock = threading.Lock()
        mgr._sessions = {}

        s = Session(
            session_id="broker-1",
            thread_id="broker-1",
            broker_pid=1,
            codex_pid=2,
            owned=False,
            start_ts=123.0,
            cwd="/tmp",
            log_path=None,
            sock_path=Path("/tmp/broker-1.sock"),
        )
        ev = {"role": "assistant", "text": "hello", "ts": 1.234}
        s.chat_index_events = [ev]
        mgr._sessions[s.session_id] = s

        mgr._append_chat_events(s.session_id, [ev], new_off=10, latest_token=None)
        self.assertEqual(len(s.chat_index_events), 1)


if __name__ == "__main__":
    unittest.main()

