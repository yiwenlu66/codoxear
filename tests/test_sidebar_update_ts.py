import threading
import unittest
from pathlib import Path

from codoxear.server import Session
from codoxear.server import SessionManager


class TestSidebarUpdateTimestamp(unittest.TestCase):
    def test_append_chat_events_does_not_advance_sidebar_ts_for_midturn_assistant(self) -> None:
        mgr = SessionManager.__new__(SessionManager)
        mgr._lock = threading.Lock()
        mgr._sessions = {}
        session = Session(
            session_id="broker-1",
            thread_id="broker-1",
            broker_pid=1,
            codex_pid=2,
            agent_backend="codex",
            owned=False,
            start_ts=100.0,
            cwd="/tmp",
            log_path=None,
            sock_path=Path("/tmp/broker-1.sock"),
            last_chat_ts=50.0,
        )
        mgr._sessions[session.session_id] = session
        mgr._append_chat_events(
            session.session_id,
            [{"role": "assistant", "text": "working", "ts": 120.0}],
            new_off=10,
            latest_token=None,
        )
        self.assertEqual(session.last_chat_ts, 50.0)

    def test_mark_log_delta_advances_sidebar_ts_on_turn_complete(self) -> None:
        mgr = SessionManager.__new__(SessionManager)
        mgr._lock = threading.Lock()
        mgr._sessions = {}
        session = Session(
            session_id="broker-1",
            thread_id="broker-1",
            broker_pid=1,
            codex_pid=2,
            agent_backend="codex",
            owned=False,
            start_ts=100.0,
            cwd="/tmp",
            log_path=None,
            sock_path=Path("/tmp/broker-1.sock"),
            last_chat_ts=50.0,
        )
        mgr._sessions[session.session_id] = session
        mgr.mark_log_delta(
            session.session_id,
            objs=[
                {
                    "type": "event_msg",
                    "payload": {"type": "turn_complete", "turn_id": "t1", "last_agent_message": "done"},
                    "ts": 130.0,
                }
            ],
            new_off=11,
        )
        self.assertEqual(session.last_chat_ts, 130.0)
