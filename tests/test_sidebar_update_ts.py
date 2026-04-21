import threading
import unittest
from pathlib import Path

from codoxear.server import Session
from codoxear.server import SessionManager


class TestSidebarUpdateTimestamp(unittest.TestCase):
    def test_mark_log_delta_does_not_advance_sidebar_ts_for_midturn_assistant(self) -> None:
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
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "working"}],
                    },
                    "ts": 120.0,
                }
            ],
            new_off=10,
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

    def test_mark_log_delta_does_not_trigger_voice_push_delivery(self) -> None:
        class _FakeVoicePush:
            def __init__(self) -> None:
                self.calls = 0

            def observe_messages(self, **_kwargs) -> None:
                self.calls += 1

        mgr = SessionManager.__new__(SessionManager)
        mgr._lock = threading.Lock()
        mgr._sessions = {}
        mgr._voice_push = _FakeVoicePush()
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
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "phase": "final_answer",
                        "content": [{"type": "output_text", "text": "old final answer"}],
                    },
                    "ts": 130.0,
                }
            ],
            new_off=11,
        )
        self.assertEqual(mgr._voice_push.calls, 0)
