import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from codoxear.server import Session
from codoxear.server import SessionManager


class TestSidebarUpdateTimestamp(unittest.TestCase):
    def _session_with_log(self, *, agent_backend: str, **kwargs):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        log_path = Path(temporary.name) / "session.jsonl"
        log_path.write_text("", encoding="utf-8")
        session = Session(
            session_id=f"broker-{agent_backend}",
            thread_id=f"thread-{agent_backend}",
            broker_pid=1,
            codex_pid=2,
            agent_backend=agent_backend,
            owned=False,
            start_ts=100.0,
            cwd=temporary.name,
            log_path=log_path,
            sock_path=Path(temporary.name) / "broker.sock",
            **kwargs,
        )
        return session, log_path

    @staticmethod
    def _mark_rows(mgr, session, log_path, rows):
        start = log_path.stat().st_size
        with log_path.open("a", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        return mgr.mark_log_delta(
            session.session_id,
            objs=rows,
            start_off=start,
            new_off=log_path.stat().st_size,
        )

    def _manager(self, session):
        mgr = SessionManager.__new__(SessionManager)
        mgr._lock = threading.Lock()
        mgr._sessions = {session.session_id: session}
        return mgr

    def test_mark_log_delta_does_not_advance_sidebar_ts_for_midturn_assistant(self) -> None:
        session, log_path = self._session_with_log(agent_backend="codex", last_chat_ts=50.0)
        mgr = self._manager(session)
        self._mark_rows(
            mgr,
            session,
            log_path,
            [
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
        )
        self.assertEqual(session.last_chat_ts, 50.0)

    def test_mark_log_delta_advances_sidebar_ts_on_turn_complete(self) -> None:
        session, log_path = self._session_with_log(agent_backend="codex", last_chat_ts=50.0)
        mgr = self._manager(session)
        self._mark_rows(
            mgr,
            session,
            log_path,
            [
                {
                    "type": "event_msg",
                    "payload": {"type": "turn_complete", "turn_id": "t1", "last_agent_message": "done"},
                    "ts": 130.0,
                }
            ],
        )
        self.assertEqual(session.last_chat_ts, 130.0)

    def test_mark_log_delta_tracks_latest_pi_model_and_thinking_changes(self) -> None:
        session, log_path = self._session_with_log(
            agent_backend="pi",
            model_provider="old-provider",
            model="old-model",
            reasoning_effort="xhigh",
        )
        mgr = self._manager(session)
        self._mark_rows(
            mgr,
            session,
            log_path,
            [
                {"type": "model_change", "provider": "first-provider", "modelId": "first-model"},
                {"type": "thinking_level_change", "thinkingLevel": "high"},
                {"type": "model_change", "provider": "new-provider", "modelId": "new-model"},
                {"type": "thinking_level_change", "thinkingLevel": "medium"},
            ],
        )
        self.assertEqual(session.model_provider, "new-provider")
        self.assertEqual(session.model, "new-model")
        self.assertEqual(session.reasoning_effort, "medium")

    def test_mark_log_delta_tracks_latest_cc_assistant_model_without_changing_effort(self) -> None:
        session, log_path = self._session_with_log(
            agent_backend="cc",
            model="launch-model",
            reasoning_effort="max",
        )
        mgr = self._manager(session)
        self._mark_rows(
            mgr,
            session,
            log_path,
            [
                {"type": "assistant", "message": {"role": "assistant", "model": "first-model"}},
                {"type": "assistant", "message": {"role": "assistant", "model": "observed-model"}},
            ],
        )
        self.assertEqual(session.model, "observed-model")
        self.assertEqual(session.reasoning_effort, "max")

    def test_mark_log_delta_does_not_trigger_voice_push_delivery(self) -> None:
        class _FakeVoicePush:
            def __init__(self) -> None:
                self.calls = 0

            def observe_messages(self, **_kwargs) -> None:
                self.calls += 1

        session, log_path = self._session_with_log(agent_backend="codex", last_chat_ts=50.0)
        mgr = self._manager(session)
        mgr._voice_push = _FakeVoicePush()
        self._mark_rows(
            mgr,
            session,
            log_path,
            [
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
        )
        self.assertEqual(mgr._voice_push.calls, 0)
