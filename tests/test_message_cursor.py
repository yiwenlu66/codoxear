import unittest
from pathlib import Path

from codoxear.server import MessageCursorError
from codoxear.server import Session
from codoxear.server import _decode_message_cursor
from codoxear.server import _encode_message_cursor


def _session(*, thread_id: str = "thread-1", log_path: str = "/tmp/rollout.jsonl") -> Session:
    return Session(
        session_id="broker-1",
        thread_id=thread_id,
        broker_pid=1,
        codex_pid=2,
        agent_backend="codex",
        owned=False,
        start_ts=0.0,
        cwd="/tmp",
        log_path=Path(log_path),
        sock_path=Path("/tmp/broker-1.sock"),
    )


class TestMessageCursor(unittest.TestCase):
    def test_live_cursor_roundtrip(self) -> None:
        session = _session()
        token = _encode_message_cursor(kind="live", session=session, pos=123)
        self.assertEqual(_decode_message_cursor(token, kind="live", session=session), 123)

    def test_history_cursor_rejects_wrong_session_identity(self) -> None:
        token = _encode_message_cursor(kind="history", session=_session(thread_id="thread-a"), pos=456)
        with self.assertRaises(MessageCursorError):
            _decode_message_cursor(token, kind="history", session=_session(thread_id="thread-b"))

    def test_cursor_rejects_wrong_kind(self) -> None:
        session = _session()
        token = _encode_message_cursor(kind="live", session=session, pos=123)
        with self.assertRaises(MessageCursorError):
            _decode_message_cursor(token, kind="history", session=session)


if __name__ == "__main__":
    unittest.main()
