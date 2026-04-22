import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from codoxear.server import Session
from codoxear.server import _message_transcript_identity


def _session(*, thread_id: str = "thread-1", log_path: Path | None = None) -> Session:
    return Session(
        session_id="broker-1",
        thread_id=thread_id,
        broker_pid=1,
        codex_pid=2,
        agent_backend="codex",
        owned=False,
        start_ts=0.0,
        cwd="/tmp",
        log_path=log_path,
        sock_path=Path("/tmp/broker-1.sock"),
    )


class TestMessageTranscriptState(unittest.TestCase):
    def test_pending_bind_identity_is_explicit_and_null(self) -> None:
        payload = _message_transcript_identity(_session(log_path=None))
        self.assertEqual(
            payload,
            {
                "transcript_state": "pending_bind",
                "thread_id": None,
                "log_path": None,
            },
        )

    def test_bound_identity_uses_real_thread_and_log(self) -> None:
        with TemporaryDirectory() as td:
            log_path = Path(td) / "rollout.jsonl"
            log_path.write_text("", encoding="utf-8")
            payload = _message_transcript_identity(_session(thread_id="thread-a", log_path=log_path))
        self.assertEqual(
            payload,
            {
                "transcript_state": "bound",
                "thread_id": "thread-a",
                "log_path": str(log_path),
            },
        )


if __name__ == "__main__":
    unittest.main()
