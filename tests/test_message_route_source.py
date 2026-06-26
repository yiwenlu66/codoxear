import hashlib
import json
import tempfile
import time
import types
import urllib.parse
import unittest
from pathlib import Path
from unittest.mock import patch

from codoxear import server
from codoxear.server import Session


class TestMessageRouteBehavior(unittest.TestCase):
    def test_messages_tail_returns_signed_live_and_history_cursors(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "rollout.jsonl"
            rows = [
                {"type": "event_msg", "payload": {"type": "user_message", "message": "hello"}, "ts": 1.0},
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "world"}],
                        "phase": "final_answer",
                    },
                    "ts": 2.0,
                },
            ]
            log_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            session = Session(
                session_id="s1",
                thread_id="thread-1",
                broker_pid=1,
                codex_pid=1,
                agent_backend="codex",
                owned=False,
                start_ts=0.0,
                cwd=td,
                log_path=log_path,
                sock_path=Path(td) / "s1.sock",
            )

            class FakeManager:
                def refresh_session_meta(self, _sid: str) -> None:
                    return None

                def get_session(self, _sid: str) -> Session:
                    return session

                def _attach_notification_texts(self, events):
                    return events

                def get_state(self, _sid: str) -> dict:
                    return {"busy": False, "queue_len": 0}

                def idle_from_log(self, _sid: str) -> bool:
                    return True

                def _queue_len(self, _sid: str) -> int:
                    return 0

            responses = []
            handler = server.Handler.__new__(server.Handler)
            parsed = urllib.parse.urlparse("/api/sessions/s1/messages/tail?limit=20")
            handler._parse_prefixed_request_path = lambda parsed=parsed: (parsed, parsed.path)  # type: ignore[attr-defined]
            handler._handle_static_get = lambda _path: False  # type: ignore[attr-defined]
            with patch.object(server, "MANAGER", FakeManager()), patch.object(server, "_require_auth", return_value=True), patch.object(
                server, "_json_response", side_effect=lambda _handler, status, obj: responses.append((status, obj))
            ):
                server.Handler.do_GET(handler)

            self.assertEqual(len(responses), 1)
            status, body = responses[0]
            self.assertEqual(status, 200)
            self.assertIsInstance(body.get("live_cursor"), str)
            self.assertEqual(body.get("history_cursor"), None)
            self.assertEqual([event["role"] for event in body["events"]], ["user", "assistant"])
            self.assertEqual(body["events"][0]["text"], "hello")
            self.assertEqual(body["events"][1]["text"], "world")
            self.assertIsInstance(body["events"][0].get("history_cursor"), str)

    def test_messages_live_rejects_bad_cursor_without_mutating_log_state(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "rollout.jsonl"
            log_path.write_text("", encoding="utf-8")
            session = Session(
                session_id="s1",
                thread_id="thread-1",
                broker_pid=1,
                codex_pid=1,
                agent_backend="codex",
                owned=False,
                start_ts=0.0,
                cwd=td,
                log_path=log_path,
                sock_path=Path(td) / "s1.sock",
            )
            marked = []

            class FakeManager:
                def refresh_session_meta(self, _sid: str) -> None:
                    return None

                def get_session(self, _sid: str) -> Session:
                    return session

                def mark_log_delta(self, *args, **kwargs) -> None:
                    marked.append((args, kwargs))

            responses = []
            handler = server.Handler.__new__(server.Handler)
            parsed = urllib.parse.urlparse("/api/sessions/s1/messages/live?cursor=not-a-valid-cursor")
            handler._parse_prefixed_request_path = lambda parsed=parsed: (parsed, parsed.path)  # type: ignore[attr-defined]
            handler._handle_static_get = lambda _path: False  # type: ignore[attr-defined]
            with patch.object(server, "MANAGER", FakeManager()), patch.object(server, "_require_auth", return_value=True), patch.object(
                server, "_json_response", side_effect=lambda _handler, status, obj: responses.append((status, obj))
            ):
                server.Handler.do_GET(handler)

            self.assertEqual(responses, [(409, {"error": "cursor_invalid"})])
            self.assertEqual(marked, [])


if __name__ == "__main__":
    unittest.main()
