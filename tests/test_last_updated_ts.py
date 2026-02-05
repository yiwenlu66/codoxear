import json
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from codoxear.server import _last_conversation_ts_from_tail


def _write_jsonl(path: Path, objs: list[dict]) -> None:
    path.write_text("".join(json.dumps(o) + "\n" for o in objs), encoding="utf-8")


class TestLastConversationTimestamp(unittest.TestCase):
    def test_ignores_reasoning_and_tools(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout-2026-02-05T00-00-00-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa.jsonl"
            t0 = time.time()
            user_ts = t0 - 20
            assistant_ts = t0 - 10

            _write_jsonl(
                p,
                [
                    {"type": "session_meta", "payload": {"id": "main", "source": "cli"}},
                    {"type": "event_msg", "payload": {"type": "agent_reasoning"}, "ts": t0 - 30},
                    {"type": "event_msg", "payload": {"type": "user_message", "message": "hi"}, "ts": user_ts},
                    {"type": "response_item", "payload": {"type": "function_call", "name": "tool"}, "ts": t0 - 15},
                    {"type": "response_item", "payload": {"type": "function_call_output"}, "ts": t0 - 14},
                    {"type": "response_item", "payload": {"type": "reasoning"}, "ts": t0 - 13},
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "ok"}],
                        },
                        "ts": assistant_ts,
                    },
                    {"type": "response_item", "payload": {"type": "reasoning"}, "ts": t0 - 1},
                ],
            )

            # Simulate frequent writes by touching the file.
            os_mtime = time.time()
            p.touch()
            self.assertGreaterEqual(p.stat().st_mtime, os_mtime - 1.0)

            ts = _last_conversation_ts_from_tail(p, max_scan_bytes=64 * 1024)
            self.assertIsInstance(ts, float)
            self.assertAlmostEqual(ts or 0.0, assistant_ts, places=3)

    def test_counts_agent_message_as_assistant(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout-2026-02-05T00-00-00-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb.jsonl"
            t0 = time.time()
            msg_ts = t0 - 5
            _write_jsonl(
                p,
                [
                    {"type": "session_meta", "payload": {"id": "main", "source": "cli"}},
                    {"type": "event_msg", "payload": {"type": "agent_message", "message": "hello"}, "ts": msg_ts},
                ],
            )
            ts = _last_conversation_ts_from_tail(p, max_scan_bytes=64 * 1024)
            self.assertAlmostEqual(ts or 0.0, msg_ts, places=3)

    def test_returns_none_when_no_conversation(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout-2026-02-05T00-00-00-cccccccc-cccc-cccc-cccc-cccccccccccc.jsonl"
            _write_jsonl(
                p,
                [
                    {"type": "session_meta", "payload": {"id": "main", "source": "cli"}},
                    {"type": "event_msg", "payload": {"type": "agent_reasoning"}, "ts": time.time()},
                    {"type": "response_item", "payload": {"type": "function_call", "name": "tool"}, "ts": time.time()},
                ],
            )
            self.assertIsNone(_last_conversation_ts_from_tail(p, max_scan_bytes=64 * 1024))


if __name__ == "__main__":
    unittest.main()

