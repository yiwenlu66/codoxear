import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from codoxear.rollout_log import _compute_idle_from_log
from codoxear.rollout_log import _extract_chat_events
from codoxear.rollout_log import _read_chat_events_from_tail


SESSION_ID = "11111111-2222-3333-4444-555555555555"


def user(content="hello"):
    return {"type": "user", "sessionId": SESSION_ID, "timestamp": "2026-06-11T00:00:00.000Z", "cwd": "/repo", "message": {"role": "user", "content": content}}


def assistant(content, stop_reason="end_turn"):
    return {"type": "assistant", "sessionId": SESSION_ID, "timestamp": "2026-06-11T00:00:01.000Z", "message": {"role": "assistant", "content": content, "stop_reason": stop_reason}}


def write_log(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class TestCcChatAndIdle(unittest.TestCase):
    def test_extract_cc_chat_events_and_meta(self) -> None:
        events, meta, flags, diag = _extract_chat_events([
            user("hello"),
            assistant([{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "Hi"}, {"type": "tool_use", "name": "Bash", "id": "toolu_1", "input": {}}], stop_reason="tool_use"),
            {"type": "system", "subtype": "turn_duration", "durationMs": 10},
        ])
        self.assertEqual(events[0]["role"], "user")
        self.assertEqual(events[0]["text"], "hello")
        self.assertEqual(events[1]["role"], "assistant")
        self.assertEqual(events[1]["text"], "Hi")
        self.assertEqual(events[1]["message_class"], "narration")
        self.assertEqual(meta["thinking"], 1)
        self.assertEqual(meta["tool"], 1)
        self.assertEqual(diag["tool_names"], ["Bash"])
        self.assertTrue(flags["turn_start"])
        self.assertTrue(flags["turn_end"])

    def test_final_cc_assistant_text_is_final_response(self) -> None:
        events, _meta, flags, _diag = _extract_chat_events([user("hello"), assistant([{"type": "text", "text": "done"}], stop_reason="end_turn")])
        self.assertEqual(events[-1]["message_class"], "final_response")
        self.assertTrue(flags["turn_end"])

    def test_cc_idle_heuristic(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "session.jsonl"
            write_log(path, [user("hello")])
            self.assertFalse(_compute_idle_from_log(path))
            write_log(path, [user("hello"), assistant([{"type": "text", "text": "done"}], stop_reason="end_turn")])
            self.assertTrue(_compute_idle_from_log(path))
            write_log(path, [user("hello"), assistant([{"type": "tool_use", "name": "Bash", "id": "toolu_1", "input": {}}], stop_reason="tool_use")])
            self.assertFalse(_compute_idle_from_log(path))

    def test_tail_reader_returns_cc_events(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "session.jsonl"
            write_log(path, [user("hello"), assistant([{"type": "text", "text": "done"}], stop_reason="end_turn")])
            events = _read_chat_events_from_tail(path, min_events=1)
        self.assertEqual([e["role"] for e in events], ["user", "assistant"])


if __name__ == "__main__":
    unittest.main()
