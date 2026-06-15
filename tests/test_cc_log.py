import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from codoxear.cc_log import cc_assistant_is_final_turn_end
from codoxear.cc_log import cc_assistant_text
from codoxear.cc_log import cc_assistant_thinking_count
from codoxear.cc_log import cc_assistant_tool_use_count
from codoxear.cc_log import cc_is_turn_end
from codoxear.cc_log import cc_message_role
from codoxear.cc_log import cc_user_text
from codoxear.cc_log import read_cc_run_settings
from codoxear.cc_log import read_cc_session_header
from codoxear.cc_log import read_cc_session_id


SESSION_ID = "11111111-2222-3333-4444-555555555555"


def cc_user(content, **extra):
    out = {
        "type": "user",
        "sessionId": SESSION_ID,
        "timestamp": "2026-06-11T00:00:00.000Z",
        "cwd": "/repo",
        "message": {"role": "user", "content": content},
    }
    out.update(extra)
    return out


def cc_assistant(content, stop_reason="end_turn", model="claude-haiku-4-5"):
    return {
        "type": "assistant",
        "sessionId": SESSION_ID,
        "timestamp": "2026-06-11T00:01:00.000Z",
        "message": {"role": "assistant", "content": content, "stop_reason": stop_reason, "model": model},
    }


class TestCcLog(unittest.TestCase):
    def test_user_text_extracts_string_and_text_parts(self) -> None:
        self.assertEqual(cc_user_text(cc_user("hello")), "hello")
        self.assertEqual(cc_user_text(cc_user([{"type": "text", "text": "hello"}, {"type": "text", "text": " world"}])), "hello world")
        self.assertEqual(cc_user_text(cc_user("<div>hello</div>")), "<div>hello</div>")

    def test_user_text_skips_tool_results_and_meta_records(self) -> None:
        self.assertIsNone(cc_user_text(cc_user([{"type": "tool_result", "content": "ok", "tool_use_id": "toolu_1"}])))
        self.assertIsNone(cc_user_text(cc_user([{"type": "tool_result", "content": "ok", "tool_use_id": "toolu_1"}, {"type": "text", "text": "transport note"}])))
        self.assertIsNone(cc_user_text(cc_user("/compact", isMeta=True)))
        self.assertEqual(cc_message_role(cc_user([{"type": "tool_result", "content": "ok"}])), "toolResult")

    def test_assistant_text_and_counts(self) -> None:
        obj = cc_assistant(
            [
                {"type": "thinking", "thinking": "hmm"},
                {"type": "text", "text": "Answer"},
                {"type": "tool_use", "name": "Bash", "id": "toolu_1", "input": {}},
            ],
            stop_reason="tool_use",
        )
        self.assertEqual(cc_assistant_text(obj), "Answer")
        self.assertEqual(cc_assistant_thinking_count(obj), 1)
        self.assertEqual(cc_assistant_tool_use_count(obj), 1)
        self.assertFalse(cc_assistant_is_final_turn_end(obj))

    def test_final_turn_end_requires_end_turn_without_tool_use(self) -> None:
        self.assertTrue(cc_assistant_is_final_turn_end(cc_assistant([{"type": "text", "text": "done"}], stop_reason="end_turn")))
        self.assertFalse(cc_assistant_is_final_turn_end(cc_assistant([{"type": "text", "text": "more"}], stop_reason="tool_use")))
        self.assertTrue(cc_is_turn_end({"type": "system", "subtype": "turn_duration", "durationMs": 123}))

    def test_read_session_header_merges_cwd_from_later_record(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "session.jsonl"
            rows = [
                {"type": "mode", "sessionId": SESSION_ID, "timestamp": "2026-06-11T00:00:00.000Z"},
                cc_user("hello"),
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            header = read_cc_session_header(path)
            self.assertEqual(header, {"id": SESSION_ID, "sessionId": SESSION_ID, "cwd": "/repo", "timestamp": "2026-06-11T00:00:00.000Z"})

    def test_read_session_header_parses_large_valid_first_record(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "session.jsonl"
            row = cc_user("x" * (600 * 1024))
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            header = read_cc_session_header(path)
            self.assertEqual(header, {"id": SESSION_ID, "sessionId": SESSION_ID, "cwd": "/repo", "timestamp": "2026-06-11T00:00:00.000Z"})

    def test_read_session_header_scan_is_bounded(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "session.jsonl"
            rows = [
                {"type": "mode", "sessionId": SESSION_ID, "timestamp": "2026-06-11T00:00:00.000Z"},
                {"type": "noise", "payload": "x" * 200},
                cc_user("hello"),
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            header = read_cc_session_header(path, max_scan_bytes=160)
            self.assertEqual(header, {"id": SESSION_ID, "sessionId": SESSION_ID, "timestamp": "2026-06-11T00:00:00.000Z"})

    def test_read_session_header_and_run_settings(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "session.jsonl"
            rows = [cc_user("hello"), cc_assistant([{"type": "text", "text": "done"}], model="claude-sonnet-4-5")]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            header = read_cc_session_header(path)
            self.assertEqual(header, {"id": SESSION_ID, "sessionId": SESSION_ID, "cwd": "/repo", "timestamp": "2026-06-11T00:00:00.000Z"})
            self.assertEqual(read_cc_session_id(path), SESSION_ID)
            self.assertEqual(read_cc_run_settings(path), (None, "claude-sonnet-4-5", None))


if __name__ == "__main__":
    unittest.main()
