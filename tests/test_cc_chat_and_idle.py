import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from codoxear.rollout_log import _cc_pending_tool_ids_before
from codoxear.rollout_log import _compute_idle_from_log
from codoxear.rollout_log import _extract_chat_events
from codoxear.rollout_log import _extract_delivery_messages
from codoxear.rollout_log import _read_chat_events_from_tail
from codoxear.rollout_log import _read_chat_live_delta
from codoxear.rollout_log import _read_chat_tail_page


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
        self.assertFalse(flags["turn_end"])

    def test_xml_looking_cc_user_prompt_remains_visible(self) -> None:
        events, _meta, flags, _diag = _extract_chat_events([user("<task>summarize</task>")])
        self.assertEqual(events[0]["role"], "user")
        self.assertEqual(events[0]["text"], "<task>summarize</task>")
        self.assertTrue(flags["turn_start"])
        self.assertFalse(flags["turn_end"])

    def test_final_cc_assistant_text_is_final_response(self) -> None:
        events, _meta, flags, _diag = _extract_chat_events([user("hello"), assistant([{"type": "text", "text": "done"}], stop_reason="end_turn")])
        self.assertEqual(events[-1]["message_class"], "final_response")
        self.assertTrue(flags["turn_end"])

    def test_cc_final_text_with_pending_tool_is_not_final_response(self) -> None:
        events, _meta, flags, _diag = _extract_chat_events(
            [
                assistant([{"type": "tool_use", "name": "Bash", "id": "toolu_1", "input": {}}], stop_reason="tool_use"),
                assistant([{"type": "text", "text": "done"}], stop_reason="end_turn"),
            ]
        )
        self.assertEqual(events[-1]["message_class"], "narration")
        self.assertFalse(flags["turn_end"])

    def test_cc_live_delta_at_eof_does_not_scan_prior_context(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "session.jsonl"
            write_log(path, [user("hello"), assistant([{"type": "text", "text": "done"}], stop_reason="end_turn")])
            eof = path.stat().st_size
            with patch("codoxear.rollout_log._cc_pending_tool_ids_before", side_effect=AssertionError("should not scan")):
                events, next_after, _meta, flags, _diag, _token = _read_chat_live_delta(path, after_byte=eof)
            self.assertEqual(events, [])
            self.assertEqual(next_after, eof)
            self.assertFalse(flags["turn_end"])

    def test_cc_multiple_idless_tool_uses_need_multiple_idless_results(self) -> None:
        rows = [
            user("hello"),
            assistant(
                [
                    {"type": "tool_use", "name": "A", "input": {}},
                    {"type": "tool_use", "name": "B", "input": {}},
                ],
                stop_reason="tool_use",
            ),
            {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "content": "one result"}]}},
            assistant([{"type": "text", "text": "done"}], stop_reason="end_turn"),
        ]
        events, _meta, flags, _diag = _extract_chat_events(rows)
        self.assertEqual(events[-1]["message_class"], "narration")
        self.assertFalse(flags["turn_end"])

    def test_cc_top_level_tool_use_result_clears_pending_tool(self) -> None:
        rows = [
            user("hello"),
            assistant([{"type": "tool_use", "name": "Bash", "id": "toolu_1", "input": {}}], stop_reason="tool_use"),
            {
                "type": "user",
                "toolUseResult": {"tool_use_id": "toolu_1", "stdout": "ok"},
                "message": {"role": "user", "content": "ok"},
            },
            assistant([{"type": "text", "text": "done"}], stop_reason="end_turn"),
        ]
        events, _meta, flags, _diag = _extract_chat_events(rows)
        self.assertEqual(events[-1]["message_class"], "final_response")
        self.assertTrue(flags["turn_end"])

    def test_cc_top_level_tool_use_result_without_id_clears_single_pending_tool(self) -> None:
        rows = [
            user("hello"),
            assistant([{"type": "tool_use", "name": "Bash", "id": "toolu_1", "input": {}}], stop_reason="tool_use"),
            {
                "type": "user",
                "toolUseResult": {"stdout": "ok"},
                "message": {"role": "user", "content": "ok"},
            },
            assistant([{"type": "text", "text": "done"}], stop_reason="end_turn"),
        ]
        events, _meta, flags, _diag = _extract_chat_events(rows)
        self.assertEqual(events[-1]["message_class"], "final_response")
        self.assertTrue(flags["turn_end"])

    def test_cc_top_level_tool_use_result_without_id_clears_single_idless_pending_tool(self) -> None:
        rows = [
            user("hello"),
            assistant([{"type": "tool_use", "name": "Bash", "input": {}}], stop_reason="tool_use"),
            {
                "type": "user",
                "toolUseResult": {"stdout": "ok"},
                "message": {"role": "user", "content": "ok"},
            },
            assistant([{"type": "text", "text": "done"}], stop_reason="end_turn"),
        ]
        events, _meta, flags, _diag = _extract_chat_events(rows)
        self.assertEqual(events[-1]["message_class"], "final_response")
        self.assertTrue(flags["turn_end"])
        with TemporaryDirectory() as td:
            path = Path(td) / "session.jsonl"
            write_log(path, rows)
            live_events, _next, _meta, live_flags, _diag, _token = _read_chat_live_delta(path, after_byte=0)
            self.assertEqual(live_events[-1]["message_class"], "final_response")
            self.assertTrue(live_flags["turn_end"])
            self.assertTrue(_compute_idle_from_log(path))

    def test_cc_split_row_idless_tool_uses_need_multiple_idless_results(self) -> None:
        rows = [
            user("hello"),
            assistant([{"type": "tool_use", "name": "A", "input": {}}], stop_reason="tool_use"),
            assistant([{"type": "tool_use", "name": "B", "input": {}}], stop_reason="tool_use"),
            {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "content": "one result"}]}},
            assistant([{"type": "text", "text": "done"}], stop_reason="end_turn"),
        ]
        events, _meta, flags, _diag = _extract_chat_events(rows)
        self.assertEqual(events[-1]["message_class"], "narration")
        self.assertFalse(flags["turn_end"])
        delivery = _extract_delivery_messages(rows)
        self.assertEqual(delivery[-1].message_class, "narration")
        with TemporaryDirectory() as td:
            path = Path(td) / "session.jsonl"
            write_log(path, rows)
            tail_events, _before, _after, _has_older = _read_chat_tail_page(path, limit=10)
            self.assertEqual(tail_events[-1]["message_class"], "narration")
            live_events, _next, _meta, live_flags, _diag, _token = _read_chat_live_delta(path, after_byte=0)
            self.assertEqual(live_events[-1]["message_class"], "narration")
            self.assertFalse(live_flags["turn_end"])
            self.assertFalse(_compute_idle_from_log(path))

    def test_cc_positioned_tail_and_live_final_text_with_pending_tool_are_not_final_response(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "session.jsonl"
            rows = [
                user("hello"),
                assistant([{"type": "tool_use", "name": "Bash", "id": "toolu_1", "input": {}}], stop_reason="tool_use"),
                assistant([{"type": "text", "text": "done"}], stop_reason="end_turn"),
            ]
            write_log(path, rows)
            tail_events, _before, _after, _has_older = _read_chat_tail_page(path, limit=10)
            self.assertEqual(tail_events[-1]["message_class"], "narration")
            live_events, _next, _meta, flags, _diag, _token = _read_chat_live_delta(path, after_byte=0)
            self.assertEqual(live_events[-1]["message_class"], "narration")
            self.assertFalse(flags["turn_end"])
            first_two_bytes = len(json.dumps(rows[0]) + "\n" + json.dumps(rows[1]) + "\n")
            split_events, _next, _meta, split_flags, _diag, _token = _read_chat_live_delta(path, after_byte=first_two_bytes)
            self.assertEqual(split_events[-1]["message_class"], "narration")
            self.assertFalse(split_flags["turn_end"])

    def test_cc_delivery_final_text_with_pending_tool_is_narration(self) -> None:
        messages = _extract_delivery_messages(
            [
                assistant([{"type": "tool_use", "name": "Bash", "id": "toolu_1", "input": {}}], stop_reason="tool_use"),
                assistant([{"type": "text", "text": "done"}], stop_reason="end_turn"),
            ]
        )
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].message_class, "narration")

    def test_cc_delivery_split_delta_uses_seeded_pending_tool_state(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "session.jsonl"
            first = json.dumps(assistant([{"type": "tool_use", "name": "Bash", "id": "toolu_1", "input": {}}], stop_reason="tool_use")) + "\n"
            second_obj = assistant([{"type": "text", "text": "done"}], stop_reason="end_turn")
            path.write_text(first + json.dumps(second_obj) + "\n", encoding="utf-8")
            pending = _cc_pending_tool_ids_before(path, len(first))
        messages = _extract_delivery_messages([second_obj], initial_cc_pending_tool_ids=pending)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].message_class, "narration")

    def test_cc_idle_heuristic(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "session.jsonl"
            write_log(path, [user("hello")])
            self.assertFalse(_compute_idle_from_log(path))
            write_log(path, [user("hello"), assistant([{"type": "text", "text": "done"}], stop_reason="end_turn")])
            self.assertTrue(_compute_idle_from_log(path))
            write_log(path, [user("hello"), assistant([{"type": "tool_use", "name": "Bash", "id": "toolu_1", "input": {}}], stop_reason="tool_use")])
            self.assertFalse(_compute_idle_from_log(path))
            write_log(
                path,
                [
                    user("hello"),
                    assistant([{"type": "tool_use", "name": "Bash", "id": "toolu_1", "input": {}}], stop_reason="tool_use"),
                    {"type": "system", "subtype": "turn_duration", "durationMs": 10},
                ],
            )
            self.assertFalse(_compute_idle_from_log(path))
            write_log(
                path,
                [
                    user("hello"),
                    assistant([{"type": "tool_use", "name": "Bash", "id": "toolu_1", "input": {}}], stop_reason="tool_use"),
                    assistant([{"type": "text", "text": "done"}], stop_reason="end_turn"),
                ],
            )
            self.assertFalse(_compute_idle_from_log(path))
            write_log(
                path,
                [
                    user("hello"),
                    assistant([{"type": "tool_use", "name": "Bash", "id": "toolu_1", "input": {}}], stop_reason="tool_use"),
                    {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "content": "ok"}]}},
                    assistant([{"type": "text", "text": "done"}], stop_reason="end_turn"),
                ],
            )
            self.assertFalse(_compute_idle_from_log(path))

    def test_cc_idle_expands_tail_for_final_text_without_context(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "session.jsonl"
            rows = [
                user("hello"),
                assistant([{"type": "tool_use", "name": "Bash", "id": "toolu_1", "input": {}}], stop_reason="tool_use"),
            ]
            rows.extend({"type": "noop", "x": "x" * 100} for _ in range(4000))
            rows.append(assistant([{"type": "text", "text": "done"}], stop_reason="end_turn"))
            write_log(path, rows)
            self.assertGreater(path.stat().st_size, 256 * 1024)
            self.assertFalse(_compute_idle_from_log(path))

    def test_cc_large_resolved_tool_result_is_idle(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "session.jsonl"
            rows = [
                user("hello"),
                assistant([{"type": "tool_use", "name": "Bash", "id": "toolu_a", "input": {}}], stop_reason="tool_use"),
                {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_a", "content": "x" * (9 * 1024 * 1024)}]}},
                assistant([{"type": "text", "text": "done"}], stop_reason="end_turn"),
            ]
            write_log(path, rows)
            self.assertTrue(_compute_idle_from_log(path))

    def test_cc_large_tool_result_does_not_hide_older_pending_tool(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "session.jsonl"
            rows = [
                user("hello"),
                assistant(
                    [
                        {"type": "tool_use", "name": "Bash", "id": "toolu_a", "input": {}},
                        {"type": "tool_use", "name": "Bash", "id": "toolu_b", "input": {}},
                    ],
                    stop_reason="tool_use",
                ),
                {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_a", "content": "x" * (9 * 1024 * 1024)}]}},
                assistant([{"type": "text", "text": "done"}], stop_reason="end_turn"),
            ]
            encoded = [json.dumps(row) + "\n" for row in rows]
            path.write_text("".join(encoded), encoding="utf-8")
            final_offset = sum(len(item.encode("utf-8")) for item in encoded[:3])
            self.assertEqual(_cc_pending_tool_ids_before(path, final_offset), {"toolu_b"})
            self.assertEqual(_cc_pending_tool_ids_before(path, final_offset, max_scan_bytes=8 * 1024 * 1024), set())
            self.assertFalse(_compute_idle_from_log(path))
            live_events, _next, _meta, flags, _diag, _token = _read_chat_live_delta(path, after_byte=final_offset)
            self.assertEqual(live_events[-1]["message_class"], "narration")
            self.assertFalse(flags["turn_end"])

    def test_cc_idle_expands_tail_for_later_context_without_user_start(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "session.jsonl"
            rows = [
                user("hello"),
                assistant(
                    [
                        {"type": "tool_use", "name": "Bash", "id": "toolu_a", "input": {}},
                        {"type": "tool_use", "name": "Bash", "id": "toolu_b", "input": {}},
                    ],
                    stop_reason="tool_use",
                ),
                {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_a", "content": "x" * 300000}]}},
                assistant([{"type": "thinking", "thinking": "still working"}], stop_reason=None),
                assistant([{"type": "text", "text": "done"}], stop_reason="end_turn"),
            ]
            write_log(path, rows)
            self.assertGreater(path.stat().st_size, 256 * 1024)
            self.assertFalse(_compute_idle_from_log(path))

    def test_cc_idle_expands_tail_for_turn_duration_without_context(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "session.jsonl"
            rows = [
                user("hello"),
                assistant([{"type": "tool_use", "name": "Bash", "id": "toolu_1", "input": {}}], stop_reason="tool_use"),
            ]
            rows.extend({"type": "noop", "x": "x" * 100} for _ in range(4000))
            rows.append({"type": "system", "subtype": "turn_duration", "durationMs": 10})
            write_log(path, rows)
            self.assertGreater(path.stat().st_size, 256 * 1024)
            self.assertFalse(_compute_idle_from_log(path))

    def test_cc_idle_scan_budget_includes_record_on_exact_line_boundary(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "session.jsonl"
            first = json.dumps({"type": "noop", "pad": "x"}) + "\n"
            tool = json.dumps(assistant([{"type": "tool_use", "name": "Bash", "id": "toolu_1", "input": {}}], stop_reason="tool_use")) + "\n"
            duration = json.dumps({"type": "system", "subtype": "turn_duration", "durationMs": 10}) + "\n"
            path.write_text(first + tool + duration, encoding="utf-8")
            self.assertFalse(_compute_idle_from_log(path, max_scan_bytes=len(tool) + len(duration)))

    def test_tail_reader_returns_cc_events(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "session.jsonl"
            write_log(path, [user("hello"), assistant([{"type": "text", "text": "done"}], stop_reason="end_turn")])
            events = _read_chat_events_from_tail(path, min_events=1)
        self.assertEqual([e["role"] for e in events], ["user", "assistant"])


if __name__ == "__main__":
    unittest.main()
