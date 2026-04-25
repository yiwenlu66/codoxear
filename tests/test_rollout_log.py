import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from codoxear.rollout_log import _extract_chat_events
from codoxear.rollout_log import _read_chat_history_page
from codoxear.rollout_log import _read_chat_live_delta
from codoxear.rollout_log import _read_chat_tail_page


def _write_jsonl(path: Path, rows: list[dict]) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    pos = 0
    chunks: list[bytes] = []
    for row in rows:
        raw = (json.dumps(row) + "\n").encode("utf-8")
        start = pos
        pos += len(raw)
        offsets.append((start, pos))
        chunks.append(raw)
    path.write_bytes(b"".join(chunks))
    return offsets


class RolloutLogInteractivePromptTests(unittest.TestCase):
    def test_pi_ask_user_tool_call_becomes_interactive_event(self):
        obj = {
            "type": "message",
            "timestamp": "2026-04-24T05:33:42.731Z",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "toolCall",
                        "id": "call_ask_1",
                        "name": "ask_user",
                        "arguments": {
                            "question": "Which path should we use?",
                            "context": "We found two safe options.",
                            "options": [
                                "Fast path",
                                {"title": "Safe path", "description": "More validation"},
                            ],
                            "allowFreeform": True,
                            "allowComment": True,
                        },
                    }
                ],
                "stopReason": "toolUse",
            },
        }

        events, meta, _flags, diag = _extract_chat_events([obj])

        self.assertEqual(meta["tool"], 1)
        self.assertEqual(diag["last_tool"], "pi_tool")
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev["interactive"], "ask_user_question")
        self.assertEqual(ev["tool_use_id"], "call_ask_1")
        question = ev["questions"][0]
        self.assertEqual(question["backend"], "pi")
        self.assertEqual(question["question"], "Which path should we use?")
        self.assertEqual(question["context"], "We found two safe options.")
        self.assertEqual(question["options"][0], {"label": "Fast path"})
        self.assertEqual(question["options"][1], {"label": "Safe path", "description": "More validation"})
        self.assertIs(question["allowFreeform"], True)
        self.assertIs(question["allowComment"], True)
        self.assertIs(question["allowMultiple"], False)

    def test_pi_ask_user_handles_json_arguments_and_bad_options(self):
        obj = {
            "type": "message",
            "timestamp": "2026-04-24T05:33:42.731Z",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "toolCall",
                        "id": "call_ask_2",
                        "name": "ask_user",
                        "arguments": '{"question":"Pick one","options":[{"label":"A"},{"bad":true},"  ","B"],"allowMultiple":true}',
                    }
                ],
            },
        }

        events, _meta, _flags, _diag = _extract_chat_events([obj])

        question = events[0]["questions"][0]
        self.assertEqual(question["options"], [{"label": "A"}, {"label": "B"}])
        self.assertIs(question["allowMultiple"], True)
        self.assertIs(question["allowFreeform"], True)

    def test_malformed_pi_ask_user_is_skipped_without_breaking_text(self):
        obj = {
            "type": "message",
            "timestamp": "2026-04-24T05:33:42.731Z",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "toolCall", "id": "bad", "name": "ask_user", "arguments": {"options": ["A"]}},
                    {"type": "text", "text": "Still visible"},
                ],
                "stopReason": "stop",
            },
        }

        events, _meta, _flags, _diag = _extract_chat_events([obj])

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["text"], "Still visible")

    def test_cursor_tail_includes_pi_ask_user_event(self):
        ask_record = {
            "type": "message",
            "timestamp": "2026-04-24T05:33:43.731Z",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "toolCall",
                        "id": "call_cursor_ask",
                        "name": "ask_user",
                        "arguments": {
                            "question": "Pick a cursor path",
                            "context": "Cursor APIs should expose this prompt.",
                            "options": ["Tail", {"title": "Live", "description": "Delta path"}],
                            "allowComment": True,
                        },
                    }
                ],
                "stopReason": "toolUse",
            },
        }
        with TemporaryDirectory() as td:
            path = Path(td) / "pi.jsonl"
            _write_jsonl(
                path,
                [
                    {
                        "type": "message",
                        "timestamp": "2026-04-24T05:33:42.731Z",
                        "message": {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                    },
                    ask_record,
                ],
            )

            events, before_byte, after_byte, has_older = _read_chat_tail_page(path, limit=20)

        self.assertFalse(has_older)
        self.assertEqual(before_byte, 0)
        self.assertGreater(after_byte, 0)
        self.assertEqual([ev.get("role") for ev in events], ["user", "assistant"])
        prompt = events[1]
        self.assertEqual(prompt["interactive"], "ask_user_question")
        self.assertEqual(prompt["tool_use_id"], "call_cursor_ask")
        question = prompt["questions"][0]
        self.assertEqual(question["backend"], "pi")
        self.assertEqual(question["question"], "Pick a cursor path")
        self.assertEqual(question["context"], "Cursor APIs should expose this prompt.")
        self.assertEqual(question["options"], [{"label": "Tail"}, {"label": "Live", "description": "Delta path"}])
        self.assertIs(question["allowComment"], True)

    def test_cursor_tail_preserves_prompt_and_text_from_same_record(self):
        obj = {
            "type": "message",
            "timestamp": "2026-04-24T05:33:42.731Z",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "toolCall", "id": "call_combo", "name": "ask_user", "arguments": {"question": "Continue?", "options": ["Yes"]}},
                    {"type": "text", "text": "I need your choice."},
                ],
                "stopReason": "toolUse",
            },
        }
        with TemporaryDirectory() as td:
            path = Path(td) / "pi.jsonl"
            _write_jsonl(path, [obj])

            events, _before_byte, _after_byte, _has_older = _read_chat_tail_page(path, limit=20)

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["interactive"], "ask_user_question")
        self.assertEqual(events[1]["text"], "I need your choice.")

    def test_cursor_history_and_live_include_pi_ask_user_event(self):
        rows = [
            {
                "type": "message",
                "timestamp": "2026-04-24T05:33:41.731Z",
                "message": {"role": "user", "content": [{"type": "text", "text": "older"}]},
            },
            {
                "type": "message",
                "timestamp": "2026-04-24T05:33:42.731Z",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "toolCall", "id": "call_history_live", "name": "ask_user", "arguments": {"question": "From history?", "options": ["A", "B"]}}],
                    "stopReason": "toolUse",
                },
            },
            {
                "type": "message",
                "timestamp": "2026-04-24T05:33:43.731Z",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "newer"}], "stopReason": "stop"},
            },
        ]
        with TemporaryDirectory() as td:
            path = Path(td) / "pi.jsonl"
            offsets = _write_jsonl(path, rows)

            history_events, _next_before, has_older = _read_chat_history_page(path, before_byte=offsets[2][0], limit=20)
            live_events, next_after, _meta, _flags, _diag, _token = _read_chat_live_delta(path, after_byte=offsets[0][1])
            live_events_again, _next_after_again, _meta2, _flags2, _diag2, _token2 = _read_chat_live_delta(path, after_byte=next_after)

        self.assertFalse(has_older)
        self.assertEqual([ev.get("role") for ev in history_events], ["user", "assistant"])
        self.assertEqual(history_events[1]["interactive"], "ask_user_question")
        self.assertEqual(history_events[1]["tool_use_id"], "call_history_live")
        self.assertEqual([ev.get("interactive") or ev.get("text") for ev in live_events], ["ask_user_question", "newer"])
        self.assertEqual(live_events_again, [])

    def test_claude_ask_user_shape_still_renders(self):
        obj = {
            "type": "assistant",
            "timestamp": "2026-04-24T05:33:42.731Z",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "AskUserQuestion",
                        "input": {
                            "questions": [
                                {
                                    "question": "Approve?",
                                    "options": [{"label": "Yes"}, {"label": "No"}],
                                }
                            ]
                        },
                    }
                ]
            },
        }

        events, _meta, _flags, _diag = _extract_chat_events([obj])

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["interactive"], "ask_user_question")
        self.assertEqual(events[0]["tool_use_id"], "toolu_1")
        self.assertEqual(events[0]["questions"][0]["question"], "Approve?")


if __name__ == "__main__":
    unittest.main()
