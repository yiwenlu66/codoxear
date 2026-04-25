import unittest

from codoxear.rollout_log import _extract_chat_events


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
