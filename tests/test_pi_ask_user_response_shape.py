import unittest

from codoxear import pi_messages


class TestPiAskUserResponseShape(unittest.TestCase):
    def test_tool_result_reads_response_shape(self) -> None:
        entries = [
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "toolCall",
                            "id": "ask-4",
                            "name": "ask_user",
                            "arguments": {
                                "question": "How should we test this?",
                                "options": ["Single", "Freeform", "Multiple"],
                                "allowFreeform": True,
                            },
                        }
                    ],
                },
            },
            {
                "type": "message",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "ask-4",
                    "toolName": "ask_user",
                    "details": {
                        "response": {
                            "kind": "selection",
                            "selections": ["Single"],
                            "comment": "web can't click it",
                        },
                        "cancelled": False,
                    },
                    "content": [{"type": "text", "text": "User answered: Single - web can't click it"}],
                },
            },
        ]

        events, _meta, _flags, _diag = pi_messages.normalize_pi_entries(entries)

        self.assertEqual(len(events), 1)
        self.assertEqual(
            events[0],
            {
                "type": "ask_user",
                "tool_call_id": "ask-4",
                "question": "How should we test this?",
                "context": "",
                "options": ["Single", "Freeform", "Multiple"],
                "allow_freeform": True,
                "allow_multiple": False,
                "timeout_ms": None,
                "resolved": True,
                "ts": 0.0,
                "answer": "Single",
                "cancelled": False,
                "was_custom": False,
            },
        )

    def test_ask_user_question_result_falls_back_to_content_text(self) -> None:
        entries = [
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "toolCall",
                            "id": "ask-5",
                            "name": "AskUserQuestion",
                            "arguments": {
                                "questions": [
                                    {
                                        "header": "Pi list",
                                        "question": "How should Pi use ~/.pi/agent/models.json?",
                                        "options": [
                                            {"label": "Provider-linked model list"},
                                            {"label": "Defaults only"},
                                        ],
                                        "multiSelect": False,
                                    }
                                ]
                            },
                        }
                    ],
                },
            },
            {
                "type": "message",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "ask-5",
                    "toolName": "AskUserQuestion",
                    "details": {
                        "action": "answered",
                    },
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "User has answered your questions: "
                                '"How should Pi use ~/.pi/agent/models.json?"='
                                '"Provider-linked model list". '
                                "You can now continue with the user's answers in mind."
                            ),
                        }
                    ],
                },
            },
        ]

        events, _meta, _flags, _diag = pi_messages.normalize_pi_entries(entries)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "ask_user")
        self.assertEqual(events[0]["context"], "Pi list")
        self.assertEqual(events[0]["answer"], "Provider-linked model list")
        self.assertTrue(events[0]["resolved"])


if __name__ == "__main__":
    unittest.main()
