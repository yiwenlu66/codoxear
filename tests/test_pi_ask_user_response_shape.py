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


if __name__ == "__main__":
    unittest.main()
