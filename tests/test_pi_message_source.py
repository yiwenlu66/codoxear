import unittest
from pathlib import Path

from codoxear import pi_log
from codoxear import pi_message


ROOT = Path(__file__).resolve().parents[1]
PI_LOG_PY = ROOT / "codoxear" / "pi_log.py"
PI_MESSAGE_PY = ROOT / "codoxear" / "pi_message.py"


class TestPiMessageSource(unittest.TestCase):

    def test_pi_terminal_no_visible_response_predicate_semantics(self) -> None:
        def row(stop_reason: object, content: object = None, role: str = "assistant") -> dict:
            return {
                "type": "message",
                "message": {
                    "role": role,
                    "stopReason": stop_reason,
                    "content": [] if content is None else content,
                },
            }

        self.assertTrue(pi_message.pi_assistant_is_terminal_no_visible_response(row("stop")))
        self.assertTrue(pi_message.pi_assistant_is_terminal_no_visible_response(row("end_turn")))
        self.assertTrue(
            pi_message.pi_assistant_is_terminal_no_visible_response(
                row("stop", [{"type": "thinking", "thinking": "internal"}])
            )
        )
        self.assertFalse(pi_message.pi_assistant_is_terminal_no_visible_response(row(None)))
        self.assertFalse(pi_message.pi_assistant_is_terminal_no_visible_response(row("")))
        self.assertFalse(pi_message.pi_assistant_is_terminal_no_visible_response(row("length")))
        self.assertFalse(pi_message.pi_assistant_is_terminal_no_visible_response(row("unknown_future_reason")))
        self.assertFalse(pi_message.pi_assistant_is_terminal_no_visible_response(row("toolUse")))
        self.assertFalse(pi_message.pi_assistant_is_terminal_no_visible_response(row("error")))
        self.assertFalse(pi_message.pi_assistant_is_terminal_no_visible_response(row("aborted")))
        with_error_message = row("stop")
        with_error_message["message"]["errorMessage"] = "provider failed"
        self.assertFalse(pi_message.pi_assistant_is_terminal_no_visible_response(with_error_message))
        with_error_flag = row("stop")
        with_error_flag["message"]["isError"] = True
        self.assertFalse(pi_message.pi_assistant_is_terminal_no_visible_response(with_error_flag))
        self.assertFalse(pi_message.pi_assistant_is_terminal_no_visible_response(row("stop", [{"type": "text", "text": "done"}])))
        self.assertFalse(
            pi_message.pi_assistant_is_terminal_no_visible_response(
                row("stop", [{"type": "toolCall", "id": "tool-1"}])
            )
        )
        self.assertFalse(pi_message.pi_assistant_is_terminal_no_visible_response(row("stop", role="toolResult")))

    def test_pi_length_visible_text_is_not_final_turn_end(self) -> None:
        def assistant(stop_reason: str, content: list[dict]) -> dict:
            return {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "stopReason": stop_reason,
                    "content": content,
                },
            }

        visible_stop = assistant("stop", [{"type": "text", "text": "done"}])
        self.assertTrue(pi_message.pi_assistant_is_final_turn_end(visible_stop))

        visible_length = assistant("length", [{"type": "text", "text": "partial before compaction"}])
        self.assertFalse(pi_message.pi_assistant_is_final_turn_end(visible_length))

        signed_length = assistant(
            "length",
            [{"type": "text", "text": "signed partial", "textSignature": '{"phase":"final_answer"}'}],
        )
        self.assertFalse(pi_message.pi_assistant_is_final_turn_end(signed_length))

        signed_tool_use_final = assistant(
            "toolUse",
            [{"type": "text", "text": "done", "textSignature": '{"phase":"final_answer"}'}],
        )
        self.assertTrue(pi_message.pi_assistant_is_final_turn_end(signed_tool_use_final))

    def test_pi_message_tool_call_identity_semantics(self) -> None:
        assistant = {
            "type": "message",
            "message": {
                "role": "assistant",
                "stopReason": "toolUse",
                "content": [
                    {"type": "thinking", "thinking": "plan"},
                    {"type": "toolCall", "id": "tool-1"},
                    {"type": "toolCall", "id": "tool-1"},
                    {"type": "toolCall"},
                    {"type": "text", "text": "done", "textSignature": '{"phase":"final_answer"}'},
                ],
            },
        }
        ids = pi_message.pi_assistant_pending_tool_call_ids(assistant)
        self.assertEqual(ids[0], "tool-1")
        self.assertIsInstance(ids[1], pi_message.PiDuplicateToolCallId)
        self.assertEqual(ids[1].tool_id, "tool-1")
        self.assertIsInstance(ids[2], pi_message.PiUnknownToolCallId)
        self.assertFalse(pi_message.pi_assistant_is_final_turn_end(assistant))

        pending = set(ids)
        pi_message.pi_apply_tool_result_to_pending({"type": "message", "message": {"role": "toolResult", "toolCallId": "tool-1"}}, pending)
        self.assertNotContains("tool-1", pending)
        self.assertTrue(any(isinstance(item, pi_message.PiDuplicateToolCallId) for item in pending))

        final = {
            "type": "message",
            "message": {
                "role": "assistant",
                "stopReason": "toolUse",
                "content": [
                    {"type": "thinking", "thinking": "plan"},
                    {"type": "text", "text": "done", "textSignature": '{"phase":"final_answer"}'},
                ],
            },
        }
        self.assertTrue(pi_message.pi_assistant_is_final_turn_end(final))
        self.assertIsNone(pi_message.pi_message_role({"type": "message", "message": {"role": ""}}))


if __name__ == "__main__":
    unittest.main()
