import unittest
from pathlib import Path

from codoxear import pi_log
from codoxear import pi_message


ROOT = Path(__file__).resolve().parents[1]
PI_LOG_PY = ROOT / "codoxear" / "pi_log.py"
PI_MESSAGE_PY = ROOT / "codoxear" / "pi_message.py"


class TestPiMessageSource(unittest.TestCase):
    def test_pi_message_parsing_has_dedicated_owner_with_pi_log_reexports(self) -> None:
        pi_log_source = PI_LOG_PY.read_text(encoding="utf-8")
        pi_message_source = PI_MESSAGE_PY.read_text(encoding="utf-8")

        for name in [
            "PiUnknownToolCallId",
            "PiDuplicateToolCallId",
            "PiPendingToolCallId",
            "pi_user_text",
            "pi_assistant_content_parts",
            "pi_assistant_text",
            "pi_assistant_error_text",
            "pi_assistant_is_aborted_turn",
            "pi_assistant_is_final_turn_end",
            "pi_assistant_is_terminal_no_visible_response",
            "pi_assistant_tool_use_count",
            "pi_assistant_pending_tool_call_ids",
            "pi_tool_result_id",
            "pi_apply_tool_result_to_pending",
            "pi_apply_assistant_tool_calls_to_pending",
            "pi_assistant_thinking_count",
            "pi_message_role",
        ]:
            self.assertIn(f"from .pi_message import {name}", pi_log_source)

        self.assertNotIn("def _text_parts(", pi_log_source)
        self.assertNotIn("def pi_user_text(", pi_log_source)
        self.assertNotIn("def pi_assistant_content_parts(", pi_log_source)
        self.assertNotIn("def pi_assistant_is_final_turn_end(", pi_log_source)
        self.assertNotIn("def pi_assistant_is_terminal_no_visible_response(", pi_log_source)
        self.assertNotIn("def pi_assistant_pending_tool_call_ids(", pi_log_source)
        self.assertNotIn("class PiUnknownToolCallId", pi_log_source)
        self.assertNotIn("class PiDuplicateToolCallId", pi_log_source)

        self.assertIn("def _text_parts(", pi_message_source)
        self.assertIn("def pi_user_text(", pi_message_source)
        self.assertIn("def pi_assistant_is_final_turn_end(", pi_message_source)
        self.assertIn("def pi_assistant_is_terminal_no_visible_response(", pi_message_source)
        self.assertIn("def pi_assistant_pending_tool_call_ids(", pi_message_source)
        self.assertIn('sig.get("phase") == "final_answer"', pi_message_source)
        self.assertIn('part.get("type") == "toolCall"', pi_message_source)

        self.assertIs(pi_log.PiUnknownToolCallId, pi_message.PiUnknownToolCallId)
        self.assertIs(pi_log.PiDuplicateToolCallId, pi_message.PiDuplicateToolCallId)
        self.assertIs(pi_log.pi_user_text, pi_message.pi_user_text)
        self.assertIs(pi_log.pi_assistant_is_final_turn_end, pi_message.pi_assistant_is_final_turn_end)
        self.assertIs(pi_log.pi_assistant_is_terminal_no_visible_response, pi_message.pi_assistant_is_terminal_no_visible_response)
        self.assertIs(pi_log.pi_assistant_pending_tool_call_ids, pi_message.pi_assistant_pending_tool_call_ids)

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
        self.assertNotIn("tool-1", pending)
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
