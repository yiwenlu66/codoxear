import unittest

from codoxear.server import _extract_chat_events


class TestServerChatFlags(unittest.TestCase):
    def test_token_count_does_not_set_turn_end(self) -> None:
        _events, _meta, flags, _diag = _extract_chat_events(
            [
                {"type": "event_msg", "payload": {"type": "user_message", "message": "hello"}},
                {"type": "event_msg", "payload": {"type": "token_count", "info": {}}},
            ]
        )
        self.assertTrue(flags["turn_start"])
        self.assertFalse(flags["turn_end"])
        self.assertFalse(flags["turn_aborted"])

    def test_task_complete_sets_turn_end(self) -> None:
        _events, _meta, flags, _diag = _extract_chat_events(
            [
                {"type": "event_msg", "payload": {"type": "user_message", "message": "hello"}},
                {"type": "event_msg", "payload": {"type": "task_complete"}},
            ]
        )
        self.assertTrue(flags["turn_start"])
        self.assertTrue(flags["turn_end"])
        self.assertFalse(flags["turn_aborted"])

    def test_turn_aborted_sets_abort_flag(self) -> None:
        _events, _meta, flags, _diag = _extract_chat_events(
            [
                {"type": "event_msg", "payload": {"type": "user_message", "message": "hello"}},
                {"type": "event_msg", "payload": {"type": "turn_aborted"}},
            ]
        )
        self.assertTrue(flags["turn_start"])
        self.assertFalse(flags["turn_end"])
        self.assertTrue(flags["turn_aborted"])

    def test_local_shell_and_web_search_increment_tool_count(self) -> None:
        _events, meta, _flags, _diag = _extract_chat_events(
            [
                {"type": "response_item", "payload": {"type": "local_shell_call", "status": "completed"}},
                {"type": "response_item", "payload": {"type": "web_search_call", "status": "completed"}},
            ]
        )
        self.assertEqual(meta["tool"], 2)

    def test_claude_turn_duration_sets_turn_end(self) -> None:
        _events, _meta, flags, _diag = _extract_chat_events(
            [
                {"type": "user", "message": {"content": [{"type": "text", "text": "hello"}]}},
                {"type": "assistant", "message": {"content": [{"type": "text", "text": "done"}]}},
                {"type": "system", "subtype": "turn_duration"},
            ]
        )
        self.assertTrue(flags["turn_start"])
        self.assertTrue(flags["turn_end"])
        self.assertFalse(flags["turn_aborted"])

    def test_claude_tool_use_and_thinking_increment_meta(self) -> None:
        _events, meta, flags, diag = _extract_chat_events(
            [
                {"type": "user", "message": {"content": [{"type": "text", "text": "hello"}]}},
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {"type": "thinking", "text": "hmm"},
                            {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file": "a"}},
                        ]
                    },
                },
                {"type": "system", "subtype": "api_error"},
            ]
        )
        self.assertEqual(meta["thinking"], 1)
        self.assertEqual(meta["tool"], 1)
        self.assertTrue(flags["turn_aborted"])
        self.assertIn("Read", diag["tool_names"])


if __name__ == "__main__":
    unittest.main()
