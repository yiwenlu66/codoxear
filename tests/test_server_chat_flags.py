import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from codoxear.pi_log import pi_token_update
from codoxear.rollout_log import _last_chat_role_ts_from_tail
from codoxear.rollout_log import _single_chat_event
from codoxear.server import _compute_idle_from_log
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

    def test_task_complete_sets_turn_end_flag(self) -> None:
        _events, _meta, flags, _diag = _extract_chat_events(
            [
                {"type": "event_msg", "payload": {"type": "user_message", "message": "hello"}},
                {"type": "event_msg", "payload": {"type": "task_complete", "turn_id": "t1"}},
            ]
        )
        self.assertTrue(flags["turn_start"])
        self.assertTrue(flags["turn_end"])
        self.assertFalse(flags["turn_aborted"])

    def test_turn_complete_sets_turn_end_flag(self) -> None:
        _events, _meta, flags, _diag = _extract_chat_events(
            [
                {"type": "event_msg", "payload": {"type": "user_message", "message": "hello"}},
                {"type": "event_msg", "payload": {"type": "turn_complete", "turn_id": "t1"}},
            ]
        )
        self.assertTrue(flags["turn_start"])
        self.assertTrue(flags["turn_end"])
        self.assertFalse(flags["turn_aborted"])

    def test_codex_error_event_is_visible_and_ends_turn(self) -> None:
        events, _meta, flags, _diag = _extract_chat_events(
            [
                {"type": "event_msg", "payload": {"type": "user_message", "message": "hello"}},
                {
                    "timestamp": "2026-04-26T01:02:03.000Z",
                    "type": "event_msg",
                    "payload": {
                        "type": "error",
                        "message": "unexpected status 400 Bad Request: invalid model",
                        "codex_error_info": "bad_request",
                    },
                },
            ]
        )
        self.assertTrue(flags["turn_end"])
        self.assertEqual(events[-1]["role"], "assistant")
        self.assertEqual(events[-1]["message_class"], "error")
        self.assertEqual(events[-1]["text"], "unexpected status 400 Bad Request: invalid model")
        self.assertIsInstance(events[-1]["message_id"], str)
        self.assertEqual(events[-1]["ts"], 1777165323.0)

    def test_codex_thread_rollback_error_is_visible_but_not_turn_end(self) -> None:
        events, _meta, flags, _diag = _extract_chat_events(
            [
                {
                    "type": "event_msg",
                    "payload": {
                        "type": "error",
                        "message": "rollback failed",
                        "codex_error_info": "thread_rollback_failed",
                    },
                }
            ]
        )
        self.assertFalse(flags["turn_end"])
        self.assertEqual(events[0]["message_class"], "error")
        self.assertEqual(events[0]["text"], "rollback failed")

    def test_codex_stream_error_and_warning_are_visible_without_turn_end(self) -> None:
        events, _meta, flags, _diag = _extract_chat_events(
            [
                {
                    "type": "event_msg",
                    "payload": {
                        "type": "stream_error",
                        "message": "stream disconnected",
                        "additional_details": "retrying request",
                    },
                },
                {"type": "event_msg", "payload": {"type": "warning", "message": "context warning"}},
            ]
        )
        self.assertFalse(flags["turn_end"])
        self.assertEqual([ev["message_class"] for ev in events], ["error", "warning"])
        self.assertEqual(events[0]["text"], "stream disconnected\n\nretrying request")
        self.assertEqual(events[1]["text"], "context warning")

    def test_local_shell_and_web_search_increment_tool_count(self) -> None:
        _events, meta, _flags, _diag = _extract_chat_events(
            [
                {"type": "response_item", "payload": {"type": "local_shell_call", "status": "completed"}},
                {"type": "response_item", "payload": {"type": "web_search_call", "status": "completed"}},
            ]
        )
        self.assertEqual(meta["tool"], 2)

    def test_assistant_message_carries_message_class(self) -> None:
        events, _meta, _flags, _diag = _extract_chat_events(
            [
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "phase": "final_answer",
                        "content": [{"type": "output_text", "text": "done"}],
                    },
                }
            ]
        )
        self.assertEqual(events[0]["message_class"], "final_response")
        self.assertIsInstance(events[0]["message_id"], str)

    def test_pi_message_sets_turn_end_for_final_text(self) -> None:
        events, meta, flags, diag = _extract_chat_events(
            [
                {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "hello"}]}},
                {"type": "message", "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}]}},
            ]
        )
        self.assertTrue(flags["turn_start"])
        self.assertTrue(flags["turn_end"])
        self.assertEqual(events[1]["message_class"], "final_response")
        self.assertEqual(meta["thinking"], 0)
        self.assertEqual(meta["tool"], 0)
        self.assertEqual(diag["tool_names"], [])

    def test_pi_message_counts_thinking_and_tool_use_as_narration(self) -> None:
        events, meta, flags, diag = _extract_chat_events(
            [
                {
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "working"},
                            {"type": "thinking", "thinking": "hmm"},
                            {"type": "toolCall", "id": "t1", "name": "bash", "arguments": {"command": "pwd"}},
                        ],
                    },
                }
            ]
        )
        self.assertFalse(flags["turn_end"])
        self.assertEqual(events[0]["message_class"], "narration")
        self.assertEqual(meta["thinking"], 1)
        self.assertEqual(meta["tool"], 1)
        self.assertEqual(diag["last_tool"], "pi_tool")

    def test_pi_final_message_with_thinking_sets_turn_end(self) -> None:
        events, meta, flags, _diag = _extract_chat_events(
            [
                {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "test"}]}},
                {
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "stopReason": "stop",
                        "content": [
                            {"type": "thinking", "thinking": ""},
                            {"type": "text", "text": "done", "textSignature": "{\"v\":1,\"phase\":\"final_answer\"}"},
                        ],
                    },
                },
            ]
        )
        self.assertTrue(flags["turn_end"])
        self.assertEqual(events[-1]["message_class"], "final_response")
        self.assertEqual(meta["thinking"], 1)

    def test_pi_error_message_is_visible_and_ends_turn(self) -> None:
        events, _meta, flags, _diag = _extract_chat_events(
            [
                {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "test"}]}},
                {
                    "type": "message",
                    "timestamp": "2026-04-26T01:02:03.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [],
                        "stopReason": "error",
                        "errorMessage": "401 Invalid API key",
                    },
                },
            ]
        )
        self.assertTrue(flags["turn_end"])
        self.assertEqual(events[-1]["role"], "assistant")
        self.assertEqual(events[-1]["message_class"], "error")
        self.assertEqual(events[-1]["text"], "401 Invalid API key")
        self.assertEqual(events[-1]["ts"], 1777165323.0)

    def test_pi_aborted_message_sets_turn_aborted(self) -> None:
        events, _meta, flags, _diag = _extract_chat_events(
            [
                {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "test"}]}},
                {"type": "message", "message": {"role": "assistant", "content": [], "stopReason": "aborted"}},
            ]
        )
        self.assertTrue(flags["turn_start"])
        self.assertTrue(flags["turn_aborted"])
        self.assertFalse(flags["turn_end"])
        self.assertEqual([event["role"] for event in events], ["user"])

    def test_pi_aborted_text_is_not_history_final_response(self) -> None:
        self.assertIsNone(
            _single_chat_event(
                {
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "partial"}],
                        "stopReason": "aborted",
                    },
                }
            )
        )

    def test_pi_aborted_text_does_not_count_as_last_assistant_chat(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "pi.jsonl"
            path.write_text(
                "\n".join(
                    [
                        '{"type":"session","id":"s1","cwd":"/tmp"}',
                        '{"type":"message","timestamp":"2026-01-01T00:00:00Z","message":{"role":"user","content":[{"type":"text","text":"test"}]}}',
                        '{"type":"message","timestamp":"2026-01-01T00:00:01Z","message":{"role":"assistant","stopReason":"aborted","content":[{"type":"text","text":"partial"}]}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(_last_chat_role_ts_from_tail(path, max_scan_bytes=512 * 1024), ("user", 1767225600.0))

    def test_compute_idle_from_log_pi_final_message_with_thinking_is_idle(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "pi.jsonl"
            path.write_text(
                "\n".join(
                    [
                        '{"type":"session","id":"s1","cwd":"/tmp"}',
                        '{"type":"message","message":{"role":"user","content":[{"type":"text","text":"test"}]}}',
                        '{"type":"message","message":{"role":"assistant","stopReason":"stop","content":[{"type":"thinking","thinking":""},{"type":"text","text":"done","textSignature":"{\\"v\\":1,\\"phase\\":\\"final_answer\\"}"}]}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertTrue(_compute_idle_from_log(path))

    def test_pi_token_update_uses_models_json_context_window(self) -> None:
        with TemporaryDirectory() as td:
            models_path = Path(td) / "models.json"
            models_path.write_text(
                '{"providers":{"macaron":{"models":[{"id":"gpt-5.4","contextWindow":1000000}]}}}\n',
                encoding="utf-8",
            )
            settings_path = Path(td) / "settings.json"
            settings_path.write_text('{"compaction":{"reserveTokens":20000}}\n', encoding="utf-8")
            obj = {
                "type": "message",
                "timestamp": "2026-03-30T08:44:03.883Z",
                "message": {
                    "role": "assistant",
                    "provider": "macaron",
                    "model": "gpt-5.4",
                    "usage": {"totalTokens": 11817},
                    "content": [{"type": "text", "text": "done"}],
                },
            }
            token = pi_token_update(obj, models_path=models_path, settings_path=settings_path)
        if token is None:
            self.fail("missing token update")
        self.assertEqual(token["context_window"], 1000000)
        self.assertEqual(token["tokens_in_context"], 11817)
        self.assertEqual(token["reserved_tokens"], 20000)
        self.assertEqual(token["max_input_tokens"], 980000)
        self.assertEqual(token["tokens_remaining"], 968183)
        self.assertEqual(token["percent_remaining"], 99)

    def test_pi_token_update_queries_pi_registry_for_builtin_context_window(self) -> None:
        with TemporaryDirectory() as td:
            models_path = Path(td) / "models.json"
            models_path.write_text(
                '{"providers":{"occ":{"models":[{"id":"gpt-5.5","contextWindow":272000}]}}}\n',
                encoding="utf-8",
            )
            settings_path = Path(td) / "settings.json"
            settings_path.write_text('{"compaction":{"reserveTokens":16000}}\n', encoding="utf-8")
            obj = {
                "type": "message",
                "timestamp": "2026-06-10T18:10:25.343Z",
                "message": {
                    "role": "assistant",
                    "provider": "openai-codex",
                    "model": "gpt-5.5",
                    "usage": {"totalTokens": 185136},
                    "content": [{"type": "text", "text": "done"}],
                },
            }
            with patch("codoxear.pi_log._default_pi_models_path", return_value=models_path), patch(
                "codoxear.pi_log._query_pi_context_windows",
                return_value={("openai-codex", "gpt-5.5"): 272000},
            ) as query:
                token = pi_token_update(obj, settings_path=settings_path)
        query.assert_called_once_with(models_path)
        if token is None:
            self.fail("missing token update")
        self.assertEqual(token["context_window"], 272000)
        self.assertEqual(token["tokens_in_context"], 185136)
        self.assertEqual(token["reserved_tokens"], 16000)
        self.assertEqual(token["max_input_tokens"], 256000)
        self.assertEqual(token["tokens_remaining"], 70864)

    def test_pi_token_update_ignores_error_usage(self) -> None:
        obj = {
            "type": "message",
            "timestamp": "2026-06-10T17:58:50.128Z",
            "message": {
                "role": "assistant",
                "provider": "openai-codex",
                "model": "gpt-5.5",
                "stopReason": "error",
                "usage": {"totalTokens": 0},
                "content": [],
                "errorMessage": "OpenAI API error (522): 522 status code (no body)",
            },
        }
        with patch("codoxear.pi_log._query_pi_context_windows") as query:
            token = pi_token_update(obj)
        query.assert_not_called()
        self.assertIsNone(token)


if __name__ == "__main__":
    unittest.main()
