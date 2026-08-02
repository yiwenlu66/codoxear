import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from codoxear.rollout_chat_batch import _extract_chat_events
from codoxear.rollout_idle import _analyze_log_chunk
from codoxear.server import _compute_idle_from_log


def _write_jsonl(path: Path, objs: list[dict]) -> None:
    path.write_text("".join(json.dumps(o) + "\n" for o in objs), encoding="utf-8")


class TestIdleHeuristics(unittest.TestCase):
    def test_fresh_session_is_idle(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout.jsonl"
            _write_jsonl(p, [{"type": "session_meta", "payload": {"id": "s"}}])
            self.assertIs(_compute_idle_from_log(p, max_scan_bytes=64 * 1024), True)

    def test_open_turn_without_assistant_candidate_is_busy(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout.jsonl"
            _write_jsonl(
                p,
                [
                    {"type": "session_meta", "payload": {"id": "s"}},
                    {"type": "event_msg", "payload": {"type": "user_message", "message": "hi"}},
                    {"type": "event_msg", "payload": {"type": "agent_reasoning"}},
                ],
            )
            self.assertIs(_compute_idle_from_log(p, max_scan_bytes=64 * 1024), False)

    def test_assistant_message_after_user_without_turn_complete_stays_busy(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout.jsonl"
            _write_jsonl(
                p,
                [
                    {"type": "session_meta", "payload": {"id": "s"}},
                    {"type": "event_msg", "payload": {"type": "user_message", "message": "hi"}},
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "done"}],
                        },
                    },
                ],
            )
            self.assertIs(_compute_idle_from_log(p, max_scan_bytes=64 * 1024), False)

    def test_tool_after_assistant_reopens_busy(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout.jsonl"
            _write_jsonl(
                p,
                [
                    {"type": "session_meta", "payload": {"id": "s"}},
                    {"type": "event_msg", "payload": {"type": "user_message", "message": "hi"}},
                    {"type": "event_msg", "payload": {"type": "agent_message", "message": "starting"}},
                    {"type": "response_item", "payload": {"type": "function_call", "call_id": "c1"}},
                ],
            )
            self.assertIs(_compute_idle_from_log(p, max_scan_bytes=64 * 1024), False)

    def test_web_search_after_assistant_reopens_busy(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout.jsonl"
            _write_jsonl(
                p,
                [
                    {"type": "session_meta", "payload": {"id": "s"}},
                    {"type": "event_msg", "payload": {"type": "user_message", "message": "hi"}},
                    {"type": "event_msg", "payload": {"type": "agent_message", "message": "starting"}},
                    {"type": "response_item", "payload": {"type": "web_search_call", "status": "completed"}},
                ],
            )
            self.assertIs(_compute_idle_from_log(p, max_scan_bytes=64 * 1024), False)

    def test_local_shell_after_assistant_reopens_busy(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout.jsonl"
            _write_jsonl(
                p,
                [
                    {"type": "session_meta", "payload": {"id": "s"}},
                    {"type": "event_msg", "payload": {"type": "user_message", "message": "hi"}},
                    {"type": "event_msg", "payload": {"type": "agent_message", "message": "starting"}},
                    {"type": "response_item", "payload": {"type": "local_shell_call", "status": "completed"}},
                ],
            )
            self.assertIs(_compute_idle_from_log(p, max_scan_bytes=64 * 1024), False)

    def test_turn_aborted_is_idle(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout.jsonl"
            _write_jsonl(
                p,
                [
                    {"type": "session_meta", "payload": {"id": "s"}},
                    {"type": "event_msg", "payload": {"type": "user_message", "message": "hi"}},
                    {"type": "event_msg", "payload": {"type": "turn_aborted"}},
                ],
            )
            self.assertIs(_compute_idle_from_log(p, max_scan_bytes=64 * 1024), True)

    def test_task_complete_is_idle(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout.jsonl"
            _write_jsonl(
                p,
                [
                    {"type": "session_meta", "payload": {"id": "s"}},
                    {"type": "event_msg", "payload": {"type": "user_message", "message": "hi"}},
                    {"type": "response_item", "payload": {"type": "function_call", "call_id": "c1"}},
                    {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "c1"}},
                    {"type": "event_msg", "payload": {"type": "task_complete"}},
                ],
            )
            self.assertIs(_compute_idle_from_log(p, max_scan_bytes=64 * 1024), True)

    def test_turn_complete_is_idle(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout.jsonl"
            _write_jsonl(
                p,
                [
                    {"type": "session_meta", "payload": {"id": "s"}},
                    {"type": "event_msg", "payload": {"type": "user_message", "message": "hi"}},
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "intermediate"}],
                        },
                    },
                    {"type": "event_msg", "payload": {"type": "turn_complete", "turn_id": "t1"}},
                ],
            )
            self.assertIs(_compute_idle_from_log(p, max_scan_bytes=64 * 1024), True)

    def test_codex_error_is_idle(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout.jsonl"
            _write_jsonl(
                p,
                [
                    {"type": "session_meta", "payload": {"id": "s"}},
                    {"type": "event_msg", "payload": {"type": "user_message", "message": "hi"}},
                    {
                        "type": "event_msg",
                        "payload": {
                            "type": "error",
                            "message": "unexpected status 400 Bad Request",
                            "codex_error_info": "bad_request",
                        },
                    },
                ],
            )
            self.assertIs(_compute_idle_from_log(p, max_scan_bytes=64 * 1024), True)

    def test_codex_thread_rollback_error_does_not_close_turn(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout.jsonl"
            _write_jsonl(
                p,
                [
                    {"type": "session_meta", "payload": {"id": "s"}},
                    {"type": "event_msg", "payload": {"type": "user_message", "message": "hi"}},
                    {
                        "type": "event_msg",
                        "payload": {
                            "type": "error",
                            "message": "rollback failed",
                            "codex_error_info": {"thread_rollback_failed": {}},
                        },
                    },
                ],
            )
            self.assertIs(_compute_idle_from_log(p, max_scan_bytes=64 * 1024), False)

    def test_assistant_commentary_after_task_complete_is_busy(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout.jsonl"
            _write_jsonl(
                p,
                [
                    {"type": "session_meta", "payload": {"id": "s"}},
                    {"type": "event_msg", "payload": {"type": "user_message", "message": "hi"}},
                    {"type": "event_msg", "payload": {"type": "task_complete", "turn_id": "t1", "last_agent_message": "done"}},
                    {"type": "event_msg", "payload": {"type": "agent_message", "message": "continuing work"}},
                ],
            )
            self.assertIs(_compute_idle_from_log(p, max_scan_bytes=64 * 1024), False)

    def test_response_item_end_turn_is_idle(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout.jsonl"
            _write_jsonl(
                p,
                [
                    {"type": "session_meta", "payload": {"id": "s"}},
                    {"type": "event_msg", "payload": {"type": "user_message", "message": "hi"}},
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "end_turn": True,
                            "content": [{"type": "output_text", "text": "done"}],
                        },
                    },
                ],
            )
            self.assertIs(_compute_idle_from_log(p, max_scan_bytes=64 * 1024), True)

    def test_pi_error_message_without_terminal_row_stays_busy(self) -> None:
        """Pi's error row has no terminal/retry discriminator, so it cannot end a turn."""
        with TemporaryDirectory() as td:
            p = Path(td) / "pi.jsonl"
            _write_jsonl(
                p,
                [
                    {"type": "session", "id": "s", "cwd": "/tmp"},
                    {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "hi"}]}},
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "content": [],
                            "stopReason": "error",
                            "errorMessage": "401 Invalid API key",
                        },
                    },
                ],
            )
            self.assertIs(_compute_idle_from_log(p, max_scan_bytes=64 * 1024), False)

    def test_pi_error_then_retry_activity_stays_busy(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "pi.jsonl"
            _write_jsonl(
                p,
                [
                    {"type": "session", "id": "s", "cwd": "/tmp"},
                    {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "hi"}]}},
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "content": [],
                            "stopReason": "error",
                            "errorMessage": "429 rate limit; retrying",
                        },
                    },
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "thinking", "thinking": "retrying"}],
                            "stopReason": "toolUse",
                        },
                    },
                ],
            )
            self.assertIs(_compute_idle_from_log(p, max_scan_bytes=64 * 1024), False)

    def test_pi_error_then_final_response_is_idle(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "pi.jsonl"
            _write_jsonl(
                p,
                [
                    {"type": "session", "id": "s", "cwd": "/tmp"},
                    {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "hi"}]}},
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "content": [],
                            "stopReason": "error",
                            "errorMessage": "503 overloaded; retrying",
                        },
                    },
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "done after retry"}],
                            "stopReason": "stop",
                        },
                    },
                ],
            )
            self.assertIs(_compute_idle_from_log(p, max_scan_bytes=64 * 1024), True)

    def test_pi_length_tool_call_message_is_not_idle(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "pi.jsonl"
            _write_jsonl(
                p,
                [
                    {"type": "session", "id": "s", "cwd": "/tmp"},
                    {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "run"}]}},
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "partial"},
                                {"type": "toolCall", "id": "tool-1", "name": "bash", "arguments": {"command": "pwd"}},
                            ],
                            "stopReason": "length",
                        },
                    },
                ],
            )
            self.assertIs(_compute_idle_from_log(p, max_scan_bytes=64 * 1024), False)

    def test_pi_length_visible_text_message_remains_busy(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "pi.jsonl"
            _write_jsonl(
                p,
                [
                    {"type": "session", "id": "s", "cwd": "/tmp"},
                    {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "run"}]}},
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "partial before compaction"}],
                            "stopReason": "length",
                        },
                    },
                ],
            )
            self.assertIs(_compute_idle_from_log(p, max_scan_bytes=64 * 1024), False)

    def test_pi_length_text_compaction_continuation_remains_busy_with_pending_tool(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "pi.jsonl"
            prefix_rows = [
                {"type": "session", "id": "s", "cwd": "/tmp"},
                {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "run"}]}},
                {
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "partial before compaction"}],
                        "stopReason": "length",
                    },
                },
            ]
            _write_jsonl(p, prefix_rows)
            self.assertIs(_compute_idle_from_log(p, max_scan_bytes=64 * 1024), False)

            _write_jsonl(
                p,
                prefix_rows
                + [
                    {"type": "compaction", "message": "compacting context"},
                    {"type": "custom_message", "message": "continuing"},
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "continuing with a tool"},
                                {"type": "toolCall", "id": "tool-1", "name": "bash", "arguments": {"command": "pwd"}},
                            ],
                            "stopReason": "toolUse",
                        },
                    },
                ],
            )
            self.assertIs(_compute_idle_from_log(p, max_scan_bytes=64 * 1024), False)

    def test_pi_length_thinking_only_message_remains_busy(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "pi.jsonl"
            _write_jsonl(
                p,
                [
                    {"type": "session", "id": "s", "cwd": "/tmp"},
                    {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "run"}]}},
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "thinking", "thinking": "internal before compaction"}],
                            "stopReason": "length",
                        },
                    },
                ],
            )
            self.assertIs(_compute_idle_from_log(p, max_scan_bytes=64 * 1024), False)

    def test_pi_stop_empty_message_is_idle(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "pi.jsonl"
            _write_jsonl(
                p,
                [
                    {"type": "session", "id": "s", "cwd": "/tmp"},
                    {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "hi"}]}},
                    {"type": "message", "message": {"role": "assistant", "content": [], "stopReason": "stop"}},
                ],
            )
            self.assertIs(_compute_idle_from_log(p, max_scan_bytes=64 * 1024), True)

    def test_pi_end_turn_empty_message_is_idle(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "pi.jsonl"
            _write_jsonl(
                p,
                [
                    {"type": "session", "id": "s", "cwd": "/tmp"},
                    {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "hi"}]}},
                    {"type": "message", "message": {"role": "assistant", "content": [], "stopReason": "end_turn"}},
                ],
            )
            self.assertIs(_compute_idle_from_log(p, max_scan_bytes=64 * 1024), True)

    def test_pi_stop_thinking_only_message_is_idle(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "pi.jsonl"
            _write_jsonl(
                p,
                [
                    {"type": "session", "id": "s", "cwd": "/tmp"},
                    {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "hi"}]}},
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "thinking", "thinking": "internal"}],
                            "stopReason": "stop",
                        },
                    },
                ],
            )
            self.assertIs(_compute_idle_from_log(p, max_scan_bytes=64 * 1024), True)

    def test_pi_nonterminal_thinking_message_remains_busy(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "pi.jsonl"
            _write_jsonl(
                p,
                [
                    {"type": "session", "id": "s", "cwd": "/tmp"},
                    {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "hi"}]}},
                    {"type": "message", "message": {"role": "assistant", "content": [{"type": "thinking", "thinking": "internal"}]}},
                ],
            )
            self.assertIs(_compute_idle_from_log(p, max_scan_bytes=64 * 1024), False)

    def test_pi_tool_use_tool_call_message_remains_busy(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "pi.jsonl"
            _write_jsonl(
                p,
                [
                    {"type": "session", "id": "s", "cwd": "/tmp"},
                    {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "run"}]}},
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "toolCall", "id": "tool-1", "name": "bash", "arguments": {}}],
                            "stopReason": "toolUse",
                        },
                    },
                ],
            )
            self.assertIs(_compute_idle_from_log(p, max_scan_bytes=64 * 1024), False)

    def test_pi_aborted_message_is_idle(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "pi.jsonl"
            _write_jsonl(
                p,
                [
                    {"type": "session", "id": "s", "cwd": "/tmp"},
                    {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "hi"}]}},
                    {"type": "message", "message": {"role": "assistant", "content": [], "stopReason": "aborted"}},
                ],
            )
            self.assertIs(_compute_idle_from_log(p, max_scan_bytes=64 * 1024), True)


if __name__ == "__main__":
    unittest.main()


class TestAnalyzeLogChunkBackendRows(unittest.TestCase):
    def test_analyze_log_chunk_counts_pi_thinking_and_tool_rows(self) -> None:
        # Regression: rollout_idle dropped pi/cc helper imports during
        # decomposition; these rows raised live NameError in mark_log_delta.
        objs = [
            {"type": "message", "message": {"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm"}]}},
            {"type": "message", "message": {"role": "assistant", "content": [{"type": "toolCall", "id": "t1", "name": "bash", "arguments": {"command": "pwd"}}]}},
        ]
        d_th, d_thinking_tokens, d_tools, d_sys, _ts, _token, _events, _turn_state = _analyze_log_chunk(objs)
        self.assertEqual(d_th, 1)
        self.assertEqual(d_thinking_tokens, 0)
        self.assertEqual(d_tools, 1)
        self.assertEqual(d_sys, 0)

    def test_analyze_log_chunk_counts_cc_thinking_and_tool_result_rows(self) -> None:
        objs = [
            {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm"}]}},
            {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}},
        ]
        d_th, d_thinking_tokens, d_tools, d_sys, _ts, _token, _events, _turn_state = _analyze_log_chunk(objs)
        self.assertEqual(d_th, 1)
        self.assertEqual(d_thinking_tokens, 0)
        self.assertGreaterEqual(d_tools, 1)

    def test_analyze_log_chunk_preserves_counts_across_pi_steer(self) -> None:
        objs = [
            {"type": "message", "message": {"role": "assistant", "content": [{"type": "thinking", "thinking": "before steer"}]}},
            {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "steer now"}]}},
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "after steer"},
                        {"type": "toolCall", "id": "t1", "name": "bash", "arguments": {"command": "pwd"}},
                    ],
                },
            },
        ]
        d_th, d_thinking_tokens, d_tools, d_sys, _ts, _token, _events, _turn_state = _analyze_log_chunk(objs)
        self.assertEqual((d_th, d_thinking_tokens), (2, 0))
        self.assertEqual(d_tools, 1)
        self.assertEqual(d_sys, 0)

    def test_analyze_log_chunk_resets_counts_after_closed_turn(self) -> None:
        objs = [
            {"type": "message", "message": {"role": "assistant", "content": [{"type": "thinking", "thinking": "old"}]}},
            {
                "type": "message",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}], "stopReason": "stop"},
            },
            {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "new turn"}]}},
            {"type": "message", "message": {"role": "assistant", "content": [{"type": "thinking", "thinking": "new"}]}},
        ]
        d_th, d_thinking_tokens, d_tools, d_sys, _ts, _token, _events, _turn_state = _analyze_log_chunk(objs)
        self.assertEqual((d_th, d_thinking_tokens), (1, 0))
        self.assertEqual(d_tools, 0)
        self.assertEqual(d_sys, 0)

    def test_analyze_log_chunk_keeps_episode_counts_across_pi_subagent_delivery(self) -> None:
        objs = [
            {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "human request"}]}},
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "thinking", "thinking": "before"}, {"type": "toolCall", "id": "t1", "name": "bash", "arguments": {}}],
                    "usage": {"reasoning": 20},
                },
            },
            {
                "type": "message",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "first response"}], "stopReason": "stop"},
            },
            {
                "type": "message",
                "message": {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "**📨 From subagent-result** (/workspace)\n"},
                        {"type": "text", "text": "subagent results"},
                    ],
                },
            },
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "thinking", "thinking": "after"}, {"type": "toolCall", "id": "t2", "name": "bash", "arguments": {}}],
                    "usage": {"reasoning": 16},
                },
            },
        ]
        d_th, d_thinking_tokens, d_tools, _d_sys, _ts, _token, _events, turn_state = _analyze_log_chunk(objs)
        self.assertEqual((d_th, d_thinking_tokens, d_tools), (2, 36, 2))
        self.assertTrue(turn_state.turn_open)
        self.assertTrue(turn_state.counters_reset)  # The episode began with the human row.

    def test_analyze_log_chunk_counts_exact_pi_reasoning_tokens(self) -> None:
        objs = [
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "thinking", "thinking": "first"}],
                    "usage": {"reasoning": 108, "input": 1, "output": 1, "cacheRead": 0, "totalTokens": 110},
                },
            },
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "thinking", "thinking": "second"}],
                    "usage": {"reasoning": 16, "input": 1, "output": 1, "cacheRead": 0, "totalTokens": 18},
                },
            },
        ]
        d_th, d_thinking_tokens, *_rest = _analyze_log_chunk(objs)
        self.assertEqual((d_th, d_thinking_tokens), (2, 124))

    def test_analyze_log_chunk_counts_exact_cc_reasoning_tokens(self) -> None:
        objs = [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "thinking", "thinking": "first"}],
                    "usage": {"output_tokens_details": {"thinking_tokens": 108}},
                },
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "thinking", "thinking": "second"}],
                    "usage": {"output_tokens_details": {"thinking_tokens": 16}},
                },
            },
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": [{"type": "thinking", "thinking": "unknown"}]},
            },
        ]
        d_th, d_thinking_tokens, *_rest = _analyze_log_chunk(objs)
        self.assertEqual((d_th, d_thinking_tokens), (3, 124))

    def test_analyze_log_chunk_counts_initial_codex_cumulative_reasoning_tokens(self) -> None:
        obj = {
            "type": "event_msg",
            "payload": {"type": "token_count", "info": {"total_token_usage": {"reasoning_output_tokens": 23}}},
        }

        _thinking, thinking_tokens, *_rest, turn_state = _analyze_log_chunk([obj])

        self.assertEqual(thinking_tokens, 23)
        self.assertEqual(turn_state.codex_reasoning_total, 23)

    def test_analyze_log_chunk_differences_codex_cumulative_reasoning_tokens(self) -> None:
        def token_count(total: int) -> dict[str, object]:
            return {
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "last_token_usage": {"reasoning_output_tokens": total},
                        "total_token_usage": {"reasoning_output_tokens": total},
                    },
                },
            }

        _thinking, thinking_tokens, *_rest, turn_state = _analyze_log_chunk(
            [token_count(130), token_count(160)],
            initial_codex_reasoning_total=100,
        )
        self.assertEqual(thinking_tokens, 60)
        self.assertEqual(turn_state.codex_reasoning_total, 160)

    def test_codex_cumulative_reasoning_fork_rebaselines_without_negative_delta(self) -> None:
        def token_count(total: int) -> dict[str, object]:
            return {
                "type": "event_msg",
                "payload": {"type": "token_count", "info": {"total_token_usage": {"reasoning_output_tokens": total}}},
            }

        _thinking, thinking_tokens, *_rest, turn_state = _analyze_log_chunk(
            [token_count(160), token_count(90), token_count(110)],
            initial_codex_reasoning_total=100,
        )
        self.assertEqual(thinking_tokens, 80)
        self.assertEqual(turn_state.codex_reasoning_total, 110)

    def test_live_meta_delta_carries_exact_pi_reasoning_tokens(self) -> None:
        _events, meta_delta, _flags, _diag = _extract_chat_events(
            [
                {
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "thinking", "thinking": "live"}],
                        "usage": {"reasoning": 1200},
                    },
                }
            ]
        )
        self.assertEqual(meta_delta["thinking"], 1)
        self.assertEqual(meta_delta["thinking_tokens"], 1200)

    def test_compute_idle_handles_pi_tool_turn_and_cc_pending_rows(self) -> None:
        with TemporaryDirectory() as td:
            pi_log = Path(td) / "pi.jsonl"
            rows = [
                {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "run"}]}},
                {"type": "message", "message": {"role": "assistant", "content": [{"type": "toolCall", "id": "t1", "name": "bash", "arguments": {"command": "sleep"}}]}},
            ]
            pi_log.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            self.assertFalse(_compute_idle_from_log(pi_log))

            cc_log = Path(td) / "cc.jsonl"
            cc_rows = [
                {"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": "run"}]}},
                {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {}}]}},
                {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}]}},
            ]
            cc_log.write_text("".join(json.dumps(r) + "\n" for r in cc_rows), encoding="utf-8")
            self.assertFalse(_compute_idle_from_log(cc_log))
