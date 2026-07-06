import gc
import json
import os
import socket
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from codoxear import broker as broker_mod
from codoxear.broker import (
    BUSY_INTERRUPT_GRACE_SECONDS,
    BUSY_QUIET_SECONDS,
    Broker,
    State,
    _apply_rollout_obj_to_state,
    _mark_busy_state_idle,
    _mark_explicit_interrupt_request,
    _maybe_detach_on_session_switch_trigger,
    _should_clear_busy_state,
    _update_busy_from_pty_text,
)
from codoxear.pi_log import pi_current_turn_state_before
from codoxear.pi_log import pi_pending_tool_call_is_duplicate
from codoxear.pi_log import pi_pending_tool_call_is_unknown


def _state() -> State:
    return State(
        codex_pid=1,
        pty_master_fd=1,
        cwd="/tmp",
        start_ts=0.0,
        codex_home=Path("/tmp"),
        sessions_dir=Path("/tmp"),
    )


class TestBrokerBusyState(unittest.TestCase):
    def test_codex_detach_trigger_detaches_rollout(self) -> None:
        st = _state()
        st.log_path = Path("/tmp/sessions/rollout-old.jsonl")
        st.session_id = "old"
        st.last_interrupt_request_ts = 10.0
        st.last_interrupted_idle_ts = 20.0
        ok = _maybe_detach_on_session_switch_trigger(
            st=st,
            tail="",
            cleaned="To continue this session, run codex resume ...\n",
            agent_backend="codex",
        )
        self.assertTrue(ok)
        self.assertIsNone(st.log_path)
        self.assertIsNone(st.session_id)
        self.assertTrue(len(st.ignored_rollout_paths) >= 1)
        self.assertEqual(st.last_interrupt_request_ts, 0.0)
        self.assertEqual(st.last_interrupted_idle_ts, 0.0)

    def test_codex_detach_trigger_matches_across_chunk_boundary(self) -> None:
        st = _state()
        st.log_path = Path("/tmp/sessions/rollout-old.jsonl")
        st.session_id = "old"
        ok = _maybe_detach_on_session_switch_trigger(
            st=st,
            tail="To continue this session, r",
            cleaned="un codex resume ...\n",
            agent_backend="codex",
        )
        self.assertTrue(ok)

    def test_pi_status_text_does_not_detach_rollout(self) -> None:
        st = _state()
        st.log_path = Path("/tmp/sessions/pi-old.jsonl")
        st.session_id = "old"
        ok = _maybe_detach_on_session_switch_trigger(
            st=st,
            tail="",
            cleaned="New session started\n",
            agent_backend="pi",
        )
        self.assertFalse(ok)
        self.assertEqual(st.log_path, Path("/tmp/sessions/pi-old.jsonl"))
        self.assertEqual(st.session_id, "old")

    def test_pi_resumed_status_text_does_not_detach_across_chunk_boundary(self) -> None:
        st = _state()
        st.log_path = Path("/tmp/sessions/pi-old.jsonl")
        st.session_id = "old"
        ok = _maybe_detach_on_session_switch_trigger(
            st=st,
            tail="Resumed sess",
            cleaned="ion abc123\n",
            agent_backend="pi",
        )
        self.assertFalse(ok)
        self.assertEqual(st.log_path, Path("/tmp/sessions/pi-old.jsonl"))
        self.assertEqual(st.session_id, "old")

    def test_detach_trigger_ignores_non_matching_backend_text(self) -> None:
        st = _state()
        st.log_path = Path("/tmp/sessions/pi-old.jsonl")
        st.session_id = "old"
        ok = _maybe_detach_on_session_switch_trigger(
            st=st,
            tail="",
            cleaned="New session started\n",
            agent_backend="codex",
        )
        self.assertFalse(ok)
        self.assertEqual(st.log_path, Path("/tmp/sessions/pi-old.jsonl"))
        self.assertEqual(st.session_id, "old")

    def test_user_message_starts_turn_and_resets_pending_calls(self) -> None:
        st = _state()
        st.pending_calls.add("old")
        _apply_rollout_obj_to_state(
            st,
            {"type": "event_msg", "payload": {"type": "user_message", "message": "hello"}},
            now_ts=10.0,
        )
        self.assertTrue(st.busy)
        self.assertEqual(st.pending_calls, set())
        self.assertEqual(st.last_turn_activity_ts, 10.0)

    def test_agent_progress_message_does_not_clear_busy(self) -> None:
        st = _state()
        _apply_rollout_obj_to_state(
            st,
            {"type": "event_msg", "payload": {"type": "user_message", "message": "hello"}},
            now_ts=10.0,
        )
        _apply_rollout_obj_to_state(
            st,
            {"type": "event_msg", "payload": {"type": "agent_message", "message": "working"}},
            now_ts=11.0,
        )
        self.assertTrue(st.busy)
        self.assertEqual(st.last_turn_activity_ts, 11.0)

    def test_call_pair_and_quiet_window_control_idle_transition(self) -> None:
        st = _state()
        _apply_rollout_obj_to_state(
            st,
            {"type": "event_msg", "payload": {"type": "user_message", "message": "hello"}},
            now_ts=10.0,
        )
        _apply_rollout_obj_to_state(
            st,
            {"type": "response_item", "payload": {"type": "function_call", "call_id": "call-1"}},
            now_ts=11.0,
        )
        _apply_rollout_obj_to_state(
            st,
            {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "call-1"}},
            now_ts=12.0,
        )
        _apply_rollout_obj_to_state(
            st,
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "done"}],
                },
            },
            now_ts=13.0,
        )
        self.assertEqual(st.pending_calls, set())
        self.assertTrue(st.busy)
        self.assertFalse(_should_clear_busy_state(st, now_ts=13.0 + max(BUSY_QUIET_SECONDS - 0.05, 0.0)))
        self.assertTrue(_should_clear_busy_state(st, now_ts=13.0 + BUSY_QUIET_SECONDS + 0.05))

    def test_long_silent_turn_without_assistant_candidate_stays_busy(self) -> None:
        st = _state()
        _apply_rollout_obj_to_state(
            st,
            {"type": "event_msg", "payload": {"type": "user_message", "message": "hello"}},
            now_ts=10.0,
        )
        self.assertFalse(_should_clear_busy_state(st, now_ts=10.0 + BUSY_QUIET_SECONDS + 60.0))

    def test_explicit_interrupt_request_can_clear_no_candidate_turn_after_quiet(self) -> None:
        st = _state()
        _apply_rollout_obj_to_state(
            st,
            {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "hello"}]}},
            now_ts=10.0,
        )
        self.assertFalse(st.turn_has_completion_candidate)
        self.assertFalse(_should_clear_busy_state(st, now_ts=100.0))
        _mark_explicit_interrupt_request(st, now_ts=20.0)
        self.assertFalse(_should_clear_busy_state(st, now_ts=20.0 + max(BUSY_INTERRUPT_GRACE_SECONDS - 0.05, 0.0)))
        self.assertTrue(
            _should_clear_busy_state(
                st,
                now_ts=20.0 + max(BUSY_INTERRUPT_GRACE_SECONDS, BUSY_QUIET_SECONDS) + 0.05,
            )
        )

    def test_explicit_interrupt_request_does_not_clear_pending_calls(self) -> None:
        st = _state()
        st.busy = True
        st.turn_open = True
        st.turn_has_completion_candidate = False
        st.pending_calls.add("call-1")
        st.last_turn_activity_ts = 10.0
        _mark_explicit_interrupt_request(st, now_ts=20.0)
        self.assertFalse(_should_clear_busy_state(st, now_ts=100.0))

    def test_pi_tool_call_blocks_explicit_interrupt_idle_until_result(self) -> None:
        st = _state()
        _apply_rollout_obj_to_state(
            st,
            {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "run"}]}},
            now_ts=10.0,
        )
        _apply_rollout_obj_to_state(
            st,
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": ""},
                        {"type": "toolCall", "id": "tool-1", "name": "bash", "arguments": {"command": "sleep 30"}},
                    ],
                    "stopReason": "toolUse",
                },
            },
            now_ts=11.0,
        )
        self.assertEqual(st.pending_calls, {"tool-1"})
        _mark_explicit_interrupt_request(st, now_ts=20.0)
        self.assertFalse(_should_clear_busy_state(st, now_ts=100.0))

        _apply_rollout_obj_to_state(
            st,
            {
                "type": "message",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "tool-1",
                    "toolName": "bash",
                    "content": [{"type": "text", "text": "interrupted"}],
                    "isError": True,
                },
            },
            now_ts=30.0,
        )
        self.assertEqual(st.pending_calls, set())
        self.assertTrue(
            _should_clear_busy_state(
                st,
                now_ts=30.0 + max(BUSY_INTERRUPT_GRACE_SECONDS, BUSY_QUIET_SECONDS) + 0.05,
            )
        )

    def test_pi_duplicate_tool_call_ids_fail_closed_after_one_result(self) -> None:
        st = _state()
        _apply_rollout_obj_to_state(
            st,
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "toolCall", "id": "tool-1", "name": "bash", "arguments": {"command": "pwd"}},
                        {"type": "toolCall", "id": "tool-1", "name": "bash", "arguments": {"command": "date"}},
                    ],
                    "stopReason": "toolUse",
                },
            },
            now_ts=10.0,
        )
        self.assertIn("tool-1", st.pending_calls)
        self.assertEqual(len(st.pending_calls), 2)
        self.assertTrue(any(pi_pending_tool_call_is_duplicate(item) for item in st.pending_calls))
        result = {
            "type": "message",
            "message": {
                "role": "toolResult",
                "toolCallId": "tool-1",
                "toolName": "bash",
                "content": [{"type": "text", "text": "done"}],
                "isError": False,
            },
        }
        _apply_rollout_obj_to_state(st, result, now_ts=20.0)
        self.assertEqual(len(st.pending_calls), 1)
        self.assertTrue(pi_pending_tool_call_is_duplicate(next(iter(st.pending_calls))))
        _mark_explicit_interrupt_request(st, now_ts=30.0)
        self.assertFalse(_should_clear_busy_state(st, now_ts=100.0))

        _apply_rollout_obj_to_state(st, result, now_ts=40.0)
        self.assertEqual(st.pending_calls, set())
        self.assertTrue(_should_clear_busy_state(st, now_ts=40.0 + max(BUSY_INTERRUPT_GRACE_SECONDS, BUSY_QUIET_SECONDS) + 0.05))

    def test_pi_duplicate_tool_call_id_across_rows_fails_closed_after_one_result(self) -> None:
        st = _state()
        for command, ts in [("pwd", 10.0), ("date", 11.0)]:
            _apply_rollout_obj_to_state(
                st,
                {
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "toolCall", "id": "tool-1", "name": "bash", "arguments": {"command": command}}],
                        "stopReason": "toolUse",
                    },
                },
                now_ts=ts,
            )
        self.assertIn("tool-1", st.pending_calls)
        self.assertEqual(len(st.pending_calls), 2)
        self.assertTrue(any(pi_pending_tool_call_is_duplicate(item) for item in st.pending_calls))
        result = {
            "type": "message",
            "message": {
                "role": "toolResult",
                "toolCallId": "tool-1",
                "toolName": "bash",
                "content": [{"type": "text", "text": "done"}],
                "isError": False,
            },
        }
        _apply_rollout_obj_to_state(st, result, now_ts=20.0)
        self.assertEqual(len(st.pending_calls), 1)
        self.assertTrue(pi_pending_tool_call_is_duplicate(next(iter(st.pending_calls))))
        _mark_explicit_interrupt_request(st, now_ts=30.0)
        self.assertFalse(_should_clear_busy_state(st, now_ts=100.0))

        _apply_rollout_obj_to_state(st, result, now_ts=40.0)
        self.assertEqual(st.pending_calls, set())
        self.assertTrue(_should_clear_busy_state(st, now_ts=40.0 + max(BUSY_INTERRUPT_GRACE_SECONDS, BUSY_QUIET_SECONDS) + 0.05))

    def test_pi_whitespace_string_tool_id_clears_with_exact_result_id(self) -> None:
        st = _state()
        tool_id = " "
        _apply_rollout_obj_to_state(
            st,
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "toolCall", "id": tool_id, "name": "bash", "arguments": {"command": "pwd"}}],
                    "stopReason": "toolUse",
                },
            },
            now_ts=10.0,
        )
        self.assertEqual(st.pending_calls, {tool_id})
        _apply_rollout_obj_to_state(
            st,
            {
                "type": "message",
                "message": {
                    "role": "toolResult",
                    "toolCallId": tool_id,
                    "toolName": "bash",
                    "content": [{"type": "text", "text": "done"}],
                    "isError": False,
                },
            },
            now_ts=20.0,
        )
        self.assertEqual(st.pending_calls, set())

    def test_pi_real_tool_id_matching_internal_prefix_is_not_duplicate_sentinel(self) -> None:
        st = _state()
        real_id = "__pi_duplicate_tool_call__:foo:bar"
        _apply_rollout_obj_to_state(
            st,
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "toolCall", "id": real_id, "name": "bash", "arguments": {"command": "pwd"}}],
                    "stopReason": "toolUse",
                },
            },
            now_ts=10.0,
        )
        self.assertEqual(st.pending_calls, {real_id})
        _apply_rollout_obj_to_state(
            st,
            {
                "type": "message",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "foo",
                    "toolName": "bash",
                    "content": [{"type": "text", "text": "wrong result"}],
                    "isError": False,
                },
            },
            now_ts=20.0,
        )
        self.assertEqual(st.pending_calls, {real_id})
        _apply_rollout_obj_to_state(
            st,
            {
                "type": "message",
                "message": {
                    "role": "toolResult",
                    "toolCallId": real_id,
                    "toolName": "bash",
                    "content": [{"type": "text", "text": "right result"}],
                    "isError": False,
                },
            },
            now_ts=30.0,
        )
        self.assertEqual(st.pending_calls, set())

    def test_pi_current_turn_seed_preserves_whitespace_string_tool_id(self) -> None:
        with TemporaryDirectory() as td:
            log_path = Path(td) / "session.jsonl"
            rows = [
                {"type": "session", "id": "sid"},
                {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "run"}]}},
                {
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "toolCall", "id": " ", "name": "bash", "arguments": {"command": "pwd"}}],
                        "stopReason": "toolUse",
                    },
                },
                {
                    "type": "message",
                    "message": {
                        "role": "toolResult",
                        "toolCallId": " ",
                        "toolName": "bash",
                        "content": [{"type": "text", "text": "pwd"}],
                        "isError": False,
                    },
                },
            ]
            log_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            pending, idle = pi_current_turn_state_before(log_path, log_path.stat().st_size)

        self.assertEqual(pending, set())
        self.assertIs(idle, False)

    def test_pi_current_turn_seed_keeps_prebind_tool_call_pending(self) -> None:
        with TemporaryDirectory() as td:
            log_path = Path(td) / "session.jsonl"
            rows = [
                {"type": "session", "id": "sid"},
                {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "run"}]}},
                {
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "toolCall", "id": "tool-1", "name": "bash", "arguments": {"command": "sleep 30"}}],
                        "stopReason": "toolUse",
                    },
                },
            ]
            log_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            pending, idle = pi_current_turn_state_before(log_path, log_path.stat().st_size)

        self.assertEqual(pending, {"tool-1"})
        self.assertIs(idle, False)

    def test_pi_current_turn_seed_preserves_duplicate_multiplicity_until_results(self) -> None:
        with TemporaryDirectory() as td:
            log_path = Path(td) / "session.jsonl"
            duplicate_rows = [
                {"type": "session", "id": "sid"},
                {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "run"}]}},
                {
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "toolCall", "id": "tool-1", "name": "bash", "arguments": {"command": "pwd"}},
                            {"type": "toolCall", "id": "tool-1", "name": "bash", "arguments": {"command": "date"}},
                        ],
                        "stopReason": "toolUse",
                    },
                },
                {
                    "type": "message",
                    "message": {
                        "role": "toolResult",
                        "toolCallId": "tool-1",
                        "toolName": "bash",
                        "content": [{"type": "text", "text": "pwd"}],
                        "isError": False,
                    },
                },
            ]
            log_path.write_text("".join(json.dumps(row) + "\n" for row in duplicate_rows), encoding="utf-8")
            pending, idle = pi_current_turn_state_before(log_path, log_path.stat().st_size)
            self.assertEqual(len(pending), 1)
            self.assertTrue(pi_pending_tool_call_is_duplicate(next(iter(pending))))
            self.assertIs(idle, False)

            duplicate_rows.append(
                {
                    "type": "message",
                    "message": {
                        "role": "toolResult",
                        "toolCallId": "tool-1",
                        "toolName": "bash",
                        "content": [{"type": "text", "text": "date"}],
                        "isError": False,
                    },
                }
            )
            log_path.write_text("".join(json.dumps(row) + "\n" for row in duplicate_rows), encoding="utf-8")
            pending, idle = pi_current_turn_state_before(log_path, log_path.stat().st_size)

        self.assertEqual(pending, set())
        self.assertIs(idle, False)

    def test_pi_current_turn_seed_keeps_idless_tool_call_unknown(self) -> None:
        with TemporaryDirectory() as td:
            log_path = Path(td) / "session.jsonl"
            rows = [
                {"type": "session", "id": "sid"},
                {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "run"}]}},
                {
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "toolCall", "name": "bash", "arguments": {"command": "pwd"}}],
                        "stopReason": "toolUse",
                    },
                },
                {
                    "type": "message",
                    "message": {
                        "role": "toolResult",
                        "toolName": "bash",
                        "content": [{"type": "text", "text": "pwd"}],
                        "isError": False,
                    },
                },
            ]
            log_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            pending, idle = pi_current_turn_state_before(log_path, log_path.stat().st_size)

        self.assertEqual(len(pending), 1)
        self.assertTrue(pi_pending_tool_call_is_unknown(next(iter(pending))))
        self.assertIs(idle, False)

    def test_interrupted_idle_marker_includes_turns_with_assistant_candidate(self) -> None:
        st = _state()
        _apply_rollout_obj_to_state(
            st,
            {"type": "event_msg", "payload": {"type": "user_message", "message": "hello"}},
            now_ts=10.0,
        )
        _apply_rollout_obj_to_state(
            st,
            {"type": "event_msg", "payload": {"type": "agent_message", "message": "partial"}},
            now_ts=11.0,
        )
        self.assertTrue(st.turn_has_completion_candidate)
        _mark_explicit_interrupt_request(st, now_ts=20.0)
        self.assertTrue(_should_clear_busy_state(st, now_ts=20.0 + max(BUSY_INTERRUPT_GRACE_SECONDS, BUSY_QUIET_SECONDS) + 0.05))
        _mark_busy_state_idle(st, now_ts=25.0)
        self.assertEqual(st.last_interrupted_idle_ts, 25.0)

    def test_pi_final_message_with_thinking_clears_busy(self) -> None:
        st = _state()
        _apply_rollout_obj_to_state(
            st,
            {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "run"}]}},
            now_ts=10.0,
        )
        _apply_rollout_obj_to_state(
            st,
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": ""},
                        {"type": "text", "text": "done", "textSignature": '{"v":1,"phase":"final_answer"}'},
                    ],
                    "stopReason": "stop",
                },
            },
            now_ts=11.0,
        )
        self.assertFalse(st.busy)
        self.assertFalse(st.turn_open)
        self.assertEqual(st.pending_calls, set())

    def test_pi_stop_empty_closes_busy_and_current_turn(self) -> None:
        st = _state()
        _apply_rollout_obj_to_state(
            st,
            {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "run"}]}},
            now_ts=10.0,
        )
        _apply_rollout_obj_to_state(
            st,
            {"type": "message", "message": {"role": "assistant", "content": [], "stopReason": "stop"}},
            now_ts=11.0,
        )
        self.assertFalse(st.busy)
        self.assertFalse(st.turn_open)
        self.assertEqual(st.pending_calls, set())

        with TemporaryDirectory() as td:
            log_path = Path(td) / "session.jsonl"
            rows = [
                {"type": "session", "id": "sid"},
                {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "run"}]}},
                {"type": "message", "message": {"role": "assistant", "content": [], "stopReason": "stop"}},
            ]
            log_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            pending, idle = pi_current_turn_state_before(log_path, log_path.stat().st_size)
        self.assertEqual(pending, set())
        self.assertIs(idle, True)

    def test_pi_end_turn_empty_closes_busy_and_current_turn(self) -> None:
        st = _state()
        _apply_rollout_obj_to_state(
            st,
            {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "run"}]}},
            now_ts=10.0,
        )
        _apply_rollout_obj_to_state(
            st,
            {"type": "message", "message": {"role": "assistant", "content": [], "stopReason": "end_turn"}},
            now_ts=11.0,
        )
        self.assertFalse(st.busy)
        self.assertFalse(st.turn_open)

        with TemporaryDirectory() as td:
            log_path = Path(td) / "session.jsonl"
            rows = [
                {"type": "session", "id": "sid"},
                {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "run"}]}},
                {"type": "message", "message": {"role": "assistant", "content": [], "stopReason": "end_turn"}},
            ]
            log_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            pending, idle = pi_current_turn_state_before(log_path, log_path.stat().st_size)
        self.assertEqual(pending, set())
        self.assertIs(idle, True)

    def test_pi_stop_thinking_only_closes_busy_and_current_turn(self) -> None:
        st = _state()
        _apply_rollout_obj_to_state(
            st,
            {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "run"}]}},
            now_ts=10.0,
        )
        _apply_rollout_obj_to_state(
            st,
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "thinking", "thinking": "internal"}],
                    "stopReason": "stop",
                },
            },
            now_ts=11.0,
        )
        self.assertFalse(st.busy)
        self.assertFalse(st.turn_open)

        with TemporaryDirectory() as td:
            log_path = Path(td) / "session.jsonl"
            rows = [
                {"type": "session", "id": "sid"},
                {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "run"}]}},
                {
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "thinking", "thinking": "internal"}],
                        "stopReason": "stop",
                    },
                },
            ]
            log_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            pending, idle = pi_current_turn_state_before(log_path, log_path.stat().st_size)
        self.assertEqual(pending, set())
        self.assertIs(idle, True)

    def test_pi_nonterminal_thinking_keeps_busy_and_current_turn(self) -> None:
        st = _state()
        _apply_rollout_obj_to_state(
            st,
            {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "run"}]}},
            now_ts=10.0,
        )
        _apply_rollout_obj_to_state(
            st,
            {"type": "message", "message": {"role": "assistant", "content": [{"type": "thinking", "thinking": "internal"}]}},
            now_ts=11.0,
        )
        self.assertTrue(st.busy)
        self.assertTrue(st.turn_open)

        with TemporaryDirectory() as td:
            log_path = Path(td) / "session.jsonl"
            rows = [
                {"type": "session", "id": "sid"},
                {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "run"}]}},
                {"type": "message", "message": {"role": "assistant", "content": [{"type": "thinking", "thinking": "internal"}]}},
            ]
            log_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            pending, idle = pi_current_turn_state_before(log_path, log_path.stat().st_size)
        self.assertEqual(pending, set())
        self.assertIs(idle, False)

    def test_pi_tool_use_tool_call_keeps_busy_and_current_turn(self) -> None:
        st = _state()
        _apply_rollout_obj_to_state(
            st,
            {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "run"}]}},
            now_ts=10.0,
        )
        _apply_rollout_obj_to_state(
            st,
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "toolCall", "id": "tool-1", "name": "bash", "arguments": {}}],
                    "stopReason": "toolUse",
                },
            },
            now_ts=11.0,
        )
        self.assertTrue(st.busy)
        self.assertTrue(st.turn_open)
        self.assertEqual(st.pending_calls, {"tool-1"})

        with TemporaryDirectory() as td:
            log_path = Path(td) / "session.jsonl"
            rows = [
                {"type": "session", "id": "sid"},
                {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "run"}]}},
                {
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "toolCall", "id": "tool-1", "name": "bash", "arguments": {}}],
                        "stopReason": "toolUse",
                    },
                },
            ]
            log_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            pending, idle = pi_current_turn_state_before(log_path, log_path.stat().st_size)
        self.assertEqual(pending, {"tool-1"})
        self.assertIs(idle, False)

    def test_pi_final_message_closes_stale_unknown_pending_call(self) -> None:
        st = _state()
        _apply_rollout_obj_to_state(
            st,
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "toolCall", "name": "bash", "arguments": {"command": "pwd"}}],
                    "stopReason": "toolUse",
                },
            },
            now_ts=10.0,
        )
        self.assertTrue(st.pending_calls)
        _apply_rollout_obj_to_state(
            st,
            {
                "type": "message",
                "message": {
                    "role": "toolResult",
                    "toolName": "bash",
                    "content": [{"type": "text", "text": "/tmp"}],
                    "isError": False,
                },
            },
            now_ts=11.0,
        )
        self.assertTrue(st.pending_calls)
        _apply_rollout_obj_to_state(
            st,
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "done"}],
                    "stopReason": "stop",
                },
            },
            now_ts=12.0,
        )
        self.assertFalse(st.busy)
        self.assertFalse(st.turn_open)
        self.assertEqual(st.pending_calls, set())

    def test_pi_length_tool_call_stays_busy_until_result(self) -> None:
        st = _state()
        _apply_rollout_obj_to_state(
            st,
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
            now_ts=11.0,
        )
        self.assertEqual(st.pending_calls, {"tool-1"})
        self.assertTrue(st.busy)
        self.assertTrue(st.turn_open)
        _mark_explicit_interrupt_request(st, now_ts=20.0)
        self.assertFalse(_should_clear_busy_state(st, now_ts=100.0))

        _apply_rollout_obj_to_state(
            st,
            {
                "type": "message",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "tool-1",
                    "toolName": "bash",
                    "content": [{"type": "text", "text": "/tmp"}],
                    "isError": False,
                },
            },
            now_ts=30.0,
        )
        self.assertEqual(st.pending_calls, set())

    def test_pi_tool_call_without_id_fails_busy_closed_after_idless_result(self) -> None:
        st = _state()
        _apply_rollout_obj_to_state(
            st,
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "toolCall", "name": "bash", "arguments": {"command": "sleep 30"}}],
                    "stopReason": "toolUse",
                },
            },
            now_ts=11.0,
        )
        self.assertEqual(len(st.pending_calls), 1)
        self.assertTrue(pi_pending_tool_call_is_unknown(next(iter(st.pending_calls))))
        _mark_explicit_interrupt_request(st, now_ts=20.0)
        _apply_rollout_obj_to_state(
            st,
            {
                "type": "message",
                "message": {
                    "role": "toolResult",
                    "toolName": "bash",
                    "content": [{"type": "text", "text": "done"}],
                    "isError": False,
                },
            },
            now_ts=30.0,
        )
        self.assertEqual(len(st.pending_calls), 1)
        self.assertFalse(_should_clear_busy_state(st, now_ts=100.0))

    def test_state_response_reports_interrupted_idle_only_after_idle_clear(self) -> None:
        client, server = socket.socketpair()
        try:
            st = _state()
            st.busy = False
            st.last_interrupted_idle_ts = 30.0
            broker = Broker.__new__(Broker)
            broker.state = st
            broker._lock = threading.Lock()
            broker._stop = threading.Event()

            client.sendall(b'{"cmd":"state"}\n')
            broker._handle_conn(server)
            resp = json.loads(client.recv(4096).decode("utf-8"))

            self.assertIs(resp["busy"], False)
            self.assertIs(resp["interrupted_idle"], True)
        finally:
            try:
                client.close()
            except OSError:
                pass

    def test_keys_interrupt_request_marks_state_after_writing_escape(self) -> None:
        read_fd, write_fd = os.pipe()
        os.set_blocking(read_fd, False)
        client, server = socket.socketpair()
        try:
            st = _state()
            st.pty_master_fd = write_fd
            broker = Broker.__new__(Broker)
            broker.state = st
            broker._lock = threading.Lock()
            broker._stop = threading.Event()

            client.sendall(b'{"cmd":"keys","seq":"\\\\x1b","interrupt":true}\n')
            broker._handle_conn(server)
            resp = json.loads(client.recv(4096).decode("utf-8"))

            self.assertTrue(resp["ok"])
            self.assertEqual(os.read(read_fd, 1), b"\x1b")
            self.assertGreater(st.last_interrupt_request_ts, 0.0)
        finally:
            for obj in (client,):
                try:
                    obj.close()
                except OSError:
                    pass
            for fd in (read_fd, write_fd):
                try:
                    os.close(fd)
                except OSError:
                    pass

    def test_new_user_message_resets_explicit_interrupt_request(self) -> None:
        st = _state()
        st.last_interrupt_request_ts = 20.0
        st.last_interrupted_idle_ts = 25.0
        _apply_rollout_obj_to_state(
            st,
            {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "next"}]}},
            now_ts=30.0,
        )
        self.assertEqual(st.last_interrupt_request_ts, 0.0)
        self.assertEqual(st.last_interrupted_idle_ts, 0.0)
        self.assertFalse(_should_clear_busy_state(st, now_ts=100.0))

    def test_tool_activity_after_assistant_resets_completion_candidate(self) -> None:
        st = _state()
        _apply_rollout_obj_to_state(
            st,
            {"type": "event_msg", "payload": {"type": "user_message", "message": "hello"}},
            now_ts=10.0,
        )
        _apply_rollout_obj_to_state(
            st,
            {"type": "event_msg", "payload": {"type": "agent_message", "message": "working"}},
            now_ts=11.0,
        )
        _apply_rollout_obj_to_state(
            st,
            {"type": "response_item", "payload": {"type": "function_call", "call_id": "call-1"}},
            now_ts=12.0,
        )
        _apply_rollout_obj_to_state(
            st,
            {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "call-1"}},
            now_ts=13.0,
        )
        self.assertFalse(_should_clear_busy_state(st, now_ts=13.0 + BUSY_QUIET_SECONDS + 5.0))

    def test_web_search_activity_after_assistant_resets_completion_candidate(self) -> None:
        st = _state()
        _apply_rollout_obj_to_state(
            st,
            {"type": "event_msg", "payload": {"type": "user_message", "message": "hello"}},
            now_ts=10.0,
        )
        _apply_rollout_obj_to_state(
            st,
            {"type": "event_msg", "payload": {"type": "agent_message", "message": "working"}},
            now_ts=11.0,
        )
        _apply_rollout_obj_to_state(
            st,
            {"type": "response_item", "payload": {"type": "web_search_call", "status": "completed"}},
            now_ts=12.0,
        )
        self.assertFalse(_should_clear_busy_state(st, now_ts=12.0 + BUSY_QUIET_SECONDS + 5.0))

    def test_local_shell_activity_after_assistant_resets_completion_candidate(self) -> None:
        st = _state()
        _apply_rollout_obj_to_state(
            st,
            {"type": "event_msg", "payload": {"type": "user_message", "message": "hello"}},
            now_ts=10.0,
        )
        _apply_rollout_obj_to_state(
            st,
            {"type": "event_msg", "payload": {"type": "agent_message", "message": "working"}},
            now_ts=11.0,
        )
        _apply_rollout_obj_to_state(
            st,
            {"type": "response_item", "payload": {"type": "local_shell_call", "status": "completed"}},
            now_ts=12.0,
        )
        self.assertFalse(_should_clear_busy_state(st, now_ts=12.0 + BUSY_QUIET_SECONDS + 5.0))

    def test_turn_aborted_clears_busy_and_pending_calls(self) -> None:
        st = _state()
        st.busy = True
        st.pending_calls.add("call-1")
        st.last_turn_activity_ts = 10.0
        _apply_rollout_obj_to_state(
            st,
            {"type": "event_msg", "payload": {"type": "turn_aborted"}},
            now_ts=11.0,
        )
        self.assertFalse(st.busy)
        self.assertEqual(st.pending_calls, set())
        self.assertEqual(st.last_turn_activity_ts, 0.0)

    def test_task_complete_clears_busy_and_pending_calls(self) -> None:
        st = _state()
        st.busy = True
        st.turn_open = True
        st.turn_has_completion_candidate = False
        st.pending_calls.add("call-1")
        st.last_turn_activity_ts = 10.0
        _apply_rollout_obj_to_state(
            st,
            {"type": "event_msg", "payload": {"type": "task_complete"}},
            now_ts=11.0,
        )
        self.assertFalse(st.busy)
        self.assertFalse(st.turn_open)
        self.assertFalse(st.turn_has_completion_candidate)
        self.assertEqual(st.pending_calls, set())
        self.assertEqual(st.last_turn_activity_ts, 0.0)

    def test_turn_complete_clears_busy_and_pending_calls(self) -> None:
        st = _state()
        st.busy = True
        st.turn_open = True
        st.turn_has_completion_candidate = True
        st.pending_calls.add("call-1")
        st.last_turn_activity_ts = 10.0
        _apply_rollout_obj_to_state(
            st,
            {"type": "event_msg", "payload": {"type": "turn_complete"}},
            now_ts=11.0,
        )
        self.assertFalse(st.busy)
        self.assertFalse(st.turn_open)
        self.assertFalse(st.turn_has_completion_candidate)
        self.assertEqual(st.pending_calls, set())
        self.assertEqual(st.last_turn_activity_ts, 0.0)

    def test_codex_error_clears_busy_and_pending_calls(self) -> None:
        st = _state()
        st.busy = True
        st.turn_open = True
        st.turn_has_completion_candidate = False
        st.pending_calls.add("call-1")
        st.last_turn_activity_ts = 10.0
        _apply_rollout_obj_to_state(
            st,
            {
                "type": "event_msg",
                "payload": {
                    "type": "error",
                    "message": "unexpected status 400 Bad Request",
                    "codex_error_info": "bad_request",
                },
            },
            now_ts=11.0,
        )
        self.assertFalse(st.busy)
        self.assertFalse(st.turn_open)
        self.assertFalse(st.turn_has_completion_candidate)
        self.assertEqual(st.pending_calls, set())
        self.assertEqual(st.last_turn_activity_ts, 0.0)

    def test_pi_error_message_clears_busy_and_pending_calls(self) -> None:
        st = _state()
        st.busy = True
        st.turn_open = True
        st.turn_has_completion_candidate = False
        st.pending_calls.add("call-1")
        st.last_turn_activity_ts = 10.0
        _apply_rollout_obj_to_state(
            st,
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [],
                    "stopReason": "error",
                    "errorMessage": "401 Invalid API key",
                },
            },
            now_ts=11.0,
        )
        self.assertFalse(st.busy)
        self.assertFalse(st.turn_open)
        self.assertFalse(st.turn_has_completion_candidate)
        self.assertEqual(st.pending_calls, set())
        self.assertEqual(st.last_turn_activity_ts, 0.0)

    def test_pi_aborted_message_clears_busy_and_pending_calls(self) -> None:
        st = _state()
        st.busy = True
        st.turn_open = True
        st.turn_has_completion_candidate = False
        st.pending_calls.add("call-1")
        st.last_turn_activity_ts = 10.0
        _apply_rollout_obj_to_state(
            st,
            {"type": "message", "message": {"role": "assistant", "content": [], "stopReason": "aborted"}},
            now_ts=11.0,
        )
        self.assertFalse(st.busy)
        self.assertFalse(st.turn_open)
        self.assertFalse(st.turn_has_completion_candidate)
        self.assertEqual(st.pending_calls, set())
        self.assertEqual(st.last_turn_activity_ts, 0.0)

    def test_thread_rollback_error_does_not_clear_busy(self) -> None:
        st = _state()
        st.busy = True
        st.turn_open = True
        st.pending_calls.add("call-1")
        st.last_turn_activity_ts = 10.0
        _apply_rollout_obj_to_state(
            st,
            {
                "type": "event_msg",
                "payload": {
                    "type": "error",
                    "message": "rollback failed",
                    "codex_error_info": "thread_rollback_failed",
                },
            },
            now_ts=11.0,
        )
        self.assertTrue(st.busy)
        self.assertTrue(st.turn_open)
        self.assertEqual(st.pending_calls, {"call-1"})
        self.assertEqual(st.last_turn_activity_ts, 10.0)

    def test_reasoning_item_can_mark_busy_without_user_message(self) -> None:
        st = _state()
        _apply_rollout_obj_to_state(
            st,
            {"type": "response_item", "payload": {"type": "reasoning"}},
            now_ts=15.0,
        )
        self.assertTrue(st.busy)
        self.assertEqual(st.last_turn_activity_ts, 15.0)

    def test_reasoning_reopens_turn_after_idle_clear(self) -> None:
        st = _state()
        _apply_rollout_obj_to_state(
            st,
            {"type": "event_msg", "payload": {"type": "user_message", "message": "hello"}},
            now_ts=10.0,
        )
        _apply_rollout_obj_to_state(
            st,
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "done"}],
                },
            },
            now_ts=11.0,
        )
        self.assertTrue(_should_clear_busy_state(st, now_ts=11.0 + BUSY_QUIET_SECONDS + 0.05))

        st.busy = False
        st.turn_open = False
        st.turn_has_completion_candidate = False
        st.last_turn_activity_ts = 0.0
        st.last_interrupt_hint_ts = 0.0

        _apply_rollout_obj_to_state(
            st,
            {"type": "event_msg", "payload": {"type": "agent_reasoning"}},
            now_ts=15.0,
        )
        self.assertTrue(st.busy)
        self.assertTrue(st.turn_open)
        self.assertFalse(st.turn_has_completion_candidate)

    def test_log_switch_resets_interrupted_idle_marker(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            old_log = root / "old.jsonl"
            new_log = root / "new.jsonl"
            old_log.write_text('{"type":"user","sessionId":"old","cwd":"/tmp","message":{"role":"user","content":"old"}}\n', encoding="utf-8")
            rows = [
                {"type": "user", "sessionId": "cc-session", "cwd": str(root), "message": {"role": "user", "content": "hi"}},
                {
                    "type": "assistant",
                    "sessionId": "cc-session",
                    "message": {"role": "assistant", "content": [{"type": "text", "text": "hello"}], "stop_reason": "end_turn"},
                },
            ]
            new_log.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            st = State(
                codex_pid=1,
                pty_master_fd=1,
                cwd=str(root),
                start_ts=0.0,
                codex_home=root,
                sessions_dir=root,
                sock_path=root / "broker.sock",
                log_path=old_log,
                last_rollout_path=old_log,
                last_interrupt_request_ts=10.0,
                last_interrupted_idle_ts=20.0,
            )
            broker = Broker.__new__(Broker)
            broker.sessions_dir = root
            broker.state = st
            broker._lock = threading.Lock()
            broker._write_meta = lambda: None  # type: ignore[method-assign]

            with patch.object(broker_mod, "AGENT_BACKEND", "cc"):
                broker._maybe_register_or_switch_rollout(log_path=new_log)

            self.assertEqual(st.log_path, new_log)
            self.assertEqual(st.last_interrupt_request_ts, 0.0)
            self.assertEqual(st.last_interrupted_idle_ts, 0.0)

    def test_pi_log_bind_replaces_stale_pending_calls(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            log_path = root / "pi-new.jsonl"
            rows = [
                {"type": "session", "id": "pi-session", "cwd": str(root)},
                {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "run"}]}},
                {
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "thinking", "thinking": "working"}],
                        "stopReason": None,
                    },
                },
            ]
            log_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            st = State(
                codex_pid=1,
                pty_master_fd=1,
                cwd=str(root),
                start_ts=0.0,
                codex_home=root,
                sessions_dir=root,
                sock_path=root / "broker.sock",
                log_path=root / "pi-old.jsonl",
                last_rollout_path=root / "pi-old.jsonl",
                pending_calls={"old"},
                busy=True,
                turn_open=True,
            )
            broker = Broker.__new__(Broker)
            broker.sessions_dir = root
            broker.state = st
            broker._lock = threading.Lock()
            broker._write_meta = lambda: None  # type: ignore[method-assign]

            with patch.object(broker_mod, "AGENT_BACKEND", "pi"):
                broker._maybe_register_or_switch_rollout(log_path=log_path)

            self.assertEqual(st.log_path, log_path)
            self.assertEqual(st.pending_calls, set())
            self.assertTrue(st.busy)
            self.assertTrue(st.turn_open)

    def test_pi_log_bind_seeds_pending_tool_state_before_log_offset_advances(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            log_path = root / "pi.jsonl"
            rows = [
                {"type": "session", "id": "pi-session", "cwd": str(root)},
                {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "run"}]}},
                {
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "toolCall", "id": "tool-1", "name": "bash", "arguments": {"command": "sleep 30"}}],
                        "stopReason": "toolUse",
                    },
                },
            ]
            log_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            st = State(codex_pid=1, pty_master_fd=1, cwd=str(root), start_ts=0.0, codex_home=root, sessions_dir=root, sock_path=root / "broker.sock")
            broker = Broker.__new__(Broker)
            broker.sessions_dir = root
            broker.state = st
            broker._lock = threading.Lock()
            broker._write_meta = lambda: None  # type: ignore[method-assign]

            with patch.object(broker_mod, "AGENT_BACKEND", "pi"):
                broker._maybe_register_or_switch_rollout(log_path=log_path)

            self.assertEqual(st.log_path, log_path)
            self.assertEqual(st.log_off, log_path.stat().st_size)
            self.assertEqual(st.pending_calls, {"tool-1"})
            self.assertTrue(st.busy)
            self.assertTrue(st.turn_open)
            _mark_explicit_interrupt_request(st, now_ts=20.0)
            self.assertFalse(_should_clear_busy_state(st, now_ts=100.0))

    def test_log_watcher_discards_batch_after_log_rebind(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            old_log = root / "old.jsonl"
            new_log = root / "new.jsonl"
            old_log.write_text("", encoding="utf-8")
            new_log.write_text("", encoding="utf-8")
            st = State(
                codex_pid=1,
                pty_master_fd=1,
                cwd=str(root),
                start_ts=0.0,
                codex_home=root,
                sessions_dir=root,
                sock_path=root / "broker.sock",
                log_path=old_log,
                log_off=0,
                pending_calls={"new"},
            )
            broker = Broker.__new__(Broker)
            broker.state = st
            broker._lock = threading.Lock()
            broker._stop = threading.Event()
            calls = 0
            old_obj = {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "toolCall", "id": "old", "name": "bash", "arguments": {"command": "pwd"}}],
                    "stopReason": "toolUse",
                },
            }

            def fake_read(path: Path, offset: int, max_bytes: int = 256 * 1024):
                nonlocal calls
                calls += 1
                if calls == 1:
                    self.assertEqual(path, old_log)
                    self.assertEqual(offset, 0)
                    with broker._lock:
                        broker.state.log_path = new_log
                        broker.state.log_off = 0
                        broker.state.pending_calls = {"new"}
                    return [old_obj], 100
                broker._stop.set()
                return [], offset

            with patch.object(broker_mod, "_read_jsonl_from_offset", side_effect=fake_read), patch.object(broker_mod.time, "sleep", lambda _seconds: broker._stop.set()):
                broker._log_watcher()

            self.assertEqual(st.log_path, new_log)
            self.assertEqual(st.log_off, 0)
            self.assertEqual(st.pending_calls, {"new"})
            self.assertGreaterEqual(calls, 2)

    def test_broker_jsonl_reader_processes_completed_row_beyond_bounded_window(self) -> None:
        with TemporaryDirectory() as td:
            log_path = Path(td) / "pi.jsonl"
            row = {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "toolCall", "id": "tool-1", "name": "bash", "arguments": {"blob": "x" * (600 * 1024)}}],
                    "stopReason": "toolUse",
                },
            }
            encoded = json.dumps(row).encode("utf-8") + b"\n"
            log_path.write_bytes(encoded)

            objs, off = broker_mod._read_jsonl_from_offset(log_path, 0, max_bytes=256 * 1024)

        self.assertEqual(len(objs), 1)
        self.assertEqual(objs[0]["message"]["content"][0]["id"], "tool-1")
        self.assertEqual(off, len(encoded))

    def test_register_from_pi_log_uses_complete_jsonl_offset(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            log_path = root / "pi.jsonl"
            complete = json.dumps({"type": "session", "id": "pi-session"}).encode("utf-8") + b"\n"
            log_path.write_bytes(complete + b'{"type":"message"')
            st = State(codex_pid=1, pty_master_fd=1, cwd=str(root), start_ts=0.0, codex_home=root, sessions_dir=root, sock_path=root / "broker.sock")
            broker = Broker.__new__(Broker)
            broker.state = st
            broker.sessions_dir = root
            broker._lock = threading.Lock()
            broker._write_meta = lambda: None  # type: ignore[method-assign]
            broker._session_id_from_rollout_path = lambda _path: "pi-session"  # type: ignore[method-assign]

            with patch.object(broker_mod, "AGENT_BACKEND", "pi"), patch.object(broker_mod, "OWNER_TAG", "web"):
                self.assertTrue(broker._register_from_log(log_path=log_path))

            self.assertEqual(st.log_path, log_path)
            self.assertEqual(st.log_off, len(complete))

    def test_broker_jsonl_reader_does_not_advance_over_large_unterminated_fragment(self) -> None:
        with TemporaryDirectory() as td:
            log_path = Path(td) / "pi.jsonl"
            complete = json.dumps({"type": "session", "id": "pi-session"}).encode("utf-8") + b"\n"
            fragment = b'{"type":"message","message":{"role":"assistant","content":[{"type":"toolCall","id":"tool-1","arguments":{"blob":"' + (b"x" * 70000)
            log_path.write_bytes(complete + fragment)

            objs, off = broker_mod._read_jsonl_from_offset(log_path, len(complete), max_bytes=1024)

            self.assertEqual(objs, [])
            self.assertEqual(off, len(complete))

            complete_tool = {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "toolCall", "id": "tool-1", "name": "bash", "arguments": {"blob": "x" * 70000}}],
                    "stopReason": "toolUse",
                },
            }
            log_path.write_bytes(complete + json.dumps(complete_tool).encode("utf-8") + b"\n")
            objs, off = broker_mod._read_jsonl_from_offset(log_path, len(complete), max_bytes=128 * 1024)

            self.assertEqual(len(objs), 1)
            self.assertEqual(objs[0]["message"]["content"][0]["id"], "tool-1")
            self.assertEqual(off, log_path.stat().st_size)

    def test_pi_log_bind_does_not_skip_trailing_partial_row(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            log_path = root / "pi.jsonl"
            complete_rows = [
                {"type": "session", "id": "pi-session", "cwd": str(root)},
                {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "run"}]}},
            ]
            partial_tool_row = {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "toolCall", "id": "tool-1", "name": "bash", "arguments": {"command": "sleep 30"}}],
                    "stopReason": "toolUse",
                },
            }
            complete_prefix = "".join(json.dumps(row) + "\n" for row in complete_rows)
            partial_line = json.dumps(partial_tool_row)
            log_path.write_text(complete_prefix + partial_line, encoding="utf-8")
            st = State(codex_pid=1, pty_master_fd=1, cwd=str(root), start_ts=0.0, codex_home=root, sessions_dir=root, sock_path=root / "broker.sock")
            broker = Broker.__new__(Broker)
            broker.sessions_dir = root
            broker.state = st
            broker._lock = threading.Lock()
            broker._write_meta = lambda: None  # type: ignore[method-assign]

            with patch.object(broker_mod, "AGENT_BACKEND", "pi"):
                broker._maybe_register_or_switch_rollout(log_path=log_path)

            self.assertEqual(st.log_off, len(complete_prefix.encode("utf-8")))
            self.assertEqual(st.pending_calls, set())
            self.assertTrue(st.busy)
            log_path.write_text(complete_prefix + partial_line + "\n", encoding="utf-8")
            objs, new_off = broker_mod._read_jsonl_from_offset(log_path, st.log_off)
            self.assertEqual(new_off, log_path.stat().st_size)
            self.assertEqual(len(objs), 1)
            _apply_rollout_obj_to_state(st, objs[0], now_ts=30.0)

        self.assertEqual(st.pending_calls, {"tool-1"})
        self.assertFalse(_should_clear_busy_state(st, now_ts=100.0))

    def test_cc_log_bind_closes_stale_busy_when_log_is_idle(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            log_path = root / "cc.jsonl"
            rows = [
                {"type": "user", "sessionId": "cc-session", "cwd": str(root), "message": {"role": "user", "content": "hi"}},
                {
                    "type": "assistant",
                    "sessionId": "cc-session",
                    "message": {"role": "assistant", "content": [{"type": "text", "text": "hello"}], "stop_reason": "end_turn"},
                },
            ]
            log_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            st = State(
                codex_pid=1,
                pty_master_fd=1,
                cwd=str(root),
                start_ts=0.0,
                codex_home=root,
                sessions_dir=root,
                sock_path=root / "broker.sock",
                busy=True,
                turn_open=True,
                turn_has_completion_candidate=False,
                last_turn_activity_ts=1.0,
            )
            broker = Broker.__new__(Broker)
            broker.sessions_dir = root
            broker.state = st
            broker._lock = threading.Lock()
            broker._write_meta = lambda: None  # type: ignore[method-assign]

            with patch.object(broker_mod, "AGENT_BACKEND", "cc"):
                broker._maybe_register_or_switch_rollout(log_path=log_path)

            self.assertFalse(st.busy)
            self.assertFalse(st.turn_open)
            self.assertEqual(st.pending_calls, set())
            self.assertFalse(_should_clear_busy_state(st, now_ts=999.0))

    def test_cc_log_bind_seeds_busy_turn_without_pending_tool(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            log_path = root / "cc.jsonl"
            rows = [
                {"type": "user", "sessionId": "cc-session", "cwd": str(root), "message": {"role": "user", "content": "run"}},
                {
                    "type": "assistant",
                    "sessionId": "cc-session",
                    "message": {"role": "assistant", "content": [{"type": "thinking", "thinking": "working"}], "stop_reason": None},
                },
            ]
            log_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            st = State(codex_pid=1, pty_master_fd=1, cwd=str(root), start_ts=0.0, codex_home=root, sessions_dir=root, sock_path=root / "broker.sock")
            broker = Broker.__new__(Broker)
            broker.sessions_dir = root
            broker.state = st
            broker._lock = threading.Lock()
            broker._write_meta = lambda: None  # type: ignore[method-assign]

            with patch.object(broker_mod, "AGENT_BACKEND", "cc"):
                broker._maybe_register_or_switch_rollout(log_path=log_path)

            self.assertEqual(st.pending_calls, set())
            self.assertTrue(st.busy)
            self.assertTrue(st.turn_open)

    def test_cc_log_bind_seeds_pending_tool_state(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            log_path = root / "cc.jsonl"
            rows = [
                {"type": "user", "sessionId": "cc-session", "cwd": str(root), "message": {"role": "user", "content": "run"}},
                {
                    "type": "assistant",
                    "sessionId": "cc-session",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "tool_use", "name": "Bash", "id": "toolu_1", "input": {}}],
                        "stop_reason": "tool_use",
                    },
                },
            ]
            log_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            st = State(codex_pid=1, pty_master_fd=1, cwd=str(root), start_ts=0.0, codex_home=root, sessions_dir=root, sock_path=root / "broker.sock")
            broker = Broker.__new__(Broker)
            broker.sessions_dir = root
            broker.state = st
            broker._lock = threading.Lock()
            broker._write_meta = lambda: None  # type: ignore[method-assign]

            with patch.object(broker_mod, "AGENT_BACKEND", "cc"):
                broker._maybe_register_or_switch_rollout(log_path=log_path)

            self.assertEqual(st.pending_calls, {"toolu_1"})
            self.assertTrue(st.busy)
            _apply_rollout_obj_to_state(st, {"type": "system", "subtype": "turn_duration", "durationMs": 10}, now_ts=2.0)
            self.assertTrue(st.busy)
            self.assertTrue(st.turn_open)

    def test_cc_turn_duration_does_not_close_unmatched_tool_use(self) -> None:
        st = _state()
        _apply_rollout_obj_to_state(
            st,
            {"type": "user", "message": {"role": "user", "content": "run pwd"}},
            now_ts=10.0,
        )
        _apply_rollout_obj_to_state(
            st,
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "name": "Bash", "id": "toolu_1", "input": {}}],
                    "stop_reason": "tool_use",
                },
            },
            now_ts=11.0,
        )
        self.assertEqual(st.pending_calls, {"toolu_1"})
        _apply_rollout_obj_to_state(
            st,
            {"type": "system", "subtype": "turn_duration", "durationMs": 10},
            now_ts=12.0,
        )
        self.assertTrue(st.busy)
        self.assertTrue(st.turn_open)
        self.assertEqual(st.pending_calls, {"toolu_1"})

    def test_cc_final_message_does_not_close_unmatched_tool_use(self) -> None:
        st = _state()
        _apply_rollout_obj_to_state(
            st,
            {"type": "user", "message": {"role": "user", "content": "run pwd"}},
            now_ts=10.0,
        )
        _apply_rollout_obj_to_state(
            st,
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "name": "Bash", "id": "toolu_1", "input": {}}],
                    "stop_reason": "tool_use",
                },
            },
            now_ts=11.0,
        )
        _apply_rollout_obj_to_state(
            st,
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn"},
            },
            now_ts=12.0,
        )
        self.assertTrue(st.busy)
        self.assertTrue(st.turn_open)
        self.assertEqual(st.pending_calls, {"toolu_1"})

    def test_cc_tool_result_without_id_does_not_clear_known_pending_tool(self) -> None:
        st = _state()
        _apply_rollout_obj_to_state(
            st,
            {"type": "user", "message": {"role": "user", "content": "run pwd"}},
            now_ts=10.0,
        )
        _apply_rollout_obj_to_state(
            st,
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "name": "Bash", "id": "toolu_1", "input": {}}],
                    "stop_reason": "tool_use",
                },
            },
            now_ts=11.0,
        )
        _apply_rollout_obj_to_state(
            st,
            {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "content": "ok"}]}},
            now_ts=12.0,
        )
        _apply_rollout_obj_to_state(
            st,
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn"},
            },
            now_ts=13.0,
        )
        self.assertTrue(st.busy)
        self.assertEqual(st.pending_calls, {"toolu_1"})

    def test_cc_multiple_idless_tool_uses_need_multiple_idless_results(self) -> None:
        st = _state()
        _apply_rollout_obj_to_state(
            st,
            {"type": "user", "message": {"role": "user", "content": "run"}},
            now_ts=10.0,
        )
        _apply_rollout_obj_to_state(
            st,
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "name": "A", "input": {}},
                        {"type": "tool_use", "name": "B", "input": {}},
                    ],
                    "stop_reason": "tool_use",
                },
            },
            now_ts=11.0,
        )
        _apply_rollout_obj_to_state(
            st,
            {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "content": "one"}]}},
            now_ts=12.0,
        )
        _apply_rollout_obj_to_state(
            st,
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn"},
            },
            now_ts=13.0,
        )
        self.assertTrue(st.busy)
        self.assertTrue(st.pending_calls)

    def test_cc_repeated_idless_tool_uses_do_not_collide_after_gc(self) -> None:
        st = _state()
        for i in range(6):
            obj = {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "tool_use", "name": "T", "input": {}}], "stop_reason": "tool_use"}}
            _apply_rollout_obj_to_state(st, obj, now_ts=float(i))
            del obj
            gc.collect()
        self.assertEqual(len(st.pending_calls), 6)
        for i in range(5):
            _apply_rollout_obj_to_state(st, {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "content": "ok"}]}}, now_ts=10.0 + i)
        _apply_rollout_obj_to_state(
            st,
            {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn"}},
            now_ts=20.0,
        )
        self.assertTrue(st.busy)
        self.assertTrue(st.pending_calls)

    def test_cc_split_row_idless_tool_uses_need_multiple_idless_results(self) -> None:
        st = _state()
        _apply_rollout_obj_to_state(st, {"type": "user", "message": {"role": "user", "content": "run"}}, now_ts=10.0)
        _apply_rollout_obj_to_state(
            st,
            {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "tool_use", "name": "A", "input": {}}], "stop_reason": "tool_use"}},
            now_ts=11.0,
        )
        _apply_rollout_obj_to_state(
            st,
            {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "tool_use", "name": "B", "input": {}}], "stop_reason": "tool_use"}},
            now_ts=12.0,
        )
        _apply_rollout_obj_to_state(st, {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "content": "one"}]}}, now_ts=13.0)
        _apply_rollout_obj_to_state(
            st,
            {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn"}},
            now_ts=14.0,
        )
        self.assertTrue(st.busy)
        self.assertTrue(st.pending_calls)

    def test_cc_mixed_malformed_tool_use_keeps_unknown_pending_after_known_result(self) -> None:
        st = _state()
        _apply_rollout_obj_to_state(
            st,
            {"type": "user", "message": {"role": "user", "content": "run pwd"}},
            now_ts=10.0,
        )
        _apply_rollout_obj_to_state(
            st,
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "name": "Bash", "id": "toolu_1", "input": {}},
                        {"type": "tool_use", "name": "Bad", "input": {}},
                    ],
                    "stop_reason": "tool_use",
                },
            },
            now_ts=11.0,
        )
        _apply_rollout_obj_to_state(
            st,
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
                },
            },
            now_ts=12.0,
        )
        _apply_rollout_obj_to_state(
            st,
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn"},
            },
            now_ts=13.0,
        )
        self.assertTrue(st.busy)
        self.assertTrue(st.pending_calls)

    def test_cc_tool_result_and_final_message_clear_busy(self) -> None:
        st = _state()
        _apply_rollout_obj_to_state(
            st,
            {"type": "user", "message": {"role": "user", "content": "run pwd"}},
            now_ts=10.0,
        )
        _apply_rollout_obj_to_state(
            st,
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "name": "Bash", "id": "toolu_1", "input": {}}],
                    "stop_reason": "tool_use",
                },
            },
            now_ts=11.0,
        )
        _apply_rollout_obj_to_state(
            st,
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
                },
            },
            now_ts=12.0,
        )
        self.assertEqual(st.pending_calls, set())
        self.assertTrue(st.busy)
        _apply_rollout_obj_to_state(
            st,
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn"},
            },
            now_ts=13.0,
        )
        self.assertFalse(st.busy)
        self.assertFalse(st.turn_open)

    def test_pi_tool_use_message_keeps_turn_busy(self) -> None:
        st = _state()
        _apply_rollout_obj_to_state(
            st,
            {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "run pwd"}]}},
            now_ts=10.0,
        )
        _apply_rollout_obj_to_state(
            st,
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": ""},
                        {"type": "toolCall", "id": "tool-1", "name": "bash", "arguments": {"command": "pwd"}},
                    ],
                },
            },
            now_ts=11.0,
        )
        self.assertTrue(st.busy)
        self.assertTrue(st.turn_open)
        self.assertFalse(st.turn_has_completion_candidate)

    def test_pi_final_message_clears_busy(self) -> None:
        st = _state()
        _apply_rollout_obj_to_state(
            st,
            {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "run pwd"}]}},
            now_ts=10.0,
        )
        _apply_rollout_obj_to_state(
            st,
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": ""},
                        {"type": "toolCall", "id": "tool-1", "name": "bash", "arguments": {"command": "pwd"}},
                    ],
                },
            },
            now_ts=11.0,
        )
        _apply_rollout_obj_to_state(
            st,
            {
                "type": "message",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "tool-1",
                    "toolName": "bash",
                    "content": [{"type": "text", "text": "/tmp\n"}],
                    "isError": False,
                },
            },
            now_ts=12.0,
        )
        _apply_rollout_obj_to_state(
            st,
            {"type": "message", "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}]}},
            now_ts=13.0,
        )
        self.assertFalse(st.busy)
        self.assertFalse(st.turn_open)
        self.assertFalse(st.turn_has_completion_candidate)
        self.assertEqual(st.pending_calls, set())
        self.assertFalse(_should_clear_busy_state(st, now_ts=15.0 + BUSY_QUIET_SECONDS + 60.0))

    def test_tool_call_reopens_turn_after_idle_clear(self) -> None:
        st = _state()
        _apply_rollout_obj_to_state(
            st,
            {"type": "response_item", "payload": {"type": "function_call", "call_id": "call-1"}},
            now_ts=20.0,
        )
        self.assertTrue(st.busy)
        self.assertTrue(st.turn_open)
        self.assertFalse(st.turn_has_completion_candidate)

    def test_agent_message_does_not_reopen_closed_turn(self) -> None:
        st = _state()
        st.turn_open = False
        st.turn_has_completion_candidate = False
        st.busy = False
        _apply_rollout_obj_to_state(
            st,
            {"type": "event_msg", "payload": {"type": "agent_message", "message": "done"}},
            now_ts=25.0,
        )
        self.assertTrue(st.busy)
        self.assertFalse(st.turn_open)
        self.assertFalse(st.turn_has_completion_candidate)

    def test_token_count_alone_does_not_start_busy(self) -> None:
        st = _state()
        _apply_rollout_obj_to_state(
            st,
            {"type": "event_msg", "payload": {"type": "token_count", "info": {}}},
            now_ts=16.0,
        )
        self.assertFalse(st.busy)
        self.assertEqual(st.last_turn_activity_ts, 0.0)

    def test_interrupt_hint_from_pty_marks_busy(self) -> None:
        st = _state()
        _update_busy_from_pty_text(st, "\x1b[2mWorking (1s • esc to interrupt)\x1b[0m", now_ts=20.0)
        self.assertTrue(st.busy)
        self.assertEqual(st.last_interrupt_hint_ts, 20.0)
        self.assertEqual(st.last_turn_activity_ts, 20.0)

    def test_interrupt_hint_grace_delays_idle_clear(self) -> None:
        st = _state()
        st.busy = True
        st.last_turn_activity_ts = 10.0
        st.last_interrupt_hint_ts = 12.0
        self.assertFalse(_should_clear_busy_state(st, now_ts=10.0 + BUSY_QUIET_SECONDS + 0.2))
        clear_ts = max(
            10.0 + BUSY_QUIET_SECONDS + 0.2,
            12.0 + BUSY_INTERRUPT_GRACE_SECONDS + 0.2,
        )
        self.assertTrue(_should_clear_busy_state(st, now_ts=clear_ts))

    def test_compacting_hint_from_pty_marks_busy(self) -> None:
        st = _state()
        _update_busy_from_pty_text(st, "\x1b[2mCompacting context...\x1b[0m", now_ts=30.0)
        self.assertTrue(st.busy)
        self.assertEqual(st.last_turn_activity_ts, 30.0)
        self.assertEqual(st.last_interrupt_hint_ts, 0.0)

    def test_stale_interrupt_tail_does_not_rearm_busy_on_unrelated_text(self) -> None:
        st = _state()
        _update_busy_from_pty_text(st, "\x1b[2mWorking (1s • esc to interrupt)\x1b[0m", now_ts=10.0)
        self.assertTrue(st.busy)
        self.assertEqual(st.last_turn_activity_ts, 10.0)
        _update_busy_from_pty_text(st, " •", now_ts=20.0)
        self.assertEqual(st.last_turn_activity_ts, 10.0)
        self.assertEqual(st.last_interrupt_hint_ts, 10.0)


if __name__ == "__main__":
    unittest.main()
