import unittest
from pathlib import Path

from codoxear.broker import (
    BUSY_INTERRUPT_GRACE_SECONDS,
    BUSY_QUIET_SECONDS,
    State,
    _apply_rollout_obj_to_state,
    _maybe_detach_on_session_switch_trigger,
    _should_clear_busy_state,
    _update_busy_from_pty_text,
)


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


class ClaudeAskUserBusyStateTests(unittest.TestCase):
    """An AskUserQuestion tool_use must keep busy=True until the matching
    tool_result arrives, regardless of how many questions are in the prompt
    or whether the broker-169813-shape "skipped" wording appears in the
    tool_result content. Coverage matters because the codoxear frontend submit
    fix on the JS side has no way to clear busy on its own; the broker is the
    single authority for that flag, and the JS fix only matters if the broker
    correctly transitions on the resulting Claude records.
    """

    def _claude_state(self) -> State:
        st = _state()
        st.busy = True
        st.turn_open = True
        st.last_turn_activity_ts = 100.0
        return st

    def _ask_user_tool_use(self, n_questions: int, tool_use_id: str = "tu-x") -> dict:
        questions = [
            {
                "header": f"H{i}",
                "question": f"Q{i}",
                "options": [{"label": "a"}, {"label": "b"}, {"label": "c"}],
                "multiSelect": False,
            }
            for i in range(n_questions)
        ]
        return {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": tool_use_id,
                        "name": "AskUserQuestion",
                        "input": {"questions": questions},
                    }
                ],
                "stop_reason": "tool_use",
            },
        }

    def _tool_result_user(self, tool_use_id: str, content_text: str) -> dict:
        return {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": content_text,
                    }
                ],
            },
        }

    def _final_assistant_text(self, text: str) -> dict:
        return {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn",
            },
        }

    def test_ask_user_tool_result_clears_busy_for_multi_question_prompt(self) -> None:
        from unittest.mock import patch

        st = self._claude_state()
        tu_id = "tu-multi"
        with patch("codoxear.broker.AGENT_BACKEND", "claude"):
            # Step 1: assistant emits AskUserQuestion tool_use with 3 questions.
            # Busy must stay true (a tool_use does not close a turn).
            _apply_rollout_obj_to_state(st, self._ask_user_tool_use(3, tu_id), now_ts=110.0)
            self.assertTrue(st.busy)
            self.assertTrue(st.turn_open)
            self.assertFalse(st.turn_has_completion_candidate)

            # Step 2: tool_result lands. By itself it does not clear busy
            # (the agent will follow up); but it must not corrupt state.
            _apply_rollout_obj_to_state(
                st,
                self._tool_result_user(
                    tu_id,
                    'Your questions have been answered: "Q0"="a", "Q1"="b", "Q2"="c". '
                    "You can now continue with these answers in mind.",
                ),
                now_ts=111.0,
            )
            self.assertTrue(st.busy)
            self.assertTrue(st.turn_open)

            # Step 3: assistant final text + end_turn closes the turn cleanly.
            _apply_rollout_obj_to_state(
                st,
                self._final_assistant_text("Got it - a, b, c."),
                now_ts=112.0,
            )
            self.assertFalse(st.busy)
            self.assertFalse(st.turn_open)
            self.assertFalse(st.turn_has_completion_candidate)

    def test_ask_user_tool_result_clears_busy_with_skipped_answers(self) -> None:
        """The broker-169813 failure shape: the user answered only one of N
        questions, so the tool_result text contains "skipped". This is the
        exact pathology the codoxear frontend was producing before the cursor
        fix; the broker must still transition cleanly so the UI is at least
        not stuck on "working" after such a partial answer.
        """
        from unittest.mock import patch

        st = self._claude_state()
        tu_id = "tu-skipped"
        with patch("codoxear.broker.AGENT_BACKEND", "claude"):
            _apply_rollout_obj_to_state(st, self._ask_user_tool_use(4, tu_id), now_ts=200.0)
            self.assertTrue(st.busy)

            # Real broker-169813 wording (truncated): partial answer + skipped notice.
            partial_content = (
                'Only the deployment question was answered. The other three were skipped.'
            )
            _apply_rollout_obj_to_state(
                st,
                self._tool_result_user(tu_id, partial_content),
                now_ts=201.0,
            )
            # tool_result alone keeps the turn open (agent will follow up).
            self.assertTrue(st.busy)
            self.assertTrue(st.turn_open)

            # The agent then either re-asks or ends. Closing path:
            _apply_rollout_obj_to_state(
                st,
                self._final_assistant_text("Recorded what you answered."),
                now_ts=202.0,
            )
            self.assertFalse(st.busy)
            self.assertFalse(st.turn_open)


if __name__ == "__main__":
    unittest.main()
