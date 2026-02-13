import unittest
from pathlib import Path

from codoxear.broker import (
    BUSY_INTERRUPT_GRACE_SECONDS,
    BUSY_QUIET_SECONDS,
    State,
    _apply_rollout_obj_to_state,
    _maybe_detach_on_new_session_hint,
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
    def test_new_session_hint_detaches_rollout(self) -> None:
        st = _state()
        st.log_path = Path("/tmp/sessions/rollout-old.jsonl")
        st.session_id = "old"
        ok = _maybe_detach_on_new_session_hint(
            st=st,
            tail="",
            cleaned="To continue this session, run codex resume ...\n",
        )
        self.assertTrue(ok)
        self.assertIsNone(st.log_path)
        self.assertIsNone(st.session_id)
        self.assertTrue(len(st.ignored_rollout_paths) >= 1)

    def test_new_session_hint_matches_across_chunk_boundary(self) -> None:
        st = _state()
        st.log_path = Path("/tmp/sessions/rollout-old.jsonl")
        st.session_id = "old"
        ok = _maybe_detach_on_new_session_hint(
            st=st,
            tail="To continue this session, r",
            cleaned="un codex resume ...\n",
        )
        self.assertTrue(ok)

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
