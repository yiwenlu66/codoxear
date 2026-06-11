import unittest
from pathlib import Path

from codoxear.broker import State
from codoxear.broker import _apply_rollout_obj_to_state


SESSION_ID = "11111111-2222-3333-4444-555555555555"


def state() -> State:
    return State(codex_pid=1, pty_master_fd=1, cwd="/repo", start_ts=1.0, codex_home=Path("/tmp/.claude"), sessions_dir=Path("/tmp/.claude/projects"))


def user(content="hello"):
    return {"type": "user", "sessionId": SESSION_ID, "message": {"role": "user", "content": content}}


def assistant(content, stop_reason="end_turn"):
    return {"type": "assistant", "sessionId": SESSION_ID, "message": {"role": "assistant", "content": content, "stop_reason": stop_reason}}


class TestCcBusyState(unittest.TestCase):
    def test_user_message_starts_turn(self) -> None:
        st = state()
        _apply_rollout_obj_to_state(st, user("hello"), now_ts=10.0)
        self.assertTrue(st.busy)
        self.assertTrue(st.turn_open)
        self.assertFalse(st.turn_has_completion_candidate)

    def test_final_assistant_closes_turn(self) -> None:
        st = state()
        _apply_rollout_obj_to_state(st, user("hello"), now_ts=10.0)
        _apply_rollout_obj_to_state(st, assistant([{"type": "text", "text": "done"}], stop_reason="end_turn"), now_ts=11.0)
        self.assertFalse(st.busy)
        self.assertFalse(st.turn_open)

    def test_tool_use_and_tool_result_keep_busy(self) -> None:
        st = state()
        _apply_rollout_obj_to_state(st, user("hello"), now_ts=10.0)
        _apply_rollout_obj_to_state(st, assistant([{"type": "tool_use", "name": "Bash", "id": "toolu_1", "input": {}}], stop_reason="tool_use"), now_ts=11.0)
        self.assertTrue(st.busy)
        self.assertTrue(st.turn_open)
        _apply_rollout_obj_to_state(st, user([{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}]), now_ts=12.0)
        self.assertTrue(st.busy)
        self.assertTrue(st.turn_open)

    def test_turn_duration_closes_turn(self) -> None:
        st = state()
        _apply_rollout_obj_to_state(st, user("hello"), now_ts=10.0)
        _apply_rollout_obj_to_state(st, {"type": "system", "subtype": "turn_duration", "durationMs": 100}, now_ts=12.0)
        self.assertFalse(st.busy)
        self.assertFalse(st.turn_open)


if __name__ == "__main__":
    unittest.main()
