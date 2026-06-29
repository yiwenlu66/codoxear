import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SESSIOND_PY = ROOT / "codoxear" / "sessiond.py"
SESSIOND_STATE_PY = ROOT / "codoxear" / "sessiond_state.py"


class TestSessiondStateSource(unittest.TestCase):
    def test_sessiond_state_and_log_reducer_have_dedicated_owner(self) -> None:
        sessiond_source = SESSIOND_PY.read_text(encoding="utf-8")
        state_source = SESSIOND_STATE_PY.read_text(encoding="utf-8")

        self.assertIn("from .sessiond_state import State", sessiond_source)
        self.assertIn("from .sessiond_state import _busy_value_after_log_batch", sessiond_source)
        self.assertIn("from .sessiond_state import _log_busy_signals", sessiond_source)
        self.assertIn("from .sessiond_state import _read_jsonl_from_offset", sessiond_source)
        self.assertNotIn("class State:", sessiond_source)
        self.assertNotIn("def _log_busy_signals(", sessiond_source)
        self.assertNotIn("def _busy_value_after_log_batch(", sessiond_source)

        self.assertIn("class State:", state_source)
        self.assertIn("def _read_jsonl_from_offset(", state_source)
        self.assertIn("def _log_busy_signals(", state_source)
        self.assertIn("def _busy_value_after_log_batch(", state_source)
        self.assertIn('raise ValueError("invalid rollout event_msg payload")', state_source)
        self.assertIn('pt in {"turn_aborted", "thread_rolled_back", "task_complete", "turn_complete"}', state_source)
        self.assertIn('isinstance(info, dict) and isinstance(info.get("total_token_usage"), dict)', state_source)
        self.assertIn("if _pi_assistant_is_aborted_turn(obj):", state_source)
        self.assertIn("if _pi_assistant_error_text(obj):", state_source)
        self.assertIn("if _pi_assistant_text(obj) and _pi_assistant_is_final_turn_end(obj):", state_source)


if __name__ == "__main__":
    unittest.main()
