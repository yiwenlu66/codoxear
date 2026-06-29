import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BROKER_PY = ROOT / "codoxear" / "broker.py"
BROKER_TURN_STATE_PY = ROOT / "codoxear" / "broker_turn_state.py"


class TestBrokerTurnStateSource(unittest.TestCase):
    def test_busy_turn_state_owner_is_dedicated_module(self) -> None:
        broker_tree = ast.parse(BROKER_PY.read_text(encoding="utf-8"))
        turn_state_tree = ast.parse(BROKER_TURN_STATE_PY.read_text(encoding="utf-8"))
        broker_defs = {node.name for node in broker_tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        turn_state_defs = {node.name for node in turn_state_tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}

        owned_names = {
            "State",
            "_strip_ansi",
            "_update_busy_from_pty_text",
            "_apply_rollout_obj_to_state",
            "_mark_explicit_interrupt_request",
            "_mark_busy_state_idle",
            "_close_turn_state",
            "_response_call_started",
            "_response_call_finished",
            "_codex_error_affects_turn_status",
        }
        self.assertTrue(owned_names <= turn_state_defs)
        self.assertFalse(owned_names & broker_defs)

    def test_broker_exports_reducer_facade_and_injects_timing_policy(self) -> None:
        broker_source = BROKER_PY.read_text(encoding="utf-8")
        self.assertIn("from codoxear.broker_turn_state import State", broker_source)
        self.assertIn("from codoxear.broker_turn_state import _apply_rollout_obj_to_state", broker_source)
        self.assertIn("from codoxear.broker_turn_state import _update_busy_from_pty_text", broker_source)
        self.assertIn("from codoxear.broker_turn_state import _should_clear_busy_state as _should_clear_busy_state_impl", broker_source)
        self.assertIn("busy_quiet_seconds=BUSY_QUIET_SECONDS", broker_source)
        self.assertIn("busy_interrupt_grace_seconds=BUSY_INTERRUPT_GRACE_SECONDS", broker_source)


if __name__ == "__main__":
    unittest.main()
