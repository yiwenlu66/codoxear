import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BROKER_PY = ROOT / "codoxear" / "broker.py"
BROKER_LOG_WATCHER_PY = ROOT / "codoxear" / "broker_log_watcher.py"


class TestBrokerLogWatcherSource(unittest.TestCase):
    def test_log_watcher_state_helpers_live_in_module(self) -> None:
        broker_tree = ast.parse(BROKER_PY.read_text(encoding="utf-8"))
        watcher_tree = ast.parse(BROKER_LOG_WATCHER_PY.read_text(encoding="utf-8"))
        broker_defs = {node.name for node in broker_tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        watcher_defs = {node.name for node in watcher_tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}

        owned_names = {
            "_pop_key_queue_if_idle",
            "_clear_resume_delivery_mute_if_idle",
            "_apply_log_objects_to_state",
        }
        self.assertTrue(owned_names <= watcher_defs)
        self.assertFalse(owned_names & broker_defs)

    def test_broker_log_watcher_loop_delegates_state_mechanics(self) -> None:
        broker_source = BROKER_PY.read_text(encoding="utf-8")
        self.assertIn("from codoxear.broker_log_watcher import _apply_log_objects_to_state", broker_source)
        self.assertIn("fd, kq = _pop_key_queue_if_idle(st3)", broker_source)
        self.assertIn("clear_meta = _clear_resume_delivery_mute_if_idle(st3)", broker_source)
        self.assertIn("_apply_log_objects_to_state(st2, objs, now=_now)", broker_source)


if __name__ == "__main__":
    unittest.main()
