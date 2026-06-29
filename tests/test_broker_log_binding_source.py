import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BROKER_PY = ROOT / "codoxear" / "broker.py"
BROKER_LOG_BINDING_PY = ROOT / "codoxear" / "broker_log_binding.py"


class TestBrokerLogBindingSource(unittest.TestCase):
    def test_log_binding_policy_owner_is_dedicated_module(self) -> None:
        broker_tree = ast.parse(BROKER_PY.read_text(encoding="utf-8"))
        binding_tree = ast.parse(BROKER_LOG_BINDING_PY.read_text(encoding="utf-8"))
        broker_defs = {node.name for node in broker_tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        binding_defs = {node.name for node in binding_tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}

        owned_names = {
            "BrokerLogBinding",
            "BrokerLogSeed",
            "BrokerLogStateApplyResult",
            "_resolve_broker_log_binding",
            "_seed_broker_log_state",
            "_apply_broker_log_binding_to_state",
        }
        self.assertTrue(owned_names <= binding_defs)
        self.assertFalse(owned_names & broker_defs)

    def test_broker_log_binding_wrapper_keeps_side_effects_local(self) -> None:
        broker_source = BROKER_PY.read_text(encoding="utf-8")
        binding_source = BROKER_LOG_BINDING_PY.read_text(encoding="utf-8")
        self.assertIn("from codoxear.broker_log_binding import _resolve_broker_log_binding", broker_source)
        self.assertIn("binding = _resolve_broker_log_binding(", broker_source)
        self.assertIn("seed = _seed_broker_log_state", broker_source)
        self.assertIn("result = _apply_broker_log_binding_to_state", broker_source)
        self.assertIn("self._register_from_log(log_path=binding.log_path)", broker_source)
        self.assertIn("self._write_meta()", broker_source)
        self.assertNotIn("threading.Thread", binding_source)
        self.assertNotIn("SOCK_DIR", binding_source)


if __name__ == "__main__":
    unittest.main()
