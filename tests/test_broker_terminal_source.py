import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BROKER_PY = ROOT / "codoxear" / "broker.py"
BROKER_TERMINAL_PY = ROOT / "codoxear" / "broker_terminal.py"


class TestBrokerTerminalSource(unittest.TestCase):
    def test_terminal_query_owner_is_dedicated_module(self) -> None:
        broker_tree = ast.parse(BROKER_PY.read_text(encoding="utf-8"))
        terminal_tree = ast.parse(BROKER_TERMINAL_PY.read_text(encoding="utf-8"))
        broker_defs = {node.name for node in broker_tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        terminal_defs = {node.name for node in terminal_tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}

        self.assertIn("_reply_to_terminal_queries", terminal_defs)
        self.assertNotIn("_reply_to_terminal_queries", broker_defs)

    def test_broker_terminal_wrapper_preserves_emulation_gate(self) -> None:
        broker_source = BROKER_PY.read_text(encoding="utf-8")
        terminal_source = BROKER_TERMINAL_PY.read_text(encoding="utf-8")
        self.assertIn("from codoxear.broker_terminal import _reply_to_terminal_queries", broker_source)
        self.assertIn("if not self._emulate_terminal:", broker_source)
        self.assertIn("self._term_query_buf = _reply_to_terminal_queries(", broker_source)
        self.assertIn("_TERMINAL_QUERY_RESPONSES", terminal_source)
        self.assertIn('b"\\x1b[?1;2c"', terminal_source)


if __name__ == "__main__":
    unittest.main()
