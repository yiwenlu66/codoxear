import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BROKER_PY = ROOT / "codoxear" / "broker.py"
BROKER_CONTROL_PY = ROOT / "codoxear" / "broker_control.py"


class TestBrokerControlSource(unittest.TestCase):
    def test_control_command_owner_is_dedicated_module(self) -> None:
        broker_tree = ast.parse(BROKER_PY.read_text(encoding="utf-8"))
        control_tree = ast.parse(BROKER_CONTROL_PY.read_text(encoding="utf-8"))
        broker_defs = {node.name for node in broker_tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        control_defs = {node.name for node in control_tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}

        self.assertIn("_handle_broker_control_connection", control_defs)
        self.assertNotIn("_handle_broker_control_connection", broker_defs)

    def test_broker_handle_conn_is_dependency_wrapper(self) -> None:
        broker_source = BROKER_PY.read_text(encoding="utf-8")
        control_source = BROKER_CONTROL_PY.read_text(encoding="utf-8")
        self.assertIn("from codoxear.broker_control import _handle_broker_control_connection", broker_source)
        self.assertIn("def _handle_conn(self, conn: socket.socket) -> None:", broker_source)
        self.assertIn("get_state=lambda: self.state", broker_source)
        self.assertIn("teardown_managed_process_group=self._teardown_managed_process_group", broker_source)
        self.assertIn("handle_control_socket_connection", control_source)
        self.assertIn("mark_interrupt = req.get(\"interrupt\") is True and b == b\"\\x1b\"", control_source)
        self.assertNotIn("pty.fork", control_source)
        self.assertNotIn("os.fork", control_source)
        self.assertNotIn("openpty", control_source)


if __name__ == "__main__":
    unittest.main()
