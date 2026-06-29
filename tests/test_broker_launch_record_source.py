import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BROKER_PY = ROOT / "codoxear" / "broker.py"
BROKER_LAUNCH_RECORD_PY = ROOT / "codoxear" / "broker_launch_record.py"


class TestBrokerLaunchRecordSource(unittest.TestCase):
    def test_launch_record_owner_is_dedicated_module(self) -> None:
        broker_tree = ast.parse(BROKER_PY.read_text(encoding="utf-8"))
        record_tree = ast.parse(BROKER_LAUNCH_RECORD_PY.read_text(encoding="utf-8"))
        broker_defs = {node.name for node in broker_tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        record_defs = {node.name for node in record_tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}

        self.assertIn("_record_broker_launch_attempt", record_defs)
        self.assertIn("_broker_launch_record", record_defs)
        self.assertIn("_record_launch_attempt", broker_defs)
        self.assertIn("_broker_launch_record", broker_defs)

    def test_broker_keeps_launch_record_wrappers(self) -> None:
        broker_source = BROKER_PY.read_text(encoding="utf-8")
        record_source = BROKER_LAUNCH_RECORD_PY.read_text(encoding="utf-8")
        self.assertIn("from codoxear.broker_launch_record import _record_broker_launch_attempt", broker_source)
        self.assertIn("owner_tag=OWNER_TAG", broker_source)
        self.assertIn("launch_attempts_path=LAUNCH_ATTEMPTS_PATH", broker_source)
        self.assertIn("stderr=sys.stderr", broker_source)
        self.assertIn("agent_backend=AGENT_BACKEND", broker_source)
        self.assertIn("_append_launch_attempt", record_source)
        self.assertIn("_read_launch_attempts", record_source)
        self.assertIn("_redacted_launch_attempt_persist_record", record_source)


if __name__ == "__main__":
    unittest.main()
