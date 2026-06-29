import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BROKER_PY = ROOT / "codoxear" / "broker.py"
BROKER_PROCESS_PY = ROOT / "codoxear" / "broker_process.py"


class TestBrokerProcessSource(unittest.TestCase):
    def test_process_helper_owner_is_dedicated_module(self) -> None:
        broker_tree = ast.parse(BROKER_PY.read_text(encoding="utf-8"))
        process_tree = ast.parse(BROKER_PROCESS_PY.read_text(encoding="utf-8"))
        broker_defs = {node.name for node in broker_tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        process_defs = {node.name for node in process_tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}

        self.assertIn("_set_pdeathsig", process_defs)
        self.assertIn("_require_proc", process_defs)
        self.assertNotIn("_set_pdeathsig", broker_defs)
        self.assertIn("_require_proc", broker_defs)

    def test_broker_preserves_require_proc_patch_seam(self) -> None:
        broker_source = BROKER_PY.read_text(encoding="utf-8")
        process_source = BROKER_PROCESS_PY.read_text(encoding="utf-8")
        self.assertIn("from codoxear.broker_process import _require_proc as _require_proc_impl", broker_source)
        self.assertIn("from codoxear.broker_process import _set_pdeathsig", broker_source)
        self.assertIn("_require_proc_impl(proc_root=PROC_ROOT, platform=sys.platform, stderr=sys.stderr)", broker_source)
        self.assertIn("PR_SET_PDEATHSIG = 1", process_source)
        self.assertIn("requires /proc", process_source)


if __name__ == "__main__":
    unittest.main()
