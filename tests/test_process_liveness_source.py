import os
import unittest
from pathlib import Path

from codoxear import process_runtime
from codoxear import util


ROOT = Path(__file__).resolve().parents[1]


class TestProcessLivenessSource(unittest.TestCase):
    def test_liveness_helpers_are_defined_once_in_process_runtime_with_util_facade(self) -> None:
        sources = {path: path.read_text(encoding="utf-8") for path in (ROOT / "codoxear").glob("*.py")}
        self.assertEqual(sum(src.count("def pid_alive(") for src in sources.values()), 1)
        self.assertEqual(sum(src.count("def process_group_alive(") for src in sources.values()), 1)
        self.assertIn("def pid_alive(pid: int) -> bool:", sources[ROOT / "codoxear" / "process_runtime.py"])
        self.assertIn("def process_group_alive(root_pid: int) -> bool:", sources[ROOT / "codoxear" / "process_runtime.py"])
        self.assertIn("from .process_runtime import pid_alive", sources[ROOT / "codoxear" / "util.py"])
        self.assertIn("from .process_runtime import process_group_alive", sources[ROOT / "codoxear" / "util.py"])
        self.assertNotIn("def pid_alive(pid: int) -> bool:", sources[ROOT / "codoxear" / "util.py"])
        self.assertIn("from .util import pid_alive as _pid_alive", sources[ROOT / "codoxear" / "server.py"])
        self.assertIn("from codoxear.util import pid_alive as _pid_alive", sources[ROOT / "codoxear" / "broker_metadata.py"])
        self.assertIn("from .util import process_group_alive as _process_group_alive", sources[ROOT / "codoxear" / "sessiond.py"])

    def test_liveness_facade_exports_runtime_functions(self) -> None:
        self.assertIs(util.pid_alive, process_runtime.pid_alive)
        self.assertIs(util.process_group_alive, process_runtime.process_group_alive)
        self.assertFalse(util.pid_alive(0))
        self.assertFalse(util.process_group_alive(0))
        self.assertTrue(util.pid_alive(os.getpid()))


if __name__ == "__main__":
    unittest.main()
