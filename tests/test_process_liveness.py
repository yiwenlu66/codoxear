import os
import unittest

from codoxear import process_runtime
from codoxear import util


class TestProcessLiveness(unittest.TestCase):
    def test_liveness_facade_exports_runtime_functions(self) -> None:
        self.assertIs(util.pid_alive, process_runtime.pid_alive)
        self.assertIs(util.process_group_alive, process_runtime.process_group_alive)

    def test_liveness_helpers_reject_invalid_pids_and_detect_current_process(self) -> None:
        self.assertFalse(util.pid_alive(0))
        self.assertFalse(util.pid_alive(-1))
        self.assertFalse(util.process_group_alive(0))
        self.assertFalse(util.process_group_alive(-1))
        self.assertTrue(util.pid_alive(os.getpid()))


if __name__ == "__main__":
    unittest.main()
