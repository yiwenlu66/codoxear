import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UTIL = ROOT / "codoxear" / "util.py"
PROCESS_LOG_PATHS = ROOT / "codoxear" / "process_log_paths.py"


if __name__ == "__main__":
    unittest.main()
