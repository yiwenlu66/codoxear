import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UTIL = ROOT / "codoxear" / "util.py"
DISCOVERY = ROOT / "codoxear" / "session_log_discovery.py"


if __name__ == "__main__":
    unittest.main()
