import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UTIL = ROOT / "codoxear" / "util.py"
JSONL_OFFSET = ROOT / "codoxear" / "jsonl_offset.py"


if __name__ == "__main__":
    unittest.main()
