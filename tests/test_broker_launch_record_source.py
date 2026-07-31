import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BROKER_PY = ROOT / "codoxear" / "broker.py"
BROKER_LAUNCH_RECORD_PY = ROOT / "codoxear" / "broker_launch_record.py"


if __name__ == "__main__":
    unittest.main()
