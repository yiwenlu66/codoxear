import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BROKER_PY = ROOT / "codoxear" / "broker.py"
BROKER_METADATA_PY = ROOT / "codoxear" / "broker_metadata.py"


if __name__ == "__main__":
    unittest.main()
