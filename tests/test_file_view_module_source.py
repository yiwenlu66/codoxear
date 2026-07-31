import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER_PY = ROOT / "codoxear" / "server.py"
FILE_VIEW_PY = ROOT / "codoxear" / "file_view.py"


if __name__ == "__main__":
    unittest.main()
