import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILE_ROUTES = ROOT / "codoxear" / "file_routes.py"
FILE_GLOBAL_ROUTES = ROOT / "codoxear" / "file_global_routes.py"


if __name__ == "__main__":
    unittest.main()
