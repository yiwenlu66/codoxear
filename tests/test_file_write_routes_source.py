import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILE_ROUTES = ROOT / "codoxear" / "file_routes.py"
FILE_WRITE_ROUTES = ROOT / "codoxear" / "file_write_routes.py"
FILE_ROUTE_COMMON = ROOT / "codoxear" / "file_route_common.py"


if __name__ == "__main__":
    unittest.main()
