import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER_PY = ROOT / "codoxear" / "server.py"
SERVER_HANDLER_PY = ROOT / "codoxear" / "server_handler.py"
SERVER_ROUTE_DEPS_PY = ROOT / "codoxear" / "server_route_deps.py"
SERVER_HTTP_PY = ROOT / "codoxear" / "server_http.py"
AUTH_ROUTES_PY = ROOT / "codoxear" / "auth_routes.py"
CONTROL_ROUTES_PY = ROOT / "codoxear" / "control_routes.py"


if __name__ == "__main__":
    unittest.main()
