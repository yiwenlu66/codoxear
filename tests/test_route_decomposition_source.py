import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER_PY = ROOT / "codoxear" / "server.py"
SERVER_HANDLER_PY = ROOT / "codoxear" / "server_handler.py"
TRANSCRIPT_SEARCH_PY = ROOT / "codoxear" / "transcript_search.py"
STATIC_ROUTES_PY = ROOT / "codoxear" / "static_routes.py"
HOOK_ROUTES_PY = ROOT / "codoxear" / "hook_routes.py"
VOICE_ROUTES_PY = ROOT / "codoxear" / "voice_routes.py"


if __name__ == "__main__":
    unittest.main()
