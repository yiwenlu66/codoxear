import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER_PY = ROOT / "codoxear" / "server.py"
SESSION_LISTING_PY = ROOT / "codoxear" / "session_listing.py"
SESSION_INPUT_PY = ROOT / "codoxear" / "session_input.py"
SESSION_CONTROL_PY = ROOT / "codoxear" / "session_control.py"
SESSION_READINESS_PY = ROOT / "codoxear" / "session_readiness.py"
SESSION_SEND_PY = ROOT / "codoxear" / "session_send.py"
SESSION_QUEUE_PY = ROOT / "codoxear" / "session_queue.py"
SERVER_ROUTE_DEPS_PY = ROOT / "codoxear" / "server_route_deps.py"
FILE_UPLOAD_PY = ROOT / "codoxear" / "file_upload.py"
CONTROL_ROUTES_PY = ROOT / "codoxear" / "control_routes.py"
BROKER_PY = ROOT / "codoxear" / "broker.py"
BROKER_METADATA_PY = ROOT / "codoxear" / "broker_metadata.py"
SESSIOND_PY = ROOT / "codoxear" / "sessiond.py"
SESSION_LAUNCHER_PY = ROOT / "codoxear" / "session_launcher.py"


if __name__ == "__main__":
    unittest.main()
