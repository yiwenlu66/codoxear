import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_LAUNCH_JS = ROOT / "codoxear" / "static" / "app_launch.js"
APP_NEW_SESSION_JS = ROOT / "codoxear" / "static" / "app_new_session.js"
APP_CSS = ROOT / "codoxear" / "static" / "app.css"
SERVER_PY = ROOT / "codoxear" / "server.py"
LAUNCH_CONFIG_PY = ROOT / "codoxear" / "launch_config.py"
AGENT_BACKEND_PY = ROOT / "codoxear" / "agent_backend.py"


if __name__ == "__main__":
    unittest.main()
