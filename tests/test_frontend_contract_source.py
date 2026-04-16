import unittest
from pathlib import Path

SERVER = Path(__file__).resolve().parents[1] / "codoxear" / "server.py"


class TestFrontendContractSource(unittest.TestCase):
    def test_server_still_exposes_frontend_required_routes(self) -> None:
        source = SERVER.read_text(encoding="utf-8")
        self.assertIn('"/api/sessions"', source)
        self.assertIn('"/api/sessions/bootstrap"', source)
        self.assertIn('"/ui_state"', source)
        self.assertIn('"/ui_response"', source)
        self.assertIn('"/diagnostics"', source)
        self.assertIn('"/queue"', source)
        self.assertIn('"/live"', source)
        self.assertIn('"/workspace"', source)
        self.assertIn('"/details"', source)
        self.assertIn('"/file/list"', source)
