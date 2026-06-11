import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
SERVER_PY = ROOT / "codoxear" / "server.py"
README = ROOT / "README.md"


class TestUnattendedModeSource(unittest.TestCase):
    def test_app_uses_unattended_user_facing_copy_and_api(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('title: "Unattended mode"', source)
        self.assertIn('"aria-label": "Unattended mode"', source)
        self.assertIn('"aria-label": "Unattended mode settings"', source)
        self.assertIn('text: "Unattended mode"', source)
        self.assertIn('api(`/api/sessions/${sid}/unattended`)', source)
        self.assertIn('api(`/api/sessions/${sid}/unattended`, {', source)
        self.assertIn('text: "unattended"', source)
        self.assertNotIn('"Harness mode"', source)
        self.assertNotIn('/harness`', source)

    def test_app_uses_unattended_session_fields_without_harness_fallback(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('s.unattended_enabled', source)
        self.assertIn('s.unattended_cooldown_minutes', source)
        self.assertIn('s.unattended_remaining_injections', source)
        self.assertIn('s.unattended_enabled = harnessCfg.enabled;', source)
        self.assertIn('s.unattended_remaining_injections = value;', source)
        self.assertNotIn('s.unattended_enabled ?? s.harness_enabled', source)
        self.assertNotIn('s.unattended_cooldown_minutes ?? s.harness_cooldown_minutes', source)
        self.assertNotIn('s.unattended_remaining_injections ?? s.harness_remaining_injections', source)

    def test_server_exposes_unattended_route_and_fields_without_harness_alias(self) -> None:
        source = SERVER_PY.read_text(encoding="utf-8")
        self.assertIn('path.endswith("/unattended")', source)
        self.assertNotIn('path.endswith("/harness") or path.endswith("/unattended")', source)
        self.assertIn('"unattended_enabled": h_enabled', source)
        self.assertIn('"unattended_cooldown_minutes": h_cooldown_minutes', source)
        self.assertIn('"unattended_remaining_injections": h_remaining_injections', source)
        self.assertIn('"unattended_enabled": False', source)
        self.assertNotIn('"harness_enabled": h_enabled', source)
        self.assertNotIn('"harness_cooldown_minutes": h_cooldown_minutes', source)
        self.assertNotIn('"harness_remaining_injections": h_remaining_injections', source)

    def test_api_validation_errors_use_unattended_term_for_user_inputs(self) -> None:
        source = SERVER_PY.read_text(encoding="utf-8")
        self.assertIn('"unattended cooldown_minutes must be an integer"', source)
        self.assertIn('"unattended remaining_injections must be an integer"', source)
        self.assertIn('APP_DIR / "unattended.json"', source)
        self.assertIn('CODEX_WEB_UNATTENDED_SWEEP_SECONDS', source)
        self.assertNotIn('"harness cooldown_minutes must', source)
        self.assertNotIn('"harness remaining_injections must', source)
        self.assertNotIn('APP_DIR / "harness.json"', source)
        self.assertNotIn('CODEX_WEB_HARNESS_SWEEP_SECONDS', source)

    def test_readme_documents_unattended_mode_not_harness_mode(self) -> None:
        readme = README.read_text(encoding="utf-8")
        self.assertIn("Enable Unattended mode", readme)
        self.assertIn("CODEX_WEB_UNATTENDED_SWEEP_SECONDS", readme)
        self.assertNotIn("Harness mode", readme)
        self.assertNotIn("CODEX_WEB_HARNESS_SWEEP_SECONDS", readme)


if __name__ == "__main__":
    unittest.main()
