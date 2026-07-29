import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_RECOVERY_JS = ROOT / "codoxear" / "static" / "app_recovery.js"
APP_TRANSCRIPT_JS = ROOT / "codoxear" / "static" / "app_transcript.js"


class TestFrontendRecoveryModuleSource(unittest.TestCase):
    """Failure projection stays in the ordinary transcript, never a second UI."""

    def test_recovery_asset_does_not_create_a_recovery_controller_or_panel(self) -> None:
        source = APP_RECOVERY_JS.read_text(encoding="utf-8")
        self.assertIn("Lifecycle failures are transcript events supplied by the session API.", source)
        self.assertIn("window.CodoxearRecovery = Object.freeze({});", source)
        self.assertNotIn("createRecoveryPanelController", source)
        self.assertNotIn("recovery-panel", source)
        self.assertNotIn("Recovery needed", source)

    def test_app_does_not_instantiate_or_render_recovery_panels(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertNotIn("recoveryController", source)
        self.assertNotIn("renderRecoveryPanel", source)
        self.assertNotIn("recovery-panel", source)
        self.assertIn("function renderSessionTail(events) {", source)
        self.assertIn("renderTranscript(events, { preserveScroll: false });", source)

    def test_orphan_recovery_uses_the_normal_tail_transcript_route(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        open_start = source.index("async function openSession(sessionId")
        open_block = source[open_start : source.index("async function pollMessages", open_start)]
        self.assertNotIn("if (s && s.orphan_recovery)", open_block)
        self.assertIn("/messages/tail?limit=${initPageLimit()}", open_block)
        self.assertIn("renderSessionTail(Array.isArray(data.events) ? data.events : []);", source)

    def test_transcript_renderer_has_no_recovery_panel_append_hook(self) -> None:
        source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        self.assertNotIn("renderRecoveryPanel", source)
        self.assertNotIn("recovery-panel", source)


if __name__ == "__main__":
    unittest.main()
