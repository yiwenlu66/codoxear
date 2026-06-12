import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"


class TestOverlayAccessibilitySource(unittest.TestCase):
    def test_custom_modals_isolate_background_app(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("const modalIsolationTargets = [", source)
        self.assertIn('app.toggleAttribute("inert", active);', source)
        self.assertIn('app.setAttribute("aria-hidden", "true");', source)
        self.assertIn("function prepareModalOpen(options = {})", source)
        self.assertIn("closeTransientOverlays(options);", source)

    def test_modal_openers_close_transient_overlays(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("if (unattendedMenuOpen) hideUnattendedMenu();", source)
        self.assertIn('if (document.body.classList.contains("sidebar-open")) setSidebarOpen(false);', source)
        self.assertIn('fileMenuOpen = false;', source)
        self.assertIn('prepareModalOpen();\n          helpBackdrop.style.display = "block";', source)
        self.assertIn('prepareModalOpen();\n          queueViewerSid = selected;', source)
        self.assertIn('prepareModalOpen();\n          voiceSettingsBackdrop.style.display = "block";', source)

    def test_help_copy_lists_claude_backend(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("Right now that is <b>Codex</b>, <b>Pi</b>, and <b>Claude</b>.", source)
        self.assertNotIn("Right now that is <b>Codex</b> and <b>Pi</b>.", source)


if __name__ == "__main__":
    unittest.main()
