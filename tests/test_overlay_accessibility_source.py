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

    def test_settings_dialog_uses_modal_and_cancel_semantics(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        show_start = source.index("function showVoiceSettingsDialog()")
        show_end = source.index("function hideVoiceSettingsDialog()", show_start)
        show_block = source[show_start:show_end]
        hide_start = show_end
        hide_end = source.index("announceBtn.onclick", hide_start)
        hide_block = source[hide_start:hide_end]
        self.assertIn("if (!voiceSettingsViewer.open) voiceSettingsViewer.showModal();", show_block)
        self.assertIn("if (voiceSettingsViewer.open) voiceSettingsViewer.close();", hide_block)
        self.assertIn('voiceSettingsViewer.addEventListener("cancel", (e) => {', source)
        self.assertIn("e.preventDefault();\n          hideVoiceSettingsDialog();", source)
        self.assertIn('if (voiceSettingsViewer.style.display === "flex") hideVoiceSettingsDialog();', source)

    def test_help_copy_lists_claude_backend(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("Right now that is <b>Codex</b>, <b>Pi</b>, and <b>Claude</b>.", source)
        self.assertNotIn("Right now that is <b>Codex</b> and <b>Pi</b>.", source)


if __name__ == "__main__":
    unittest.main()
