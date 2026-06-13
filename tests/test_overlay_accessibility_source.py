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
        self.assertIn('prepareModalOpen();\n          voiceSettingsReturnFocusEl = document.activeElement instanceof HTMLElement ? document.activeElement : null;\n          voiceSettingsBackdrop.style.display = "block";', source)

    def test_settings_dialog_uses_modal_and_cancel_semantics(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        show_start = source.index("function showVoiceSettingsDialog()")
        show_end = source.index("function hideVoiceSettingsDialog()", show_start)
        show_block = source[show_start:show_end]
        hide_start = show_end
        hide_end = source.index("announceBtn.onclick", hide_start)
        hide_block = source[hide_start:hide_end]
        self.assertIn("voiceSettingsReturnFocusEl = document.activeElement instanceof HTMLElement ? document.activeElement : null;", show_block)
        self.assertIn("if (!voiceSettingsViewer.open) voiceSettingsViewer.showModal();", show_block)
        self.assertIn("const focusTarget = voiceSettingsReturnFocusEl;", hide_block)
        self.assertIn("voiceSettingsReturnFocusEl = null;", hide_block)
        self.assertIn("if (voiceSettingsViewer.open) voiceSettingsViewer.close();", hide_block)
        self.assertIn('requestAnimationFrame(() => focusTarget.focus({ preventScroll: true }));', hide_block)
        self.assertIn('voiceSettingsViewer.addEventListener("cancel", (e) => {', source)
        self.assertIn("e.preventDefault();\n          hideVoiceSettingsDialog();", source)
        self.assertIn('if (voiceSettingsViewer.style.display === "flex") hideVoiceSettingsDialog();', source)

    def test_new_session_dialog_restores_focus_and_sets_initial_focus(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('role: "dialog", "aria-modal": "true", "aria-label": "New session"', source)
        self.assertIn("let newSessionReturnFocusEl = null;", source)
        open_start = source.index("function openNewSessionDialog(")
        open_end = source.index("editPriorityRange.oninput", open_start)
        open_block = source[open_start:open_end]
        self.assertIn("newSessionReturnFocusEl = document.activeElement instanceof HTMLElement ? document.activeElement : null;", open_block)
        self.assertIn("prepareModalOpen();", open_block)
        self.assertIn("focusNewSessionInitialControl();", open_block)
        self.assertIn("function focusNewSessionInitialControl()", source)
        focus_start = source.index("function focusNewSessionInitialControl()")
        focus_end = source.index("function hideNewSessionDialog", focus_start)
        focus_block = source[focus_start:focus_end]
        self.assertIn("const target = isMobile() ? newSessionCloseBtn : newSessionCwdInput;", focus_block)
        self.assertIn("target.focus({ preventScroll: true });", focus_block)
        self.assertIn("newSessionCwdInput.setSelectionRange(end, end);", focus_block)
        hide_start = source.index("function hideNewSessionDialog")
        hide_end = source.index("function openNewSessionDialog", hide_start)
        hide_block = source[hide_start:hide_end]
        self.assertIn("const wasOpen = isModalTargetOpen(newSessionViewer);", hide_block)
        self.assertIn("if (wasOpen) restoreNewSessionFocus();", hide_block)
        restore_start = source.index("function restoreNewSessionFocus()")
        restore_end = source.index("function focusNewSessionInitialControl()", restore_start)
        restore_block = source[restore_start:restore_end]
        self.assertIn("const target = newSessionReturnFocusEl;", restore_block)
        self.assertIn("newSessionReturnFocusEl = null;", restore_block)
        self.assertIn("target.focus({ preventScroll: true });", restore_block)
        self.assertIn("if (isModalTargetOpen(newSessionViewer)) return;", restore_block)
        self.assertIn('if (newSessionViewer.style.display === "flex") hideNewSessionDialog();', source)
        self.assertIn('if (brokerPid) hideNewSessionDialog();', source)

    def test_help_copy_lists_claude_backend(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("Right now that is <b>Codex</b>, <b>Pi</b>, and <b>Claude</b>.", source)
        self.assertNotIn("Right now that is <b>Codex</b> and <b>Pi</b>.", source)


if __name__ == "__main__":
    unittest.main()
