import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
SERVER_PY = ROOT / "codoxear" / "server.py"
UNATTENDED_PY = ROOT / "codoxear" / "unattended.py"
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

    def test_unattended_popover_has_keyboard_and_focus_semantics(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('"aria-controls": "unattendedMenu"', source)
        self.assertIn('"aria-expanded": "false"', source)
        self.assertIn('"aria-haspopup": "dialog"', source)
        self.assertIn("let unattendedMenuToken = 0;", source)
        self.assertIn("let unattendedMenuSessionId = null;", source)
        self.assertIn("let unattendedReturnFocusEl = null;", source)
        self.assertIn("function setUnattendedMenuExpanded(open)", source)
        self.assertIn('unattendedBtn.setAttribute("aria-expanded", unattendedMenuOpen ? "true" : "false");', source)
        self.assertIn("function restoreUnattendedFocus()", source)
        self.assertIn("restoreModalFocus(target, () => unattendedMenuOpen);", source)
        self.assertIn("function focusUnattendedInitialControl()", source)
        self.assertIn('const target = $("#unattendedEnabled") || unattendedMenu;', source)
        self.assertIn("target.focus({ preventScroll: true });", source)
        self.assertIn("function hideUnattendedMenu({ restoreFocus = false } = {})", source)
        self.assertIn("unattendedMenuToken += 1;", source)
        self.assertIn("unattendedMenuSessionId = null;", source)
        self.assertIn("if (restoreFocus && wasOpen) restoreUnattendedFocus();", source)
        self.assertIn("async function loadUnattendedCfgForSelected({ sid = selected, openToken = null } = {})", source)
        self.assertIn("if (openToken !== null && (unattendedMenuToken !== openToken || unattendedMenuSessionId !== sid || !unattendedMenuOpen)) return;", source)
        load_start = source.index("async function loadUnattendedCfgForSelected")
        load_end = source.index("function scheduleUnattendedSave", load_start)
        load_block = source[load_start:load_end]
        self.assertLess(load_block.index("if (openToken !== null"), load_block.index("unattendedCfg = {"))
        self.assertIn("async function showUnattendedMenu({ opener = null } = {})", source)
        self.assertIn("const sid = selected;", source)
        self.assertIn("const openToken = unattendedMenuToken + 1;", source)
        self.assertIn("unattendedMenuSessionId = sid;", source)
        self.assertIn("unattendedReturnFocusEl = opener instanceof HTMLElement ? opener : document.activeElement instanceof HTMLElement ? document.activeElement : null;", source)
        self.assertIn("if (unattendedMenuOpen && unattendedMenuToken === openToken && unattendedMenuSessionId === sid && selected === sid) focusUnattendedInitialControl();", source)
        self.assertIn("if (unattendedMenuToken !== openToken || unattendedMenuSessionId !== sid || selected !== sid) return;", source)
        self.assertIn("function toggleUnattendedMenu({ opener = null } = {})", source)
        self.assertIn("toggleUnattendedMenu({ opener: e.currentTarget });", source)
        self.assertIn('const onUnattendedKeydown = (e) => {', source)
        self.assertIn('if (e.key !== "Escape" || !unattendedMenuOpen) return;', source)
        self.assertIn('hideUnattendedMenu({ restoreFocus: true });', source)
        self.assertIn('addAppEvent(document, "keydown", onUnattendedKeydown, true);', source)
        self.assertIn("if (unattendedMenuOpen && (!selected || unattendedMenuSessionId !== selected)) hideUnattendedMenu();", source)
        self.assertIn("if (unattendedMenuOpen && unattendedMenuSessionId !== sessionId) hideUnattendedMenu();", source)
        self.assertIn("if (unattendedMenuOpen) hideUnattendedMenu();", source)

    def test_app_uses_unattended_session_fields_without_harness_fallback(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('s.unattended_enabled', source)
        self.assertIn('s.unattended_cooldown_minutes', source)
        self.assertIn('s.unattended_remaining_injections', source)
        self.assertIn('s.unattended_enabled = unattendedCfg.enabled;', source)
        self.assertIn('s.unattended_remaining_injections = value;', source)
        self.assertNotIn('s.unattended_enabled ?? s.harness_enabled', source)
        self.assertNotIn('s.unattended_cooldown_minutes ?? s.harness_cooldown_minutes', source)
        self.assertNotIn('s.unattended_remaining_injections ?? s.harness_remaining_injections', source)

    def test_server_exposes_unattended_route_and_fields_without_harness_alias(self) -> None:
        source = SERVER_PY.read_text(encoding="utf-8")
        self.assertIn('_match_session_route(path, "unattended")', source)
        self.assertNotIn('path.endswith("/harness") or path.endswith("/unattended")', source)
        self.assertIn('"unattended_enabled": unattended_enabled', source)
        self.assertIn('"unattended_cooldown_minutes": unattended_cooldown_minutes', source)
        self.assertIn('"unattended_remaining_injections": unattended_remaining_injections', source)
        self.assertIn('"unattended_enabled": False', source)
        self.assertNotIn('"harness_enabled": h_enabled', source)
        self.assertNotIn('"harness_cooldown_minutes": h_cooldown_minutes', source)
        self.assertNotIn('"harness_remaining_injections": h_remaining_injections', source)

    def test_api_validation_errors_use_unattended_term_for_user_inputs(self) -> None:
        source = SERVER_PY.read_text(encoding="utf-8")
        unattended_source = UNATTENDED_PY.read_text(encoding="utf-8")
        self.assertIn('"unattended cooldown_minutes must be an integer"', unattended_source)
        self.assertIn('"unattended remaining_injections must be an integer"', unattended_source)
        self.assertIn('APP_DIR / "unattended.json"', source)
        self.assertIn('CODEX_WEB_UNATTENDED_SWEEP_SECONDS', source)
        self.assertNotIn('"harness cooldown_minutes must', source + unattended_source)
        self.assertNotIn('"harness remaining_injections must', source + unattended_source)
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
