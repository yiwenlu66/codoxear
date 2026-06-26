import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
SERVER_PY = ROOT / "codoxear" / "server.py"
SESSION_LISTING_PY = ROOT / "codoxear" / "session_listing.py"
UNATTENDED_PY = ROOT / "codoxear" / "unattended.py"
CONTROL_ROUTES_PY = ROOT / "codoxear" / "control_routes.py"
SESSION_ROUTES_PY = ROOT / "codoxear" / "session_routes.py"
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
        self.assertIn("function setUnattendedControlsDisabled(disabled)", source)
        self.assertIn('["unattendedEnabled", "unattendedCooldownMinutes", "unattendedRemainingInjections", "unattendedRequest"].forEach', source)
        self.assertIn("if (node) node.disabled = value;", source)
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
        self.assertIn("setUnattendedControlsDisabled(true);", source)
        self.assertIn("if (unattendedMenuOpen && unattendedMenuToken === openToken && unattendedMenuSessionId === sid && selected === sid) {", source)
        self.assertIn("setUnattendedControlsDisabled(false);\n              focusUnattendedInitialControl();", source)
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
        self.assertIn("unattendedCfg.enabled = false;", source)
        self.assertIn('const enabledEl = $("#unattendedEnabled");\n                if (enabledEl) enabledEl.checked = false;', source)

    def test_unattended_saves_are_session_scoped_and_queued(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("const unattendedSaveTimers = new Map();", source)
        self.assertIn("const unattendedSaveInFlight = new Map();", source)
        self.assertIn("const unattendedSavePending = new Map();", source)
        self.assertIn("function unattendedSaveSnapshot(patch = {})", source)
        self.assertIn('if (has("remaining_injections")) {', source)
        self.assertIn('if (Number.isFinite(remaining) && remaining <= 0) out.enabled = false;', source)
        self.assertIn('out.enabled = Boolean(patch.enabled) && Number.isFinite(remaining) && remaining > 0;', source)
        self.assertIn("function applySavedUnattendedCfg(saved, sid)", source)
        apply_start = source.index("function applySavedUnattendedCfg(saved, sid)")
        apply_end = source.index("async function flushUnattendedSave", apply_start)
        apply_block = source[apply_start:apply_end]
        self.assertIn("if (selected !== sid) return;", apply_block)
        self.assertIn("if (unattendedMenuOpen && unattendedMenuSessionId !== sid) return;", apply_block)
        self.assertIn("const s = sessionIndex.get(sid);", apply_block)
        self.assertIn("s.unattended_enabled = Boolean(saved.enabled);", apply_block)
        self.assertIn("s.unattended_remaining_injections = saved.remaining_injections;", apply_block)
        self.assertIn("syncUnattendedNumberInputs();", apply_block)
        self.assertIn('const enabledEl = $("#unattendedEnabled");', apply_block)
        self.assertIn('const requestEl = $("#unattendedRequest");', apply_block)
        self.assertIn("async function flushUnattendedSave(sid)", source)
        flush_start = source.index("async function flushUnattendedSave(sid)")
        flush_end = source.index("function scheduleUnattendedSave", flush_start)
        flush_block = source[flush_start:flush_end]
        self.assertIn("if (!sid || appDisposed || unattendedSaveInFlight.get(sid)) return;", flush_block)
        self.assertIn("const snapshot = unattendedSavePending.get(sid);", flush_block)
        self.assertIn("unattendedSavePending.delete(sid);", flush_block)
        self.assertIn("unattendedSaveInFlight.set(sid, true);", flush_block)
        self.assertIn("body: snapshot", flush_block)
        self.assertIn("if (!unattendedSavePending.has(sid)) applySavedUnattendedCfg(saved, sid);", flush_block)
        self.assertIn("if (!appDisposed && unattendedSavePending.has(sid)) void flushUnattendedSave(sid);", flush_block)
        schedule_start = source.index("function scheduleUnattendedSave(patch = {})")
        schedule_end = source.index("function setUnattendedMenuExpanded", schedule_start)
        schedule_block = source[schedule_start:schedule_end]
        self.assertIn("const snapshot = unattendedSaveSnapshot(patch);", schedule_block)
        self.assertIn("if (!Object.keys(snapshot).length) return;", schedule_block)
        self.assertIn("unattendedSavePending.set(sid, { ...(unattendedSavePending.get(sid) || {}), ...snapshot });", schedule_block)
        self.assertIn("const existing = unattendedSaveTimers.get(sid);", schedule_block)
        self.assertIn("unattendedSaveTimers.set(sid, timer);", schedule_block)
        self.assertNotIn("if (selected !== sid) return;", schedule_block)

    def test_app_uses_unattended_session_fields_without_harness_fallback(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('s.unattended_enabled', source)
        self.assertIn('s.unattended_cooldown_minutes', source)
        self.assertIn('s.unattended_remaining_injections', source)
        self.assertIn('s.unattended_enabled = unattendedCfg.enabled;', source)
        self.assertIn('s.unattended_remaining_injections = value;', source)
        self.assertIn('scheduleUnattendedSave({ enabled: unattendedCfg.enabled });', source)
        self.assertIn('scheduleUnattendedSave({ cooldown_minutes: value });', source)
        self.assertIn('scheduleUnattendedSave({ remaining_injections: value, ...(value <= 0 ? { enabled: false } : {}) });', source)
        self.assertIn('scheduleUnattendedSave({ request: unattendedCfg.request });', source)
        self.assertNotIn('s.unattended_enabled ?? s.harness_enabled', source)
        self.assertNotIn('s.unattended_cooldown_minutes ?? s.harness_cooldown_minutes', source)
        self.assertNotIn('s.unattended_remaining_injections ?? s.harness_remaining_injections', source)

    def test_server_exposes_unattended_route_and_fields_without_harness_alias(self) -> None:
        source = SERVER_PY.read_text(encoding="utf-8")
        listing_source = SESSION_LISTING_PY.read_text(encoding="utf-8")
        route_source = CONTROL_ROUTES_PY.read_text(encoding="utf-8")
        session_route_source = SESSION_ROUTES_PY.read_text(encoding="utf-8")
        combined_route_source = source + listing_source + route_source + session_route_source
        self.assertIn('match_session_route(path, "unattended")', combined_route_source)
        self.assertIn('("unattended", None, _handle_unattended)', route_source)
        self.assertNotIn('path.endswith("/harness") or path.endswith("/unattended")', combined_route_source)
        self.assertIn('unattended_enabled = bool(cfg0.get("enabled")) and unattended_remaining_injections > 0', source)
        self.assertIn('unattended_enabled=unattended_enabled', source)
        self.assertIn('elif isinstance(enabled_raw, bool):', route_source)
        self.assertIn('"enabled must be a boolean"', route_source)
        self.assertIn('unattended_cooldown_minutes=unattended_cooldown_minutes', source)
        self.assertIn('unattended_remaining_injections=unattended_remaining_injections', source)
        self.assertIn('"unattended_enabled": facts.unattended_enabled', listing_source)
        self.assertIn('"unattended_cooldown_minutes": facts.unattended_cooldown_minutes', listing_source)
        self.assertIn('"unattended_remaining_injections": facts.unattended_remaining_injections', listing_source)
        self.assertIn('"unattended_enabled": False', listing_source)
        self.assertIn('"unattended_cooldown_minutes": unattended_default_idle_minutes', listing_source)
        self.assertIn('"unattended_remaining_injections": unattended_default_max_injections', listing_source)
        self.assertNotIn('"harness_enabled": h_enabled', combined_route_source)
        self.assertNotIn('"harness_cooldown_minutes": h_cooldown_minutes', combined_route_source)
        self.assertNotIn('"harness_remaining_injections": h_remaining_injections', combined_route_source)

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
