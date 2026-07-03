import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_UNATTENDED_JS = ROOT / "codoxear" / "static" / "app_unattended.js"
SERVER_PY = ROOT / "codoxear" / "server.py"
SESSION_LISTING_PY = ROOT / "codoxear" / "session_listing.py"
UNATTENDED_PY = ROOT / "codoxear" / "unattended.py"
CONTROL_ROUTES_PY = ROOT / "codoxear" / "control_routes.py"
SESSION_ROUTES_PY = ROOT / "codoxear" / "session_routes.py"
README = ROOT / "README.md"


class TestUnattendedModeSource(unittest.TestCase):
    def test_app_uses_unattended_user_facing_copy_and_api(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        module = APP_UNATTENDED_JS.read_text(encoding="utf-8")
        # DOM construction + user-facing copy stayed in app.js.
        self.assertIn('title: "Unattended mode"', source)
        self.assertIn('"aria-label": "Unattended mode"', source)
        self.assertIn('"aria-label": "Unattended mode settings"', source)
        self.assertIn('text: "Unattended mode"', source)
        self.assertIn('text: "unattended"', source)
        self.assertNotIn('"Harness mode"', source)
        self.assertNotIn('/harness`', source)
        # The unattended API calls moved into the controller module.
        self.assertIn("api(`/api/sessions/${sid}/unattended`)", module)
        self.assertIn("api(`/api/sessions/${sid}/unattended`, {", module)

    def test_app_delegates_unattended_behavior_to_controller_module(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        module = APP_UNATTENDED_JS.read_text(encoding="utf-8")
        # app.js instantiates the controller and injects the unattended nodes.
        self.assertIn("window.CodoxearUnattended", source)
        self.assertIn("createUnattendedController", source)
        self.assertIn("unattendedController.syncButtonState()", source)
        self.assertIn("unattendedController.isOpen()", source)
        self.assertIn("unattendedController.menuSessionId()", source)
        # Thin delegating wrappers retained for existing callers.
        self.assertIn("return unattendedController.hide(opts);", source)
        self.assertIn("return unattendedController.show(opts);", source)
        self.assertIn("return unattendedController.toggle(opts);", source)
        # The controller owns the open token / session id / focus state.
        self.assertIn("let unattendedMenuToken = 0;", module)
        self.assertIn("let unattendedMenuSessionId = null;", module)
        self.assertIn("let unattendedReturnFocusEl = null;", module)
        # app.js must no longer own the extracted locals/functions.
        self.assertNotIn("let unattendedMenuOpen = false;", source)
        self.assertNotIn("const unattendedSaveTimers = new Map();", source)
        self.assertNotIn("function loadUnattendedCfgForSelected", source)
        self.assertNotIn("function scheduleUnattendedSave", source)

    def test_unattended_button_blocks_failed_launches_in_module(self) -> None:
        module = APP_UNATTENDED_JS.read_text(encoding="utf-8")
        self.assertIn("const unattendedBlocked = Boolean(selected && sessionLaunchFailed(s));", module)
        self.assertIn('unattendedBlocked ? "Failed launch has no unattended mode" : "Unattended mode"', module)
        self.assertIn("unattendedBtn.disabled = !selected || unattendedBlocked;", module)
        self.assertIn('unattendedBtn.setAttribute("aria-label", unattendedLabel);', module)
        self.assertIn('setToast("failed launch has no unattended mode");', module)

    def test_unattended_popover_has_keyboard_and_focus_semantics(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        module = APP_UNATTENDED_JS.read_text(encoding="utf-8")
        # DOM dialog semantics stayed in app.js.
        self.assertIn('"aria-controls": "unattendedMenu"', source)
        self.assertIn('"aria-expanded": "false"', source)
        self.assertIn('"aria-haspopup": "dialog"', source)
        # Focus / disable / token-guard behavior moved into the controller.
        self.assertIn("function setUnattendedMenuExpanded(open)", module)
        self.assertIn('unattendedBtn.setAttribute("aria-expanded", unattendedMenuOpen ? "true" : "false");', module)
        self.assertIn("function restoreUnattendedFocus()", module)
        self.assertIn("restoreModalFocus(target, () => unattendedMenuOpen, requestFrame);", module)
        self.assertIn("function setUnattendedControlsDisabled(disabled)", module)
        self.assertIn("function focusUnattendedInitialControl()", module)
        self.assertIn("target.focus({ preventScroll: true });", module)
        self.assertIn("function hideUnattendedMenu({ restoreFocus = false } = {})", module)
        self.assertIn("unattendedMenuToken += 1;", module)
        self.assertIn("if (restoreFocus && wasOpen) restoreUnattendedFocus();", module)
        self.assertIn("async function loadUnattendedCfgForSelected({ sid = getSelected(), openToken = null } = {})", module)
        self.assertIn("if (openToken !== null && (unattendedMenuToken !== openToken || unattendedMenuSessionId !== sid || !unattendedMenuOpen)) return;", module)
        load_start = module.index("async function loadUnattendedCfgForSelected")
        load_end = module.index("function unattendedSaveSnapshot", load_start)
        load_block = module[load_start:load_end]
        self.assertLess(load_block.index("if (openToken !== null"), load_block.index("unattendedCfg = {"))
        self.assertIn("async function showUnattendedMenu({ opener = null } = {})", module)
        self.assertIn("const openToken = unattendedMenuToken + 1;", module)
        self.assertIn("unattendedMenuSessionId = sid;", module)
        self.assertIn("setUnattendedControlsDisabled(true);", module)
        self.assertIn("if (unattendedMenuOpen && unattendedMenuToken === openToken && unattendedMenuSessionId === sid && getSelected() === sid) {", module)
        self.assertIn("setUnattendedControlsDisabled(false);\n          focusUnattendedInitialControl();", module)
        self.assertIn("if (unattendedMenuToken !== openToken || unattendedMenuSessionId !== sid || getSelected() !== sid) return;", module)
        self.assertIn("function toggleUnattendedMenu({ opener = null } = {})", module)
        self.assertIn("toggleUnattendedMenu({ opener: e.currentTarget });", module)
        self.assertIn("const onUnattendedKeydown = (e) => {", module)
        self.assertIn('if (e.key !== "Escape" || !unattendedMenuOpen) return;', module)
        self.assertIn('hideUnattendedMenu({ restoreFocus: true });', module)
        self.assertIn('addAppEvent(documentTarget, "keydown", onUnattendedKeydown, true);', module)
        # The button onclick + menu click-stop are wired inside the controller.
        self.assertIn("unattendedBtn.onclick = (e) => {", module)
        self.assertIn("unattendedMenu.onclick = (e) => e.stopPropagation();", module)
        # Hide-when-session-changes guard is owned by the controller projection.
        self.assertIn("if (unattendedMenuOpen && (!selected || unattendedMenuSessionId !== selected)) hideUnattendedMenu();", module)
        # app.js callers delegate via isOpen()/menuSessionId().
        self.assertIn("if (unattendedController.isOpen()) hideUnattendedMenu();", source)
        self.assertIn("if (unattendedController.isOpen() && unattendedController.menuSessionId() !== sessionId) hideUnattendedMenu();", source)

    def test_unattended_saves_are_session_scoped_and_queued(self) -> None:
        module = APP_UNATTENDED_JS.read_text(encoding="utf-8")
        self.assertIn("const unattendedSaveTimers = new Map();", module)
        self.assertIn("const unattendedSaveInFlight = new Map();", module)
        self.assertIn("const unattendedSavePending = new Map();", module)
        self.assertIn("function unattendedSaveSnapshot(patch = {})", module)
        self.assertIn('if (has("remaining_injections")) {', module)
        self.assertIn('if (Number.isFinite(remaining) && remaining <= 0) out.enabled = false;', module)
        self.assertIn('out.enabled = Boolean(patch.enabled) && Number.isFinite(remaining) && remaining > 0;', module)
        self.assertIn("function applySavedUnattendedCfg(saved, sid)", module)
        apply_start = module.index("function applySavedUnattendedCfg(saved, sid)")
        apply_end = module.index("async function flushUnattendedSave", apply_start)
        apply_block = module[apply_start:apply_end]
        self.assertIn("if (getSelected() !== sid) return;", apply_block)
        self.assertIn("if (unattendedMenuOpen && unattendedMenuSessionId !== sid) return;", apply_block)
        self.assertIn("s.unattended_enabled = Boolean(saved.enabled);", apply_block)
        self.assertIn("s.unattended_remaining_injections = saved.remaining_injections;", apply_block)
        self.assertIn("syncUnattendedNumberInputs();", apply_block)
        self.assertIn("async function flushUnattendedSave(sid)", module)
        flush_start = module.index("async function flushUnattendedSave(sid)")
        flush_end = module.index("function scheduleUnattendedSave", flush_start)
        flush_block = module[flush_start:flush_end]
        self.assertIn("if (!sid || isAppDisposed() || unattendedSaveInFlight.get(sid)) return;", flush_block)
        self.assertIn("const snapshot = unattendedSavePending.get(sid);", flush_block)
        self.assertIn("unattendedSavePending.delete(sid);", flush_block)
        self.assertIn("unattendedSaveInFlight.set(sid, true);", flush_block)
        self.assertIn("body: snapshot", flush_block)
        self.assertIn("if (!unattendedSavePending.has(sid)) applySavedUnattendedCfg(saved, sid);", flush_block)
        self.assertIn("if (!isAppDisposed() && unattendedSavePending.has(sid)) void flushUnattendedSave(sid);", flush_block)
        schedule_start = module.index("function scheduleUnattendedSave(patch = {})")
        schedule_end = module.index("function projectButtonState", schedule_start)
        schedule_block = module[schedule_start:schedule_end]
        self.assertIn("const snapshot = unattendedSaveSnapshot(patch);", schedule_block)
        self.assertIn("if (!Object.keys(snapshot).length) return;", schedule_block)
        self.assertIn("unattendedSavePending.set(sid, { ...(unattendedSavePending.get(sid) || {}), ...snapshot });", schedule_block)
        self.assertIn("const existing = unattendedSaveTimers.get(sid);", schedule_block)
        self.assertIn("unattendedSaveTimers.set(sid, timer);", schedule_block)
        self.assertNotIn("if (getSelected() !== sid) return;", schedule_block)

    def test_app_uses_unattended_session_fields_without_harness_fallback(self) -> None:
        module = APP_UNATTENDED_JS.read_text(encoding="utf-8")
        self.assertIn('s.unattended_enabled', module)
        self.assertIn('s.unattended_cooldown_minutes', module)
        self.assertIn('s.unattended_remaining_injections', module)
        self.assertIn('s.unattended_enabled = unattendedCfg.enabled;', module)
        self.assertIn('s.unattended_remaining_injections = value;', module)
        self.assertIn('scheduleUnattendedSave({ enabled: unattendedCfg.enabled });', module)
        self.assertIn('scheduleUnattendedSave({ cooldown_minutes: value });', module)
        self.assertIn('scheduleUnattendedSave({ remaining_injections: value, ...(value <= 0 ? { enabled: false } : {}) });', module)
        self.assertIn('scheduleUnattendedSave({ request: unattendedCfg.request });', module)
        self.assertNotIn('s.unattended_enabled ?? s.harness_enabled', module)
        self.assertNotIn('s.unattended_cooldown_minutes ?? s.harness_cooldown_minutes', module)
        self.assertNotIn('s.unattended_remaining_injections ?? s.harness_remaining_injections', module)

    def test_server_exposes_unattended_route_and_fields_without_harness_alias(self) -> None:
        source = SERVER_PY.read_text(encoding="utf-8")
        listing_source = SESSION_LISTING_PY.read_text(encoding="utf-8")
        route_source = CONTROL_ROUTES_PY.read_text(encoding="utf-8")
        session_route_source = SESSION_ROUTES_PY.read_text(encoding="utf-8")
        combined_route_source = source + listing_source + route_source + session_route_source
        self.assertIn('match_session_route(path, "unattended")', combined_route_source)
        self.assertIn('("unattended", None, _handle_unattended)', route_source)
        self.assertNotIn('path.endswith("/harness") or path.endswith("/unattended")', combined_route_source)
        self.assertIn('unattended_enabled = bool(cfg0.get("enabled")) and unattended_remaining_injections > 0', listing_source)
        self.assertIn('unattended_enabled=unattended_enabled', listing_source)
        self.assertIn('elif isinstance(enabled_raw, bool):', route_source)
        self.assertIn('"enabled must be a boolean"', route_source)
        self.assertIn('unattended_cooldown_minutes=unattended_cooldown_minutes', listing_source)
        self.assertIn('unattended_remaining_injections=unattended_remaining_injections', listing_source)
        self.assertIn('"unattended_enabled": facts.unattended_enabled', listing_source)
        self.assertIn('"unattended_cooldown_minutes": facts.unattended_cooldown_minutes', listing_source)
        self.assertIn('"unattended_remaining_injections": facts.unattended_remaining_injections', listing_source)
        self.assertIn('"unattended_enabled": False', listing_source)
        self.assertIn('"unattended_cooldown_minutes": unattended_default_idle_minutes', listing_source)
        self.assertIn('"unattended_remaining_injections": unattended_default_max_injections', listing_source)
        self.assertNotIn('"harness_enabled": h_enabled', combined_route_source)
        self.assertNotIn('"harness_cooldown_minutes": h_cooldown_minutes', combined_route_source)
        self.assertNotIn('"harness_remaining_injections": h_remaining', combined_route_source)

    def test_api_validation_errors_use_unattended_term_for_user_inputs(self) -> None:
        source = SERVER_PY.read_text(encoding="utf-8")
        config_source = (SERVER_PY.parent / "server_config.py").read_text(encoding="utf-8")
        unattended_source = UNATTENDED_PY.read_text(encoding="utf-8")
        self.assertIn('"unattended cooldown_minutes must be an integer"', unattended_source)
        self.assertIn('"unattended remaining_injections must be an integer"', unattended_source)
        self.assertIn('UNATTENDED_PATH=app_dir / "unattended.json"', config_source)
        self.assertIn("_export_server_config(globals(), _SERVER_CONFIG)", source)
        self.assertIn('CODEX_WEB_UNATTENDED_SWEEP_SECONDS', config_source)
        self.assertNotIn('"harness cooldown_minutes must', source + unattended_source)
        self.assertNotIn('"harness remaining_injections must', source + unattended_source)
        self.assertNotIn('APP_DIR / "harness.json"', source + config_source)
        self.assertNotIn('CODEX_WEB_HARNESS_SWEEP_SECONDS', source + config_source)

    def test_readme_documents_unattended_mode_not_harness_mode(self) -> None:
        readme = README.read_text(encoding="utf-8")
        self.assertIn("Enable Unattended mode", readme)
        self.assertIn("CODEX_WEB_UNATTENDED_SWEEP_SECONDS", readme)
        self.assertNotIn("Harness mode", readme)
        self.assertNotIn("CODEX_WEB_HARNESS_SWEEP_SECONDS", readme)


if __name__ == "__main__":
    unittest.main()
