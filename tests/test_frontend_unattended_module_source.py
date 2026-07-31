import json
import os
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_UNATTENDED_JS = ROOT / "codoxear" / "static" / "app_unattended.js"
APP_SESSION_HELPERS_JS = ROOT / "codoxear" / "static" / "app_session_helpers.js"
APP_MODAL_JS = ROOT / "codoxear" / "static" / "app_modal.js"


def run_node_json(js: str) -> dict:
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={"PATH": os.environ.get("PATH", ""), "TZ": "UTC"},
    )
    return json.loads(proc.stdout)


HARNESS = r"""
const vm = require("vm");
const calls = [];
const toasts = [];
const rafCalls = [];
const shellProjections = [];
const sessions = new Map();
let selected = null;
let disposed = false;

// Per-target/type event handler registry populated by addAppEvent so tests can
// synthesize Escape / click / resize.
const eventRegistry = new Map();
function addAppEvent(target, type, handler, options) {
  calls.push(["addAppEvent", type, !!options]);
  const key = `${target && target.__id || "doc"}:${type}`;
  eventRegistry.set(key, handler);
  return handler;
}
function dispatchEvent(target, type, event) {
  const key = `${target && target.__id || "doc"}:${type}`;
  const handler = eventRegistry.get(key);
  if (handler) handler(event || {});
}

// Controllable API response queue. Each entry is a value, an Error, or a
// { promise } wrapper for in-flight control.
let apiResponseQueue = [];
function setApiResponses(list) { apiResponseQueue = list.slice(); }

const pendingTimers = new Map();
let timerHandle = 0;
function fakeSetTimeout(fn, ms) {
  calls.push(["setTimeout", ms]);
  const handle = ++timerHandle;
  pendingTimers.set(handle, fn);
  return handle;
}
function fakeClearTimeout(handle) {
  pendingTimers.delete(handle);
  calls.push(["clearTimeout", handle]);
}
function runPendingTimers() {
  let guard = 0;
  while (pendingTimers.size && guard < 50) {
    guard += 1;
    const [handle, fn] = Array.from(pendingTimers.entries())[0];
    pendingTimers.delete(handle);
    fn();
  }
}

function fakeNode(extra = {}) {
  return {
    style: {},
    _attrs: {},
    _children: [],
    value: "",
    checked: false,
    disabled: false,
    textContent: "",
    offsetWidth: 0,
    isConnected: true,
    set innerHTML(v) { this._children = []; },
    get innerHTML() { return ""; },
    setAttribute(name, value) { this._attrs[name] = String(value); },
    getAttribute(name) { return this._attrs[name]; },
    removeAttribute(name) { delete this._attrs[name]; },
    appendChild(child) { this._children.push(child); return child; },
    addEventListener() {},
    focus(opts) { calls.push(["focus", this.__id || "node"]); },
    getBoundingClientRect() { return { bottom: 100, right: 200, width: 60, height: 30 }; },
    classList: { add() {}, remove() {}, toggle(c, f) { this._last = c; this._flag = f; }, contains() { return false; } },
    ...extra,
  };
}

const unattendedBtn = fakeNode({ __id: "unattendedBtn" });
const unattendedMenu = fakeNode({ __id: "unattendedMenu", offsetWidth: 300 });
const enabledEl = fakeNode({ __id: "unattendedEnabled" });
const cooldownEl = fakeNode({ __id: "unattendedCooldownMinutes" });
const remainingEl = fakeNode({ __id: "unattendedRemainingInjections" });
const requestEl = fakeNode({ __id: "unattendedRequest" });

const documentTarget = { __id: "doc", activeElement: null };
const windowTarget = { __id: "win", innerHeight: 600, innerWidth: 400 };

const deps = {
  unattendedBtn,
  unattendedMenu,
  enabledEl,
  cooldownEl,
  remainingEl,
  requestEl,
  getSelected: () => selected,
  getSessionInfo: (sid) => sessions.get(sid) || null,
  isAppDisposed: () => disposed,
  api: (url, options = {}) => {
    const body = options && options.body ? JSON.parse(JSON.stringify(options.body)) : null;
    calls.push(["api", url, body]);
    if (apiResponseQueue.length) {
      const next = apiResponseQueue.shift();
      if (next instanceof Error) return Promise.reject(next);
      if (next && next.__promise) return next.__promise;
      return Promise.resolve(next);
    }
    return Promise.resolve({});
  },
  refreshSessions: async () => { calls.push(["refreshSessions"]); },
  handleAppAuthLoss: () => { calls.push(["handleAppAuthLoss"]); },
  setToast: (t) => { toasts.push(t); calls.push(["setToast", t]); },
  addAppEvent,
  documentTarget,
  windowTarget,
  requestFrame: (fn) => { rafCalls.push(fn); fn(); },
  setTimeout: fakeSetTimeout,
  clearTimeout: fakeClearTimeout,
  requestShellProjection: () => { shellProjections.push(1); calls.push(["shellProjection"]); },
};

const ctx = {
  HTMLElement: function HTMLElement() {},
  document: documentTarget,
  window: windowTarget,
  console,
};
vm.createContext(ctx);
vm.runInContext(MODAL_SOURCE, ctx);
vm.runInContext(HELPERS_SOURCE, ctx);
vm.runInContext(UNATTENDED_SOURCE, ctx);
const controller = ctx.window.CodoxearUnattended.createUnattendedController(deps);

globalThis.__harness = {
  controller,
  calls,
  toasts,
  rafCalls,
  shellProjections,
  HTMLElementCtor: ctx.HTMLElement,
  sessions,
  select: (sid) => { selected = sid; },
  setDisposed: (v) => { disposed = v; },
  setApiResponses,
  runPendingTimers,
  pendingTimerCount: () => pendingTimers.size,
  dispatchEvent,
  documentTarget,
  windowTarget,
  dom: { unattendedBtn, unattendedMenu, enabledEl, cooldownEl, remainingEl, requestEl },
  setBtnRect: (rect) => { unattendedBtn.getBoundingClientRect = () => rect; },
  setWindowSize: (h, w) => { windowTarget.innerHeight = h; windowTarget.innerWidth = w; },
  setActiveElement: (el) => { documentTarget.activeElement = el; },
};
"""


def harness_script(epilogue: str) -> str:
    unattended_source = APP_UNATTENDED_JS.read_text(encoding="utf-8")
    helpers_source = APP_SESSION_HELPERS_JS.read_text(encoding="utf-8")
    modal_source = APP_MODAL_JS.read_text(encoding="utf-8")
    js = (
        textwrap.dedent(
            f"""
        const MODAL_SOURCE = {json.dumps(modal_source)};
        const HELPERS_SOURCE = {json.dumps(helpers_source)};
        const UNATTENDED_SOURCE = {json.dumps(unattended_source)};
        """
        )
        + HARNESS
        + "\n(async () => {\n"
        + textwrap.dedent(epilogue)
        + "\n})().then(() => {\n"
        + "  process.stdout.write(JSON.stringify(globalThis.__result || {}));\n"
        + "}).catch((err) => {\n"
        + "  console.error(err && err.stack || err);\n"
        + "  process.exit(1);\n"
        + "});\n"
    )
    return js


class TestFrontendUnattendedModuleBehavior(unittest.TestCase):
    # --- 1. frozen export + missing dep failures ---

    def test_module_export_is_frozen_createUnattended_controller(self) -> None:
        unattended_source = APP_UNATTENDED_JS.read_text(encoding="utf-8")
        helpers_source = APP_SESSION_HELPERS_JS.read_text(encoding="utf-8")
        modal_source = APP_MODAL_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const vm = require("vm");
            const ctx = {{ window: {{}}, HTMLElement: function HTMLElement() {{}} }};
            vm.createContext(ctx);
            vm.runInContext({json.dumps(modal_source)}, ctx);
            vm.runInContext({json.dumps(helpers_source)}, ctx);
            vm.runInContext({json.dumps(unattended_source)}, ctx);
            process.stdout.write(JSON.stringify({{
              frozen: Object.isFrozen(ctx.window.CodoxearUnattended),
              keys: Object.keys(ctx.window.CodoxearUnattended),
              hasCreate: typeof ctx.window.CodoxearUnattended.createUnattendedController === "function",
            }}));
            """
        )
        result = run_node_json(js)
        self.assertTrue(result["frozen"])
        self.assertEqual(result["keys"], ["createUnattendedController"])
        self.assertTrue(result["hasCreate"])

    def test_create_throws_on_missing_deps(self) -> None:
        unattended_source = APP_UNATTENDED_JS.read_text(encoding="utf-8")
        helpers_source = APP_SESSION_HELPERS_JS.read_text(encoding="utf-8")
        modal_source = APP_MODAL_JS.read_text(encoding="utf-8")
        head = textwrap.dedent(
            f"""
            const vm = require("vm");
            const ctx = {{ window: {{}}, HTMLElement: function HTMLElement() {{}} }};
            vm.createContext(ctx);
            vm.runInContext({json.dumps(modal_source)}, ctx);
            vm.runInContext({json.dumps(helpers_source)}, ctx);
            vm.runInContext({json.dumps(unattended_source)}, ctx);
            """
        )
        body = textwrap.dedent(
            r'''
            const U = ctx.window.CodoxearUnattended;
            const errors = [];
            const attempts = [
              ["options not object", () => U.createUnattendedController(null)],
              ["missing DOM node", () => U.createUnattendedController({})],
            ];
            for (const [label, fn] of attempts) {
              try { fn(); errors.push({ label, threw: false }); }
              catch (e) { errors.push({ label, threw: true, type: e.name === "TypeError", msg: String(e.message) }); }
            }
            // A fully-wired controller with one function dep swapped for null
            // must throw a TypeError naming the missing dep.
            const node = { style: {}, setAttribute() {}, appendChild() {}, classList: { toggle() {} }, getBoundingClientRect() {} };
            const wiredExceptApi = {
              unattendedBtn: node, unattendedMenu: node,
              enabledEl: node, cooldownEl: node, remainingEl: node, requestEl: node,
              getSelected: () => null, getSessionInfo: () => null, isAppDisposed: () => false,
              api: null,
              refreshSessions: async () => {}, handleAppAuthLoss: () => {}, setToast: () => {},
              addAppEvent: () => {},
            };
            try {
              U.createUnattendedController(wiredExceptApi);
              errors.push({ label: "missing api", threw: false });
            } catch (e) {
              errors.push({ label: "missing api", threw: true, type: e.name === "TypeError", msg: String(e.message) });
            }
            process.stdout.write(JSON.stringify(errors));
            '''
        )
        result = run_node_json(head + body)
        by_label = {row["label"]: row for row in result}
        self.assertTrue(by_label["options not object"]["threw"])
        self.assertTrue(by_label["options not object"]["type"])
        self.assertTrue(by_label["missing DOM node"]["threw"])
        self.assertTrue(by_label["missing DOM node"]["type"])
        self.assertTrue(by_label["missing api"]["threw"])
        self.assertTrue(by_label["missing api"]["type"])
        self.assertContains("api", by_label["missing api"]["msg"])

    def test_module_load_fails_loud_without_helpers_or_modal(self) -> None:
        unattended_source = APP_UNATTENDED_JS.read_text(encoding="utf-8")
        js_only_unattended = textwrap.dedent(
            f"""
            const vm = require("vm");
            const ctx = {{ window: {{}}, HTMLElement: function HTMLElement() {{}} }};
            vm.createContext(ctx);
            let threw = false;
            let msg = "";
            try {{ vm.runInContext({json.dumps(unattended_source)}, ctx); }}
            catch (e) {{ threw = true; msg = String(e.message); }}
            process.stdout.write(JSON.stringify({{ threw, msg }}));
            """
        )
        result = run_node_json(js_only_unattended)
        self.assertTrue(result["threw"])
        self.assertContains("failed to load", result["msg"])

    # --- 2. button projection: no selected / failed launch / active session ---

    def test_button_projection_for_no_selected_failed_launch_active(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            // No selection.
            h.controller.syncButtonState();
            const none = { disabled: h.dom.unattendedBtn.disabled, title: h.dom.unattendedBtn.title, aria: h.dom.unattendedBtn._attrs["aria-label"], activeFlag: h.dom.unattendedBtn.classList._flag };

            // Failed launch.
            h.sessions.set("sid-1", { launch_state: "failed" });
            h.select("sid-1");
            h.controller.syncButtonState();
            const failed = { disabled: h.dom.unattendedBtn.disabled, title: h.dom.unattendedBtn.title, aria: h.dom.unattendedBtn._attrs["aria-label"], activeFlag: h.dom.unattendedBtn.classList._flag };

            // Active unattended session.
            h.sessions.set("sid-2", { launch_state: "ready", unattended_enabled: true, unattended_cooldown_minutes: 7, unattended_remaining_injections: 3 });
            h.select("sid-2");
            h.controller.syncButtonState();
            const active = { disabled: h.dom.unattendedBtn.disabled, title: h.dom.unattendedBtn.title, aria: h.dom.unattendedBtn._attrs["aria-label"], activeFlag: h.dom.unattendedBtn.classList._flag };
            globalThis.__result = { none, failed, active };
            """
        )
        result = run_node_json(js)
        self.assertTrue(result["none"]["disabled"])
        self.assertEqual(result["none"]["title"], "Select a session for unattended mode")
        self.assertFalse(result["none"]["activeFlag"])
        self.assertTrue(result["failed"]["disabled"])
        self.assertEqual(result["failed"]["title"], "Failed launch has no unattended mode")
        self.assertFalse(result["active"]["disabled"])
        self.assertEqual(result["active"]["title"], "Unattended mode")
        self.assertTrue(result["active"]["activeFlag"])

    # --- 3. show blocked for failed launch (exact toast) + no-selected no-op ---

    def test_show_blocked_for_failed_launch_and_no_selection(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            // No selection: no-op, no toast, menu stays closed.
            h.controller.show({ opener: null });
            const noneOpen = h.controller.isOpen();
            const noneToastCount = h.toasts.length;

            // Failed launch: exact toast, no open.
            h.sessions.set("sid-1", { launch_state: "failed" });
            h.select("sid-1");
            await h.controller.show({ opener: null });
            const failedOpen = h.controller.isOpen();
            globalThis.__result = { noneOpen, noneToastCount, failedOpen, toasts: h.toasts };
            """
        )
        result = run_node_json(js)
        self.assertFalse(result["noneOpen"])
        self.assertEqual(result["noneToastCount"], 0)
        self.assertFalse(result["failedOpen"])
        self.assertEqual(result["toasts"], ["failed launch has no unattended mode"])

    # --- 4. show opens menu, disables controls, positions, fetches, validates,
    #       enables controls, focuses initial control, sets token/session guards ---

    def test_show_opens_menu_loads_and_focuses(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-1", { launch_state: "ready", unattended_enabled: false });
            h.select("sid-1");
            h.setBtnRect({ bottom: 100, right: 200 });
            h.setWindowSize(600, 400);
            const opener = new h.HTMLElementCtor();
            opener.isConnected = true;
            opener.disabled = false;
            opener.focus = () => { h.calls.push(["opener-focus"]); };
            h.setApiResponses([{ enabled: true, request: "please continue", cooldown_minutes: 5, remaining_injections: 4 }]);
            await h.controller.show({ opener });
            const menuSession = h.controller.menuSessionId();
            const open = h.controller.isOpen();
            const display = h.dom.unattendedMenu.style.display;
            const top = h.dom.unattendedMenu.style.top;
            const left = h.dom.unattendedMenu.style.left;
            const right = h.dom.unattendedMenu.style.right;
            const apiCalls = h.calls.filter((c) => c[0] === "api").map((c) => c[1]);
            const enabledChecked = h.dom.enabledEl.checked;
            const cooldownValue = h.dom.cooldownEl.value;
            const remainingValue = h.dom.remainingEl.value;
            const requestValue = h.dom.requestEl.value;
            const focusCalls = h.calls.filter((c) => c[0] === "focus");
            const ariaExpanded = h.dom.unattendedBtn._attrs["aria-expanded"];
            globalThis.__result = { menuSession, open, display, top, left, right, apiCalls, enabledChecked, cooldownValue, remainingValue, requestValue, focusCalls, ariaExpanded };
            """
        )
        result = run_node_json(js)
        self.assertTrue(result["open"])
        self.assertEqual(result["menuSession"], "sid-1")
        self.assertEqual(result["display"], "block")
        self.assertEqual(result["ariaExpanded"], "true")
        # top = min(600-12, 100+8) = 108; left: w=300, max(12, min(400-12-300, 200-300)) = max(12, min(88, -100)) = 12
        self.assertEqual(result["top"], "108px")
        self.assertEqual(result["left"], "12px")
        self.assertEqual(result["right"], "auto")
        self.assertEqual(result["apiCalls"], ["/api/sessions/sid-1/unattended"])
        self.assertTrue(result["enabledChecked"])
        self.assertEqual(result["cooldownValue"], "5")
        self.assertEqual(result["remainingValue"], "4")
        self.assertEqual(result["requestValue"], "please continue")
        # focusUnattendedInitialControl focuses the enabled checkbox via rAF.
        self.assertTrue(any("unattendedEnabled" in (c[1] if len(c) > 1 else "") for c in result["focusCalls"]))

    def test_show_disables_controls_during_load(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-1", { launch_state: "ready" });
            h.select("sid-1");
            let resolveLoad;
            const load = new Promise((resolve) => { resolveLoad = resolve; });
            h.setApiResponses([{ __promise: load }]);
            const p = h.controller.show({ opener: null });
            // While load is pending, controls are disabled.
            const disabledDuring = [
              h.dom.enabledEl.disabled, h.dom.cooldownEl.disabled, h.dom.remainingEl.disabled, h.dom.requestEl.disabled,
            ];
            resolveLoad({ enabled: false, request: "", cooldown_minutes: 5, remaining_injections: 10 });
            await p;
            const disabledAfter = [
              h.dom.enabledEl.disabled, h.dom.cooldownEl.disabled, h.dom.remainingEl.disabled, h.dom.requestEl.disabled,
            ];
            globalThis.__result = { disabledDuring, disabledAfter };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["disabledDuring"], [True, True, True, True])
        self.assertEqual(result["disabledAfter"], [False, False, False, False])

    def test_show_rejects_invalid_load_response(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-1", { launch_state: "ready" });
            h.select("sid-1");
            h.setApiResponses([{ enabled: "yes", request: "", cooldown_minutes: 5, remaining_injections: 10 }]);
            await h.controller.show({ opener: null });
            globalThis.__result = { toasts: h.toasts, open: h.controller.isOpen() };
            """
        )
        result = run_node_json(js)
        self.assertFalse(result["open"])
        self.assertTrue(any(t.startswith("unattended load error:") for t in result["toasts"]))

    # --- 5. stale load ignored when selected/session/token changes or menu closes ---

    def test_stale_load_ignored_when_selected_changes_before_response(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-1", { launch_state: "ready" });
            h.sessions.set("sid-2", { launch_state: "ready" });
            h.select("sid-1");
            let resolveLoad;
            const load = new Promise((resolve) => { resolveLoad = resolve; });
            h.setApiResponses([{ __promise: load }]);
            const p = h.controller.show({ opener: null });
            // Selected changes before the load resolves.
            h.select("sid-2");
            resolveLoad({ enabled: true, request: "stale", cooldown_minutes: 9, remaining_injections: 9 });
            await p;
            // The stale response must not mutate inputs / enable controls / focus.
            const requestValue = h.dom.requestEl.value;
            const focusCalls = h.calls.filter((c) => c[0] === "focus").length;
            globalThis.__result = { requestValue, focusCalls, menuSession: h.controller.menuSessionId() };
            """
        )
        result = run_node_json(js)
        # selected changed mid-load -> load returns early at the selected guard;
        # the catch-guard also short-circuits, so no toast and menu stays open
        # for sid-1 (hide is not triggered by the selected-change guard alone).
        self.assertNotEqual(result["requestValue"], "stale")

    def test_stale_load_ignored_when_menu_closes_before_response(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-1", { launch_state: "ready" });
            h.select("sid-1");
            let resolveLoad;
            const load = new Promise((resolve) => { resolveLoad = resolve; });
            h.setApiResponses([{ __promise: load }]);
            const p = h.controller.show({ opener: null });
            h.controller.hide({ restoreFocus: false });
            resolveLoad({ enabled: true, request: "stale", cooldown_minutes: 9, remaining_injections: 9 });
            await p;
            globalThis.__result = { requestValue: h.dom.requestEl.value, open: h.controller.isOpen() };
            """
        )
        result = run_node_json(js)
        self.assertNotEqual(result["requestValue"], "stale")
        self.assertFalse(result["open"])

    # --- 6. load error toast + hides with focus restore ---

    def test_load_error_toasts_and_hides_with_focus_restore(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-1", { launch_state: "ready" });
            h.select("sid-1");
            const opener = new h.HTMLElementCtor();
            opener.isConnected = true;
            opener.disabled = false;
            opener.focus = () => { h.calls.push(["opener-focus"]); };
            h.setApiResponses([new Error("boom")]);
            await h.controller.show({ opener });
            const toasts = h.toasts.slice();
            const open = h.controller.isOpen();
            const openerRestored = h.calls.some((c) => c[0] === "opener-focus");
            const display = h.dom.unattendedMenu.style.display;
            globalThis.__result = { toasts, open, openerRestored, display };
            """
        )
        result = run_node_json(js)
        self.assertFalse(result["open"])
        self.assertEqual(result["display"], "none")
        self.assertTrue(any(t == "unattended load error: boom" for t in result["toasts"]))
        self.assertTrue(result["openerRestored"])

    # --- 7. number draft preservation while dirty + invalid blur restore ---

    def test_number_draft_preserved_and_invalid_blur_restores(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-1", { launch_state: "ready", unattended_cooldown_minutes: 5, unattended_remaining_injections: 10 });
            h.select("sid-1");
            // Load an initial cfg via show.
            h.setApiResponses([{ enabled: false, request: "", cooldown_minutes: 5, remaining_injections: 10 }]);
            await h.controller.show({ opener: null });

            // Invalid draft: parseUnattendedDraftInt returns null -> no save, cfg unchanged.
            h.dom.cooldownEl.value = "abc";
            h.dom.cooldownEl.oninput({ target: h.dom.cooldownEl });
            const valueDuringInvalid = h.dom.cooldownEl.value;
            const saveCallsAfterInvalid = h.calls.filter((c) => c[0] === "api" && c[1].indexOf("/unattended") !== -1 && c[2]).length;

            // Valid draft updates cfg and schedules a save.
            h.dom.cooldownEl.value = "8";
            h.dom.cooldownEl.oninput({ target: h.dom.cooldownEl });
            const timersBefore = h.pendingTimerCount();
            const debounceMs = h.calls.filter((c) => c[0] === "setTimeout").slice(-1)[0][1];

            // Type an invalid draft again, then blur: the invalid draft is
            // restored to the last saved/current cfg value.
            h.dom.cooldownEl.value = "xyz";
            h.dom.cooldownEl.oninput({ target: h.dom.cooldownEl });
            h.dom.cooldownEl.onblur();
            const restoredValue = h.dom.cooldownEl.value;
            globalThis.__result = { valueDuringInvalid, saveCallsAfterInvalid, timersBefore, debounceMs, restoredValue };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["saveCallsAfterInvalid"], 0)
        self.assertGreaterEqual(result["timersBefore"], 1)
        self.assertEqual(result["debounceMs"], 450)
        self.assertEqual(result["restoredValue"], "8")

    # --- 8. remaining_injections <= 0 disables enabled + schedules save with enabled false ---

    def test_remaining_zero_disables_enabled_and_schedules_save(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-1", { launch_state: "ready", unattended_enabled: true, unattended_cooldown_minutes: 5, unattended_remaining_injections: 10 });
            h.select("sid-1");
            h.setApiResponses([{ enabled: true, request: "", cooldown_minutes: 5, remaining_injections: 10 }]);
            await h.controller.show({ opener: null });
            h.dom.enabledEl.checked = true;

            // Drive remaining injections to 0.
            h.dom.remainingEl.value = "0";
            h.dom.remainingEl.oninput({ target: h.dom.remainingEl });
            const enabledChecked = h.dom.enabledEl.checked;
            const sessionRemaining = h.sessions.get("sid-1").unattended_remaining_injections;
            const sessionEnabled = h.sessions.get("sid-1").unattended_enabled;
            // Fire the debounced save and inspect the merged snapshot body.
            h.runPendingTimers();
            await new Promise((r) => setTimeout(r, 0));
            const saveBody = h.calls.find((c) => c[0] === "api" && c[1].indexOf("/unattended") !== -1 && c[2])[2];
            globalThis.__result = { enabledChecked, sessionRemaining, sessionEnabled, saveBody };
            """
        )
        result = run_node_json(js)
        self.assertFalse(result["enabledChecked"])
        self.assertEqual(result["sessionRemaining"], 0)
        self.assertFalse(result["sessionEnabled"])
        self.assertEqual(result["saveBody"], { "remaining_injections": 0, "enabled": False })

    # --- 9. save snapshot coercion/merge/debounce/per-session; in-flight + pending drain ---

    def test_save_debounced_450ms_and_merges_patches(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-1", { launch_state: "ready" });
            h.select("sid-1");
            // Two saves merge into one debounced flush.
            h.controller.scheduleUnattendedSave ? null : null;
            // Use the request input handler to schedule a save (request snapshot path).
            h.dom.requestEl.value = "do work";
            h.dom.requestEl.oninput({ target: h.dom.requestEl });
            h.dom.requestEl.value = "do more";
            h.dom.requestEl.oninput({ target: h.dom.requestEl });
            const timersBeforeFlush = h.pendingTimerCount();
            const debounceMs = h.calls.filter((c) => c[0] === "setTimeout").slice(-1)[0][1];
            h.setApiResponses([{ enabled: false, request: "do more", cooldown_minutes: 5, remaining_injections: 10 }]);
            h.runPendingTimers();
            await new Promise((r) => setTimeout(r, 0));
            const saveBodies = h.calls.filter((c) => c[0] === "api" && c[1].indexOf("/unattended") !== -1 && c[2]).map((c) => c[2]);
            globalThis.__result = { timersBeforeFlush, debounceMs, saveBodies };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["debounceMs"], 450)
        self.assertGreaterEqual(result["timersBeforeFlush"], 1)
        # Only one POST fired with the merged request value.
        self.assertEqual(len(result["saveBodies"]), 1)
        self.assertEqual(result["saveBodies"][0], { "request": "do more" })

    def test_save_is_per_session(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-1", { launch_state: "ready" });
            h.sessions.set("sid-2", { launch_state: "ready" });
            h.select("sid-1");
            h.dom.requestEl.value = "for sid-1";
            h.dom.requestEl.oninput({ target: h.dom.requestEl });
            h.select("sid-2");
            h.dom.requestEl.value = "for sid-2";
            h.dom.requestEl.oninput({ target: h.dom.requestEl });
            // Two independent timers for two sessions.
            const timers = h.pendingTimerCount();
            h.setApiResponses([
              { enabled: false, request: "for sid-1", cooldown_minutes: 5, remaining_injections: 10 },
              { enabled: false, request: "for sid-2", cooldown_minutes: 5, remaining_injections: 10 },
            ]);
            h.runPendingTimers();
            await new Promise((r) => setTimeout(r, 0));
            const saveUrls = h.calls.filter((c) => c[0] === "api" && c[1].indexOf("/unattended") !== -1 && c[2]).map((c) => c[1]);
            globalThis.__result = { timers, saveUrls };
            """
        )
        result = run_node_json(js)
        self.assertGreaterEqual(result["timers"], 2)
        self.assertEqual(len(result["saveUrls"]), 2)
        self.assertContains("/api/sessions/sid-1/unattended", result["saveUrls"])
        self.assertContains("/api/sessions/sid-2/unattended", result["saveUrls"])

    def test_in_flight_blocks_duplicate_flush_and_pending_drains(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-1", { launch_state: "ready" });
            h.select("sid-1");
            h.dom.requestEl.value = "first";
            h.dom.requestEl.oninput({ target: h.dom.requestEl });
            // Hold the first POST in flight.
            let resolveFirst;
            const first = new Promise((resolve) => { resolveFirst = resolve; });
            h.setApiResponses([
              { __promise: first },
              { enabled: false, request: "second", cooldown_minutes: 5, remaining_injections: 10 },
            ]);
            h.runPendingTimers();                // fires first POST, in-flight lock acquired
            const apiCountDuringInFlight = h.calls.filter((c) => c[0] === "api" && c[1].indexOf("/unattended") !== -1 && c[2]).length;

            // Schedule a second save while the first is in flight.
            h.dom.requestEl.value = "second";
            h.dom.requestEl.oninput({ target: h.dom.requestEl });
            h.runPendingTimers();                // timer fires but flush short-circuits (in-flight)
            const apiCountAfterBlockedFlush = h.calls.filter((c) => c[0] === "api" && c[1].indexOf("/unattended") !== -1 && c[2]).length;

            // Resolve the first POST -> finally sees pending -> drains the second flush.
            resolveFirst({ enabled: false, request: "first", cooldown_minutes: 5, remaining_injections: 10 });
            await new Promise((r) => setTimeout(r, 0));
            const saveBodies = h.calls.filter((c) => c[0] === "api" && c[1].indexOf("/unattended") !== -1 && c[2]).map((c) => c[2]);
            globalThis.__result = { apiCountDuringInFlight, apiCountAfterBlockedFlush, saveBodies };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["apiCountDuringInFlight"], 1)
        self.assertEqual(result["apiCountAfterBlockedFlush"], 1)
        # Second flush drained with the merged pending body.
        self.assertEqual(len(result["saveBodies"]), 2)
        self.assertEqual(result["saveBodies"][0], { "request": "first" })
        self.assertEqual(result["saveBodies"][1], { "request": "second" })

    # --- 10. 401 -> handleAppAuthLoss; non-401 -> "unattended save error: ..." ---

    def test_401_save_triggers_auth_loss(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-1", { launch_state: "ready" });
            h.select("sid-1");
            h.dom.requestEl.value = "x";
            h.dom.requestEl.oninput({ target: h.dom.requestEl });
            const err = new Error("no auth"); err.status = 401;
            h.setApiResponses([err]);
            h.runPendingTimers();
            await new Promise((r) => setTimeout(r, 0));
            const order = h.calls.map((c) => c[0]);
            const authIdx = order.indexOf("handleAppAuthLoss");
            const errorToasts = h.toasts.filter((t) => String(t).indexOf("unattended save error") !== -1);
            globalThis.__result = { authIdx, errorToasts };
            """
        )
        result = run_node_json(js)
        self.assertGreaterEqual(result["authIdx"], 0)
        self.assertEqual(result["errorToasts"], [])

    def test_non401_save_toasts_error(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-1", { launch_state: "ready" });
            h.select("sid-1");
            h.dom.requestEl.value = "x";
            h.dom.requestEl.oninput({ target: h.dom.requestEl });
            h.setApiResponses([new Error("server down")]);
            h.runPendingTimers();
            await new Promise((r) => setTimeout(r, 0));
            globalThis.__result = { toasts: h.toasts, authCalled: h.calls.some((c) => c[0] === "handleAppAuthLoss") };
            """
        )
        result = run_node_json(js)
        self.assertTrue(any(t == "unattended save error: server down" for t in result["toasts"]))
        self.assertFalse(result["authCalled"])

    # --- 11. applySavedCfg updates session fields + inputs only when guards allow ---

    def test_apply_saved_updates_selected_session_fields_when_menu_closed(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-1", { launch_state: "ready", unattended_enabled: false, unattended_cooldown_minutes: 5, unattended_remaining_injections: 10 });
            h.select("sid-1");
            h.dom.requestEl.value = "x";
            h.dom.requestEl.oninput({ target: h.dom.requestEl });
            h.setApiResponses([{ enabled: true, request: "saved", cooldown_minutes: 7, remaining_injections: 3 }]);
            h.runPendingTimers();
            await new Promise((r) => setTimeout(r, 0));
            const s = h.sessions.get("sid-1");
            globalThis.__result = {
              enabled: s.unattended_enabled, cooldown: s.unattended_cooldown_minutes, remaining: s.unattended_remaining_injections,
              requestValue: h.dom.requestEl.value, enabledChecked: h.dom.enabledEl.checked,
            };
            """
        )
        result = run_node_json(js)
        self.assertTrue(result["enabled"])
        self.assertEqual(result["cooldown"], 7)
        self.assertEqual(result["remaining"], 3)
        self.assertEqual(result["requestValue"], "saved")
        self.assertTrue(result["enabledChecked"])

    def test_apply_saved_skipped_when_selected_changed(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-1", { launch_state: "ready", unattended_enabled: false, unattended_cooldown_minutes: 5, unattended_remaining_injections: 10 });
            h.sessions.set("sid-2", { launch_state: "ready" });
            h.select("sid-1");
            h.dom.requestEl.value = "x";
            h.dom.requestEl.oninput({ target: h.dom.requestEl });
            // Change selection before the save resolves.
            let resolveSave;
            const save = new Promise((resolve) => { resolveSave = resolve; });
            h.setApiResponses([{ __promise: save }]);
            h.runPendingTimers();
            h.select("sid-2");
            resolveSave({ enabled: true, request: "saved", cooldown_minutes: 7, remaining_injections: 3 });
            await new Promise((r) => setTimeout(r, 0));
            const s = h.sessions.get("sid-1");
            globalThis.__result = { enabled: s.unattended_enabled, cooldown: s.unattended_cooldown_minutes, remaining: s.unattended_remaining_injections };
            """
        )
        result = run_node_json(js)
        # selected no longer sid-1 -> applySavedUnattendedCfg short-circuits.
        self.assertFalse(result["enabled"])
        self.assertEqual(result["cooldown"], 5)
        self.assertEqual(result["remaining"], 10)

    # --- 12. Escape / outside click / resize hide + focus restore ---

    def test_escape_outside_click_and_resize_hide(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-1", { launch_state: "ready" });
            h.select("sid-1");
            const opener = new h.HTMLElementCtor();
            opener.isConnected = true;
            opener.disabled = false;
            opener.focus = () => { h.calls.push(["opener-focus"]); };
            h.setApiResponses([{ enabled: false, request: "", cooldown_minutes: 5, remaining_injections: 10 }]);

            // Escape hides with focus restore.
            await h.controller.show({ opener });
            h.calls.length = 0;
            h.dispatchEvent(h.documentTarget, "keydown", { key: "Escape", preventDefault() {}, stopPropagation() {} });
            const afterEscape = { open: h.controller.isOpen(), restored: h.calls.some((c) => c[0] === "opener-focus") };

            // Outside click hides (no focus restore).
            await h.controller.show({ opener });
            h.calls.length = 0;
            h.dispatchEvent(h.documentTarget, "click", {});
            const afterClick = { open: h.controller.isOpen() };

            // Resize hides.
            await h.controller.show({ opener });
            h.calls.length = 0;
            h.dispatchEvent(h.windowTarget, "resize", {});
            const afterResize = { open: h.controller.isOpen() };

            globalThis.__result = { afterEscape, afterClick, afterResize };
            """
        )
        result = run_node_json(js)
        self.assertFalse(result["afterEscape"]["open"])
        self.assertTrue(result["afterEscape"]["restored"])
        self.assertFalse(result["afterClick"]["open"])
        self.assertFalse(result["afterResize"]["open"])

    def test_escape_does_not_hide_when_menu_closed(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            // Dispatch Escape with menu closed: no-op (no error, stays closed).
            h.dispatchEvent(h.documentTarget, "keydown", { key: "Escape", preventDefault() {}, stopPropagation() {} });
            globalThis.__result = { open: h.controller.isOpen() };
            """
        )
        result = run_node_json(js)
        self.assertFalse(result["open"])

    # --- 13. dispose clears timers/pending/in-flight and invalidates menu ---

    def test_dispose_clears_state_and_invalidates_menu(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-1", { launch_state: "ready" });
            h.select("sid-1");
            h.dom.requestEl.value = "pending";
            h.dom.requestEl.oninput({ target: h.dom.requestEl });
            const timersBefore = h.pendingTimerCount();
            // Open the menu so dispose must invalidate it.
            h.setApiResponses([{ __promise: new Promise(() => {}) }]);  // never resolves
            const showP = h.controller.show({ opener: null });
            const openBefore = h.controller.isOpen();
            h.controller.dispose();
            const openAfter = h.controller.isOpen();
            const menuSessionAfter = h.controller.menuSessionId();
            const timersAfter = h.pendingTimerCount();
            const display = h.dom.unattendedMenu.style.display;
            const ariaExpanded = h.dom.unattendedBtn._attrs["aria-expanded"];
            // After dispose, firing any pending timer must not call the API.
            h.runPendingTimers();
            const apiAfterDispose = h.calls.some((c) => c[0] === "api" && c[1].indexOf("/unattended") !== -1 && c[2]);
            globalThis.__result = { timersBefore, timersAfter, openBefore, openAfter, menuSessionAfter, display, ariaExpanded, apiAfterDispose };
            """
        )
        result = run_node_json(js)
        self.assertGreater(result["timersBefore"], 0)
        self.assertEqual(result["timersAfter"], 0)
        self.assertFalse(result["openAfter"])
        self.assertIsNone(result["menuSessionAfter"])
        self.assertEqual(result["display"], "none")
        self.assertEqual(result["ariaExpanded"], "false")
        self.assertFalse(result["apiAfterDispose"])

if __name__ == "__main__":
    unittest.main()
