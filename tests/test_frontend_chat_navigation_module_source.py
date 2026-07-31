import json
import os
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_CHAT_NAVIGATION_JS = ROOT / "codoxear" / "static" / "app_chat_navigation.js"
INDEX_HTML = ROOT / "codoxear" / "static" / "index.html"
APP_JS = ROOT / "codoxear" / "static" / "app.js"


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
const openSearchCalls = [];
const pulseCalls = [];

let selected = null;
let scrollTop = 0;
let reducedMotion = false;
let sidebarOpen = false;
let modalOpen = false;
let textEntryTarget = null;

// Fake row factory. Rows carry an offsetTop (set per-test) and a name marker
// so assertions can inspect which row was scrolled/pulsed.
let nextRowId = 0;
function makeRow(name, offsetTop) {
  return {
    name,
    offsetTop,
    scrollIntoView(opts) { calls.push(["scrollIntoView", name, JSON.stringify(opts)]); },
  };
}

function userJumpTarget(rows, direction, threshold) {
  if (!rows.length) return { reason: "none", target: null };
  if (direction < 0) {
    for (let i = rows.length - 1; i >= 0; i -= 1) {
      if (rows[i].offsetTop < threshold) return { reason: "target", target: rows[i] };
    }
    return { reason: "first", target: null };
  }
  for (const row of rows) {
    if (row.offsetTop > threshold) return { reason: "target", target: row };
  }
  return { reason: "last", target: null };
}

function copyJumpTarget(rows, direction, threshold) {
  // Simplified copy target: index-based boundary semantics matching real shape.
  if (!rows.length) return { reason: "none", target: null };
  if (direction < 0) {
    for (let i = rows.length - 1; i >= 0; i -= 1) {
      if (rows[i].offsetTop < threshold) return { reason: "target", target: rows[i] };
    }
    return { reason: "first", target: null };
  }
  for (const row of rows) {
    if (row.offsetTop > threshold) return { reason: "target", target: row };
  }
  return { reason: "last", target: null };
}

let userRows = [];
let copyRows = [];

function addAppEvent(target, type, handler, options) {
  calls.push(["addAppEvent", type, !!options]);
  if (target && target.__id === "doc" && type === "keydown") keydownHandler = handler;
  return handler;
}
let keydownHandler = null;

const documentTarget = { __id: "doc", body: { classList: { contains: (c) => c === "sidebar-open" && sidebarOpen } } };

const prevUserBtn = { _id: "prevUserBtn", disabled: false, style: {}, _attrs: {}, setAttribute(n, v) { this._attrs[n] = String(v); }, getAttribute(n) { return this._attrs[n]; }, onclick: null };
const nextUserBtn = { _id: "nextUserBtn", disabled: false, style: {}, _attrs: {}, setAttribute(n, v) { this._attrs[n] = String(v); }, getAttribute(n) { return this._attrs[n]; }, onclick: null };

const modalIsolationTargets = [{ __modal: true }];
const deps = {
  prevUserBtn,
  nextUserBtn,
  getSelected: () => selected,
  loadedUserMessageRows: () => userRows.slice(),
  loadedCopyMessageRows: () => copyRows.slice(),
  loadedUserJumpTarget: userJumpTarget,
  loadedCopyJumpTarget: copyJumpTarget,
  getScrollTop: () => scrollTop,
  prefersReducedMotion: () => reducedMotion,
  pulseNavigatedRow: (row) => { pulseCalls.push(row ? row.name : null); },
  setToast: (t) => { toasts.push(t); calls.push(["setToast", t]); },
  openChatSearch: () => { openSearchCalls.push(1); calls.push(["openChatSearch"]); },
  isTextEntryElement: (target) => Boolean(target && target.__textEntry),
  modalIsolationTargets,
  isModalTargetOpen: (node) => modalOpen,
  addAppEvent,
  documentTarget,
  isSidebarOpen: () => sidebarOpen,
};

const ctx = { HTMLElement: function HTMLElement() {}, document: documentTarget, window: {}, console };
vm.createContext(ctx);
vm.runInContext(MODULE_SOURCE, ctx);
const controller = ctx.window.CodoxearChatNavigation.createChatNavigationController(deps);

function dispatchKey(opts) {
  if (!keydownHandler) throw new Error("no keydown handler wired");
  const e = {
    defaultPrevented: !!opts.defaultPrevented,
    key: opts.key,
    altKey: !!opts.altKey,
    shiftKey: !!opts.shiftKey,
    ctrlKey: !!opts.ctrlKey,
    metaKey: !!opts.metaKey,
    target: opts.target || null,
    _preventDefaultCalled: false,
    _stopPropagationCalled: false,
    preventDefault() { this._preventDefaultCalled = true; },
    stopPropagation() { this._stopPropagationCalled = true; },
  };
  keydownHandler(e);
  return e;
}

globalThis.__harness = {
  controller,
  calls,
  toasts,
  openSearchCalls,
  pulseCalls,
  dom: { prevUserBtn, nextUserBtn },
  select: (sid) => { selected = sid; },
  setUserRows: (rows) => { userRows = rows; },
  setCopyRows: (rows) => { copyRows = rows; },
  setScrollTop: (v) => { scrollTop = v; },
  setReducedMotion: (v) => { reducedMotion = v; },
  setSidebarOpen: (v) => { sidebarOpen = v; },
  setModalOpen: (v) => { modalOpen = v; },
  setTextEntryTarget: (v) => { textEntryTarget = v; },
  textEntryTarget: () => textEntryTarget,
  makeRow,
  dispatchKey,
  scrollIntoViewCalls: () => calls.filter((c) => c[0] === "scrollIntoView"),
  resetCalls: () => { calls.length = 0; toasts.length = 0; openSearchCalls.length = 0; pulseCalls.length = 0; },
};
"""


def harness_script(epilogue: str) -> str:
    module_source = APP_CHAT_NAVIGATION_JS.read_text(encoding="utf-8")
    js = (
        textwrap.dedent(
            f"""
        const MODULE_SOURCE = {json.dumps(module_source)};
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


class TestFrontendChatNavigationModuleSource(unittest.TestCase):
    # --- 1. frozen export + missing dep failures ---

    # --- 2. syncButtons disabled/enabled ---

    def test_sync_buttons_disabled_when_no_selected(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.setUserRows([h.makeRow("u1", 0)]);
            h.controller.syncButtons();
            globalThis.__result = { prev: h.dom.prevUserBtn.disabled, next: h.dom.nextUserBtn.disabled };
            """
        )
        result = run_node_json(js)
        self.assertTrue(result["prev"])
        self.assertTrue(result["next"])

    def test_sync_buttons_disabled_when_no_rows(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.select("sid-1");
            h.setUserRows([]);
            h.controller.syncButtons();
            globalThis.__result = { prev: h.dom.prevUserBtn.disabled, next: h.dom.nextUserBtn.disabled };
            """
        )
        result = run_node_json(js)
        self.assertTrue(result["prev"])
        self.assertTrue(result["next"])

    def test_sync_buttons_enabled_with_selected_and_rows(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.select("sid-1");
            h.setUserRows([h.makeRow("u1", 0), h.makeRow("u2", 100)]);
            h.controller.syncButtons();
            globalThis.__result = { prev: h.dom.prevUserBtn.disabled, next: h.dom.nextUserBtn.disabled };
            """
        )
        result = run_node_json(js)
        self.assertFalse(result["prev"])
        self.assertFalse(result["next"])

    # --- 3. user navigation no rows / boundary / target scroll+pulse ---

    def test_user_navigation_no_rows_toast(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.select("sid-1");
            h.setUserRows([]);
            h.controller.jumpToLoadedUserMessage(-1);
            globalThis.__result = { toasts: h.toasts.slice(), scrolls: h.scrollIntoViewCalls(), pulses: h.pulseCalls.slice() };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["toasts"], ["No loaded user messages"])
        self.assertEqual(result["scrolls"], [])
        self.assertEqual(result["pulses"], [])

    def test_user_navigation_boundary_first_toast(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.select("sid-1");
            h.setUserRows([h.makeRow("u1", 200)]);
            h.setScrollTop(0);
            h.controller.jumpToLoadedUserMessage(-1);
            globalThis.__result = { toasts: h.toasts.slice(), threshold: h.scrollIntoViewCalls().length, pulses: h.pulseCalls.length };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["toasts"], ["At first loaded user message"])
        self.assertEqual(result["threshold"], 0)
        self.assertEqual(result["pulses"], 0)

    def test_user_navigation_boundary_last_toast(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.select("sid-1");
            h.setUserRows([h.makeRow("u1", 0)]);
            h.setScrollTop(100);
            h.controller.jumpToLoadedUserMessage(1);
            globalThis.__result = { toasts: h.toasts.slice(), scrolls: h.scrollIntoViewCalls(), pulses: h.pulseCalls.length };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["toasts"], ["At last loaded user message"])
        self.assertEqual(result["scrolls"], [])
        self.assertEqual(result["pulses"], 0)

    def test_user_navigation_target_scrolls_and_pulses(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.select("sid-1");
            h.setUserRows([h.makeRow("u1", 0), h.makeRow("u2", 400)]);
            h.setScrollTop(0);
            h.setReducedMotion(false);
            h.controller.jumpToLoadedUserMessage(1);
            globalThis.__result = { scrolls: h.scrollIntoViewCalls(), pulses: h.pulseCalls.slice() };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["pulses"], ["u2"])
        self.assertEqual(result["scrolls"], [["scrollIntoView", "u2", json.dumps({"block": "start", "behavior": "auto"}, separators=(",", ":"))]])

    def test_user_navigation_target_uses_auto_when_reduced_motion(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.select("sid-1");
            h.setUserRows([h.makeRow("u1", 0), h.makeRow("u2", 400)]);
            h.setScrollTop(0);
            h.setReducedMotion(true);
            h.controller.jumpToLoadedUserMessage(1);
            globalThis.__result = { scrolls: h.scrollIntoViewCalls() };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["scrolls"], [["scrollIntoView", "u2", json.dumps({"block": "start", "behavior": "auto"}, separators=(",", ":"))]])

    def test_user_navigation_threshold_uses_scroll_top_plus_24(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.select("sid-1");
            // Single row far below scrollTop; with scrollTop=0 + 24 it is the
            // next target. Push scrollTop past it so prev boundary fires.
            h.setUserRows([h.makeRow("u1", 50), h.makeRow("u2", 100)]);
            h.setScrollTop(200);
            h.controller.jumpToLoadedUserMessage(-1);
            globalThis.__result = { toasts: h.toasts.slice(), scrolls: h.scrollIntoViewCalls() };
            """
        )
        result = run_node_json(js)
        # threshold = 224; u1@50 and u2@100 are both < 224 so prev target = u2.
        self.assertEqual(result["scrolls"], [["scrollIntoView", "u2", json.dumps({"block": "start", "behavior": "auto"}, separators=(",", ":"))]])

    # --- 4. copy navigation no rows / boundary / target scroll+pulse ---

    def test_copy_navigation_no_rows_toast(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.select("sid-1");
            h.setCopyRows([]);
            h.controller.jumpToLoadedMessage(1);
            globalThis.__result = { toasts: h.toasts.slice(), scrolls: h.scrollIntoViewCalls(), pulses: h.pulseCalls.length };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["toasts"], ["No loaded messages"])
        self.assertEqual(result["scrolls"], [])
        self.assertEqual(result["pulses"], 0)

    def test_copy_navigation_boundary_first_toast(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.select("sid-1");
            h.setCopyRows([h.makeRow("c1", 200)]);
            h.setScrollTop(0);
            h.controller.jumpToLoadedMessage(-1);
            globalThis.__result = { toasts: h.toasts.slice(), scrolls: h.scrollIntoViewCalls() };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["toasts"], ["At first loaded message"])
        self.assertEqual(result["scrolls"], [])

    def test_copy_navigation_boundary_last_toast(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.select("sid-1");
            h.setCopyRows([h.makeRow("c1", 0)]);
            h.setScrollTop(100);
            h.controller.jumpToLoadedMessage(1);
            globalThis.__result = { toasts: h.toasts.slice(), scrolls: h.scrollIntoViewCalls() };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["toasts"], ["At last loaded message"])
        self.assertEqual(result["scrolls"], [])

    def test_copy_navigation_target_scrolls_and_pulses(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.select("sid-1");
            h.setCopyRows([h.makeRow("c1", 0), h.makeRow("c2", 500)]);
            h.setScrollTop(0);
            h.controller.jumpToLoadedMessage(1);
            globalThis.__result = { scrolls: h.scrollIntoViewCalls(), pulses: h.pulseCalls.slice() };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["pulses"], ["c2"])
        self.assertEqual(result["scrolls"], [["scrollIntoView", "c2", json.dumps({"block": "start", "behavior": "auto"}, separators=(",", ":"))]])

    # --- 5. prev/next click handlers prevent/stop and call directions ---

    def test_prev_next_click_handlers_wired_and_delegating(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.select("sid-1");
            h.setUserRows([h.makeRow("u1", 0), h.makeRow("u2", 400)]);
            h.setScrollTop(0);
            const prevEvent = { _prevent: false, _stop: false, preventDefault() { this._prevent = true; }, stopPropagation() { this._stop = true; } };
            h.dom.prevUserBtn.onclick(prevEvent);
            const afterPrev = { scrolls: h.scrollIntoViewCalls(), pulses: h.pulseCalls.slice(), prevent: prevEvent._prevent, stop: prevEvent._stop };
            h.resetCalls();
            const nextEvent = { _prevent: false, _stop: false, preventDefault() { this._prevent = true; }, stopPropagation() { this._stop = true; } };
            h.dom.nextUserBtn.onclick(nextEvent);
            globalThis.__result = {
              prev: afterPrev,
              next: { scrolls: h.scrollIntoViewCalls(), pulses: h.pulseCalls.slice(), prevent: nextEvent._prevent, stop: nextEvent._stop },
            };
            """
        )
        result = run_node_json(js)
        # prev (direction -1) from scrollTop 0 with a single row at 0 -> first boundary.
        self.assertEqual(result["prev"]["prevent"], True)
        self.assertEqual(result["prev"]["stop"], True)
        # prev (direction -1) from scrollTop 0: threshold=24; u2@400 not <24,
        # u1@0 <24 -> target u1.
        self.assertEqual(result["prev"]["scrolls"], [["scrollIntoView", "u1", json.dumps({"block": "start", "behavior": "auto"}, separators=(",", ":"))]])
        self.assertEqual(result["prev"]["pulses"], ["u1"])
        # next (direction +1) -> target u2.
        self.assertEqual(result["next"]["scrolls"], [["scrollIntoView", "u2", json.dumps({"block": "start", "behavior": "auto"}, separators=(",", ":"))]])
        self.assertEqual(result["next"]["pulses"], ["u2"])

    # --- 6. `/` shortcut opens search only when not blocked ---

    def test_slash_opens_chat_search_when_unblocked(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.select("sid-1");
            const e = h.dispatchKey({ key: "/" });
            globalThis.__result = { open: h.openSearchCalls.length, prevented: e._preventDefaultCalled };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["open"], 1)
        self.assertTrue(result["prevented"])

    def test_slash_blocked_when_no_selected(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            const e = h.dispatchKey({ key: "/" });
            globalThis.__result = { open: h.openSearchCalls.length, prevented: e._preventDefaultCalled };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["open"], 0)
        self.assertFalse(result["prevented"])

    def test_slash_blocked_when_text_entry_target(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.select("sid-1");
            const e = h.dispatchKey({ key: "/", target: { __textEntry: true } });
            globalThis.__result = { open: h.openSearchCalls.length, prevented: e._preventDefaultCalled };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["open"], 0)
        self.assertFalse(result["prevented"])

    def test_slash_blocked_when_sidebar_open(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.select("sid-1");
            h.setSidebarOpen(true);
            const e = h.dispatchKey({ key: "/" });
            globalThis.__result = { open: h.openSearchCalls.length, prevented: e._preventDefaultCalled };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["open"], 0)
        self.assertFalse(result["prevented"])

    def test_slash_blocked_when_modal_open(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.select("sid-1");
            h.setModalOpen(true);
            const e = h.dispatchKey({ key: "/" });
            globalThis.__result = { open: h.openSearchCalls.length, prevented: e._preventDefaultCalled };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["open"], 0)
        self.assertFalse(result["prevented"])

    # --- 7. Alt+Shift (copy) and Alt (user) arrow shortcuts ---

    def test_alt_shift_arrow_down_navigates_copy(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.select("sid-1");
            h.setCopyRows([h.makeRow("c1", 0), h.makeRow("c2", 500)]);
            h.setScrollTop(0);
            const e = h.dispatchKey({ key: "ArrowDown", altKey: true, shiftKey: true });
            globalThis.__result = { scrolls: h.scrollIntoViewCalls(), pulses: h.pulseCalls.slice(), prevented: e._preventDefaultCalled };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["scrolls"], [["scrollIntoView", "c2", json.dumps({"block": "start", "behavior": "auto"}, separators=(",", ":"))]])
        self.assertEqual(result["pulses"], ["c2"])
        self.assertTrue(result["prevented"])

    def test_alt_shift_arrow_up_navigates_copy(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.select("sid-1");
            h.setCopyRows([h.makeRow("c1", 0), h.makeRow("c2", 500)]);
            h.setScrollTop(300);
            const e = h.dispatchKey({ key: "ArrowUp", altKey: true, shiftKey: true });
            globalThis.__result = { scrolls: h.scrollIntoViewCalls(), pulses: h.pulseCalls.slice() };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["pulses"], ["c1"])

    def test_alt_arrow_down_navigates_user(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.select("sid-1");
            h.setUserRows([h.makeRow("u1", 0), h.makeRow("u2", 400)]);
            h.setScrollTop(0);
            const e = h.dispatchKey({ key: "ArrowDown", altKey: true });
            globalThis.__result = { scrolls: h.scrollIntoViewCalls(), pulses: h.pulseCalls.slice(), prevented: e._preventDefaultCalled };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["scrolls"], [["scrollIntoView", "u2", json.dumps({"block": "start", "behavior": "auto"}, separators=(",", ":"))]])
        self.assertEqual(result["pulses"], ["u2"])
        self.assertTrue(result["prevented"])

    def test_alt_arrow_up_navigates_user(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.select("sid-1");
            h.setUserRows([h.makeRow("u1", 0), h.makeRow("u2", 400)]);
            h.setScrollTop(200);
            const e = h.dispatchKey({ key: "ArrowUp", altKey: true });
            globalThis.__result = { scrolls: h.scrollIntoViewCalls(), pulses: h.pulseCalls.slice() };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["pulses"], ["u1"])

    def test_arrow_shortcuts_ignore_default_prevented(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.select("sid-1");
            h.setUserRows([h.makeRow("u1", 0), h.makeRow("u2", 400)]);
            h.setScrollTop(0);
            const e = h.dispatchKey({ key: "ArrowDown", altKey: true, defaultPrevented: true });
            globalThis.__result = { scrolls: h.scrollIntoViewCalls().length, pulses: h.pulseCalls.length };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["scrolls"], 0)
        self.assertEqual(result["pulses"], 0)

    def test_arrow_shortcuts_ignore_extra_modifiers(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.select("sid-1");
            h.setUserRows([h.makeRow("u1", 0), h.makeRow("u2", 400)]);
            const cases = [
              { key: "ArrowDown", altKey: true, ctrlKey: true },
              { key: "ArrowDown", altKey: true, metaKey: true },
              { key: "ArrowDown", altKey: true, shiftKey: true, ctrlKey: true },
            ];
            let scrolls = 0; let pulses = 0;
            for (const c of cases) {
              h.dispatchKey(c);
              scrolls += h.scrollIntoViewCalls().length;
              pulses += h.pulseCalls.length;
            }
            globalThis.__result = { scrolls, pulses };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["scrolls"], 0)
        self.assertEqual(result["pulses"], 0)

    def test_arrow_shortcuts_blocked_when_no_selected_or_text_entry_or_sidebar_or_modal(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.select("sid-1");
            h.setUserRows([h.makeRow("u1", 0), h.makeRow("u2", 400)]);
            h.setCopyRows([h.makeRow("c1", 0), h.makeRow("c2", 400)]);
            // No selected: unselect, both arrows noop.
            h.select(null);
            h.dispatchKey({ key: "ArrowDown", altKey: true });
            h.dispatchKey({ key: "ArrowDown", altKey: true, shiftKey: true });
            const noSel = { scrolls: h.scrollIntoViewCalls().length };
            h.resetCalls();
            h.select("sid-1");
            h.dispatchKey({ key: "ArrowDown", altKey: true, target: { __textEntry: true } });
            h.dispatchKey({ key: "ArrowDown", altKey: true, shiftKey: true, target: { __textEntry: true } });
            const textEntry = { scrolls: h.scrollIntoViewCalls().length };
            h.resetCalls();
            h.setSidebarOpen(true);
            h.dispatchKey({ key: "ArrowDown", altKey: true });
            const sidebar = { scrolls: h.scrollIntoViewCalls().length };
            h.resetCalls();
            h.setSidebarOpen(false);
            h.setModalOpen(true);
            h.dispatchKey({ key: "ArrowDown", altKey: true });
            const modal = { scrolls: h.scrollIntoViewCalls().length };
            globalThis.__result = { noSel, textEntry, sidebar, modal };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["noSel"]["scrolls"], 0)
        self.assertEqual(result["textEntry"]["scrolls"], 0)
        self.assertEqual(result["sidebar"]["scrolls"], 0)
        self.assertEqual(result["modal"]["scrolls"], 0)

    def test_other_keys_are_ignored(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.select("sid-1");
            h.setUserRows([h.makeRow("u1", 0), h.makeRow("u2", 400)]);
            const e = h.dispatchKey({ key: "Enter" });
            globalThis.__result = { scrolls: h.scrollIntoViewCalls().length, prevented: e._preventDefaultCalled };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["scrolls"], 0)
        self.assertFalse(result["prevented"])

    # --- 8. load order and app.js delegation checks ---

    # --- dispose ---

    def test_dispose_clears_button_handlers(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.controller.dispose();
            globalThis.__result = { prev: h.dom.prevUserBtn.onclick, next: h.dom.nextUserBtn.onclick };
            """
        )
        result = run_node_json(js)
        self.assertIsNone(result["prev"])
        self.assertIsNone(result["next"])


if __name__ == "__main__":
    unittest.main()
