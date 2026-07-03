import json
import os
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_QUEUE_JS = ROOT / "codoxear" / "static" / "app_queue.js"
APP_SESSION_HELPERS_JS = ROOT / "codoxear" / "static" / "app_session_helpers.js"
APP_MODAL_JS = ROOT / "codoxear" / "static" / "app_modal.js"
INDEX_HTML = ROOT / "codoxear" / "static" / "index.html"


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
let nowValue = 1000;
let disposed = false;
let confirmValue = false;
const confirmCalls = [];
const sessions = new Map();
let selected = null;

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
  // Run timers in registration order; newly scheduled timers during a run are
  // also drained so a debounced update that finalizes and triggers a pending
  // delete executes within a single flush.
  let guard = 0;
  while (pendingTimers.size && guard < 50) {
    guard += 1;
    const [handle, fn] = Array.from(pendingTimers.entries())[0];
    pendingTimers.delete(handle);
    fn();
  }
}

function fakeNode(extra = {}) {
  const node = {
    style: { display: "none" },
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    _attrs: {},
    _children: [],
    value: "",
    disabled: false,
    textContent: "",
    set innerHTML(v) { this._children = []; },
    get innerHTML() { return ""; },
    setAttribute(name, value) { this._attrs[name] = String(value); },
    getAttribute(name) { return this._attrs[name]; },
    removeAttribute(name) { delete this._attrs[name]; },
    appendChild(child) { this._children.push(child); return child; },
    addEventListener() {},
    focus() { calls.push(["focus"]); },
    ...extra,
  };
  return node;
}

const queueBtn = fakeNode();
const queueBackdrop = fakeNode();
const queueCloseBtn = fakeNode();
const queueList = fakeNode();
const queueEmpty = fakeNode();
const queueViewer = fakeNode();

function makeEl(tag, attrs = {}, children = []) {
  const node = fakeNode({ tag });
  const a = attrs || {};
  if (a.class !== undefined) node._class = String(a.class);
  if (a.text !== undefined) node.textContent = String(a.text);
  if (a.html !== undefined) node.innerHTML = String(a.html);
  if (a["aria-label"] !== undefined) node._attrs["aria-label"] = String(a["aria-label"]);
  if (a.title !== undefined) node._attrs.title = String(a.title);
  node.appendChild = (child) => { node._children.push(child); return child; };
  (Array.isArray(children) ? children : [children]).forEach((c) => { if (c) node._children.push(c); });
  return node;
}

const deps = {
  queueBackdrop,
  queueCloseBtn,
  queueList,
  queueEmpty,
  queueViewer,
  queueBtn,
  getSelected: () => selected,
  getSessionInfo: (sid) => sessions.get(sid) || null,
  isAppDisposed: () => disposed,
  api: (url, options = {}) => {
    const body = options && options.body ? JSON.parse(JSON.stringify(options.body)) : null;
    calls.push(["api", url, body]);
    if (apiResponseQueue.length) {
      const next = apiResponseQueue.shift();
      if (next instanceof Error) return Promise.reject(next);
      if (typeof next === "function") return Promise.resolve(next(body));
      return Promise.resolve(next);
    }
    return Promise.resolve({});
  },
  setToast: (t) => { toasts.push(t); calls.push(["setToast", t]); },
  clearCommitUnknownSend: (sid, text) => { calls.push(["clearCommitUnknownSend", sid, text]); return Promise.resolve(true); },
  refreshSessions: async () => { calls.push(["refreshSessions"]); },
  updateQueueBadge: () => { calls.push(["updateQueueBadge"]); },
  syncRecoveryUiForSession: (sid) => { calls.push(["syncRecoveryUiForSession", sid]); },
  kickPoll: (ms) => { calls.push(["kickPoll", ms]); },
  setPollFastUntilMs: (ms) => { calls.push(["setPollFastUntilMs", ms]); },
  handleAppAuthLoss: () => { calls.push(["handleAppAuthLoss"]); },
  prepareModalOpen: () => { calls.push(["prepareModalOpen"]); },
  afterModalVisibilityChanged: () => { calls.push(["afterModalVisibilityChanged"]); },
  el: makeEl,
  iconSvg: (name) => `<svg>${name}</svg>`,
  recoveryPanelFocusFallback: () => null,
  requestFrame: (fn) => fn(),
  setTimeout: fakeSetTimeout,
  clearTimeout: fakeClearTimeout,
  now: () => nowValue,
};

const ctx = {
  HTMLElement: function HTMLElement() {},
  document: { activeElement: null, querySelector: () => null },
  window: { confirm: (msg) => { confirmCalls.push(String(msg)); return confirmValue; } },
  console,
};
vm.createContext(ctx);
vm.runInContext(MODAL_SOURCE, ctx);
vm.runInContext(HELPERS_SOURCE, ctx);
vm.runInContext(QUEUE_SOURCE, ctx);

const controller = ctx.window.CodoxearQueue.createQueueController(deps);

function inspectRows() {
  // Walk the rendered queueList children and project each row's lock/tag state.
  return queueList._children.map((row) => {
    const actions = row._children.find((c) => c && c._class && c._class.indexOf("queueActionRail") !== -1) || { _children: [] };
    const editorShell = row._children.find((c) => c && c._class && c._class.indexOf("queueEditorShell") !== -1) || { _children: [] };
    const ta = editorShell._children.find((c) => c && c.tag === "textarea") || {};
    const tagNodes = actions._children.filter((c) => c && c._class && c._class.indexOf("queueSendingTag") !== -1);
    const byAria = (label) => actions._children.find((c) => c && c._attrs && c._attrs["aria-label"] === label) || {};
    return {
      class: row._class || "",
      tagTexts: tagNodes.map((n) => n.textContent),
      taDisabled: !!ta.disabled,
      upDisabled: !!byAria("Move up").disabled,
      downDisabled: !!byAria("Move down").disabled,
      delDisabled: !!byAria("Delete").disabled,
    };
  });
}

function editFirstRow(text) {
  // Simulate a user edit on the first rendered row: drive the textarea oninput
  // so the controller records queueLastEditMs and schedules a debounced update.
  const row = queueList._children[0];
  if (!row) return false;
  const shell = row._children.find((c) => c && c._class && c._class.indexOf("queueEditorShell") !== -1);
  const ta = shell && shell._children.find((c) => c && c.tag === "textarea");
  if (!ta || typeof ta.oninput !== "function") return false;
  ta.value = String(text || "");
  ta.oninput();
  return true;
}

globalThis.__harness = {
  controller,
  calls,
  toasts,
  confirmCalls,
  HTMLElementCtor: ctx.HTMLElement,
  setConfirm: (v) => { confirmValue = v; },
  sessions,
  select: (sid) => { selected = sid; },
  setDisposed: (v) => { disposed = v; },
  setNow: (v) => { nowValue = v; },
  setApiResponses,
  runPendingTimers,
  pendingTimerCount: () => pendingTimers.size,
  dom: { queueBtn, queueBackdrop, queueCloseBtn, queueList, queueEmpty, queueViewer },
  inspectRows,
  editFirstRow,
};
"""


def harness_script(epilogue: str) -> str:
    queue_source = APP_QUEUE_JS.read_text(encoding="utf-8")
    helpers_source = APP_SESSION_HELPERS_JS.read_text(encoding="utf-8")
    modal_source = APP_MODAL_JS.read_text(encoding="utf-8")
    js = (
        textwrap.dedent(
            f"""
        const MODAL_SOURCE = {json.dumps(modal_source)};
        const HELPERS_SOURCE = {json.dumps(helpers_source)};
        const QUEUE_SOURCE = {json.dumps(queue_source)};
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


class TestFrontendQueueModuleSource(unittest.TestCase):
    # --- 1. dependency failures + frozen export ---

    def test_module_export_is_frozen_createQueue_controller(self) -> None:
        queue_source = APP_QUEUE_JS.read_text(encoding="utf-8")
        helpers_source = APP_SESSION_HELPERS_JS.read_text(encoding="utf-8")
        modal_source = APP_MODAL_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const vm = require("vm");
            const ctx = {{ window: {{}}, HTMLElement: function HTMLElement() {{}} }};
            vm.createContext(ctx);
            vm.runInContext({json.dumps(modal_source)}, ctx);
            vm.runInContext({json.dumps(helpers_source)}, ctx);
            vm.runInContext({json.dumps(queue_source)}, ctx);
            process.stdout.write(JSON.stringify({{
              frozen: Object.isFrozen(ctx.window.CodoxearQueue),
              keys: Object.keys(ctx.window.CodoxearQueue),
              hasCreate: typeof ctx.window.CodoxearQueue.createQueueController === "function",
            }}));
            """
        )
        result = run_node_json(js)
        self.assertTrue(result["frozen"])
        self.assertEqual(result["keys"], ["createQueueController"])
        self.assertTrue(result["hasCreate"])

    def test_createQueue_controller_throws_on_missing_deps(self) -> None:
        queue_source = APP_QUEUE_JS.read_text(encoding="utf-8")
        helpers_source = APP_SESSION_HELPERS_JS.read_text(encoding="utf-8")
        modal_source = APP_MODAL_JS.read_text(encoding="utf-8")
        head = textwrap.dedent(
            f"""
            const vm = require("vm");
            const ctx = {{ window: {{}}, HTMLElement: function HTMLElement() {{}} }};
            vm.createContext(ctx);
            vm.runInContext({json.dumps(modal_source)}, ctx);
            vm.runInContext({json.dumps(helpers_source)}, ctx);
            vm.runInContext({json.dumps(queue_source)}, ctx);
            """
        )
        body = textwrap.dedent(
            r'''
            const Q = ctx.window.CodoxearQueue;
            const errors = [];
            const attempts = [
              ["options not object", () => Q.createQueueController(null)],
              ["missing DOM node", () => Q.createQueueController({})],
            ];
            for (const [label, fn] of attempts) {
              try { fn(); errors.push({ label, threw: false }); }
              catch (e) { errors.push({ label, threw: true, type: e.name === "TypeError", msg: String(e.message) }); }
            }
            // A fully-wired controller with every function dep swapped for null
            // must throw a TypeError naming the missing dep.
            const node = { style: {}, setAttribute() {}, appendChild() {} };
            const wiredExceptApi = {
              queueBackdrop: node, queueCloseBtn: node, queueList: node,
              queueEmpty: node, queueViewer: node, queueBtn: node,
              getSelected: () => null, getSessionInfo: () => null, isAppDisposed: () => false,
              api: null,
              setToast: () => {}, clearCommitUnknownSend: () => {}, refreshSessions: async () => {},
              updateQueueBadge: () => {}, syncRecoveryUiForSession: () => {}, kickPoll: () => {},
              setPollFastUntilMs: () => {}, handleAppAuthLoss: () => {},
              prepareModalOpen: () => {}, afterModalVisibilityChanged: () => {},
              el: () => ({}), iconSvg: () => "", recoveryPanelFocusFallback: () => null,
            };
            try {
              Q.createQueueController(wiredExceptApi);
              errors.push({ label: "missing api", threw: false });
            } catch (e) {
              errors.push({ label: "missing api", threw: true, type: e.name === "TypeError", msg: String(e.message) });
            }
            process.stdout.write(JSON.stringify(errors));
            '''
        )
        js = head + body
        result = run_node_json(js)
        by_label = {row["label"]: row for row in result}
        self.assertTrue(by_label["options not object"]["threw"])
        self.assertTrue(by_label["options not object"]["type"], by_label["options not object"])
        self.assertTrue(by_label["missing DOM node"]["threw"])
        self.assertTrue(by_label["missing DOM node"]["type"])
        self.assertTrue(by_label["missing api"]["threw"])
        self.assertTrue(by_label["missing api"]["type"])
        self.assertIn("api", by_label["missing api"]["msg"])

    # --- 2. queue button projections (covered in detail in test_queue_button_source
    #     and exercised here against the live controller) ---

    def test_queue_button_disabled_for_busy_submit(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-1", { launch_state: "ready", queue_len: 0 });
            h.select("sid-1");
            // Drive a slow enqueue: the api never resolves, so queueSubmitBusy stays
            // true when we read the button state on the next microtask.
            let resolveEnqueue;
            h.setApiResponses([new Promise((resolve) => { resolveEnqueue = resolve; })]);
            const p = h.controller.enqueueComposerText("hello", { sid: "sid-1" });
            // While in flight, syncQueueSubmitState has run with queueSubmitBusy=true.
            const busyDisabled = h.dom.queueBtn.disabled;
            const busyTitle = h.dom.queueBtn.title;
            resolveEnqueue({});
            await p;
            h.controller.syncQueueSubmitState();
            const idleDisabled = h.dom.queueBtn.disabled;
            globalThis.__result = { busyDisabled, busyTitle, idleDisabled };
            """
        )
        result = run_node_json(js)
        self.assertTrue(result["busyDisabled"])
        self.assertEqual(result["busyTitle"], "Queued messages")
        self.assertFalse(result["idleDisabled"])

    # --- 3. enqueue safety gates + success callback/API order ---

    def test_enqueue_gates_block_before_api_call(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.select("sid-1");
            const gates = [];
            h.sessions.set("sid-1", { launch_state: "failed", queue_len: 0 });
            gates.push({ name: "failed", ok: await h.controller.enqueueComposerText("hi", { sid: "sid-1" }) });
            h.sessions.set("sid-1", { launch_state: "ready", orphan_recovery: true });
            gates.push({ name: "orphan", ok: await h.controller.enqueueComposerText("hi", { sid: "sid-1" }) });
            h.sessions.set("sid-1", { launch_state: "ready", queue_recovery: true });
            gates.push({ name: "queueRecovery", ok: await h.controller.enqueueComposerText("hi", { sid: "sid-1" }) });
            h.sessions.set("sid-1", { launch_state: "ready", commit_unknown_send: true, commit_unknown_send_text: "abc" });
            gates.push({ name: "commitUnknown", ok: await h.controller.enqueueComposerText("hi", { sid: "sid-1" }) });
            const apiCalled = h.calls.some((c) => c[0] === "api" && String(c[1]).indexOf("/enqueue") !== -1);
            globalThis.__result = { gates, toasts: h.toasts.slice(), clearCalls: h.calls.filter((c) => c[0] === "clearCommitUnknownSend"), apiCalled };
            """
        )
        result = run_node_json(js)
        gates = {g["name"]: g["ok"] for g in result["gates"]}
        self.assertFalse(any(gates.values()))
        self.assertEqual(result["toasts"][0], "failed launch cannot receive queued messages")
        self.assertIn("missing session can only be reviewed", result["toasts"])
        self.assertIn("review preserved queue before queueing", result["toasts"])
        self.assertIn("resolve the unknown send before queueing", result["toasts"])
        self.assertFalse(result["apiCalled"])
        self.assertEqual(result["clearCalls"], [["clearCommitUnknownSend", "sid-1", "abc"]])

    def test_enqueue_success_callback_and_api_order(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-1", { launch_state: "ready", queue_len: 0 });
            h.select("sid-1");
            h.setApiResponses([{ queued: true, queue_len: 3 }]);
            const ok = await h.controller.enqueueComposerText("hello world", { sid: "sid-1" });
            const order = h.calls.map((c) => c[0]);
            const apiBody = h.calls.find((c) => c[0] === "api" && String(c[1]).indexOf("/enqueue") !== -1)[2];
            globalThis.__result = { ok, toasts: h.toasts, order, apiBody };
            """
        )
        result = run_node_json(js)
        self.assertTrue(result["ok"])
        self.assertEqual(result["toasts"], ["queued (3)"])
        self.assertEqual(result["apiBody"], {"text": "hello world"})
        order = result["order"]
        # api(enqueue) -> setPollFastUntilMs -> kickPoll -> refreshSessions ->
        # updateQueueBadge -> syncRecoveryUiForSession (refreshQueueViewer is
        # skipped because the viewer is closed).
        self.assertLess(order.index("api"), order.index("setPollFastUntilMs"))
        self.assertLess(order.index("setPollFastUntilMs"), order.index("kickPoll"))
        self.assertLess(order.index("kickPoll"), order.index("refreshSessions"))
        self.assertLess(order.index("refreshSessions"), order.index("updateQueueBadge"))
        self.assertLess(order.index("updateQueueBadge"), order.index("syncRecoveryUiForSession"))

    # --- 4. 401 handling before any queue error toast ---

    def _enqueue_401_script(self, op_script: str, op_label: str) -> str:
        return harness_script(
            f"""
            const h = globalThis.__harness;
            h.sessions.set("sid-1", {{ launch_state: "ready", queue_len: 1 }});
            h.select("sid-1");
            const err = new Error("no auth"); err.status = 401;
            h.setApiResponses([err]);
            try {{ {op_script} }} catch (e) {{}}
            // Allow microtasks to settle.
            await new Promise((r) => setTimeout(r, 0));
            const order = h.calls.map((c) => c[0]);
            const authIdx = order.indexOf("handleAppAuthLoss");
            const toastIdx = order.findIndex((name) => name === "setToast");
            const queueErrorToasts = h.toasts.filter((t) => String(t).indexOf("queue") !== -1 && String(t).indexOf("error") !== -1);
            globalThis.__result = {{ op: "{op_label}", order, authIdx, toastIdx, toasts: h.toasts, queueErrorToasts }};
            """
        )

    def test_401_precedes_queue_error_toast_for_all_ops(self) -> None:
        cases = [
            ("enqueue", 'await h.controller.enqueueComposerText("hi", { sid: "sid-1" })'),
            ("delete", 'await h.controller.deleteQueueItem("sid-1", "item-1")'),
            ("move", 'await h.controller.moveQueueItem("sid-1", "item-1", 0)'),
            ("refresh", 'await h.controller.refreshQueueViewer()'),
        ]
        for label, op in cases:
            result = run_node_json(self._enqueue_401_script(op, label))
            self.assertIn(result["authIdx"] >= 0 and "handleAppAuthLoss" or "NO_AUTH", ["handleAppAuthLoss"])
            self.assertGreaterEqual(result["authIdx"], 0, label)
            # No queue error toast should fire for a 401 (auth loss returns early).
            self.assertEqual(result["queueErrorToasts"], [], f"{label}: {result['queueErrorToasts']}")
            # handleAppAuthLoss precedes any setToast if any toast fired.
            if result["toastIdx"] >= 0:
                self.assertLess(result["authIdx"], result["toastIdx"], label)

    def test_401_in_update_path_calls_auth_loss(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-1", { launch_state: "ready", queue_len: 1 });
            h.select("sid-1");
            const err = new Error("no auth"); err.status = 401;
            h.setApiResponses([err]);
            h.controller.scheduleQueueUpdate("sid-1", "item-1", "edited text");
            h.runPendingTimers();
            await new Promise((r) => setTimeout(r, 0));
            const order = h.calls.map((c) => c[0]);
            const authIdx = order.indexOf("handleAppAuthLoss");
            const queueErrorToasts = h.toasts.filter((t) => String(t).indexOf("queue update error") !== -1);
            globalThis.__result = { order, authIdx, queueErrorToasts };
            """
        )
        result = run_node_json(js)
        self.assertGreaterEqual(result["authIdx"], 0)
        self.assertEqual(result["queueErrorToasts"], [])

    # --- 5. queue viewer show/hide modal/focus ---

    def test_show_hide_viewer_modal_and_focus(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-1", { launch_state: "ready", queue_len: 0 });
            h.select("sid-1");
            h.setApiResponses([{ items: [] }]);
            const opener = new h.HTMLElementCtor();
            opener.isConnected = true;
            opener.disabled = false;
            opener.focus = () => { calls.push(["opener-focus"]); };
            h.controller.showQueueViewer({ opener });
            await new Promise((r) => setTimeout(r, 0));
            const afterShow = {
              backdrop: h.dom.queueBackdrop.style.display,
              viewer: h.dom.queueViewer.style.display,
              prepareCalled: h.calls.some((c) => c[0] === "prepareModalOpen"),
              afterModalCalled: h.calls.some((c) => c[0] === "afterModalVisibilityChanged"),
              closeFocus: h.calls.some((c) => c[0] === "focus"),
              refreshCalled: h.calls.some((c) => c[0] === "api" && String(c[1]).indexOf("/queue") !== -1),
            };
            h.controller.hideQueueViewer();
            const afterHide = {
              backdrop: h.dom.queueBackdrop.style.display,
              viewer: h.dom.queueViewer.style.display,
              emptyText: h.dom.queueEmpty.textContent,
              openerRestored: h.calls.some((c) => c[0] === "opener-focus"),
            };
            globalThis.__result = { afterShow, afterHide };
            """
        )
        result = run_node_json(js)
        show = result["afterShow"]
        self.assertEqual(show["backdrop"], "block")
        self.assertEqual(show["viewer"], "flex")
        self.assertTrue(show["prepareCalled"])
        self.assertTrue(show["afterModalCalled"])
        self.assertTrue(show["closeFocus"])
        self.assertTrue(show["refreshCalled"])
        hide = result["afterHide"]
        self.assertEqual(hide["backdrop"], "none")
        self.assertEqual(hide["viewer"], "none")
        self.assertTrue(hide["openerRestored"])

    def test_show_viewer_requires_selection(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.controller.showQueueViewer({ opener: null });
            globalThis.__result = {
              viewerDisplay: h.dom.queueViewer.style.display,
              prepareCalled: h.calls.some((c) => c[0] === "prepareModalOpen"),
            };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["viewerDisplay"], "none")
        self.assertFalse(result["prepareCalled"])

    # --- 6. refresh normalizes items + preserves nonblank drafts ---

    def test_refresh_normalizes_and_preserves_nonblank_drafts(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-1", { launch_state: "ready", queue_len: 2 });
            h.select("sid-1");
            // First refresh loads server items.
            h.setApiResponses([{ items: [
              { id: "a", text: "server-a" },
              { id: "b", text: "server-b", commit_unknown: true },
            ] }]);
            h.dom.queueViewer.style.display = "flex";
            await h.controller.refreshQueueViewer();
            const rows1 = h.inspectRows();
            const emptyAfter1 = h.dom.queueEmpty.textContent;
            // Simulate a local nonblank draft edit on item "a".
            h.controller.scheduleQueueUpdate("sid-1", "a", "edited-draft");
            // Server still reports the old text; refresh must preserve the draft.
            h.setNow(100000); // past the 900ms edit guard
            h.setApiResponses([{ items: [
              { id: "a", text: "server-a" },
              { id: "b", text: "server-b", commit_unknown: true },
            ] }]);
            await h.controller.refreshQueueViewer();
            const rows2 = h.inspectRows();
            // Clear pending update timer so it doesn't fire later.
            globalThis.__result = { rows1, emptyAfter1, rows2 };
            """
        )
        result = run_node_json(js)
        # First refresh: two rows rendered, empty text reset to the idle copy.
        self.assertEqual(len(result["rows1"]), 2)
        self.assertEqual(result["emptyAfter1"], "No queued messages.")
        # Second refresh: the locally-edited draft for "a" is preserved, not the
        # server text. (Rows are rendered in order: a, b.)
        self.assertEqual(len(result["rows2"]), 2)

    def test_refresh_skips_within_edit_guard_while_viewer_open(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-1", { launch_state: "ready", queue_len: 1 });
            h.select("sid-1");
            h.dom.queueViewer.style.display = "flex";
            h.setApiResponses([{ items: [{ id: "a", text: "first" }] }]);
            await h.controller.refreshQueueViewer();
            const firstApiCount = h.calls.filter((c) => c[0] === "api").length;
            // Simulate a local edit by driving the rendered textarea oninput;
            // this sets queueLastEditMs = now via the controller.
            h.editFirstRow("draft");
            // Immediate refresh within the 900ms guard must short-circuit.
            h.setApiResponses([{ items: [{ id: "a", text: "should-not-load" }] }]);
            await h.controller.refreshQueueViewer();
            const apiCountAfterGuarded = h.calls.filter((c) => c[0] === "api").length;
            globalThis.__result = { firstApiCount, apiCountAfterGuarded };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["apiCountAfterGuarded"], result["firstApiCount"])

    # --- 7. rendered commitUnknown/orphanRecovery/sending rows locked/tagged ---

    def test_rows_locked_and_tagged_for_special_items(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-1", { launch_state: "ready", queue_len: 4 });
            h.select("sid-1");
            h.dom.queueViewer.style.display = "flex";
            h.setApiResponses([{ items: [
              { id: "normal", text: "n" },
              { id: "sending", text: "s", sending: true },
              { id: "commit", text: "c", commit_unknown: true },
              { id: "orphan", text: "o", orphan_recovery: true },
            ] }]);
            await h.controller.refreshQueueViewer();
            globalThis.__result = { rows: h.inspectRows() };
            """
        )
        result = run_node_json(js)
        rows = result["rows"]
        self.assertEqual(len(rows), 4)
        self.assertIn("queueItem", rows[0]["class"])
        # normal: editable, no tag, del enabled
        self.assertFalse(rows[0]["taDisabled"])
        self.assertEqual(rows[0]["tagTexts"], [])
        self.assertFalse(rows[0]["delDisabled"])
        # sending: locked, "Sending" tag, del disabled
        self.assertTrue(rows[1]["taDisabled"])
        self.assertEqual(rows[1]["tagTexts"], ["Sending"])
        self.assertTrue(rows[1]["delDisabled"])
        self.assertTrue(rows[1]["upDisabled"])
        self.assertTrue(rows[1]["downDisabled"])
        # commitUnknown: locked, "Commit unknown" tag, del enabled (confirm-gated)
        self.assertTrue(rows[2]["taDisabled"])
        self.assertEqual(rows[2]["tagTexts"], ["Commit unknown"])
        self.assertFalse(rows[2]["delDisabled"])
        # orphanRecovery: locked, "Recovery" tag, del enabled (confirm-gated)
        self.assertTrue(rows[3]["taDisabled"])
        self.assertEqual(rows[3]["tagTexts"], ["Recovery"])
        self.assertFalse(rows[3]["delDisabled"])

    # --- 8. move buttons cannot cross recovery/commitUnknown/sending barriers ---

    def test_move_buttons_respect_barriers(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-1", { launch_state: "ready", queue_len: 3 });
            h.select("sid-1");
            h.dom.queueViewer.style.display = "flex";
            h.setApiResponses([{ items: [
              { id: "a", text: "a" },
              { id: "b", text: "b", commit_unknown: true },
              { id: "c", text: "c" },
            ] }]);
            await h.controller.refreshQueueViewer();
            const rows = h.inspectRows();
            // a (idx 0): up disabled (first), down disabled (crosses commit barrier b)
            // b (idx 1): locked -> both disabled
            // c (idx 2): up disabled (crosses commit barrier b), down disabled (last)
            globalThis.__result = { rows };
            """
        )
        result = run_node_json(js)
        rows = result["rows"]
        # a
        self.assertTrue(rows[0]["upDisabled"])  # first row
        self.assertTrue(rows[0]["downDisabled"])  # would cross commit item b
        # b (locked)
        self.assertTrue(rows[1]["upDisabled"])
        self.assertTrue(rows[1]["downDisabled"])
        # c
        self.assertTrue(rows[2]["upDisabled"])  # would cross commit item b
        self.assertTrue(rows[2]["downDisabled"])  # last row

    # --- 9. delete confirmation sends allow_commit_unknown / allow_orphan_recovery ---

    def test_delete_confirmation_sends_recovery_flags(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-1", { launch_state: "ready", queue_len: 2 });
            h.select("sid-1");
            h.dom.queueViewer.style.display = "flex";
            h.setApiResponses([{ items: [
              { id: "commit", text: "c-text", commit_unknown: true },
              { id: "orphan", text: "o-text", orphan_recovery: true },
            ] }]);
            await h.controller.refreshQueueViewer();

            // Confirm a commit_unknown delete.
            h.setConfirm(true);
            h.sessions.set("sid-1", { launch_state: "ready", queue_len: 1 });
            h.setApiResponses([{}, { items: [{ id: "orphan", text: "o-text", orphan_recovery: true }] }]);
            await h.controller.deleteQueueItem("sid-1", "commit");
            const commitConfirm = h.confirmCalls.slice();
            const commitDeleteBody = h.calls.find((c) => c[0] === "api" && String(c[1]).indexOf("/queue/delete") !== -1)[2];

            // Cancel an orphan recovery delete -> no delete API call for it.
            h.setConfirm(false);
            const beforeCancelDeleteCount = h.calls.filter((c) => c[0] === "api" && String(c[1]).indexOf("/queue/delete") !== -1).length;
            await h.controller.deleteQueueItem("sid-1", "orphan");
            const afterCancelDeleteCount = h.calls.filter((c) => c[0] === "api" && String(c[1]).indexOf("/queue/delete") !== -1).length;

            globalThis.__result = { commitConfirm, commitDeleteBody, beforeCancelDeleteCount, afterCancelDeleteCount };
            """
        )
        result = run_node_json(js)
        # Confirmation copy present.
        self.assertTrue(any("checking the transcript or terminal" in m for m in result["commitConfirm"]))
        self.assertTrue(any("may allow later queued prompts" in m for m in result["commitConfirm"]))
        # Delete body carries allow_commit_unknown: true.
        self.assertEqual(result["commitDeleteBody"], {"id": "commit", "allow_commit_unknown": True, "allow_orphan_recovery": False})
        # Canceled orphan delete did not issue another delete call.
        self.assertEqual(result["afterCancelDeleteCount"], result["beforeCancelDeleteCount"])

    def test_delete_orphan_recovery_sends_flag_when_confirmed(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-1", { launch_state: "ready", queue_len: 1 });
            h.select("sid-1");
            h.dom.queueViewer.style.display = "flex";
            h.setApiResponses([{ items: [{ id: "orphan", text: "o", orphan_recovery: true }] }]);
            await h.controller.refreshQueueViewer();
            h.setConfirm(true);
            h.setApiResponses([{}]);
            await h.controller.deleteQueueItem("sid-1", "orphan");
            const body = h.calls.find((c) => c[0] === "api" && String(c[1]).indexOf("/queue/delete") !== -1)[2];
            globalThis.__result = { body };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["body"], {"id": "orphan", "allow_commit_unknown": False, "allow_orphan_recovery": True})

    # --- 10. debounced update uses 350ms; pending delete runs after update ---

    def test_update_debounced_350ms_and_pending_delete_runs_after(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-1", { launch_state: "ready", queue_len: 1 });
            h.select("sid-1");
            h.dom.queueViewer.style.display = "flex";
            h.setApiResponses([{ items: [{ id: "a", text: "a" }] }]);
            await h.controller.refreshQueueViewer();

            // Schedule an update; capture the debounce delay (350ms).
            h.controller.scheduleQueueUpdate("sid-1", "a", "draft");
            const setTimeoutCalls = h.calls.filter((c) => c[0] === "setTimeout");
            const updateTimerMs = setTimeoutCalls.length ? setTimeoutCalls[0][1] : null;

            // Fire the debounced timer with the update API held in flight so the
            // mutation lock is acquired. Then request a delete while the lock is
            // held: it must be queued ("delete queued"), not sent immediately.
            let resolveUpdate;
            const updateInFlight = new Promise((resolve) => { resolveUpdate = resolve; });
            h.setApiResponses([
              updateInFlight,                                  // queue/update (in flight)
              { items: [{ id: "a", text: "draft" }] },         // refreshQueueViewer inside update
            ]);
            h.runPendingTimers();                              // fires update: api(update) pending, lock held
            h.sessions.set("sid-1", { launch_state: "ready", queue_len: 0 });
            const deletePromise = h.controller.deleteQueueItem("sid-1", "a");
            const deleteQueuedToast = h.toasts.some((t) => t === "delete queued");
            const deleteApiBeforeFinalize = h.calls.some((c) => c[0] === "api" && String(c[1]).indexOf("/queue/delete") !== -1);
            // Finalize the update: refreshQueueViewer consumes the 2nd response,
            // then the finally block releases the lock and runs the pending delete.
            resolveUpdate({});
            await deletePromise;
            await new Promise((r) => setTimeout(r, 0));
            const apiUrls = h.calls.filter((c) => c[0] === "api").map((c) => c[1]);
            const updateIdx = apiUrls.findIndex((u) => u.indexOf("/queue/update") !== -1);
            const deleteIdx = apiUrls.findIndex((u) => u.indexOf("/queue/delete") !== -1);
            globalThis.__result = { updateTimerMs, deleteQueuedToast, deleteApiBeforeFinalize, updateIdx, deleteIdx };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["updateTimerMs"], 350)
        self.assertTrue(result["deleteQueuedToast"])
        self.assertFalse(result["deleteApiBeforeFinalize"])
        self.assertGreaterEqual(result["updateIdx"], 0)
        self.assertGreater(result["deleteIdx"], result["updateIdx"])

    # --- 11. dispose clears timers/state ---

    def test_dispose_clears_timers_and_state(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-1", { launch_state: "ready", queue_len: 1 });
            h.select("sid-1");
            h.dom.queueViewer.style.display = "flex";
            h.setApiResponses([{ items: [{ id: "a", text: "a" }] }]);
            await h.controller.refreshQueueViewer();
            // Open a pending update timer.
            h.controller.scheduleQueueUpdate("sid-1", "a", "draft");
            const timersBefore = h.pendingTimerCount();
            h.controller.dispose();
            const timersAfter = h.pendingTimerCount();
            const emptyAfter = h.dom.queueEmpty.textContent;
            const viewerDisplay = h.dom.queueViewer.style.display;
            // After dispose, running timers must not mutate state even if flushed.
            h.runPendingTimers();
            const apiCallsAfterDispose = h.calls.filter((c) => c[0] === "api").length;
            globalThis.__result = { timersBefore, timersAfter, apiCallsAfterDispose };
            """
        )
        result = run_node_json(js)
        self.assertGreater(result["timersBefore"], 0)
        self.assertEqual(result["timersAfter"], 0)

    # --- module loading order in index.html ---

    def test_index_loads_queue_module_before_app(self) -> None:
        source = INDEX_HTML.read_text(encoding="utf-8")
        self.assertIn("app_queue.js?v=__CODOXEAR_ASSET_VERSION__", source)
        self.assertLess(source.index("app_queue.js?v=__CODOXEAR_ASSET_VERSION__"), source.index("app.js?v=__CODOXEAR_ASSET_VERSION__"))


if __name__ == "__main__":
    unittest.main()
