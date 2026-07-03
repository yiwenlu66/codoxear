import json
import os
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_DIAGNOSTICS_JS = ROOT / "codoxear" / "static" / "app_diagnostics.js"
APP_SESSION_HELPERS_JS = ROOT / "codoxear" / "static" / "app_session_helpers.js"
APP_MODAL_JS = ROOT / "codoxear" / "static" / "app_modal.js"
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
const opens = [];
const sessions = new Map();
let selected = null;
let apiResponse = null;
let apiError = null;
let copyError = null;

function fakeNode(extra = {}) {
  return {
    style: { display: "none" },
    _attrs: {},
    _children: [],
    disabled: false,
    textContent: "",
    set innerHTML(v) { this._children = []; },
    get innerHTML() { return ""; },
    setAttribute(name, value) { this._attrs[name] = String(value); },
    appendChild(child) { this._children.push(child); return child; },
    addEventListener() {},
    focus() { calls.push(["focus"]); },
    ...extra,
  };
}

function makeEl(tag, attrs = {}, children = []) {
  const node = fakeNode({ tag });
  const a = attrs || {};
  if (a.class !== undefined) node._class = String(a.class);
  if (a.text !== undefined) node.textContent = String(a.text);
  node.appendChild = (child) => { node._children.push(child); return child; };
  (Array.isArray(children) ? children : [children]).forEach((c) => { if (c) node._children.push(c); });
  return node;
}

const diagBackdrop = fakeNode();
const diagViewer = fakeNode();
const diagContent = fakeNode();
const diagStatus = fakeNode();
const diagCloseBtn = fakeNode();
const diagNewLikeBtn = fakeNode();
const diagCopyBtn = fakeNode();

const deps = {
  diagBackdrop, diagViewer, diagContent, diagStatus, diagCloseBtn, diagNewLikeBtn, diagCopyBtn,
  getSelected: () => selected,
  getSessionInfo: (sid) => sessions.get(sid) || null,
  api: (url) => {
    calls.push(["api", url]);
    if (apiError) { const e = apiError; apiError = null; return Promise.reject(e); }
    return Promise.resolve(apiResponse || {});
  },
  setToast: (t) => { toasts.push(t); calls.push(["setToast", t]); },
  copyToClipboard: (text) => {
    calls.push(["copyToClipboard", text]);
    if (copyError) return Promise.reject(copyError);
    return Promise.resolve(true);
  },
  openNewSessionDialog: (opts) => { opens.push(opts); calls.push(["openNewSessionDialog", opts]); },
  recoveryDetailsText: (sid, s) => `RECOVERY:${sid}:${s && s.session_id}`,
  launchPresetFromSessionInfo: (s) => (s && typeof s === "object" ? { session_id: s.session_id, cwd: s.cwd } : null),
  redactedLaunchErrorText: (e) => `REDACTED:${e}`,
  sessionLaunchLabel: (s) => (s && s.launch_state === "failed" ? "session launch failed" : "web-owned session"),
  agentBackendDisplayName: (b) => `BACKEND:${b}`,
  diagnosticsProviderDisplay: (d) => `PROV:${d && d.provider_choice}`,
  diagnosticsCopyText: (sid, rows) => rows.map((r) => `${r[0]}=${r[1]}`).join("|"),
  fmtTs: (ts) => `TS:${ts}`,
  fmtRelativeAge: (s) => `${s}s`,
  formatPriorityOffset: (v) => `OFF:${v}`,
  prepareModalOpen: () => { calls.push(["prepareModalOpen"]); },
  afterModalVisibilityChanged: () => { calls.push(["afterModalVisibilityChanged"]); },
  el: makeEl,
  uiVersion: "test-ver",
  requestFrame: (fn) => { calls.push(["raf"]); fn(); },
};

const ctx = {
  HTMLElement: function HTMLElement() {},
  document: { activeElement: null },
  window: {},
  console,
};
vm.createContext(ctx);
vm.runInContext(MODAL_SOURCE, ctx);
vm.runInContext(HELPERS_SOURCE, ctx);
vm.runInContext(DIAG_SOURCE, ctx);
const controller = ctx.window.CodoxearDiagnostics.createDiagnosticsController(deps);

function labelValues() {
  return diagContent._children.map((row) => {
    const lab = row._children[0] && row._children[0].textContent;
    const val = row._children[1] && row._children[1].textContent;
    return [lab, val];
  });
}

globalThis.__harness = {
  controller,
  calls,
  toasts,
  opens,
  dom: { diagBackdrop, diagViewer, diagContent, diagStatus, diagCloseBtn, diagNewLikeBtn, diagCopyBtn },
  HTMLElementCtor: ctx.HTMLElement,
  sessions,
  select: (sid) => { selected = sid; },
  setApiResponse: (v) => { apiResponse = v; },
  setApiError: (e) => { apiError = e; },
  setCopyError: (e) => { copyError = e; },
  labelValues,
};
"""


def harness_script(epilogue: str) -> str:
    diag_source = APP_DIAGNOSTICS_JS.read_text(encoding="utf-8")
    helpers_source = APP_SESSION_HELPERS_JS.read_text(encoding="utf-8")
    modal_source = APP_MODAL_JS.read_text(encoding="utf-8")
    js = (
        textwrap.dedent(
            f"""
        const MODAL_SOURCE = {json.dumps(modal_source)};
        const HELPERS_SOURCE = {json.dumps(helpers_source)};
        const DIAG_SOURCE = {json.dumps(diag_source)};
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


class TestFrontendDiagnosticsModuleSource(unittest.TestCase):
    # --- 1. frozen export + missing dep failures ---

    def test_module_export_is_frozen_createDiagnostics_controller(self) -> None:
        diag_source = APP_DIAGNOSTICS_JS.read_text(encoding="utf-8")
        helpers_source = APP_SESSION_HELPERS_JS.read_text(encoding="utf-8")
        modal_source = APP_MODAL_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const vm = require("vm");
            const ctx = {{ window: {{}}, HTMLElement: function HTMLElement() {{}} }};
            vm.createContext(ctx);
            vm.runInContext({json.dumps(modal_source)}, ctx);
            vm.runInContext({json.dumps(helpers_source)}, ctx);
            vm.runInContext({json.dumps(diag_source)}, ctx);
            process.stdout.write(JSON.stringify({{
              frozen: Object.isFrozen(ctx.window.CodoxearDiagnostics),
              keys: Object.keys(ctx.window.CodoxearDiagnostics),
              hasCreate: typeof ctx.window.CodoxearDiagnostics.createDiagnosticsController === "function",
            }}));
            """
        )
        result = run_node_json(js)
        self.assertTrue(result["frozen"])
        self.assertEqual(result["keys"], ["createDiagnosticsController"])
        self.assertTrue(result["hasCreate"])

    def test_createDiagnostics_controller_throws_on_missing_deps(self) -> None:
        diag_source = APP_DIAGNOSTICS_JS.read_text(encoding="utf-8")
        helpers_source = APP_SESSION_HELPERS_JS.read_text(encoding="utf-8")
        modal_source = APP_MODAL_JS.read_text(encoding="utf-8")
        head = textwrap.dedent(
            f"""
            const vm = require("vm");
            const ctx = {{ window: {{}}, HTMLElement: function HTMLElement() {{}} }};
            vm.createContext(ctx);
            vm.runInContext({json.dumps(modal_source)}, ctx);
            vm.runInContext({json.dumps(helpers_source)}, ctx);
            vm.runInContext({json.dumps(diag_source)}, ctx);
            """
        )
        body = textwrap.dedent(
            r'''
            const D = ctx.window.CodoxearDiagnostics;
            const errors = [];
            const attempts = [
              ["options not object", () => D.createDiagnosticsController(null)],
              ["missing DOM node", () => D.createDiagnosticsController({})],
            ];
            for (const [label, fn] of attempts) {
              try { fn(); errors.push({ label, threw: false }); }
              catch (e) { errors.push({ label, threw: true, type: e.name === "TypeError", msg: String(e.message) }); }
            }
            // Fully wired except `api` (a function dep) -> TypeError naming api.
            const node = { style: {}, setAttribute() {}, appendChild() {} };
            const wiredExceptApi = {
              diagBackdrop: node, diagViewer: node, diagContent: node, diagStatus: node,
              diagCloseBtn: node, diagNewLikeBtn: node, diagCopyBtn: node,
              getSelected: () => null, getSessionInfo: () => null,
              api: null,
              setToast: () => {}, copyToClipboard: () => {}, openNewSessionDialog: () => {},
              recoveryDetailsText: () => "", launchPresetFromSessionInfo: () => null,
              redactedLaunchErrorText: () => "", sessionLaunchLabel: () => "",
              agentBackendDisplayName: () => "", diagnosticsProviderDisplay: () => "",
              diagnosticsCopyText: () => "", fmtTs: () => "", fmtRelativeAge: () => "",
              formatPriorityOffset: () => "", prepareModalOpen: () => {},
              afterModalVisibilityChanged: () => {}, el: () => ({}), uiVersion: "v",
            };
            try {
              D.createDiagnosticsController(wiredExceptApi);
              errors.push({ label: "missing api", threw: false });
            } catch (e) {
              errors.push({ label: "missing api", threw: true, type: e.name === "TypeError", msg: String(e.message) });
            }
            // Fully wired except `uiVersion` (a string dep) -> TypeError naming uiVersion.
            const wiredExceptVersion = Object.assign({}, wiredExceptApi, { api: () => Promise.resolve({}), uiVersion: null });
            try {
              D.createDiagnosticsController(wiredExceptVersion);
              errors.push({ label: "missing uiVersion", threw: false });
            } catch (e) {
              errors.push({ label: "missing uiVersion", threw: true, type: e.name === "TypeError", msg: String(e.message) });
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
        self.assertIn("api", by_label["missing api"]["msg"])
        self.assertTrue(by_label["missing uiVersion"]["threw"])
        self.assertTrue(by_label["missing uiVersion"]["type"])
        self.assertIn("uiVersion", by_label["missing uiVersion"]["msg"])

    def test_module_load_fails_loud_without_helpers_or_modal(self) -> None:
        diag_source = APP_DIAGNOSTICS_JS.read_text(encoding="utf-8")
        js_only_diag = textwrap.dedent(
            f"""
            const vm = require("vm");
            const ctx = {{ window: {{}}, HTMLElement: function HTMLElement() {{}} }};
            vm.createContext(ctx);
            let err = null;
            try {{ vm.runInContext({json.dumps(diag_source)}, ctx); }}
            catch (e) {{ err = String(e.message); }}
            process.stdout.write(JSON.stringify({{ err }}));
            """
        )
        result = run_node_json(js_only_diag)
        self.assertIn("failed to load", result["err"])

    # --- 2. failed-launch / local path: no API, recovery rows, copy/new-like enabled ---

    def test_failed_launch_path_renders_locally_without_api(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("launch-dead", {
              session_id: "launch-dead",
              launch_state: "failed",
              launch_stage: "pty_fork",
              launch_error: "boom",
              cwd: "/tmp/work",
              agent_backend: "pi",
              provider_choice: "macaron",
              model: "claude-haiku-4-5",
              reasoning_effort: "medium",
              tmux_session: "codex",
              tmux_window: "1",
            });
            h.select("launch-dead");
            await h.controller.show({ opener: null });
            const apiCalled = h.calls.some((c) => c[0] === "api");
            const rows = h.labelValues();
            const labels = rows.map((r) => r[0]);
            const statusText = h.dom.diagStatus.textContent;
            const copyDisabled = h.dom.diagCopyBtn.disabled;
            const newLikeDisabled = h.dom.diagNewLikeBtn.disabled;
            const copyCall = h.calls.find((c) => c[0] === "copyToClipboard");
            await h.controller.onCopyClick({});
            const copyAfter = h.calls.find((c) => c[0] === "copyToClipboard");
            globalThis.__result = { apiCalled, labels, statusText, copyDisabled, newLikeDisabled, copyAfter };
            """
        )
        result = run_node_json(js)
        self.assertFalse(result["apiCalled"])
        for label in ["Session", "State", "Stage", "Error", "CWD", "Agent", "Provider", "Model", "Reasoning", "tmux"]:
            self.assertIn(label, result["labels"], label)
        # State row says "launch failed".
        state_row = next(r for r in result["labels"] if True)  # placeholder; check below
        rows_map = dict(zip(result["labels"], [None] * len(result["labels"])))
        # Re-fetch with values.
        # (The harness labelValues returns [label, value]; we kept labels only here.)
        self.assertEqual(result["statusText"], "")
        self.assertFalse(result["copyDisabled"])
        self.assertFalse(result["newLikeDisabled"])
        # Copy details on the failed-launch path copies the recoveryDetailsText output.
        self.assertIsNotNone(result["copyAfter"])
        self.assertEqual(result["copyAfter"][1], "RECOVERY:launch-dead:launch-dead")

    def test_failed_launch_path_with_values(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("launch-dead", {
              session_id: "launch-dead",
              launch_state: "failed",
              launch_stage: "pty_fork",
              launch_error: "boom SECRET: abc",
              cwd: "/tmp/work",
              agent_backend: "pi",
              provider_choice: "macaron",
              model: "claude-haiku-4-5",
              reasoning_effort: "medium",
              tmux_session: "codex",
              tmux_window: "1",
            });
            h.select("launch-dead");
            await h.controller.show({ opener: null });
            const rows = {};
            h.labelValues().forEach(([k, v]) => { rows[k] = v; });
            globalThis.__result = { rows };
            """
        )
        result = run_node_json(js)
        rows = result["rows"]
        self.assertEqual(rows["Session"], "launch-dead")
        self.assertEqual(rows["State"], "launch failed")
        self.assertEqual(rows["Stage"], "pty_fork")
        self.assertEqual(rows["Error"], "REDACTED:boom SECRET: abc")
        self.assertEqual(rows["CWD"], "/tmp/work")
        self.assertEqual(rows["Agent"], "BACKEND:pi")
        self.assertEqual(rows["Provider"], "PROV:macaron")
        self.assertEqual(rows["Model"], "claude-haiku-4-5")
        self.assertEqual(rows["Reasoning"], "medium")
        self.assertEqual(rows["tmux"], "codex:1")

    # --- 3. live path: /diagnostics, row labels including Provider/Model/Reasoning/Context/UI, copy text, new-like preset ---

    def test_live_path_fetches_diagnostics_and_renders_rows(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-1", { session_id: "sid-1", launch_state: "ready", agent_backend: "codex" });
            h.select("sid-1");
            h.setApiResponse({
              session_id: "sid-1",
              thread_id: "thr-1",
              busy: true,
              queue_len: 2,
              cwd: "/repo",
              start_ts: 1000,
              updated_ts: 2000,
              broker_pid: 11,
              agent_backend: "codex",
              codex_pid: 22,
              log_path: "/log/path",
              tmux_session: "t",
              tmux_window: "2",
              git_branch: "main",
              provider_choice: "chatgpt",
              model: "gpt-5.4",
              reasoning_effort: "high",
              service_tier: "fast",
              final_priority: 0.12,
              priority_offset: 0.05,
              snooze_until: 3000,
              dependency_session_id: "dep-1",
              token: { context_window: 200000, tokens_in_context: 1000, percent_remaining: 99, max_input_tokens: 195000, reserved_tokens: 5000 },
            });
            await h.controller.show({ opener: null });
            const apiCalls = h.calls.filter((c) => c[0] === "api").map((c) => c[1]);
            const rows = {};
            h.labelValues().forEach(([k, v]) => { rows[k] = v; });
            // Capture copy text via the injected copyToClipboard.
            await h.controller.onCopyClick({});
            const copyCall = h.calls.filter((c) => c[0] === "copyToClipboard").pop();
            const newLikeDisabled = h.dom.diagNewLikeBtn.disabled;
            const copyDisabled = h.dom.diagCopyBtn.disabled;
            globalThis.__result = { apiCalls, rows, copyText: copyCall && copyCall[1], newLikeDisabled, copyDisabled };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["apiCalls"], ["/api/sessions/sid-1/diagnostics"])
        rows = result["rows"]
        for label in ["Session", "Thread", "Owned", "Busy", "Queue", "CWD", "Started", "Updated", "Broker PID", "Agent", "Agent PID", "Log", "tmux", "Branch", "Provider", "Model", "Reasoning", "Service tier", "Priority", "Priority offset", "Snooze", "Depends on", "UI", "Context"]:
            self.assertIn(label, rows, label)
        self.assertEqual(rows["Provider"], "PROV:chatgpt")
        self.assertEqual(rows["Model"], "gpt-5.4")
        self.assertEqual(rows["Reasoning"], "high")
        self.assertEqual(rows["UI"], "test-ver")
        self.assertIn("Context", rows)
        # Copy text includes the Session row via diagnosticsCopyText semantics.
        self.assertIn("Session=sid-1", result["copyText"])
        self.assertIn("Provider=PROV:chatgpt", result["copyText"])
        self.assertFalse(result["copyDisabled"])
        self.assertFalse(result["newLikeDisabled"])

    def test_live_path_creates_new_like_preset_from_diagnostics_response(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-1", { session_id: "sid-1", launch_state: "ready", agent_backend: "pi" });
            h.select("sid-1");
            h.setApiResponse({
              session_id: "sid-1",
              cwd: "/repo",
              agent_backend: "pi",
              provider_choice: "macaron",
              model_provider: "anthropic",
              preferred_auth_method: "api_key",
              model: "claude-haiku-4-5",
              reasoning_effort: "medium",
              service_tier: "fast",
              transport: "tmux",
              tmux_session: "codex",
              tmux_window: "1",
            });
            await h.controller.show({ opener: null });
            const newLikeDisabled = h.dom.diagNewLikeBtn.disabled;
            await h.controller.onNewLikeClick({});
            const openCall = h.opens.pop();
            globalThis.__result = { newLikeDisabled, openCall };
            """
        )
        result = run_node_json(js)
        self.assertFalse(result["newLikeDisabled"])
        self.assertIsNotNone(result["openCall"])
        preset = result["openCall"]["likeSession"]
        self.assertEqual(preset["session_id"], "sid-1")
        self.assertEqual(preset["cwd"], "/repo")
        self.assertEqual(preset["agent_backend"], "pi")
        self.assertEqual(preset["provider_choice"], "macaron")
        self.assertEqual(preset["model_provider"], "anthropic")
        self.assertEqual(preset["preferred_auth_method"], "api_key")
        self.assertEqual(preset["model"], "claude-haiku-4-5")
        self.assertEqual(preset["reasoning_effort"], "medium")
        self.assertEqual(preset["service_tier"], "fast")
        self.assertEqual(preset["transport"], "tmux")
        self.assertEqual(preset["tmux_session"], "codex")
        self.assertEqual(preset["tmux_window"], "1")
        self.assertEqual(result["openCall"]["statusText"], "Review copied launch settings before starting.")

    # --- 4. stale response ignored when selected changes ---

    def test_stale_response_ignored_when_selected_changes(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-1", { session_id: "sid-1", launch_state: "ready", agent_backend: "codex" });
            h.sessions.set("sid-2", { session_id: "sid-2", launch_state: "ready", agent_backend: "codex" });
            h.select("sid-1");
            // Hold the API in flight, then change selection before it resolves.
            let resolveApi;
            h.setApiResponse(new Promise((resolve) => { resolveApi = resolve; }));
            const showPromise = h.controller.show({ opener: null });
            // Selection changes while waiting.
            h.select("sid-2");
            resolveApi({ session_id: "sid-1", agent_backend: "codex", provider_choice: "chatgpt", model: "stale-model" });
            await showPromise;
            const rows = {};
            h.labelValues().forEach(([k, v]) => { rows[k] = v; });
            const copyDisabled = h.dom.diagCopyBtn.disabled;
            const newLikeDisabled = h.dom.diagNewLikeBtn.disabled;
            const statusText = h.dom.diagStatus.textContent;
            globalThis.__result = { rows, copyDisabled, newLikeDisabled, statusText };
            """
        )
        result = run_node_json(js)
        # No rows rendered for the stale session; status stays at "Loading..." and
        # buttons remain disabled because renderLiveRows never ran.
        self.assertEqual(result["rows"], {})
        self.assertTrue(result["copyDisabled"])
        self.assertTrue(result["newLikeDisabled"])
        self.assertEqual(result["statusText"], "Loading...")

    def test_stale_response_after_error_ignored_when_selected_changes(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-1", { session_id: "sid-1", launch_state: "ready", agent_backend: "codex" });
            h.sessions.set("sid-2", { session_id: "sid-2", launch_state: "ready", agent_backend: "codex" });
            h.select("sid-1");
            let rejectApi;
            // Hold the API in flight via a pending promise that will reject.
            h.setApiResponse(new Promise((_, reject) => { rejectApi = reject; }));
            const showPromise = h.controller.show({ opener: null });
            h.select("sid-2");
            rejectApi(new Error("late boom"));
            await showPromise;
            const statusText = h.dom.diagStatus.textContent;
            globalThis.__result = { statusText };
            """
        )
        result = run_node_json(js)
        # Error path also short-circuits when selection changed: status not overwritten.
        self.assertEqual(result["statusText"], "Loading...")

    # --- 5. error path: disables copy/new-like, writes "error: ..." status ---

    def test_error_path_disables_buttons_and_writes_status(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-1", { session_id: "sid-1", launch_state: "ready", agent_backend: "codex" });
            h.select("sid-1");
            h.setApiError(new Error("kaboom"));
            await h.controller.show({ opener: null });
            const copyDisabled = h.dom.diagCopyBtn.disabled;
            const newLikeDisabled = h.dom.diagNewLikeBtn.disabled;
            const statusText = h.dom.diagStatus.textContent;
            // Copy with cleared text -> "details not loaded" toast, no clipboard call.
            const copyCallsBefore = h.calls.filter((c) => c[0] === "copyToClipboard").length;
            await h.controller.onCopyClick({});
            const toastsAfter = h.toasts.slice();
            const copyCallsAfter = h.calls.filter((c) => c[0] === "copyToClipboard").length;
            globalThis.__result = { copyDisabled, newLikeDisabled, statusText, toastsAfter, copyCallsBefore, copyCallsAfter };
            """
        )
        result = run_node_json(js)
        self.assertTrue(result["copyDisabled"])
        self.assertTrue(result["newLikeDisabled"])
        self.assertEqual(result["statusText"], "error: kaboom")
        self.assertEqual(result["toastsAfter"][-1], "details not loaded")
        self.assertEqual(result["copyCallsAfter"], result["copyCallsBefore"])

    # --- 6. copy button copies current diagCopyText and toasts exact success ---

    def test_copy_button_copies_current_text_and_toasts_success(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-1", { session_id: "sid-1", launch_state: "ready", agent_backend: "codex" });
            h.select("sid-1");
            h.setApiResponse({ session_id: "sid-1", agent_backend: "codex", provider_choice: "chatgpt" });
            await h.controller.show({ opener: null });
            const copyCallsBefore = h.calls.filter((c) => c[0] === "copyToClipboard").length;
            await h.controller.onCopyClick({});
            const copyCall = h.calls.filter((c) => c[0] === "copyToClipboard").pop();
            const lastToast = h.toasts[h.toasts.length - 1];
            globalThis.__result = { copyCallsBefore, copiedText: copyCall && copyCall[1], lastToast };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["copyCallsBefore"], 0)
        self.assertIn("Session=sid-1", result["copiedText"])
        self.assertEqual(result["lastToast"], "Copied details")

    def test_copy_button_toasts_copy_failed_on_error(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-1", { session_id: "sid-1", launch_state: "ready", agent_backend: "codex" });
            h.select("sid-1");
            h.setApiResponse({ session_id: "sid-1", agent_backend: "codex" });
            await h.controller.show({ opener: null });
            h.setCopyError(new Error("denied"));
            await h.controller.onCopyClick({});
            globalThis.__result = { lastToast: h.toasts[h.toasts.length - 1] };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["lastToast"], "copy failed: denied")

    # --- 7. New-like click: applies preset, hides without restoring focus, opens New Session ---

    def test_new_like_click_hides_without_focus_restore_and_opens_dialog(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-1", { session_id: "sid-1", launch_state: "ready", agent_backend: "pi" });
            h.select("sid-1");
            h.setApiResponse({ session_id: "sid-1", cwd: "/repo", agent_backend: "pi", provider_choice: "macaron" });
            const opener = new h.HTMLElementCtor();
            opener.isConnected = true;
            opener.disabled = false;
            opener.focus = () => { calls.push(["opener-focus"]); };
            await h.controller.show({ opener });
            const backdropBefore = h.dom.diagBackdrop.style.display;
            // New-like click should hide the diag modal WITHOUT focus restore.
            h.controller.onNewLikeClick({});
            const backdropAfter = h.dom.diagBackdrop.style.display;
            const viewerAfter = h.dom.diagViewer.style.display;
            const openCall = h.opens.pop();
            const openerFocusCalled = h.calls.some((c) => c[0] === "opener-focus");
            globalThis.__result = { backdropBefore, backdropAfter, viewerAfter, openCall, openerFocusCalled };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["backdropBefore"], "block")
        self.assertEqual(result["backdropAfter"], "none")
        self.assertEqual(result["viewerAfter"], "none")
        self.assertIsNotNone(result["openCall"])
        self.assertEqual(result["openCall"]["statusText"], "Review copied launch settings before starting.")
        # hide({ restoreFocus: false }) must not invoke the opener's focus.
        self.assertFalse(result["openerFocusCalled"])

    def test_new_like_click_with_no_preset_toasts_not_loaded(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-1", { session_id: "sid-1", launch_state: "ready", agent_backend: "codex" });
            h.select("sid-1");
            h.setApiError(new Error("x"));
            await h.controller.show({ opener: null });
            const opensBefore = h.opens.length;
            h.controller.onNewLikeClick({});
            globalThis.__result = { lastToast: h.toasts[h.toasts.length - 1], opensAfter: h.opens.length - opensBefore };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["lastToast"], "details not loaded")
        self.assertEqual(result["opensAfter"], 0)

    # --- 8. show/hide modal focus + backdrop behavior ---

    def test_show_requires_selection_and_does_not_open_modal(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            await h.controller.show({ opener: null });
            globalThis.__result = {
              backdrop: h.dom.diagBackdrop.style.display,
              viewer: h.dom.diagViewer.style.display,
              prepareCalled: h.calls.some((c) => c[0] === "prepareModalOpen"),
            };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["backdrop"], "none")
        self.assertEqual(result["viewer"], "none")
        self.assertFalse(result["prepareCalled"])

    def test_show_hide_modal_focus_and_backdrop(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-1", { session_id: "sid-1", launch_state: "failed", agent_backend: "codex" });
            h.select("sid-1");
            const opener = new h.HTMLElementCtor();
            opener.isConnected = true;
            opener.disabled = false;
            opener.focus = () => { calls.push(["opener-focus"]); };
            await h.controller.show({ opener });
            const afterShow = {
              backdrop: h.dom.diagBackdrop.style.display,
              viewer: h.dom.diagViewer.style.display,
              statusText: h.dom.diagStatus.textContent,
              prepareCalled: h.calls.some((c) => c[0] === "prepareModalOpen"),
              afterModalCalled: h.calls.some((c) => c[0] === "afterModalVisibilityChanged"),
              closeFocus: h.calls.some((c) => c[0] === "raf"),
            };
            h.controller.hide({});
            const afterHide = {
              backdrop: h.dom.diagBackdrop.style.display,
              viewer: h.dom.diagViewer.style.display,
              openerRestored: h.calls.some((c) => c[0] === "opener-focus"),
            };
            globalThis.__result = { afterShow, afterHide };
            """
        )
        result = run_node_json(js)
        show = result["afterShow"]
        self.assertEqual(show["backdrop"], "block")
        self.assertEqual(show["viewer"], "flex")
        self.assertEqual(show["statusText"], "")  # failed-launch clears status
        self.assertTrue(show["prepareCalled"])
        self.assertTrue(show["afterModalCalled"])
        self.assertTrue(show["closeFocus"])
        hide = result["afterHide"]
        self.assertEqual(hide["backdrop"], "none")
        self.assertEqual(hide["viewer"], "none")
        self.assertTrue(hide["openerRestored"])

    def test_hide_default_does_not_restore_when_modal_was_closed(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            // hide() on an already-closed modal must not restore focus.
            const opener = new h.HTMLElementCtor();
            opener.isConnected = true;
            opener.disabled = false;
            opener.focus = () => { calls.push(["opener-focus"]); };
            h.controller.hide({});
            globalThis.__result = { openerRestored: h.calls.some((c) => c[0] === "opener-focus") };
            """
        )
        result = run_node_json(js)
        self.assertFalse(result["openerRestored"])

    def test_dispose_resets_state_and_disables_buttons(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-1", { session_id: "sid-1", launch_state: "failed", agent_backend: "codex" });
            h.select("sid-1");
            await h.controller.show({ opener: null });
            // Before dispose, copy button is enabled (recovery text exists).
            const copyEnabledBefore = !h.dom.diagCopyBtn.disabled;
            h.controller.dispose();
            const copyDisabledAfter = h.dom.diagCopyBtn.disabled;
            const newLikeDisabledAfter = h.dom.diagNewLikeBtn.disabled;
            // After dispose, a copy click must toast "details not loaded".
            const copyCallsBefore = h.calls.filter((c) => c[0] === "copyToClipboard").length;
            await h.controller.onCopyClick({});
            const toast = h.toasts[h.toasts.length - 1];
            const copyCallsAfter = h.calls.filter((c) => c[0] === "copyToClipboard").length;
            globalThis.__result = { copyEnabledBefore, copyDisabledAfter, newLikeDisabledAfter, toast, copyCallsBefore, copyCallsAfter };
            """
        )
        result = run_node_json(js)
        self.assertTrue(result["copyEnabledBefore"])
        self.assertTrue(result["copyDisabledAfter"])
        self.assertTrue(result["newLikeDisabledAfter"])
        self.assertEqual(result["toast"], "details not loaded")
        self.assertEqual(result["copyCallsAfter"], result["copyCallsBefore"])

    # --- index.html load order ---

    def test_index_loads_diagnostics_module_after_deps_before_app(self) -> None:
        source = INDEX_HTML.read_text(encoding="utf-8")
        self.assertIn("app_diagnostics.js?v=__CODOXEAR_ASSET_VERSION__", source)
        self.assertLess(source.index("app_session_helpers.js?v=__CODOXEAR_ASSET_VERSION__"), source.index("app_diagnostics.js?v=__CODOXEAR_ASSET_VERSION__"))
        self.assertLess(source.index("app_modal.js?v=__CODOXEAR_ASSET_VERSION__"), source.index("app_diagnostics.js?v=__CODOXEAR_ASSET_VERSION__"))
        self.assertLess(source.index("app_diagnostics.js?v=__CODOXEAR_ASSET_VERSION__"), source.index("app.js?v=__CODOXEAR_ASSET_VERSION__"))

    # --- app.js integration: app.js delegates and keeps DOM construction ---

    def test_app_js_delegates_diag_behavior_to_controller(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        # Controller instantiation with fail-loud guard.
        self.assertIn("const codoxearDiagnostics = window.CodoxearDiagnostics;", source)
        self.assertIn('throw new Error("Codoxear diagnostics controller failed to load")', source)
        self.assertIn("codoxearDiagnostics.createDiagnosticsController({", source)
        # Thin delegating wrappers.
        self.assertIn("return diagController.show(opts);", source)
        self.assertIn("return diagController.hide(opts);", source)
        # Button handlers delegate to the controller.
        self.assertIn("diagNewLikeBtn.onclick = (e) => diagController.onNewLikeClick(e);", source)
        self.assertIn("diagCopyBtn.onclick = (e) => diagController.onCopyClick(e);", source)
        # DOM construction stays in app.js.
        self.assertIn('id: "diagBackdrop"', source)
        self.assertIn('id: "diagNewLikeBtn"', source)
        self.assertIn('id: "diagCopyBtn"', source)
        self.assertIn('id: "diagCloseBtn"', source)
        self.assertIn('id: "diagStatus"', source)
        self.assertIn('id: "diagContent"', source)
        self.assertIn('id: "diagViewer", role: "dialog", "aria-modal": "true", "aria-label": "Details"', source)
        # app.js keeps the diagBtn (Details opener) and disables it when no selection.
        self.assertIn("diagBtn.disabled = !selected;", source)
        self.assertIn("void showDiagViewer({ opener: e.currentTarget });", source)
        # Dispose hook.
        self.assertIn("if (diagController) diagController.dispose();", source)
        # The old diag state locals moved out of app.js into the controller.
        self.assertNotIn("let diagReturnFocusEl = null;", source)
        self.assertNotIn("let diagCopyText = \"\";", source)
        self.assertNotIn("let diagNewLikeSession = null;", source)
        # The old inline rendering authority moved out of app.js.
        self.assertNotIn("async function showDiagViewer({ opener = null } = {}) {", source)
        self.assertNotIn("diagCopyText = recoveryDetailsText(sid, selectedInfo);", source)
        self.assertNotIn("const d = await api(`/api/sessions/${sid}/diagnostics`);", source)


if __name__ == "__main__":
    unittest.main()
