import json
import os
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_RECOVERY_JS = ROOT / "codoxear" / "static" / "app_recovery.js"
APP_SESSION_HELPERS_JS = ROOT / "codoxear" / "static" / "app_session_helpers.js"
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


# The recovery controller renders into a fake DOM tree. The harness builds a
# minimal tree model so the controller's querySelector / querySelectorAll /
# closest / insertBefore calls behave like real DOM for the selectors and
# relationships the recovery panel uses (.recovery-panel-row, .recovery-panel,
# .icon-btn, button). Focus is observed through a calls log.
HARNESS = r"""
const vm = require("vm");
const calls = [];
const toasts = [];
const opens = [];
const queueOpens = [];
const sessions = new Map();
let copyError = null;
let activeElement = null;

function hasClass(node, name) {
  const cls = node && typeof node._class === "string" ? node._class : "";
  return cls.split(/\s+/).indexOf(name) !== -1;
}

function matchesSelector(node, selector) {
  if (!node) return false;
  if (selector === "button") return node.tag === "button";
  if (selector === "ul") return node.tag === "ul";
  if (selector === "pre") return node.tag === "pre";
  if (selector === ".recovery-panel-row") return hasClass(node, "recovery-panel-row");
  if (selector === ".recovery-panel") return hasClass(node, "recovery-panel");
  if (selector === ".icon-btn") return hasClass(node, "icon-btn");
  if (selector === ".recovery-panel-row button") return node.tag === "button" && hasAncestorClass(node, "recovery-panel-row");
  if (selector === ".recovery-panel .icon-btn") return hasClass(node, "icon-btn") && hasAncestorClass(node, "recovery-panel");
  return false;
}

function hasAncestorClass(node, name) {
  let cur = node && node._parent ? node._parent : null;
  while (cur) {
    if (hasClass(cur, name)) return true;
    cur = cur._parent ? cur._parent : null;
  }
  return false;
}

function queryAll(root, selector) {
  const out = [];
  const walk = (n) => {
    if (!n) return;
    const children = Array.isArray(n._children) ? n._children : [];
    for (const c of children) {
      if (matchesSelector(c, selector)) out.push(c);
      walk(c);
    }
  };
  walk(root);
  return out;
}

function makeEl(tag, attrs = {}, children = []) {
  const a = attrs || {};
  const node = {
    tag,
    _class: a.class !== undefined ? String(a.class) : "",
    _attrs: {},
    _children: [],
    _parent: null,
    style: { display: "" },
    dataset: {},
    disabled: false,
    isConnected: true,
    textContent: a.text !== undefined ? String(a.text) : "",
    value: "",
    setAttribute(name, value) { this._attrs[name] = String(value); },
    getAttribute(name) { return Object.prototype.hasOwnProperty.call(this._attrs, name) ? this._attrs[name] : null; },
    removeAttribute(name) { delete this._attrs[name]; },
    appendChild(child) {
      if (!child) return child;
      child._parent = this;
      this._children.push(child);
      return child;
    },
    insertBefore(child, ref) {
      if (!child) return child;
      child._parent = this;
      const idx = ref ? this._children.indexOf(ref) : -1;
      if (idx === -1) this._children.push(child);
      else this._children.splice(idx, 0, child);
      return child;
    },
    remove() {
      if (this._parent) {
        const i = this._parent._children.indexOf(this);
        if (i !== -1) this._parent._children.splice(i, 1);
        this._parent = null;
      }
      this.isConnected = false;
    },
    querySelector(selector) { const all = queryAll(this, selector); return all.length ? all[0] : null; },
    querySelectorAll(selector) { return queryAll(this, selector); },
    closest(selector) {
      let cur = this;
      while (cur) {
        if (matchesSelector(cur, selector)) return cur;
        cur = cur._parent ? cur._parent : null;
      }
      return null;
    },
    addEventListener() {},
    focus(opts) { calls.push(["focus", this.textContent || this._class || this.tag, opts && opts.preventScroll ? "preventScroll" : ""]); },
  };
  for (const [k, v] of Object.entries(a)) {
    if (k === "class" || k === "text") continue;
    node._attrs[k] = String(v);
  }
  (Array.isArray(children) ? children : [children]).forEach((c) => { if (c) node.appendChild(c); });
  return node;
}

const chatInner = makeEl("div", { class: "chatInner" });
const queueBtn = makeEl("button", { class: "icon-btn", text: "Queue" });
const typingAnchor = makeEl("div", { class: "typing-row" });
chatInner.appendChild(typingAnchor);

const deps = {
  chatInner,
  queueBtn,
  typingRowAnchor: () => typingAnchor,
  getSessionInfo: (sid) => sessions.get(sid) || null,
  el: makeEl,
  recoveryPromptPreview: (text, maxLen) => `[PREVIEW:${maxLen || 320}:${String(text || "")}]`,
  redactedLaunchErrorText: (e) => `REDACTED:${e}`,
  recoveryDetailsText: (sid, s) => `RECOVERY_DETAILS:${sid}:${s && s.session_id}`,
  launchPresetFromSessionInfo: (s) => (s && typeof s === "object" ? { session_id: s.session_id, cwd: s.cwd } : null),
  showQueueViewer: (opts) => { queueOpens.push(opts); calls.push(["showQueueViewer", opts]); },
  clearCommitUnknownSend: (sid, text) => { calls.push(["clearCommitUnknownSend", sid, text]); return Promise.resolve(true); },
  openNewSessionDialog: (opts) => { opens.push(opts); calls.push(["openNewSessionDialog", opts]); },
  dismissFailedLaunchRecord: (sid) => { calls.push(["dismissFailedLaunchRecord", sid]); return Promise.resolve(true); },
  copyToClipboard: (text) => {
    calls.push(["copyToClipboard", text]);
    if (copyError) return Promise.reject(copyError);
    return Promise.resolve(true);
  },
  setToast: (t) => { toasts.push(t); calls.push(["setToast", t]); },
  requestFrame: (fn) => { calls.push(["raf"]); fn(); },
};

const ctx = {
  HTMLElement: function HTMLElement() {},
  document: {
    get activeElement() { return activeElement; },
    set activeElement(v) { activeElement = v; },
    querySelector: (sel) => null,
  },
  window: {},
  console,
};
vm.createContext(ctx);
vm.runInContext(HELPERS_SOURCE, ctx);
vm.runInContext(RECOVERY_SOURCE, ctx);
const controller = ctx.window.CodoxearRecovery.createRecoveryPanelController(deps);

function resetTree() {
  // Remove any rendered recovery rows but keep the typing anchor in place.
  for (const row of Array.from(chatInner.querySelectorAll(".recovery-panel-row"))) row.remove();
}

function currentRow() {
  return chatInner.querySelector(".recovery-panel-row") || null;
}

function actionButtons() {
  const row = currentRow();
  if (!row) return [];
  return row.querySelectorAll("button");
}

function findButton(text) {
  return actionButtons().find((b) => String(b.textContent || "").trim() === text) || null;
}

function listTexts() {
  const row = currentRow();
  if (!row) return [];
  const ul = queryAll(row, "ul").find((n) => hasClass(n, "recoveryPanelList")) || queryAll(row, "ul")[0];
  if (!ul) return [];
  return ul._children.filter((c) => c.tag === "li").map((c) => c.textContent);
}

function previewTexts() {
  const row = currentRow();
  if (!row) return [];
  return queryAll(row, "pre").map((p) => p.textContent);
}

globalThis.__harness = {
  controller,
  calls,
  toasts,
  opens,
  queueOpens,
  dom: { chatInner, queueBtn, typingAnchor },
  HTMLElementCtor: ctx.HTMLElement,
  sessions,
  setActive: (el) => { activeElement = el; },
  setCopyError: (e) => { copyError = e; },
  resetTree,
  currentRow,
  actionButtons,
  findButton,
  listTexts,
  previewTexts,
  makeEl,
  createControllerWithDeps: (overrides) => ctx.window.CodoxearRecovery.createRecoveryPanelController(Object.assign({}, deps, overrides)),
};
"""


def harness_script(epilogue: str) -> str:
    recovery_source = APP_RECOVERY_JS.read_text(encoding="utf-8")
    helpers_source = APP_SESSION_HELPERS_JS.read_text(encoding="utf-8")
    js = (
        textwrap.dedent(
            f"""
        const HELPERS_SOURCE = {json.dumps(helpers_source)};
        const RECOVERY_SOURCE = {json.dumps(recovery_source)};
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


class TestFrontendRecoveryModuleSource(unittest.TestCase):
    # --- 1. frozen export + missing dep failures ---

    def test_module_export_is_frozen_createRecovery_controller(self) -> None:
        recovery_source = APP_RECOVERY_JS.read_text(encoding="utf-8")
        helpers_source = APP_SESSION_HELPERS_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const vm = require("vm");
            const ctx = {{ window: {{}}, HTMLElement: function HTMLElement() {{}} }};
            vm.createContext(ctx);
            vm.runInContext({json.dumps(helpers_source)}, ctx);
            vm.runInContext({json.dumps(recovery_source)}, ctx);
            process.stdout.write(JSON.stringify({{
              frozen: Object.isFrozen(ctx.window.CodoxearRecovery),
              keys: Object.keys(ctx.window.CodoxearRecovery),
              hasCreate: typeof ctx.window.CodoxearRecovery.createRecoveryPanelController === "function",
            }}));
            """
        )
        result = run_node_json(js)
        self.assertTrue(result["frozen"])
        self.assertEqual(result["keys"], ["createRecoveryPanelController"])
        self.assertTrue(result["hasCreate"])

    def test_createRecovery_controller_throws_on_missing_deps(self) -> None:
        recovery_source = APP_RECOVERY_JS.read_text(encoding="utf-8")
        helpers_source = APP_SESSION_HELPERS_JS.read_text(encoding="utf-8")
        head = textwrap.dedent(
            f"""
            const vm = require("vm");
            const ctx = {{ window: {{}}, HTMLElement: function HTMLElement() {{}} }};
            vm.createContext(ctx);
            vm.runInContext({json.dumps(helpers_source)}, ctx);
            vm.runInContext({json.dumps(recovery_source)}, ctx);
            """
        )
        body = textwrap.dedent(
            r'''
            const R = ctx.window.CodoxearRecovery;
            const errors = [];
            const attempts = [
              ["options not object", () => R.createRecoveryPanelController(null)],
              ["missing DOM node", () => R.createRecoveryPanelController({})],
            ];
            for (const [label, fn] of attempts) {
              try { fn(); errors.push({ label, threw: false }); }
              catch (e) { errors.push({ label, threw: true, type: e.name === "TypeError", msg: String(e.message) }); }
            }
            // Fully wired except `setToast` (a function dep) -> TypeError naming setToast.
            const node = { style: {}, dataset: {}, setAttribute() {}, appendChild() {}, querySelector: () => null, querySelectorAll: () => [] };
            const wiredExceptToast = {
              chatInner: node, queueBtn: node,
              typingRowAnchor: () => null, getSessionInfo: () => null, el: () => ({}),
              recoveryPromptPreview: () => "", redactedLaunchErrorText: () => "",
              recoveryDetailsText: () => "", launchPresetFromSessionInfo: () => null,
              showQueueViewer: () => {}, clearCommitUnknownSend: () => {},
              openNewSessionDialog: () => {}, dismissFailedLaunchRecord: () => {},
              copyToClipboard: () => Promise.resolve(true),
              setToast: null,
            };
            try {
              R.createRecoveryPanelController(wiredExceptToast);
              errors.push({ label: "missing setToast", threw: false });
            } catch (e) {
              errors.push({ label: "missing setToast", threw: true, type: e.name === "TypeError", msg: String(e.message) });
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
        self.assertTrue(by_label["missing setToast"]["threw"])
        self.assertTrue(by_label["missing setToast"]["type"])
        self.assertIn("setToast", by_label["missing setToast"]["msg"])

    def test_module_load_fails_loud_without_helpers(self) -> None:
        recovery_source = APP_RECOVERY_JS.read_text(encoding="utf-8")
        js_only = textwrap.dedent(
            f"""
            const vm = require("vm");
            const ctx = {{ window: {{}}, HTMLElement: function HTMLElement() {{}} }};
            vm.createContext(ctx);
            let err = null;
            try {{ vm.runInContext({json.dumps(recovery_source)}, ctx); }}
            catch (e) {{ err = String(e.message); }}
            process.stdout.write(JSON.stringify({{ err }}));
            """
        )
        result = run_node_json(js_only)
        self.assertIn("failed to load", result["err"])

    # --- 2. no recovery state removes existing rows, returns false, attempts fallback ---

    def test_no_recovery_state_removes_rows_returns_false_and_focuses_fallback(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-healthy", { session_id: "sid-healthy" });
            // Seed a fake existing recovery row so we can prove it is removed.
            const existing = h.makeEl("div", { class: "msg-row assistant recovery-panel-row" });
            const existingIcon = h.makeEl("button", { class: "icon-btn", text: "Copy details" });
            existing.appendChild(existingIcon);
            h.dom.chatInner.insertBefore(existing, h.dom.typingAnchor);
            const hadRowBefore = Boolean(h.currentRow());
            const result = h.controller.renderRecoveryPanelIfNeeded("sid-healthy");
            const hasRowAfter = Boolean(h.currentRow());
            // Because the existing row was removed and no new panel rendered, the
            // fallback candidate is null and no focus call is made.
            const focusCalls = h.calls.filter((c) => c[0] === "focus");
            globalThis.__result = { result, hadRowBefore, hasRowAfter, focusCalls };
            """
        )
        result = run_node_json(js)
        self.assertFalse(result["result"])
        self.assertTrue(result["hadRowBefore"])
        self.assertFalse(result["hasRowAfter"])
        # No fallback icon button remained after removing the stale row, so no focus.
        self.assertEqual(result["focusCalls"], [])

    # --- 3. failed launch renders panel title/list/preview/actions before typing anchor ---

    def test_failed_launch_renders_panel_and_inserts_before_typing_anchor(self) -> None:
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
              model_provider: "macaron",
              model: "claude-haiku-4-5",
              reasoning_effort: "medium",
            });
            const ok = h.controller.renderRecoveryPanelIfNeeded("launch-dead");
            const row = h.currentRow();
            const bubble = row && row._children[0];
            const title = bubble && bubble._children.find((c) => c._class === "recoveryPanelTitle");
            const list = h.listTexts();
            const previews = h.previewTexts();
            const buttons = h.actionButtons().map((b) => b.textContent);
            const titles = {};
            h.actionButtons().forEach((b) => { titles[b.textContent] = b.getAttribute("title"); });
            // The row must be inserted before the typing anchor.
            const rowIdx = h.dom.chatInner._children.indexOf(row);
            const anchorIdx = h.dom.chatInner._children.indexOf(h.dom.typingAnchor);
            globalThis.__result = { ok, titleText: title && title.textContent, list, previews, buttons, titles, rowIdx, anchorIdx };
            """
        )
        result = run_node_json(js)
        self.assertTrue(result["ok"])
        self.assertEqual(result["titleText"], "Launch failed")
        self.assertIn("This web-owned session failed before a usable session log was bound.", result["list"])
        self.assertIn("Stage: pty_fork", result["list"])
        self.assertIn("Launch settings: macaron/claude-haiku-4-5 · medium", result["list"])
        # Redacted launch error preview is rendered.
        self.assertTrue(any("REDACTED:boom" in p for p in result["previews"]), result["previews"])
        for label in ["New like this", "Dismiss launch", "Copy details"]:
            self.assertIn(label, result["buttons"], result["buttons"])
        self.assertEqual(result["titles"]["New like this"], "Review copied launch settings before starting")
        self.assertEqual(result["titles"]["Dismiss launch"], "Dismiss failed launch record")
        self.assertEqual(result["titles"]["Copy details"], "Copy recovery details")
        self.assertLess(result["rowIdx"], result["anchorIdx"])

    def test_post_log_failed_launch_uses_after_log_recovery_copy(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("post-log-dead", {
              session_id: "post-log-dead",
              launch_state: "failed",
              launch_stage: "broker_exit_after_log_bind",
              launch_error: "broker control socket went stale after binding a transcript log before the turn completed",
              cwd: "/tmp/work",
              agent_backend: "pi",
            });
            const ok = h.controller.renderRecoveryPanelIfNeeded("post-log-dead");
            globalThis.__result = { ok, list: h.listTexts() };
            """
        )
        result = run_node_json(js)
        self.assertTrue(result["ok"])
        self.assertIn("This web-owned session stopped after binding a transcript log, before the turn completed.", result["list"])
        self.assertIn("Stage: broker_exit_after_log_bind", result["list"])
        self.assertNotIn("This web-owned session failed before a usable session log was bound.", result["list"])

    # --- 4. queue/orphan/commit-unknown renders recovery-needed list + actions ---

    def test_orphan_queue_commit_unknown_renders_recovery_needed(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-recover", {
              session_id: "sid-recover",
              orphan_recovery: true,
              queue_recovery: true,
              commit_unknown_send: true,
              commit_unknown_send_text: "did it land?",
              queue_len: 2,
            });
            const ok = h.controller.renderRecoveryPanelIfNeeded("sid-recover");
            const row = h.currentRow();
            const bubble = row && row._children[0];
            const title = bubble && bubble._children.find((c) => c._class === "recoveryPanelTitle");
            const list = h.listTexts();
            const previews = h.previewTexts();
            const buttons = h.actionButtons().map((b) => b.textContent);
            globalThis.__result = { ok, titleText: title && title.textContent, list, previews, buttons };
            """
        )
        result = run_node_json(js)
        self.assertTrue(result["ok"])
        self.assertEqual(result["titleText"], "Recovery needed")
        self.assertIn("The original session is missing; preserved prompts can be reviewed here before you decide what to discard.", result["list"])
        self.assertIn("A direct send may or may not have reached the terminal. Check the transcript or terminal before clearing the marker.", result["list"])
        self.assertIn("2 queued recovery items preserved for review.", result["list"])
        self.assertTrue(any("did it land?" in p for p in result["previews"]), result["previews"])
        for label in ["Review queue", "Clear unknown marker", "Copy details"]:
            self.assertIn(label, result["buttons"], result["buttons"])

    # --- 5. Review queue click calls showQueueViewer({ opener }) ---

    def test_review_queue_click_calls_show_queue_viewer(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-recover", { session_id: "sid-recover", queue_recovery: true, queue_len: 1 });
            h.controller.renderRecoveryPanelIfNeeded("sid-recover");
            const btn = h.findButton("Review queue");
            btn.onclick({ preventDefault: () => {}, stopPropagation: () => {}, currentTarget: btn });
            const open = h.queueOpens[0] || null;
            globalThis.__result = { count: h.queueOpens.length, openerText: open && open.opener && open.opener.textContent };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["openerText"], "Review queue")

    # --- 6. Clear unknown click calls clearCommitUnknownSend(sessionId, text) ---

    def test_clear_unknown_click_calls_clear_commit_unknown_send(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-recover", { session_id: "sid-recover", commit_unknown_send: true, commit_unknown_send_text: "the prompt" });
            h.controller.renderRecoveryPanelIfNeeded("sid-recover");
            const btn = h.findButton("Clear unknown marker");
            await btn.onclick({ preventDefault: () => {}, stopPropagation: () => {}, currentTarget: btn });
            const call = h.calls.find((c) => c[0] === "clearCommitUnknownSend");
            globalThis.__result = { call };
            """
        )
        result = run_node_json(js)
        self.assertIsNotNone(result["call"])
        self.assertEqual(result["call"][1], "sid-recover")
        self.assertEqual(result["call"][2], "the prompt")

    # --- 7. New-like click applies preset and opens New Session with exact text ---

    def test_new_like_click_opens_new_session_with_preset(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("launch-dead", {
              session_id: "launch-dead",
              launch_state: "failed",
              cwd: "/tmp/work",
              agent_backend: "pi",
            });
            h.controller.renderRecoveryPanelIfNeeded("launch-dead");
            const btn = h.findButton("New like this");
            btn.onclick({ preventDefault: () => {}, stopPropagation: () => {}, currentTarget: btn });
            const open = h.opens.pop() || null;
            globalThis.__result = {
              statusText: open && open.statusText,
              presetSid: open && open.likeSession && open.likeSession.session_id,
              presetCwd: open && open.likeSession && open.likeSession.cwd,
              returnFocusText: open && open.returnFocusEl && open.returnFocusEl.textContent,
            };
            """
        )
        result = run_node_json(js)
        self.assertIsNotNone(result["statusText"])
        self.assertEqual(result["statusText"], "Review copied launch settings before starting.")
        self.assertEqual(result["presetSid"], "launch-dead")
        self.assertEqual(result["presetCwd"], "/tmp/work")
        self.assertEqual(result["returnFocusText"], "New like this")

    def test_new_like_click_with_no_preset_toasts_launch_details_not_available(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            // Recreate the controller with a null-returning launchPreset factory.
            const ctrl = h.createControllerWithDeps({ launchPresetFromSessionInfo: () => null });
            h.sessions.set("launch-dead", { session_id: "launch-dead", launch_state: "failed" });
            ctrl.renderRecoveryPanelIfNeeded("launch-dead");
            const btn = h.findButton("New like this");
            btn.onclick({ preventDefault: () => {}, stopPropagation: () => {}, currentTarget: btn });
            globalThis.__result = { lastToast: h.toasts[h.toasts.length - 1], opensCount: h.opens.length };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["lastToast"], "launch details not available")

    # --- 8. Dismiss launch click calls dismissFailedLaunchRecord(sessionId) ---

    def test_dismiss_launch_click_calls_dismiss(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("launch-dead", { session_id: "launch-dead", launch_state: "failed" });
            h.controller.renderRecoveryPanelIfNeeded("launch-dead");
            const btn = h.findButton("Dismiss launch");
            await btn.onclick({ preventDefault: () => {}, stopPropagation: () => {}, currentTarget: btn });
            const call = h.calls.find((c) => c[0] === "dismissFailedLaunchRecord");
            globalThis.__result = { call };
            """
        )
        result = run_node_json(js)
        self.assertIsNotNone(result["call"])
        self.assertEqual(result["call"][1], "launch-dead")

    # --- 9. Copy details copies recoveryDetailsText and toasts; copy-failed surfaces error ---

    def test_copy_details_copies_and_toasts_success(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-recover", { session_id: "sid-recover", queue_recovery: true, queue_len: 1 });
            h.controller.renderRecoveryPanelIfNeeded("sid-recover");
            const btn = h.findButton("Copy details");
            await btn.onclick({ preventDefault: () => {}, stopPropagation: () => {}, currentTarget: btn });
            const call = h.calls.find((c) => c[0] === "copyToClipboard");
            globalThis.__result = { copied: call && call[1], lastToast: h.toasts[h.toasts.length - 1] };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["copied"], "RECOVERY_DETAILS:sid-recover:sid-recover")
        self.assertEqual(result["lastToast"], "Copied recovery details")

    def test_copy_details_toasts_copy_failed_on_error(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-recover", { session_id: "sid-recover", queue_recovery: true, queue_len: 1 });
            h.controller.renderRecoveryPanelIfNeeded("sid-recover");
            h.setCopyError(new Error("denied"));
            const btn = h.findButton("Copy details");
            await btn.onclick({ preventDefault: () => {}, stopPropagation: () => {}, currentTarget: btn });
            globalThis.__result = { lastToast: h.toasts[h.toasts.length - 1] };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["lastToast"], "copy failed: denied")

    # --- 10. focus preservation across rerender ---

    def test_focused_action_descriptor_restored_across_rerender(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("launch-dead", { session_id: "launch-dead", launch_state: "failed", cwd: "/x" });
            h.controller.renderRecoveryPanelIfNeeded("launch-dead");
            // Simulate the user focusing the "Copy details" button within the panel.
            const copyBtn = h.findButton("Copy details");
            h.setActive(copyBtn);
            // Re-render: descriptor is captured from the active element, the row is
            // rebuilt, and the matching button is refocused via requestFrame.
            h.controller.renderRecoveryPanelIfNeeded("launch-dead");
            const focusCalls = h.calls.filter((c) => c[0] === "focus").map((c) => c[1]);
            const rafCalls = h.calls.filter((c) => c[0] === "raf").length;
            globalThis.__result = { focusCalls, rafCalls };
            """
        )
        result = run_node_json(js)
        # The refocus targets the "Copy details" button by text+title match.
        self.assertIn("Copy details", result["focusCalls"])
        self.assertGreaterEqual(result["rafCalls"], 1)

    def test_focus_falls_back_to_first_icon_button_when_no_match(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            // First render as launch-failed, which owns New like this / Dismiss
            // launch / Copy details. Focus "Dismiss launch" — a button that ONLY
            // a launch-failed panel renders, so the captured descriptor cannot
            // match a panel of any other recovery kind.
            h.sessions.set("sid-a", { session_id: "sid-a", launch_state: "failed", cwd: "/a" });
            h.controller.renderRecoveryPanelIfNeeded("sid-a");
            const dismissBtn = h.findButton("Dismiss launch");
            h.setActive(dismissBtn);
            // Re-render the same session as a queue-recovery panel: Review queue +
            // Copy details only. "Dismiss launch" is absent, so focusRecoveryAction
            // returns false and the controller must fall back to the FIRST recovery
            // panel icon button of the new panel (Review queue) — not the stale
            // descriptor and not the queue button.
            h.sessions.set("sid-a", { session_id: "sid-a", queue_recovery: true, queue_len: 1 });
            h.controller.renderRecoveryPanelIfNeeded("sid-a");
            const focusCalls = h.calls.filter((c) => c[0] === "focus").map((c) => c[1]);
            const newButtons = h.actionButtons().map((b) => b.textContent);
            globalThis.__result = { focusCalls, newButtons };
            """
        )
        result = run_node_json(js)
        # The new panel has Review queue + Copy details but no Dismiss launch.
        self.assertNotIn("Dismiss launch", result["newButtons"], result["newButtons"])
        self.assertIn("Review queue", result["newButtons"], result["newButtons"])
        # Fallback targeted the first recovery-panel icon button of the new panel.
        self.assertIn("Review queue", result["focusCalls"], result["focusCalls"])
        # The stale Dismiss-launch descriptor did not match/refocus.
        self.assertNotIn("Dismiss launch", result["focusCalls"], result["focusCalls"])
        # The queue button is only the fallback when no panel remains; here a panel
        # exists, so the panel's own icon button must be preferred over the queue btn.
        self.assertNotIn("Queue", result["focusCalls"], result["focusCalls"])

    def test_focus_fallback_uses_queue_button_when_no_panel_remains(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.sessions.set("sid-a", { session_id: "sid-a", launch_state: "failed", cwd: "/a" });
            h.controller.renderRecoveryPanelIfNeeded("sid-a");
            const copyBtn = h.findButton("Copy details");
            h.setActive(copyBtn);
            // Recovery clears (healthy session): no panel remains, fallback should
            // use the queue button.
            h.sessions.set("sid-a", { session_id: "sid-a" });
            h.controller.renderRecoveryPanelIfNeeded("sid-a");
            const focusCalls = h.calls.filter((c) => c[0] === "focus").map((c) => c[1]);
            globalThis.__result = { focusCalls, hasRow: Boolean(h.currentRow()) };
            """
        )
        result = run_node_json(js)
        self.assertFalse(result["hasRow"])
        # The queue button is the fallback target.
        self.assertIn("Queue", result["focusCalls"])

    # --- 11. focusFallbackCandidate() returns first recovery icon button or null ---

    def test_focus_fallback_candidate_returns_first_icon_button(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            const beforeNull = h.controller.focusFallbackCandidate();
            h.sessions.set("sid-a", { session_id: "sid-a", launch_state: "failed" });
            h.controller.renderRecoveryPanelIfNeeded("sid-a");
            const candidate = h.controller.focusFallbackCandidate();
            globalThis.__result = {
              beforeNull: beforeNull === null,
              candidateText: candidate && candidate.textContent,
              candidateClass: candidate && candidate._class,
            };
            """
        )
        result = run_node_json(js)
        self.assertTrue(result["beforeNull"])
        self.assertIsNotNone(result["candidateText"])
        # The first action button rendered is "New like this" for a launch-failed panel.
        self.assertEqual(result["candidateText"], "New like this")
        self.assertIn("icon-btn", result["candidateClass"])

    # --- 12. dispose() clears pending focus descriptor ---

    def test_dispose_clears_pending_focus_descriptor(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            // The default harness requestFrame runs synchronously, which clears
            // pendingRecoveryFocusDescriptor inside the rAF callback. To prove
            // dispose() clears a RETAINED pending descriptor, swap in a deferred
            // requestFrame that queues the callback without running it.
            const pendingFrames = [];
            const ctrl = h.createControllerWithDeps({
              requestFrame: (fn) => { pendingFrames.push(fn); },
            });
            // Render launch-failed and focus its Dismiss button, then render again:
            // the second render captures the descriptor from the active button and
            // sets pendingRecoveryFocusDescriptor (rAF deferred => retained).
            h.sessions.set("sid-a", { session_id: "sid-a", launch_state: "failed", cwd: "/a" });
            ctrl.renderRecoveryPanelIfNeeded("sid-a");
            const dismissBtn = h.findButton("Dismiss launch");
            h.setActive(dismissBtn);
            ctrl.renderRecoveryPanelIfNeeded("sid-a");
            const rafBefore = pendingFrames.length;
            // Move focus off the panel so the ONLY way a descriptor resurfaces on
            // the next render is via the retained pending descriptor.
            h.setActive(null);
            // dispose() must clear pendingRecoveryFocusDescriptor. With activeElement
            // null and pending cleared, the next render finds no descriptor and must
            // not schedule any fallback/refocus rAF.
            ctrl.dispose();
            ctrl.renderRecoveryPanelIfNeeded("sid-a");
            const rafAfter = pendingFrames.length;
            globalThis.__result = { rafBefore, rafAfter };
            """
        )
        result = run_node_json(js)
        # No new rAF was scheduled after dispose: the retained pending descriptor
        # was cleared, so with activeElement null there was nothing to refocus.
        # (Without dispose, focusedRecoveryActionDescriptor would have returned the
        # stale pending descriptor and focusRecoveryAction would have queued a rAF.)
        self.assertEqual(result["rafAfter"], result["rafBefore"])

    # --- index.html load order ---

    def test_index_loads_recovery_module_after_deps_before_app(self) -> None:
        source = INDEX_HTML.read_text(encoding="utf-8")
        self.assertIn("app_recovery.js?v=__CODOXEAR_ASSET_VERSION__", source)
        self.assertLess(source.index("app_session_helpers.js?v=__CODOXEAR_ASSET_VERSION__"), source.index("app_recovery.js?v=__CODOXEAR_ASSET_VERSION__"))
        self.assertLess(source.index("app_diagnostics.js?v=__CODOXEAR_ASSET_VERSION__"), source.index("app_recovery.js?v=__CODOXEAR_ASSET_VERSION__"))
        self.assertLess(source.index("app_recovery.js?v=__CODOXEAR_ASSET_VERSION__"), source.index("app.js?v=__CODOXEAR_ASSET_VERSION__"))

    # --- app.js integration: app.js delegates and keeps DOM/pure helpers ---

    def test_app_js_delegates_recovery_behavior_to_controller(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        # Controller instantiation with fail-loud guard, before the queue controller.
        self.assertIn("const recoveryController = (function instantiateRecoveryController() {", source)
        self.assertIn('throw new Error("Codoxear recovery controller failed to load")', source)
        self.assertIn("codoxearRecovery.createRecoveryPanelController({", source)
        # Thin delegating wrapper.
        self.assertIn("return recoveryController.renderRecoveryPanelIfNeeded(sessionId);", source)
        # Queue controller receives the recovery-panel focus fallback via the controller.
        self.assertIn("recoveryPanelFocusFallback: () => recoveryController.focusFallbackCandidate(),", source)
        # Dispose hook.
        self.assertIn("if (recoveryController) recoveryController.dispose();", source)
        # Pure helpers stay in app.js (shared with diagnostics).
        self.assertIn("function launchPresetFromSessionInfo(s)", source)
        self.assertIn("function recoveryDetailsText(sessionId, s)", source)
        self.assertIn("function recoveryPromptPreview(text, maxLen = 320)", source)
        self.assertIn("function redactedLaunchErrorText(value)", source)
        self.assertIn("function dismissFailedLaunchRecord(sessionId)", source)
        # The moved state/helpers are gone from app.js.
        self.assertNotIn("let pendingRecoveryFocusDescriptor = null;", source)
        self.assertNotIn("function focusedRecoveryActionDescriptor(sessionId)", source)
        self.assertNotIn("function focusRecoveryAction(row, descriptor)", source)
        self.assertNotIn("function focusRecoveryFallback(descriptor)", source)
        self.assertNotIn("function recoverySessionInfo(sessionId)", source)


if __name__ == "__main__":
    unittest.main()
