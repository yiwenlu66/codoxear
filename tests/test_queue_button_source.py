import json
import os
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_QUEUE_JS = ROOT / "codoxear" / "static" / "app_queue.js"
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


def _bootstrap_controller(controller_options_overrides: str = "") -> str:
    """VM harness: load helper + queue modules and instantiate a controller.

    The controller is wired with fakes that record calls; the returned script
    writes the recorded behavior to stdout as JSON.
    """
    queue_source = APP_QUEUE_JS.read_text(encoding="utf-8")
    helpers_source = APP_SESSION_HELPERS_JS.read_text(encoding="utf-8")
    modal_source = APP_MODAL_JS.read_text(encoding="utf-8")
    return textwrap.dedent(
        f"""
        const vm = require("vm");
        const calls = [];
        const toasts = [];
        const apiCalls = [];
        let apiResponseQueue = [];
        function setApiResponses(list) {{ apiResponseQueue = list.slice(); }}
        const fakeNode = (extra = {{}}) => ({{
          style: {{ display: "none" }},
          classList: {{ add() {{}}, remove() {{}}, toggle() {{}} }},
          setAttribute(name, value) {{ this["attr_" + name] = value; }},
          getAttribute(name) {{ return this["attr_" + name]; }},
          removeAttribute() {{}},
          appendChild(child) {{ (this.children ||= []).push(child); return child; }},
          focus() {{ calls.push(["focus"]); }},
          ...extra,
        }});
        const queueBtn = fakeNode();
        const queueBackdrop = fakeNode();
        const queueCloseBtn = fakeNode();
        const queueList = fakeNode();
        const queueEmpty = fakeNode();
        const queueViewer = fakeNode();
        const sessions = new Map();
        let selected = null;
        let disposed = false;
        const api = (url, options = {{}}) => {{
          apiCalls.push([url, options]);
          calls.push(["api", url, JSON.parse(JSON.stringify(options.body || null))]);
          if (apiResponseQueue.length) {{
            const next = apiResponseQueue.shift();
            if (next instanceof Error) return Promise.reject(next);
            return Promise.resolve(next);
          }}
          return Promise.resolve({{}});
        }};
        const deps = {{
          queueBackdrop,
          queueCloseBtn,
          queueList,
          queueEmpty,
          queueViewer,
          queueBtn,
          getSelected: () => selected,
          getSessionInfo: (sid) => sessions.get(sid) || null,
          isAppDisposed: () => disposed,
          api,
          setToast: (t) => {{ toasts.push(t); calls.push(["setToast", t]); }},
          clearCommitUnknownSend: (sid, text) => {{ calls.push(["clearCommitUnknownSend", sid, text]); return Promise.resolve(); }},
          refreshSessions: async () => calls.push(["refreshSessions"]),
          updateQueueBadge: () => calls.push(["updateQueueBadge"]),
          syncRecoveryUiForSession: (sid) => calls.push(["syncRecoveryUiForSession", sid]),
          kickPoll: (ms) => calls.push(["kickPoll", ms]),
          setPollFastUntilMs: (ms) => calls.push(["setPollFastUntilMs", ms]),
          handleAppAuthLoss: () => calls.push(["handleAppAuthLoss"]),
          prepareModalOpen: () => calls.push(["prepareModalOpen"]),
          afterModalVisibilityChanged: () => calls.push(["afterModalVisibilityChanged"]),
          el: (tag, attrs = {{}}, children = []) => {{
            const node = fakeNode({{ tag, ...(attrs || {{}}) }});
            if (attrs && attrs.html !== undefined) node.innerHTML = attrs.html;
            if (attrs && attrs.text !== undefined) node.textContent = attrs.text;
            node.appendChild = (child) => {{ (node.children ||= []).push(child); return child; }};
            (Array.isArray(children) ? children : [children]).forEach((c) => {{ if (c) (node.children ||= []).push(c); }});
            return node;
          }},
          iconSvg: (name) => `<svg>${{name}}</svg>`,
          recoveryPanelFocusFallback: () => null,
          confirmAction: async () => true,
          requestFrame: (fn) => fn(),
          setTimeout: (fn, ms) => {{ calls.push(["setTimeout", ms]); const handle = ++timerHandle; pendingTimers.set(handle, {{ fn, ms }}); return handle; }},
          clearTimeout: (handle) => {{ pendingTimers.delete(handle); }},
          now: () => nowValue,
        }};
        let timerHandle = 0;
        const pendingTimers = new Map();
        let nowValue = 1000;
        function runPendingTimers() {{
          for (const [handle, entry] of Array.from(pendingTimers.entries())) {{
            pendingTimers.delete(handle);
            entry.fn();
          }}
        }}
        const ctx = {{
          HTMLElement: function HTMLElement() {{}},
          window: {{}},
          console,
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(modal_source)}, ctx);
        vm.runInContext({json.dumps(helpers_source)}, ctx);
        vm.runInContext({json.dumps(queue_source)}, ctx);
        const CodoxearQueue = ctx.window.CodoxearQueue;
        const controller = CodoxearQueue.createQueueController(deps);
        const session = (overrides = {{}}) => ({{ session_id: "sid-1", ...overrides }});
        globalThis.__harness = {{ controller, deps, calls, toasts, apiCalls, sessions, selectedRef: {{ get: () => selected, set: (v) => {{ selected = v; }} }} , setApiResponses, runPendingTimers, setNow: (v) => {{ nowValue = v; }}, queueViewer, queueBtn }};
        {controller_options_overrides}
        process.stdout.write(JSON.stringify(globalThis.__result || {{}}));
        """
    )


class TestQueueButtonSource(unittest.TestCase):
    def test_app_js_requires_queue_module_and_delegates(self) -> None:
        """Integration: app.js fails loud without CodoxearQueue and routes
        queue authority through queueController, not local queue state."""
        source = APP_JS.read_text(encoding="utf-8")

        # Fail-loud module dependency, mirroring the other frontend modules.
        self.assertIn('const codoxearQueue = window.CodoxearQueue;', source)
        self.assertIn('throw new Error("Codoxear queue controller failed to load")', source)
        # app.js must not own queue state/decision internals anymore.
        for removed in [
            "const queueUpdateTimers = new Map();",
            "const queueMutationLocks = new Set();",
            "const queuePendingDeletes = new Set();",
            "const queueDraftTexts = new Map();",
            "let queueSubmitBusy = false;",
            "let queueViewerSid = null;",
            "let queueViewerItems = [];",
            "async function deleteQueueItem(sid, itemId) {",
            "async function moveQueueItem(sid, itemId, toIndex) {",
            "function scheduleQueueUpdate(sid, itemId, text) {",
            "function renderQueueList() {",
            "let queueReturnFocusEl = null;",
        ]:
            self.assertNotIn(removed, source, removed)
        # app.js keeps thin delegating wrappers and the dispose hook.
        self.assertIn("queueController.syncQueueSubmitState();", source)
        self.assertIn("return queueController.enqueueComposerText(raw, opts);", source)
        self.assertIn("return queueController.refreshQueueViewer();", source)
        self.assertIn("return queueController.showQueueViewer(opts);", source)
        self.assertIn("return queueController.hideQueueViewer();", source)
        self.assertIn("if (queueController) queueController.dispose();", source)
        # The session predicates that feed both queue + send/composer projection
        # stay in app.js (send/composer projection is not part of this contract).
        for helper in [
            "function selectedSessionHasUnknownSend() {",
            "function selectedSessionIsOrphanRecovery() {",
            "function selectedSessionHasOrphanQueueRecovery() {",
            "function selectedSessionLaunchFailed() {",
        ]:
            self.assertIn(helper, source)

    def test_queue_button_disabled_title_aria_projections(self) -> None:
        js = _bootstrap_controller(
            """
            const h = globalThis.__harness;
            const projections = [];
            const record = (label) => {
              h.controller.syncQueueSubmitState();
              projections.push({ label, disabled: h.deps.queueBtn.disabled, title: h.deps.queueBtn.title, aria: h.deps.queueBtn.getAttribute("aria-label") });
            };
            // No selection.
            record("none");
            // Normal selected session.
            h.sessions.set("sid-1", h.deps.getSessionInfo ? { launch_state: "ready", queue_len: 0 } : {});
            h.selectedRef.set("sid-1");
            record("normal");
            // Failed launch.
            h.sessions.set("sid-1", { launch_state: "failed", queue_len: 0 });
            record("failed");
            // Unknown send (commit_unknown_send) blocks queue.
            h.sessions.set("sid-1", { launch_state: "ready", commit_unknown_send: true, queue_len: 0 });
            record("unknown");
            // Orphan queue recovery re-enables the queue button despite unknown send.
            h.sessions.set("sid-1", { launch_state: "ready", commit_unknown_send: true, queue_recovery: true, queue_len: 2 });
            record("orphanQueueRecovery");
            globalThis.__result = { projections };
            """
        )
        result = run_node_json(js)
        p = {row["label"]: row for row in result["projections"]}
        self.assertTrue(p["none"]["disabled"])
        self.assertEqual(p["none"]["title"], "Select a session to view queued messages")
        self.assertFalse(p["normal"]["disabled"])
        self.assertEqual(p["normal"]["title"], "Queued messages")
        self.assertTrue(p["failed"]["disabled"])
        self.assertEqual(p["failed"]["title"], "Failed launch cannot receive queued messages")
        self.assertTrue(p["unknown"]["disabled"])
        self.assertEqual(p["unknown"]["title"], "Resolve the unknown send before queueing")
        self.assertFalse(p["orphanQueueRecovery"]["disabled"])
        self.assertEqual(p["orphanQueueRecovery"]["title"], "Review preserved queued recovery items")
        # aria-label always mirrors title.
        for row in result["projections"]:
            self.assertEqual(row["aria"], row["title"])


if __name__ == "__main__":
    unittest.main()
