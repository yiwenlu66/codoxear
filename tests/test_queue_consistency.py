from frontend_module_loader import module_path
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_QUEUE_JS = module_path("app_queue.js")
APP_SESSION_HELPERS_JS = module_path("app_session_helpers.js")
APP_MODAL_JS = module_path("app_modal.js")
APP_SESSION_STATE_JS = module_path("app_session_state.js")


def run_node_json(script: str) -> dict:
    result = subprocess.run(
        ["node", "-e", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={"PATH": os.environ.get("PATH", ""), "TZ": "UTC"},
    )
    if result.returncode:
        raise AssertionError(result.stderr)
    return json.loads(result.stdout)


def test_queue_snapshots_reconcile_sidebar_header_badge_and_panel_after_each_operation() -> None:
    """The controller must project every queue-store response through one count.

    This VM executes the real queue controller.  The harness represents the app
    shell's session-index projection as the sidebar/header state and the
    controller's own badge/list DOM as the composer badge/queue panel.
    """
    script = f"""
    (async () => {{
      const vm = require("vm");
      const node = (extra = {{}}) => ({{
        style: {{ display: "none" }}, _children: [], _attrs: {{}}, disabled: false,
        textContent: "", value: "", classList: {{ add() {{}}, remove() {{}}, toggle() {{}} }},
        appendChild(child) {{ this._children.push(child); return child; }},
        set innerHTML(_value) {{ this._children = []; }}, get innerHTML() {{ return ""; }},
        setAttribute(name, value) {{ this._attrs[name] = String(value); }},
        getAttribute(name) {{ return this._attrs[name]; }}, focus() {{}}, ...extra,
      }});
      const queueBtn = node();
      const queueBackdrop = node();
      const queueCloseBtn = node();
      const queueList = node();
      const queueEmpty = node();
      const queueViewer = node();
      const sessions = new Map([["sid", {{ session_id: "sid", launch_state: "ready", queue_len: 0 }}]]);
      let selected = "sid";
      let nextResponses = [];
      let nextSessionQueueLens = [];
      const shell = {{ sidebar: 0, header: 0 }};
      const ctx = {{
        HTMLElement: function HTMLElement() {{}},
        document: {{ activeElement: null }},
        window: {{}}, console, setTimeout, clearTimeout,
        requestAnimationFrame: (fn) => fn(),
      }};
      vm.createContext(ctx);
      vm.runInContext({json.dumps(APP_MODAL_JS.read_text(encoding="utf-8"))}, ctx);
      vm.runInContext({json.dumps(APP_SESSION_HELPERS_JS.read_text(encoding="utf-8"))}, ctx);
      vm.runInContext({json.dumps(APP_SESSION_STATE_JS.read_text(encoding="utf-8"))}, ctx);
      vm.runInContext({json.dumps(APP_QUEUE_JS.read_text(encoding="utf-8"))}, ctx);
      const sessionState = ctx.window.CodoxearSessionState.createSessionState({{ consoleError: () => {{}} }});
sessionState.set("selected", selected);
      const controller = ctx.window.CodoxearQueue.createQueueController({{
        queueBackdrop, queueCloseBtn, queueList, queueEmpty, queueViewer, queueBtn,

        getSessionInfo: (sid) => sessions.get(sid) || null,
        sessionState,
        isAppDisposed: () => false,
        api: async () => nextResponses.shift(),
        setToast() {{}}, clearCommitUnknownSend: async () => true,
        refreshSessions: async () => {{
          const queueLen = nextSessionQueueLens.shift();
          if (queueLen === undefined) throw new Error("missing authoritative session snapshot");
          sessions.get("sid").queue_len = queueLen;
          sessionState.applyRuntime({{ queueLen }});
          shell.sidebar = queueLen;
          shell.header = queueLen;
        }},
        syncRecoveryUiForSession() {{}}, kickPoll() {{}}, setPollFastUntilMs() {{}},
        handleAppAuthLoss() {{}}, prepareModalOpen() {{}}, afterModalVisibilityChanged() {{}},
        el: (tag, attrs = {{}}, children = []) => {{
          const result = node({{ tag }});
          if (attrs.text !== undefined) result.textContent = String(attrs.text);
          if (attrs.class !== undefined) result._class = String(attrs.class);
          if (attrs["aria-label"] !== undefined) result._attrs["aria-label"] = String(attrs["aria-label"]);
          if (attrs.title !== undefined) result.title = String(attrs.title);
          (Array.isArray(children) ? children : [children]).filter(Boolean).forEach((child) => result.appendChild(child));
          return result;
        }},
        iconSvg: () => "", recoveryPanelFocusFallback: () => null,
        confirmAction: async () => true, requestFrame: (fn) => fn(),
      }});
      const badge = () => queueBtn._children.find((child) => child && child._class === "attachBadge queueBadge");
      const snapshot = (name) => {{
        controller.updateQueueBadge();
        return {{
          name,
          sidebar: shell.sidebar,
          header: shell.header,
          composerBadge: Number((badge() && badge().textContent) || 0),
          panel: queueViewer.style.display === "flex" ? queueList._children.length : (shell.header === 0 ? 0 : null),
        }};
      }};
      const results = [];
      nextResponses = [{{ ok: true, items: [] }}];
      controller.showQueueViewer();
      await new Promise((resolve) => setTimeout(resolve, 0));
      nextResponses = [
        {{ queued: true }},
        {{ ok: true, items: [{{ id: "q1", text: "after current" }}] }},
      ];
      nextSessionQueueLens = [1];
      await controller.enqueueComposerText("after current", {{ sid: "sid" }});
      results.push(snapshot("enqueue"));
      results.push(snapshot("open-panel"));
      nextResponses = [{{ ok: true }}];
      nextSessionQueueLens = [0];
      await controller.deleteQueueItem("sid", "q1");
      nextResponses = [{{ ok: true, items: [] }}];
      controller.showQueueViewer();
      await new Promise((resolve) => setTimeout(resolve, 0));
      results.push(snapshot("delete"));
      process.stdout.write(JSON.stringify({{ results, finalDisabled: queueBtn.disabled }}));
    }})().catch((error) => {{ console.error(error); process.exit(1); }});
    """
    result = run_node_json(script)
    assert result["finalDisabled"] is False
    for state in result["results"]:
        expected = state["sidebar"]
        assert state["header"] == expected, state
        assert state["composerBadge"] == expected, state
        assert state["panel"] == expected, state
    assert [state["sidebar"] for state in result["results"]] == [1, 1, 0]
