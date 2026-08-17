"""Behavioral coverage for the sidebar's store-owned active-card projection."""

import json
import subprocess
import textwrap

from frontend_module_loader import module_path


APP_SESSIONS_JS = module_path("app_sessions.js")
APP_SESSION_STATE_JS = module_path("app_session_state.js")


def test_sidebar_active_class_tracks_selected_store_without_lifecycle_writer() -> None:
    source = APP_SESSIONS_JS.read_text(encoding="utf-8")
    state_source = APP_SESSION_STATE_JS.read_text(encoding="utf-8")
    script = textwrap.dedent(
        f"""
        const vm = require("vm");
        function node(attrs = {{}}) {{
          const out = {{ attrs, children: [], dataset: {{}}, style: {{}}, parentNode: null }};
          if (attrs["data-session-id"]) out.dataset.sessionId = attrs["data-session-id"];
          out.appendChild = (child) => {{ if (child) {{ child.parentNode = out; out.children.push(child); }} return child; }};
          out.addEventListener = () => {{}};
          out.remove = () => {{ if (out.parentNode) out.parentNode.children = out.parentNode.children.filter((child) => child !== out); }};
          Object.defineProperty(out, "childElementCount", {{ get: () => out.children.length }});
          Object.defineProperty(out, "innerHTML", {{ get: () => "", set: () => {{ out.children = []; }} }});
          out.querySelectorAll = (selector) => {{
            const found = [];
            const walk = (current) => {{
              const classes = String(current.attrs.class || "").split(/\\s+/);
              if (selector === ".session[data-session-id]" && classes.includes("session") && current.dataset.sessionId) found.push(current);
              current.children.forEach(walk);
            }};
            walk(out);
            return found;
          }};
          out.classList = {{
            add(name) {{ if (!String(out.attrs.class || "").split(/\\s+/).includes(name)) out.attrs.class = `${{out.attrs.class || ""}} ${{name}}`.trim(); }},
            toggle(name, on) {{
              const names = new Set(String(out.attrs.class || "").split(/\\s+/).filter(Boolean));
              if (on) names.add(name); else names.delete(name);
              out.attrs.class = [...names].join(" ");
            }},
          }};
          return out;
        }}
        const ctx = {{ window: {{}}, performance: {{ now: () => 0 }} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(state_source)}, ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        const sessionState = ctx.window.CodoxearSessionState.createSessionState({{ consoleError: () => {{}} }});
        const wrap = node();
        const el = (_tag, attrs = {{}}, children = []) => {{
          const out = node(attrs);
          (Array.isArray(children) ? children : [children]).filter(Boolean).forEach((child) => out.appendChild(child));
          return out;
        }};
        const controller = ctx.window.CodoxearSessions.createSessionsController({{
          sessionState, sessionsWrap: wrap, sidebarEmptyHint: node(), el, iconSvg: () => "",
          sidebarRenderSignature: () => "cards", sidebarSessionEntries: (sessions) => sessions.map((session) => ({{ type: "session", session }})),
          sessionDisplayName: (session) => session.session_id, sessionLaunchFailed: () => false, sessionLaunchPending: () => false,
          redactedLaunchErrorText: () => "", fmtRelativeAge: () => "now", sidebarEffortCode: () => "", sidebarModelText: () => "",
          baseName: () => "repo", sessionIsFast: () => false, agentBackendLogoPath: () => "", agentBackendDisplayName: () => "",
          sessionAgentBackend: () => "pi", sessionLaunchIcon: () => "", sessionLaunchLabel: () => "", confirmAction: async () => false,
          api: async () => ({{}}), clearDeletedSessionClientState: () => {{}}, refreshSessions: async () => [], setToast: () => {{}},
          openEditSession: () => {{}}, duplicateSession: async () => {{}}, selectSession: async () => {{}}, setSidebarOpen: () => {{}}, now: () => 0,
        }});
        controller.renderSessions([{{ session_id: "a" }}, {{ session_id: "b" }}], {{ swipeActions: false }});
        const active = () => wrap.querySelectorAll(".session[data-session-id]").filter((card) => String(card.attrs.class).split(/\\s+/).includes("active")).map((card) => card.dataset.sessionId);
        sessionState.set("selected", "a");
        const afterA = active();
        sessionState.set("selected", "b");
        const afterB = active();
        controller.dispose();
        sessionState.set("selected", "a");
        process.stdout.write(JSON.stringify({{ afterA, afterB, afterDispose: active() }}));
        """
    )
    result = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == {"afterA": ["a"], "afterB": ["b"], "afterDispose": ["b"]}
