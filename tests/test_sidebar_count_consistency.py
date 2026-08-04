import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_SESSION_HELPERS_JS = ROOT / "codoxear" / "static" / "app_session_helpers.js"
APP_SESSIONS_JS = ROOT / "codoxear" / "static" / "app_sessions.js"


def render_grouped_sidebar() -> dict:
    helpers_source = APP_SESSION_HELPERS_JS.read_text(encoding="utf-8")
    sessions_source = APP_SESSIONS_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");

        class Node {{
          constructor(tag, attributes = {{}}, children = []) {{
            this.tag = tag;
            this.attributes = attributes;
            this.children = [];
            this.parentNode = null;
            this.style = {{}};
            this.dataset = {{}};
            this.classList = {{
              add: (...names) => {{
                const classes = new Set(String(this.attributes.class || "").split(/\\s+/).filter(Boolean));
                names.forEach((name) => classes.add(name));
                this.attributes.class = Array.from(classes).join(" ");
              }},
            }};
            this.textContent = attributes.text || "";
            this.append(...children);
          }}
          get parentElement() {{ return this.parentNode; }}
          get childElementCount() {{ return this.children.length; }}
          set innerHTML(value) {{
            if (value !== "") throw new Error("harness only supports clearing innerHTML");
            this.children.forEach((child) => {{ child.parentNode = null; }});
            this.children = [];
          }}
          append(...children) {{ children.filter(Boolean).forEach((child) => this.appendChild(child)); }}
          appendChild(child) {{
            if (child.parentNode) child.remove();
            this.children.push(child);
            child.parentNode = this;
            return child;
          }}
          remove() {{
            if (!this.parentNode) return;
            const siblings = this.parentNode.children;
            siblings.splice(siblings.indexOf(this), 1);
            this.parentNode = null;
          }}
          addEventListener() {{}}
          setPointerCapture() {{}}
          releasePointerCapture() {{}}
        }}

        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(helpers_source)}, ctx);
        vm.runInContext({json.dumps(sessions_source)}, ctx);
        const helpers = ctx.window.CodoxearSessionHelpers;
        const sessionsWrap = new Node("div");
        const sidebarEmptyHint = new Node("div");
        const el = (tag, attributes = {{}}, children = []) => new Node(tag, attributes, children);
        const controller = ctx.window.CodoxearSessions.createSessionsController({{
          sessionsWrap,
          sidebarEmptyHint,
          el,
          iconSvg: () => "",
          sidebarRenderSignature: helpers.sidebarRenderSignature,
          sidebarSessionEntries: helpers.sidebarSessionEntries,
          sessionDisplayName: (session) => session.session_id,
          sessionLaunchFailed: helpers.sessionLaunchFailed,
          sessionLaunchPending: helpers.sessionLaunchPending,
          redactedLaunchErrorText: () => "",
          fmtRelativeAge: () => "now",
          sidebarEffortCode: () => "",
          sidebarModelText: () => "",
          baseName: () => "",
          sessionIsFast: helpers.sessionIsFast,
          agentBackendLogoPath: () => "",
          agentBackendDisplayName: () => "",
          sessionAgentBackend: () => "pi",
          sessionLaunchIcon: () => "",
          sessionLaunchLabel: () => "",
          confirmAction: async () => false,
          api: async () => ({{}}),
          clearDeletedSessionClientState: () => {{}},
          refreshSessions: async () => {{}},
          setToast: () => {{}},
          openEditSession: () => {{}},
          duplicateSession: async () => {{}},
          selectSession: async () => {{}},
          setSidebarOpen: () => {{}},
          now: () => 0,
        }});
        controller.renderSessions([
          {{ session_id: "now-1" }},
          {{ session_id: "now-2" }},
          {{ session_id: "waiting-1", blocked: true }},
          {{ session_id: "later-1", snoozed: true }},
          {{ session_id: "later-2", snoozed: true }},
        ]);

        const result = [];
        let current = null;
        for (const child of sessionsWrap.children) {{
          const classes = String(child.attributes.class || "").split(/\\s+/);
          if (classes.includes("sessionGroupHeader")) {{
            current = {{
              key: child.attributes["data-session-group"],
              headerCount: Number(child.children.find((node) => String(node.attributes.class || "").split(/\\s+/).includes("sessionGroupCount")).textContent),
              cardCount: 0,
            }};
            result.push(current);
          }} else if (classes.includes("session")) {{
            if (!current) throw new Error("session card rendered before a group header");
            current.cardCount += 1;
          }}
        }}
        process.stdout.write(JSON.stringify({{ groups: result, rendered: sessionsWrap.dataset.codoxearSessionsRendered }}));
        """
    )
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(proc.stdout)


class TestSidebarCountConsistency(unittest.TestCase):
    def test_group_header_count_matches_rendered_session_cards(self) -> None:
        result = render_grouped_sidebar()

        self.assertEqual(
            result["groups"],
            [
                {"key": "now", "headerCount": 2, "cardCount": 2},
                {"key": "waiting", "headerCount": 1, "cardCount": 1},
                {"key": "later", "headerCount": 2, "cardCount": 2},
            ],
        )
        self.assertTrue(result["rendered"])


if __name__ == "__main__":
    unittest.main()
