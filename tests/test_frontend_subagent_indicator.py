import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_SHELL_JS = ROOT / "codoxear" / "static" / "app_shell.js"


class TestFrontendSubagentIndicator(unittest.TestCase):
    def test_sidebar_meta_appends_subagent_suffix_only_when_running(self) -> None:
        source = APP_SHELL_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const vm = require("vm");
            class Node {{
              constructor(tag, attrs = {{}}) {{ this.tag = tag; this.attrs = attrs; this.className = attrs.class || ""; this.children = []; this.parentElement = null; this.dataset = {{}}; this.style = {{}}; this.classList = {{ add: () => {{}} }}; }}
              get childElementCount() {{ return this.children.length; }}
              set innerHTML(_value) {{ this.children = []; }}
              appendChild(child) {{ if (!child) return child; child.parentElement = this; this.children.push(child); return child; }}
              append(...children) {{ children.forEach((child) => this.appendChild(child)); }}
              addEventListener() {{}}
              querySelectorAll() {{ return []; }}
              remove() {{ if (!this.parentElement) return; const i = this.parentElement.children.indexOf(this); if (i >= 0) this.parentElement.children.splice(i, 1); this.parentElement = null; }}
            }}
            const el = (tag, attrs = {{}}, children = []) => {{ const node = new Node(tag, attrs); children.forEach((child) => node.appendChild(child)); return node; }};
            const ctx = {{ window: {{}} }};
            vm.createContext(ctx);
            vm.runInContext({json.dumps(source)}, ctx);
            const sessionsWrap = new Node("div");
            const controller = ctx.window.CodoxearShell.createSidebarController({{
              sessionsWrap, sidebarEmptyHint: new Node("div"), el, iconSvg: () => "", sidebarRenderSignature: (entries) => JSON.stringify(entries),
              sessionDisplayName: () => "session", sessionLaunchFailed: () => false, sessionLaunchPending: () => false, redactedLaunchErrorText: () => "",
              fmtRelativeAge: () => "now", sidebarEffortCode: () => "", sidebarModelText: () => "", baseName: () => "repo", sessionIsFast: () => false,
              agentBackendLogoPath: () => "", agentBackendDisplayName: () => "Pi", sessionAgentBackend: () => "pi", sessionLaunchIcon: () => "terminal", sessionLaunchLabel: () => "terminal",
              confirmAction: async () => false, api: async () => ({{}}), clearDeletedSessionClientState: () => {{}}, refreshSessions: async () => {{}}, setToast: () => {{}},
              openEditSession: () => {{}}, duplicateSession: async () => {{}}, selectSession: async () => {{}}, setSidebarOpen: () => {{}}, now: () => 0,
            }});
            const meta = (node) => {{
              if (node.className === "metaText") return node.children.map((child) => child.attrs.text || "").join("");
              for (const child of node.children) {{ const found = meta(child); if (found !== null) return found; }}
              return null;
            }};
            const textOf = (node) => (node.attrs.text || "") + node.children.map((child) => textOf(child)).join("");
            const title = (node) => {{
              if (node.className === "sessionTitleRow") return node.children.map((child) => textOf(child)).join("");
              for (const child of node.children) {{ const found = title(child); if (found !== null) return found; }}
              return null;
            }};
            const row = (n) => ({{ session_id: `s-${{n}}`, cwd: "/repo", start_ts: 0, updated_ts: 0, transport: "pty", owned: false, subagents_running: n }});
            controller.render([{{ type: "session", session: row(2) }}], {{ swipeActions: false }});
            const active = {{ meta: meta(sessionsWrap), title: title(sessionsWrap) }};
            controller.render([{{ type: "session", session: row(0) }}], {{ swipeActions: false }});
            const inactive = {{ meta: meta(sessionsWrap), title: title(sessionsWrap) }};
            process.stdout.write(JSON.stringify({{ active, inactive }}));
            """
        )
        proc = subprocess.run(["node", "-e", js], check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(proc.stdout), {
            "active": {"meta": "now | repo", "title": "▸2session"},
            "inactive": {"meta": "now | repo", "title": "session"},
        })


if __name__ == "__main__":
    unittest.main()
