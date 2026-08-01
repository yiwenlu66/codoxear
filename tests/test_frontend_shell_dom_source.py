import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_SHELL_JS = ROOT / "codoxear" / "static" / "app_shell.js"


def render_shell_parents() -> dict[str, str | None]:
    source = APP_SHELL_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        class Node {{
          constructor(tag, attrs = {{}}) {{
            this.tag = tag;
            this.attrs = attrs;
            this.className = attrs.class;
            this.children = [];
            this.parent = null;
            this.style = {{}};
          }}
          appendChild(child) {{
            if (child.parent) {{
              const index = child.parent.children.indexOf(child);
              if (index >= 0) child.parent.children.splice(index, 1);
            }}
            child.parent = this;
            this.children.push(child);
            return child;
          }}
          append(...children) {{ children.forEach((child) => this.appendChild(child)); }}
          set innerHTML(_value) {{ this.children = []; }}
        }}
        const el = (tag, attrs = {{}}, children = []) => {{
          const node = new Node(tag, attrs);
          children.forEach((child) => node.appendChild(child));
          return node;
        }};
        const ctx = {{ window: {{}}, document: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        const root = new Node("root");
        const {{ elements }} = ctx.window.CodoxearShell.createShellDOM({{
          root,
          el,
          iconSvg: (name) => `<${{name}}>`,
          resolveAppUrl: (value) => value,
          versionedShellAssetPath: (value) => value,
        }});
        const parentClass = (node) => node.parent && node.parent.className;
        process.stdout.write(JSON.stringify({{
          chatSearch: parentClass(elements.chatSearchBtn),
          prev: parentClass(elements.prevUserBtn),
          next: parentClass(elements.nextUserBtn),
          hasComposerStop: Object.hasOwn(elements, "composerStopBtn"),
        }}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, capture_output=True, text=True)
    return json.loads(proc.stdout)


class TestFrontendShellDomSource(unittest.TestCase):
    def test_message_navigation_controls_mount_in_chat_nav_rail(self) -> None:
        self.assertEqual(
            render_shell_parents(),
            {
                "chatSearch": "chatNavRail",
                "prev": "chatMessageNavControls",
                "next": "chatMessageNavControls",
                "hasComposerStop": False,
            },
        )


if __name__ == "__main__":
    unittest.main()
