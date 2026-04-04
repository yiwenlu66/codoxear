import json
import subprocess
import textwrap
import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


def eval_rebuild_decorations_with_nested_rows() -> dict[str, object]:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("        function rebuildDecorations({ preserveScroll }) {")
    end = source.index("\n        function trimRenderedRows(", start)
    snippet = source[start:end]
    injected = json.dumps(snippet + "\nglobalThis.__test_rebuildDecorations = rebuildDecorations;\n")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");

        function makeClassList(initial) {{
          const set = new Set(initial || []);
          return {{
            contains(name) {{ return set.has(name); }},
            add(name) {{ set.add(name); }},
            remove(name) {{ set.delete(name); }},
          }};
        }}

        function makeRow(parent, ts, classes) {{
          return {{
            parentNode: parent,
            dataset: {{ ts: String(ts) }},
            classList: makeClassList(["msg-row", ...(classes || [])]),
          }};
        }}

        const chatInner = {{
          children: [],
          querySelectorAll(selector) {{
            if (selector === ".day-sep") return [];
            if (selector === ".msg-row") return [topRow, nestedRow];
            return [];
          }},
          insertBefore(node, anchor) {{
            if (anchor != null && anchor.parentNode !== chatInner) {{
              throw new Error("anchor is not a direct child");
            }}
            node.parentNode = chatInner;
            return node;
          }},
        }};

        const nestedParent = {{ parentNode: chatInner }};
        const topRow = makeRow(chatInner, 0, ["assistant"]);
        const nestedRow = makeRow(nestedParent, 1711929601, ["assistant", "tool-marker"]);
        chatInner.children = [topRow, nestedParent];

        const ctx = {{
          chatInner,
          chat: {{ scrollTop: 0, scrollHeight: 0 }},
          directChatRows: () => [topRow],
          autoScroll: false,
          requestAnimationFrame: (fn) => fn(),
          scrollToBottom: () => {{}},
          syncJumpButton: () => {{}},
          ymd: () => "2026-03-30",
          dayLabel: () => "Today",
          el: (tag, attrs = {{}}) => {{
            const node = {{
              tagName: String(tag || "div").toUpperCase(),
              className: attrs.class || "",
              textContent: attrs.text || "",
              dataset: {{}},
              parentNode: null,
            }};
            node.classList = makeClassList(String(node.className || "").split(/\s+/).filter(Boolean));
            return node;
          }},
          Date,
        }};

        vm.createContext(ctx);
        vm.runInContext({injected}, ctx);
        try {{
          ctx.__test_rebuildDecorations({{ preserveScroll: false }});
          process.stdout.write(JSON.stringify({{ ok: true }}));
        }} catch (err) {{
          process.stdout.write(JSON.stringify({{
            ok: false,
            error: err && err.message ? String(err.message) : String(err),
          }}));
        }}
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


class TestChatDecorationsUi(unittest.TestCase):
    def test_rebuild_decorations_ignores_nested_msg_rows(self) -> None:
        result = eval_rebuild_decorations_with_nested_rows()

        self.assertTrue(result["ok"], result.get("error", "rebuildDecorations threw"))


if __name__ == "__main__":
    unittest.main()
