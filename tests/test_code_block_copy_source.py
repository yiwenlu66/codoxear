import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_CSS = ROOT / "codoxear" / "static" / "app.css"
APP_CODE_COPY_JS = ROOT / "codoxear" / "static" / "app_code_copy.js"
APP_MARKDOWN_JS = ROOT / "codoxear" / "static" / "app_markdown.js"
INDEX_HTML = ROOT / "codoxear" / "static" / "index.html"


def render_markdown(markdown: str) -> str:
    source = APP_MARKDOWN_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{
          URL,
          location: {{ origin: "http://localhost", href: "http://localhost/" }},
          console,
          window: {{
            CodoxearUrls: {{
              resolveAppUrl: (path) => new URL(String(path ?? "").replace(/^\\//, ""), "http://localhost/").toString(),
            }},
          }},
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        process.stdout.write(ctx.window.CodoxearMarkdown.mdToHtml({json.dumps(markdown)}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.stdout


def css_media_block(css: str, marker: str) -> str:
    start = css.index(marker)
    open_brace = css.index("{", start)
    depth = 1
    pos = open_brace + 1
    while pos < len(css) and depth:
        if css[pos] == "{":
            depth += 1
        elif css[pos] == "}":
            depth -= 1
        pos += 1
    return css[open_brace + 1 : pos - 1]


def eval_code_copy_runtime() -> dict:
    source = APP_CODE_COPY_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        const calls = [];
        function classList() {{
          const values = new Set();
          return {{
            values,
            add(name) {{ values.add(name); calls.push(["class-add", name]); }},
            remove(name) {{ values.delete(name); calls.push(["class-remove", name]); }},
            contains(name) {{ return values.has(name); }},
          }};
        }}
        const code = {{ textContent: "first <block> & only\\n" }};
        const otherCode = {{ textContent: "second block" }};
        const pre = {{ querySelector: (selector) => selector === "code" ? code : null }};
        const otherPre = {{ querySelector: (selector) => selector === "code" ? otherCode : null }};
        const attrs = {{ "aria-label": "Copy code", title: "Copy code" }};
        const button = {{
          classList: classList(),
          closest: (selector) => selector === "pre" ? pre : null,
          getAttribute: (name) => attrs[name] || "",
          setAttribute: (name, value) => {{ attrs[name] = String(value); calls.push(["attr", name, String(value)]); }},
        }};
        const child = {{ closest: (selector) => selector === ".code-copy-btn" ? button : null }};
        const otherButton = {{ closest: (selector) => selector === "pre" ? otherPre : null }};
        const runtime = ctx.window.CodoxearCodeCopy.createCodeBlockCopyRuntime({{
          copyToClipboard: async (text) => calls.push(["copy", text]),
          setToast: (text) => calls.push(["toast", text]),
          setTimeout: (fn, ms) => {{ calls.push(["timeout", ms]); ctx.resetCopy = fn; return 7; }},
          clearTimeout: (id) => calls.push(["clearTimeout", id]),
        }});
        let prevented = 0;
        let stopped = 0;
        let fileRefCalls = 0;
        const event = {{
          target: child,
          preventDefault: () => prevented += 1,
          stopPropagation: () => stopped += 1,
        }};
        if (!runtime.handleClick(event)) fileRefCalls += 1;
        const miss = runtime.handleClick({{ target: {{ closest: () => null }} }});
        setImmediate(() => {{
          ctx.resetCopy();
          process.stdout.write(JSON.stringify({{
            frozen: Object.isFrozen(ctx.window.CodoxearCodeCopy),
            copiedText: calls.find((call) => call[0] === "copy")[1],
            directText: ctx.window.CodoxearCodeCopy.codeTextForCopyButton(otherButton),
            prevented,
            stopped,
            fileRefCalls,
            miss,
            calls,
            ariaLabel: attrs["aria-label"],
            title: attrs.title,
            copiedClassAfterReset: button.classList.contains("copied"),
          }}));
        }});
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


class TestCodeBlockCopySource(unittest.TestCase):
    def test_markdown_code_blocks_render_accessible_button_without_code_attributes(self) -> None:
        html = render_markdown("before\n\n```sh-script\nalpha <tag> & \"quote\"\n```\n\nafter")
        self.assertIn('<pre><button class="code-copy-btn" type="button" aria-label="Copy code" title="Copy code"></button><code data-lang="sh-script">alpha &lt;tag&gt; &amp; &quot;quote&quot;</code></pre>', html)
        button_start = html.index('<button class="code-copy-btn"')
        button_end = html.index("</button>", button_start)
        button_html = html[button_start:button_end]
        self.assertNotIn("alpha", button_html)
        self.assertNotIn("sh-script", button_html)

    def test_code_copy_runtime_copies_only_nearest_code_text(self) -> None:
        result = eval_code_copy_runtime()
        self.assertTrue(result["frozen"])
        self.assertEqual(result["copiedText"], "first <block> & only\n")
        self.assertEqual(result["directText"], "second block")
        self.assertEqual(result["prevented"], 1)
        self.assertEqual(result["stopped"], 1)
        self.assertEqual(result["fileRefCalls"], 0)
        self.assertFalse(result["miss"])
        self.assertIn(["toast", "Copied code"], result["calls"])
        self.assertIn(["timeout", 1200], result["calls"])
        self.assertEqual(result["ariaLabel"], "Copy code")
        self.assertEqual(result["title"], "Copy code")

    def test_code_copy_runtime_restores_button_after_feedback(self) -> None:
        result = eval_code_copy_runtime()
        self.assertFalse(result["copiedClassAfterReset"])
        self.assertEqual(result["ariaLabel"], "Copy code")
        self.assertEqual(result["title"], "Copy code")
        self.assertIn(["class-add", "copied"], result["calls"])
        self.assertIn(["class-remove", "copied"], result["calls"])

    def test_code_copy_runtime_ignores_non_copy_clicks(self) -> None:
        result = eval_code_copy_runtime()
        self.assertFalse(result["miss"])
        self.assertEqual(result["fileRefCalls"], 0)
        self.assertEqual(result["directText"], "second block")


if __name__ == "__main__":
    unittest.main()
