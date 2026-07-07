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
          setTimeout: (fn, ms) => {{ calls.push(["timeout", ms]); return 7; }},
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
        self.assertEqual(result["ariaLabel"], "Copied code")
        self.assertEqual(result["title"], "Copied code")

    def test_app_js_delegates_code_copy_before_file_references(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("const codoxearCodeCopy = window.CodoxearCodeCopy;", source)
        self.assertIn('throw new Error("Codoxear code copy helpers failed to load")', source)
        self.assertIn("const codeBlockCopyRuntime = codoxearCodeCopy.createCodeBlockCopyRuntime({", source)
        handler_start = source.index('chatInner.addEventListener("click", (e) => {')
        handler_end = source.index("});", handler_start)
        handler = source[handler_start:handler_end]
        self.assertIn("if (codeBlockCopyRuntime.handleClick(e)) return;", handler)
        self.assertIn("void fileReferenceRuntime.handleClick(e);", handler)
        self.assertLess(handler.index("codeBlockCopyRuntime.handleClick(e)"), handler.index("fileReferenceRuntime.handleClick(e)"))

    def test_code_copy_css_preserves_pre_layout_and_mobile_touch_target(self) -> None:
        css = APP_CSS.read_text(encoding="utf-8")
        pre_start = css.index(".md pre {")
        pre_end = css.index("}", pre_start)
        pre_block = css[pre_start:pre_end]
        self.assertIn("position: relative;", pre_block)
        self.assertIn("overflow: auto;", pre_block)
        self.assertIn("max-width: 100%;", pre_block)
        self.assertIn("width: 100%;", pre_block)
        self.assertIn("box-sizing: border-box;", pre_block)
        self.assertIn("white-space: pre-wrap;", pre_block)
        self.assertIn("overflow-wrap: anywhere;", pre_block)
        button_start = css.index(".code-copy-btn {")
        button_end = css.index("}", button_start)
        button_block = css[button_start:button_end]
        self.assertIn("position: absolute;", button_block)
        self.assertIn("right: 6px;", button_block)
        self.assertNotIn("right: -", button_block)
        self.assertIn("width: 30px", button_block)
        mobile_start = css.index("@media (max-width: 520px)")
        mobile_block = css[mobile_start:]
        next_media = mobile_block.find("@media", 1)
        mobile_block = mobile_block if next_media == -1 else mobile_block[:next_media]
        self.assertIn(".code-copy-btn", mobile_block)
        mobile_button_start = mobile_block.index(".code-copy-btn")
        mobile_button_end = mobile_block.index("}", mobile_button_start)
        mobile_button = mobile_block[mobile_button_start:mobile_button_end]
        self.assertIn("width: 44px", mobile_button)
        self.assertIn("height: 44px", mobile_button)
        self.assertIn("min-width: 44px", mobile_button)
        self.assertIn("min-height: 44px", mobile_button)
        mobile_pre_start = mobile_block.index(".md pre")
        mobile_pre_end = mobile_block.index("}", mobile_pre_start)
        mobile_pre = mobile_block[mobile_pre_start:mobile_pre_end]
        self.assertIn("padding-right: 58px", mobile_pre)

    def test_code_copy_asset_loads_before_app_js(self) -> None:
        source = INDEX_HTML.read_text(encoding="utf-8")
        self.assertIn('app_code_copy.js?v=__CODOXEAR_ASSET_VERSION__', source)
        self.assertLess(source.index('app_clipboard.js?v=__CODOXEAR_ASSET_VERSION__'), source.index('app_code_copy.js?v=__CODOXEAR_ASSET_VERSION__'))
        self.assertLess(source.index('app_code_copy.js?v=__CODOXEAR_ASSET_VERSION__'), source.index('app.js?v=__CODOXEAR_ASSET_VERSION__'))


if __name__ == "__main__":
    unittest.main()
