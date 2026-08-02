import json
import re
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
        const pre = {{
          classList: classList(),
          querySelector: (selector) => selector === "code" || selector === ":scope > .code-copy-btn" ? (selector === "code" ? code : button) : null,
        }};
        const otherPre = {{
          classList: classList(),
          querySelector: (selector) => selector === "code" || selector === ":scope > .code-copy-btn" ? (selector === "code" ? otherCode : otherButton) : null,
        }};
        const attrs = {{ "aria-label": "Copy code", title: "Copy code" }};
        const button = {{
          classList: classList(),
          closest: (selector) => selector === "pre" ? pre : null,
          getAttribute: (name) => attrs[name] || "",
          setAttribute: (name, value) => {{ attrs[name] = String(value); calls.push(["attr", name, String(value)]); }},
        }};
        const child = {{ closest: (selector) => selector === ".code-copy-btn" ? button : null }};
        const otherButton = {{ closest: (selector) => selector === "pre" ? otherPre : null }};
        const toggleRoot = {{
          querySelectorAll: (selector) => selector === "pre.show-copy" ? [pre, otherPre].filter((item) => item.classList.contains("show-copy")) : [],
        }};
        const runtime = ctx.window.CodoxearCodeCopy.createCodeBlockCopyRuntime({{
          copyToClipboard: async (text) => calls.push(["copy", text]),
          setToast: (text) => calls.push(["toast", text]),
          setTimeout: (fn, ms) => {{ calls.push(["timeout", ms]); ctx.resetCopy = fn; return 7; }},
          clearTimeout: (id) => calls.push(["clearTimeout", id]),
        }});
        const touchFirst = runtime.toggleTouchPre(pre, toggleRoot);
        const touchFirstState = [pre, otherPre].map((item) => item.classList.contains("show-copy"));
        const touchSecond = runtime.toggleTouchPre(otherPre, toggleRoot);
        const touchSecondState = [pre, otherPre].map((item) => item.classList.contains("show-copy"));
        const touchHidden = runtime.toggleTouchPre(otherPre, toggleRoot);
        const touchHiddenState = [pre, otherPre].map((item) => item.classList.contains("show-copy"));
        const codePreFound = ctx.window.CodoxearCodeCopy.codePreFromTarget({{ closest: (selector) => selector === "pre" ? pre : null }});
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
            touchFirst: touchFirst === pre,
            touchFirstState,
            touchSecond: touchSecond === otherPre,
            touchSecondState,
            touchHidden,
            touchHiddenState,
            codePreFound: codePreFound === pre,
          }}));
        }});
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


class TestCodeBlockCopySource(unittest.TestCase):
    def test_markdown_code_blocks_are_decorated_after_marked_parsing(self) -> None:
        source = APP_MARKDOWN_JS.read_text(encoding="utf-8")
        self.assertIn("function decorateCodeBlocks", source)
        self.assertIn('root.querySelectorAll("pre")', source)
        self.assertIn('button.className = "code-copy-btn"', source)
        self.assertIn('button.setAttribute("aria-label", "Copy code")', source)
        self.assertIn('button.title = "Copy code"', source)
        self.assertIn('code.dataset.lang = languageClass.slice', source)

    def test_code_copy_runtime_copies_only_nearest_code_text(self) -> None:
        result = eval_code_copy_runtime()
        self.assertTrue(result["frozen"])
        self.assertEqual(result["copiedText"], "first <block> & only\n")
        self.assertEqual(result["directText"], "second block")
        self.assertEqual(result["prevented"], 1)
        self.assertEqual(result["stopped"], 1)
        self.assertEqual(result["fileRefCalls"], 0)
        self.assertFalse(result["miss"])
        self.assertContains(["toast", "Copied code"], result["calls"])
        self.assertContains(["timeout", 1200], result["calls"])
        self.assertEqual(result["ariaLabel"], "Copy code")
        self.assertEqual(result["title"], "Copy code")

    def test_code_copy_runtime_restores_button_after_feedback(self) -> None:
        result = eval_code_copy_runtime()
        self.assertFalse(result["copiedClassAfterReset"])
        self.assertEqual(result["ariaLabel"], "Copy code")
        self.assertEqual(result["title"], "Copy code")
        self.assertContains(["class-add", "copied"], result["calls"])
        self.assertContains(["class-remove", "copied"], result["calls"])

    def test_code_copy_runtime_ignores_non_copy_clicks(self) -> None:
        result = eval_code_copy_runtime()
        self.assertFalse(result["miss"])
        self.assertEqual(result["fileRefCalls"], 0)
        self.assertEqual(result["directText"], "second block")

    def test_code_copy_runtime_toggles_one_touch_block_at_a_time(self) -> None:
        result = eval_code_copy_runtime()
        self.assertTrue(result["touchFirst"])
        self.assertEqual(result["touchFirstState"], [True, False])
        self.assertTrue(result["touchSecond"])
        self.assertEqual(result["touchSecondState"], [False, True])
        self.assertIsNone(result["touchHidden"])
        self.assertEqual(result["touchHiddenState"], [False, False])
        self.assertTrue(result["codePreFound"])

    def test_code_copy_reveal_css_reclaims_code_width(self) -> None:
        css = APP_CSS.read_text(encoding="utf-8")
        button = re.search(r"\.code-copy-btn\s*\{(?P<body>[^}]*)\}", css)
        self.assertIsNotNone(button)
        self.assertIn("opacity: 0", button.group("body"))
        self.assertIn("pointer-events: none", button.group("body"))
        self.assertIn(".md pre:hover .code-copy-btn", css)
        self.assertIn(".md pre:focus-within .code-copy-btn", css)
        self.assertIn(".md pre.show-copy .code-copy-btn", css)
        pre = re.search(r"\.md pre\s*\{(?P<body>[^}]*)\}", css)
        self.assertIsNotNone(pre)
        self.assertIn("padding: 12px", pre.group("body"))
        self.assertNotIn("padding-right", pre.group("body"))
        touch = css_media_block(css, "@media (max-width: 700px), (pointer: coarse)")
        self.assertNotIn(".md pre {", touch)
        self.assertIn(".code-copy-btn::after", touch)
        self.assertIn("inset: -7px", touch)

    def test_touch_code_copy_toggle_is_wired_before_message_copy_toggle(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("window.getSelection().toString()", source)
        self.assertIn("a, button, input, select, textarea, [role='link'], mark", source)
        self.assertIn("const pre = codoxearCodeCopy.codePreFromTarget(target);", source)
        self.assertIn("codeBlockCopyRuntime.toggleTouchPre(pre, chatInner);", source)
        self.assertLess(
            source.index("codeBlockCopyRuntime.toggleTouchPre(pre, chatInner);"),
            source.index("messageCopyNavigationRuntime.toggleTouchRow(row);"),
        )


if __name__ == "__main__":
    unittest.main()
