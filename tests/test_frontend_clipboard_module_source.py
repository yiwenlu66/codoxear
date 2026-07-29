import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_CLIPBOARD_JS = ROOT / "codoxear" / "static" / "app_clipboard.js"
INDEX_HTML = ROOT / "codoxear" / "static" / "index.html"


def eval_clipboard_helpers() -> dict:
    source = APP_CLIPBOARD_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const events = [];
        class FakeHTMLElement {{
          focus(opts) {{ events.push(["active-focus", Boolean(opts && opts.preventScroll)]); }}
        }}
        const active = new FakeHTMLElement();
        const textareas = [];
        const document = {{
          activeElement: active,
          body: {{ appendChild(node) {{ events.push(["append", node.tagName]); }} }},
          createElement(tag) {{
            const node = {{
              tagName: tag,
              value: "",
              attrs: {{}},
              style: {{}},
              setAttribute(name, value) {{ this.attrs[name] = String(value); }},
              focus(opts) {{ events.push(["textarea-focus", Boolean(opts && opts.preventScroll)]); }},
              select() {{ events.push(["select"]); }},
              setSelectionRange(start, end) {{ events.push(["range", start, end]); }},
              remove() {{ events.push(["remove"]); }},
            }};
            textareas.push(node);
            return node;
          }},
          execCommand(command) {{ events.push(["exec", command]); return true; }},
        }};
        const writes = [];
        const ctx = {{
          HTMLElement: FakeHTMLElement,
          document,
          navigator: {{ clipboard: {{ writeText(value) {{ writes.push(value); return Promise.resolve(); }} }} }},
          window: {{ isSecureContext: true, navigator: null }},
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        const helpers = ctx.window.CodoxearClipboard;
        (async () => {{
          await helpers.copyToClipboard("secure text");
          ctx.window.isSecureContext = false;
          await helpers.copyToClipboard("fallback text");
          process.stdout.write(JSON.stringify({{
            writes,
            events,
            textarea: textareas[0],
            frozen: Object.isFrozen(helpers),
          }}));
        }})().catch((err) => {{ console.error(err && err.stack || err); process.exit(1); }});
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


class TestFrontendClipboardModuleSource(unittest.TestCase):
    def test_index_loads_clipboard_before_app(self) -> None:
        source = INDEX_HTML.read_text(encoding="utf-8")
        self.assertIn('app_clipboard.js?v=__CODOXEAR_ASSET_VERSION__', source)
        self.assertLess(source.index('app_modal.js?v=__CODOXEAR_ASSET_VERSION__'), source.index('app_clipboard.js?v=__CODOXEAR_ASSET_VERSION__'))
        self.assertLess(source.index('app_clipboard.js?v=__CODOXEAR_ASSET_VERSION__'), source.index('app.js?v=__CODOXEAR_ASSET_VERSION__'))

    def test_app_js_requires_clipboard_helpers_without_fallback(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        helper_source = APP_CLIPBOARD_JS.read_text(encoding="utf-8")
        self.assertIn("const codoxearClipboard = window.CodoxearClipboard;", source)
        self.assertIn('throw new Error("Codoxear clipboard helpers failed to load")', source)
        self.assertIn("return codoxearClipboard.copyTextViaSelection(text);", source)
        self.assertIn("return await codoxearClipboard.copyToClipboard(text);", source)
        self.assertIn('document.execCommand("copy")', helper_source)
        self.assertIn('const nav = typeof navigator !== "undefined" ? navigator : window.navigator;', helper_source)
        self.assertIn("nav.clipboard", helper_source)

    def test_clipboard_module_preserves_secure_and_selection_copy_paths(self) -> None:
        result = eval_clipboard_helpers()
        self.assertEqual(result["writes"], ["secure text"])
        self.assertEqual(
            result["events"],
            [
                ["append", "textarea"],
                ["textarea-focus", True],
                ["select"],
                ["range", 0, 13],
                ["exec", "copy"],
                ["remove"],
                ["active-focus", True],
            ],
        )
        self.assertEqual(result["textarea"]["value"], "fallback text")
        self.assertEqual(result["textarea"]["attrs"], {"aria-hidden": "true", "readonly": ""})
        self.assertEqual(result["textarea"]["style"]["position"], "fixed")
        self.assertTrue(result["frozen"])


if __name__ == "__main__":
    unittest.main()
