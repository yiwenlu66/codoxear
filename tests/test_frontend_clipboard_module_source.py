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
        let insecureError = null;
        (async () => {{
          await helpers.copyToClipboard("secure text");
          ctx.window.isSecureContext = false;
          try {{
            await helpers.copyToClipboard("fallback text");
          }} catch (e) {{
            insecureError = e.message || String(e);
          }}
          process.stdout.write(JSON.stringify({{
            writes,
            events,
            insecureError,
            frozen: Object.isFrozen(helpers),
          }}));
        }})().catch((err) => {{ console.error(err && err.stack || err); process.exit(1); }});
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


class TestFrontendClipboardModuleSource(unittest.TestCase):


    def test_clipboard_module_uses_secure_clipboard_only(self) -> None:
        result = eval_clipboard_helpers()
        self.assertEqual(result["writes"], ["secure text"])
        self.assertEqual(result["events"], [])
        self.assertContains("Clipboard API unavailable", result.get("insecureError", ""))
        self.assertTrue(result["frozen"])


if __name__ == "__main__":
    unittest.main()
