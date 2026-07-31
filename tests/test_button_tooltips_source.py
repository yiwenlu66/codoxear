import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_DISPLAY_JS = ROOT / "codoxear" / "static" / "app_display.js"
APP_DOM_JS = ROOT / "codoxear" / "static" / "app_dom.js"


def eval_dom_helper() -> dict:
    source = APP_DOM_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const created = [];
        const document = {{
          createElement(tag) {{
            const node = {{
              tag,
              className: "",
              textContent: "",
              innerHTML: "",
              attrs: {{}},
              children: [],
              setAttribute(k, v) {{ this.attrs[k] = String(v); }},
              getAttribute(k) {{ return Object.prototype.hasOwnProperty.call(this.attrs, k) ? this.attrs[k] : null; }},
              appendChild(child) {{ this.children.push(child.name); }},
            }};
            created.push(node);
            return node;
          }},
        }};
        const ctx = {{ document, window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        const helper = ctx.window.CodoxearDom;
        const child = {{ name: "child" }};
        const titled = helper.createElement("button", {{ title: "Already", text: "Save" }}, [], () => "Fallback");
        const untitled = helper.createElement("button", {{ class: "primary", text: "Save", "data-x": "1" }}, [child], (attrs, node) => `${{attrs.text}} tooltip ${{node.className}}`);
        const html = helper.createElement("div", {{ html: "<b>x</b>" }});
        process.stdout.write(JSON.stringify({{
          titled,
          untitled,
          html,
          frozen: Object.isFrozen(helper),
        }}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


class TestButtonTooltipsSource(unittest.TestCase):
    def test_dom_helper_preserves_element_creation_contract(self) -> None:
        result = eval_dom_helper()
        self.assertEqual(result["titled"]["attrs"]["title"], "Already")
        self.assertEqual(result["untitled"]["className"], "primary")
        self.assertEqual(result["untitled"]["textContent"], "Save")
        self.assertEqual(result["untitled"]["attrs"]["data-x"], "1")
        self.assertEqual(result["untitled"]["attrs"]["title"], "Save tooltip primary")
        self.assertEqual(result["untitled"]["children"], ["child"])
        self.assertEqual(result["html"]["innerHTML"], "<b>x</b>")
        self.assertTrue(result["frozen"])


if __name__ == "__main__":
    unittest.main()
