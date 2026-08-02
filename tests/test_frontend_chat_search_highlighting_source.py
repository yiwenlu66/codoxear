import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MESSAGE_ROWS = ROOT / "codoxear" / "static" / "app_message_rows.js"


def run_highlight_dom() -> dict:
    source = json.dumps(MESSAGE_ROWS.read_text(encoding="utf-8"))
    js = f"""
    const vm = require("vm");
    const ctx = {{ window: {{}} }};
    vm.createContext(ctx);
    vm.runInContext({source}, ctx);

    class ClassList {{
      constructor() {{ this.values = new Set(); }}
      add(...names) {{ for (const name of names) this.values.add(name); }}
      remove(...names) {{ for (const name of names) this.values.delete(name); }}
      contains(name) {{ return this.values.has(name); }}
    }}

    class FakeNode {{
      constructor(doc, type, tag = "", value = "") {{
        this.ownerDocument = doc;
        this.nodeType = type;
        this.tagName = tag;
        this.nodeValue = type === 3 ? value : null;
        this.childNodes = [];
        this.parentNode = null;
        this.classList = new ClassList();
      }}
      get className() {{ return Array.from(this.classList.values).join(" "); }}
      set className(value) {{ this.classList = new ClassList(); this.classList.add(...String(value || "").split(/\\s+/).filter(Boolean)); }}
      get textContent() {{ return this.nodeType === 3 ? this.nodeValue : this.childNodes.map((child) => child.textContent).join(""); }}
      set textContent(value) {{
        if (this.nodeType === 3) {{ this.nodeValue = String(value); return; }}
        this.childNodes = [];
        if (value) this.appendChild(this.ownerDocument.createTextNode(String(value)));
      }}
      removeChild(child) {{
        const index = this.childNodes.indexOf(child);
        if (index >= 0) this.childNodes.splice(index, 1);
        child.parentNode = null;
        return child;
      }}
      appendChild(child) {{
        if (child.nodeType === 11) {{
          for (const item of [...child.childNodes]) this.appendChild(item);
          return child;
        }}
        if (child.parentNode) child.parentNode.removeChild(child);
        this.childNodes.push(child);
        child.parentNode = this;
        return child;
      }}
      replaceChild(replacement, oldChild) {{
        const index = this.childNodes.indexOf(oldChild);
        if (index < 0) throw new Error("missing replacement child");
        const inserted = replacement.nodeType === 11 ? [...replacement.childNodes] : [replacement];
        if (replacement.nodeType === 11) replacement.childNodes = [];
        for (const child of inserted) {{
          if (child.parentNode) child.parentNode.removeChild(child);
          child.parentNode = this;
        }}
        this.childNodes.splice(index, 1, ...inserted);
        oldChild.parentNode = null;
        return oldChild;
      }}
      normalize() {{
        for (const child of [...this.childNodes]) child.normalize();
        for (let index = 0; index < this.childNodes.length - 1;) {{
          const left = this.childNodes[index];
          const right = this.childNodes[index + 1];
          if (left.nodeType === 3 && right.nodeType === 3) {{
            left.nodeValue += right.nodeValue;
            this.removeChild(right);
          }} else index += 1;
        }}
      }}
      querySelectorAll(selector) {{
        const parts = selector.trim().split(/\\s+/);
        const descendants = [];
        const visit = (node) => {{ for (const child of node.childNodes) {{ descendants.push(child); visit(child); }} }};
        visit(this);
        const matches = (node, part) => {{
          const [tag, ...classes] = part.split(".");
          return node.nodeType === 1 && (!tag || node.tagName === tag) && classes.every((name) => node.classList.contains(name));
        }};
        return descendants.filter((node) => {{
          if (!matches(node, parts[parts.length - 1])) return false;
          let ancestor = node.parentNode;
          for (let index = parts.length - 2; index >= 0; index -= 1) {{
            while (ancestor && !matches(ancestor, parts[index])) ancestor = ancestor.parentNode;
            if (!ancestor) return false;
            ancestor = ancestor.parentNode;
          }}
          return true;
        }});
      }}
      querySelector(selector) {{ return this.querySelectorAll(selector)[0] || null; }}
    }}

    class FakeDocument {{
      constructor() {{ this.defaultView = {{ NodeFilter: {{ SHOW_TEXT: 4 }} }}; }}
      createTextNode(value) {{ return new FakeNode(this, 3, "", value); }}
      createElement(tag) {{ return new FakeNode(this, 1, tag); }}
      createDocumentFragment() {{ return new FakeNode(this, 11); }}
      createTreeWalker(root) {{
        const textNodes = [];
        const visit = (node) => {{ for (const child of node.childNodes) {{ if (child.nodeType === 3) textNodes.push(child); visit(child); }} }};
        visit(root);
        let index = 0;
        return {{ nextNode: () => textNodes[index++] || null }};
      }}
    }}

    const doc = new FakeDocument();
    const row = doc.createElement("div");
    row.className = "msg-row assistant";
    const msg = doc.createElement("div");
    msg.className = "msg assistant";
    const md = doc.createElement("div");
    md.className = "md";
    md.appendChild(doc.createTextNode("Alpha needle "));
    const emphasis = doc.createElement("em");
    emphasis.appendChild(doc.createTextNode("NEEDLE"));
    md.appendChild(emphasis);
    msg.appendChild(md);
    row.appendChild(msg);

    const rows = ctx.window.CodoxearMessageRows;
    rows.applyChatSearchMarks([row], row, "needle");
    const firstPass = {{
      text: md.textContent,
      hits: row.querySelectorAll("mark.searchHit").length,
      current: row.querySelectorAll("mark.searchHit.searchHitCurrent").length,
      currentRow: row.classList.contains("chat-search-current"),
    }};
    rows.clearChatSearchMarks([row]);
    const cleared = {{
      text: md.textContent,
      hits: row.querySelectorAll("mark.searchHit").length,
      currentRow: row.classList.contains("chat-search-current"),
    }};
    rows.applyChatSearchMarks([row], row, "alpha");
    const secondPass = {{
      hits: row.querySelectorAll("mark.searchHit").length,
      hitText: row.querySelector("mark.searchHit").textContent,
      current: row.querySelectorAll("mark.searchHit.searchHitCurrent").length,
    }};
    process.stdout.write(JSON.stringify({{ firstPass, cleared, secondPass }}));
    """
    proc = subprocess.run(["node", "-e", textwrap.dedent(js)], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return json.loads(proc.stdout)


class TestFrontendChatSearchHighlighting(unittest.TestCase):
    def test_text_node_hits_current_state_and_unwrap_round_trip(self) -> None:
        result = run_highlight_dom()
        self.assertEqual(result["firstPass"], {
            "text": "Alpha needle NEEDLE",
            "hits": 2,
            "current": 2,
            "currentRow": True,
        })
        self.assertEqual(result["cleared"], {
            "text": "Alpha needle NEEDLE",
            "hits": 0,
            "currentRow": False,
        })
        self.assertEqual(result["secondPass"], {"hits": 1, "hitText": "Alpha", "current": 1})


if __name__ == "__main__":
    unittest.main()
