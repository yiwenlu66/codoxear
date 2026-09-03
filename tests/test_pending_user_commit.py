"""Behavioral tests for the pending-echo commit path.

A web-sent message renders an optimistic echo from the composer text. When the
backend-committed log event arrives, consumePendingUserIfMatches patches the
echo bubble in place. The committed text can differ from the echoed text
(server-injected "Attachment N: /path" prefix lines), so the commit must
re-establish everything makeRow derived from the event text: the copy
authority and the candidate file-ref upgrade. These tests pin that contract.
"""

from __future__ import annotations
from frontend_module_loader import module_path

import json
import subprocess
import textwrap
import unittest


ROWS_JS = module_path("app_message_rows.js")
TRANSCRIPT_JS = module_path("app_transcript.js")


def run_node(script: str) -> dict:
    result = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


FAKE_DOM = """
    function fakeEl(tag, props = {}) {
      const node = {
        tag,
        className: props.class || "",
        dataset: {},
        style: {},
        children: [],
        attrs: {},
        textContent: props.text || "",
        innerHTML: props.html || "",
        appendChild(child) { this.children.push(child); return child; },
        setAttribute(name, value) { this.attrs[name] = String(value); },
        removeAttribute(name) { delete this.attrs[name]; },
        querySelector(sel) {
          if (sel === ".md") return this.mdChild || null;
          if (sel === ".ts") return this.tsChild || null;
          return null;
        },
        closest() { return this.parentRow || null; },
      };
      return node;
    }
"""


class TestMessageRowCopyAuthority(unittest.TestCase):
    def test_copy_button_reads_row_copy_text_at_click_time(self) -> None:
        rows_source = ROWS_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const vm = require("vm");
            const buttons = [];
            {FAKE_DOM}
            const el = (tag, props) => {{
              const node = fakeEl(tag, props);
              if (tag === "button") buttons.push(node);
              return node;
            }};
            const copied = [];
            const deps = {{
              el,
              chatMarkdownHtmlCached: () => "<p>rendered</p>",
              upgradeCandidateFileRefs: () => {{}},
              time24: () => "00:00",
              iconSvg: () => "",
              copyToClipboard: async (text) => {{ copied.push(text); }},
              setToast: () => {{}},
              chatAssistantDedupeKey: () => "",
              setTimeout: () => {{}},
              selectedSessionId: "sid",
            }};
            const ctx = {{ window: {{}}, console }};
            vm.createContext(ctx);
            vm.runInContext({json.dumps(rows_source)}, ctx);
            const makeRow = ctx.window.CodoxearMessageRows ? null : null;
            const rows = ctx.window.CodoxearMessageRows;
            (async () => {{
              // Pending echo: composer text only, no attachment prefix.
              const pending = rows.makeRow({{ role: "user", text: "look at this" }}, {{ pending: true }}, deps);
              // The pending-commit path rewrites the same slot when the
              // backend-committed event lands with the injected prefix.
              pending.row.copyText = "Attachment 1: /uploads/shot.png\\nlook at this";
              await buttons[0].onclick({{ preventDefault() {{}}, stopPropagation() {{}} }});
              process.stdout.write(JSON.stringify({{
                copied,
                seeded: pending.row.copyText,
                buttonCount: buttons.length,
              }}));
            }})();
            """
        )
        out = run_node(js)
        self.assertEqual(out["buttonCount"], 1)
        self.assertEqual(out["copied"], ["Attachment 1: /uploads/shot.png\nlook at this"])

    def test_pending_commit_rewrites_copy_text_and_upgrades_file_refs(self) -> None:
        transcript_source = TRANSCRIPT_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const vm = require("vm");
            {FAKE_DOM}
            const md = fakeEl("div");
            const ts = fakeEl("div");
            const pendingElement = fakeEl("div");
            pendingElement.mdChild = md;
            pendingElement.tsChild = ts;
            const row = fakeEl("div");
            row.dataset = {{}};
            row.copyText = "look at this";
            pendingElement.parentRow = row;
            const chatInner = {{ querySelector: () => pendingElement }};
            const upgrades = [];
            const rendered = [];
            const rebuilt = [];
            const seen = [];
            const controllerOptions = {{
              sessionState: {{ get: () => "sid" }},
              takePendingUserMatch: () => ({{ id: 7, text: "look at this" }}),
              chatInner,
              markdownHtml: (text, sid) => {{ rendered.push([text, sid]); return "<p>committed</p>"; }},
              time24: () => "12:00",
              rebuildDecorations: (opts) => rebuilt.push(opts),
              markEventSeen: (ev) => seen.push(ev.text),
              upgradeCandidateFileRefs: (rootEl) => {{ upgrades.push(rootEl === md); }},
            }};
            const ctx = {{ window: {{}}, console }};
            vm.createContext(ctx);
            vm.runInContext({json.dumps(transcript_source)}, ctx);
            const controller = ctx.window.CodoxearPendingUser.createPendingUserController(controllerOptions);
            const committed = {{
              role: "user",
              text: "Attachment 1: /uploads/shot.png\\nAttachment 2: /uploads/other.png\\nlook at this",
              ts: 1234.5,
            }};
            const consumed = controller.consumePendingUserIfMatches(committed, "sid");
            let missingDep = "";
            try {{
              const {{ upgradeCandidateFileRefs, ...rest }} = controllerOptions;
              ctx.window.CodoxearPendingUser.createPendingUserController(rest);
            }} catch (err) {{ missingDep = err && err.message ? err.message : String(err); }}
            process.stdout.write(JSON.stringify({{
              consumed,
              copyText: row.copyText,
              rowTs: row.dataset.ts,
              mdHtml: md.innerHTML,
              rendered,
              upgrades,
              rebuilt,
              seen,
              missingDep,
              pendingCleared: !("local-id" in pendingElement.attrs) && !("data-pending" in pendingElement.attrs),
            }}));
            """
        )
        out = run_node(js)
        committed_text = "Attachment 1: /uploads/shot.png\nAttachment 2: /uploads/other.png\nlook at this"
        self.assertTrue(out["consumed"])
        self.assertEqual(out["copyText"], committed_text)
        self.assertEqual(out["rendered"], [[committed_text, "sid"]])
        self.assertEqual(out["mdHtml"], "<p>committed</p>")
        self.assertEqual(out["upgrades"], [True])
        self.assertEqual(out["rowTs"], "1234.5")
        self.assertEqual(out["rebuilt"], [{"preserveScroll": True}])
        self.assertEqual(out["seen"], [committed_text])
        self.assertTrue(out["pendingCleared"])
        self.assertIn("upgradeCandidateFileRefs", out["missingDep"])


if __name__ == "__main__":
    unittest.main()
