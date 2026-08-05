import json
import os
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_HINT_MODE_JS = ROOT / "codoxear" / "static" / "app_hint_mode.js"
APP_SHELL_JS = ROOT / "codoxear" / "static" / "app_shell.js"


def run_shell_hint_coverage() -> dict:
    hint_source = APP_HINT_MODE_JS.read_text(encoding="utf-8")
    shell_source = APP_SHELL_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("node:vm");
        const hintSource = {json.dumps(hint_source)};
        const shellSource = {json.dumps(shell_source)};

        class Node {{
          constructor(tag, attrs = {{}}) {{
            this.tagName = String(tag).toUpperCase();
            this.attrs = {{}};
            this.children = [];
            this.parentNode = null;
            this.style = {{ display: "block", visibility: "visible" }};
            this.disabled = false;
            this.offsetParent = {{}};
            this.className = "";
            this.textContent = "";
            for (const [name, value] of Object.entries(attrs)) this.setAttribute(name, value);
          }}
          appendChild(child) {{
            if (child.parentNode) child.parentNode.removeChild(child);
            child.parentNode = this;
            this.children.push(child);
            return child;
          }}
          append(...children) {{ children.forEach((child) => this.appendChild(child)); }}
          removeChild(child) {{
            const index = this.children.indexOf(child);
            if (index >= 0) this.children.splice(index, 1);
            child.parentNode = null;
            return child;
          }}
          setAttribute(name, value) {{
            const text = String(value);
            this.attrs[name] = text;
            if (name === "class") this.className = text;
            if (name === "text") this.textContent = text;
            if (name === "disabled") this.disabled = true;
            if (name === "style" && text.includes("display:none")) {{
              this.style.display = "none";
              this.offsetParent = null;
            }}
          }}
          getAttribute(name) {{ return Object.hasOwn(this.attrs, name) ? this.attrs[name] : null; }}
          hasAttribute(name) {{ return Object.hasOwn(this.attrs, name); }}
          remove() {{ if (this.parentNode) this.parentNode.removeChild(this); }}
          getBoundingClientRect() {{ return {{ left: 0, top: 0, right: 20, bottom: 20, width: 20, height: 20 }}; }}
          click() {{ this.clicked = true; }}
          focus() {{ this.focused = true; }}
          contains(target) {{
            return target === this || this.children.some((child) => child.contains(target));
          }}
          closest(selector) {{
            let node = this;
            while (node) {{
              if (selector === ".chat" && String(node.className).split(/\\s+/).includes("chat")) return node;
              node = node.parentNode;
            }}
            return null;
          }}
          querySelectorAll(selector) {{ return query(this, selector); }}
          set innerHTML(_value) {{ this.children = []; }}
        }}

        function descendants(root) {{
          const nodes = [];
          for (const child of root.children) {{
            nodes.push(child, ...descendants(child));
          }}
          return nodes;
        }}
        function matches(node, selector) {{
          if (selector === "[data-hint]") return node.hasAttribute("data-hint");
          if (selector === "#sessions .session[data-session-id]") {{
            return node.hasAttribute("data-session-id") && String(node.className).split(/\\s+/).includes("session");
          }}
          if (selector === ".chat a[data-file-path]" || selector === ".chat a[data-file-picker-query]") {{
            return node.tagName === "A" && Boolean(node.closest(".chat")) && (node.hasAttribute("data-file-path") || node.hasAttribute("data-file-picker-query"));
          }}
          return ["BUTTON", "INPUT", "TEXTAREA", "SELECT", "A"].includes(node.tagName) && (selector === node.tagName.toLowerCase() || selector === "a[href]" && node.hasAttribute("href"));
        }}
        function query(root, selector) {{
          if (selector === ".chat a[data-file-path], .chat a[data-file-picker-query]") {{
            return descendants(root).filter((node) => matches(node, ".chat a[data-file-path]") || matches(node, ".chat a[data-file-picker-query]"));
          }}
          if (selector.includes(",")) return descendants(root).filter((node) => selector.split(",").some((part) => matches(node, part.trim())));
          return descendants(root).filter((node) => matches(node, selector));
        }}

        const body = new Node("body");
        const documentTarget = {{
          body,
          activeElement: null,
          defaultView: {{
            innerHeight: 1000,
            innerWidth: 1000,
            getComputedStyle: (node) => node.style,
          }},
          createElement: (tag) => new Node(tag),
          querySelectorAll(selector) {{ return query(body, selector); }},
        }};
        const el = (tag, attrs = {{}}, children = []) => {{
          const node = new Node(tag, attrs);
          children.forEach((child) => node.appendChild(child));
          return node;
        }};
        const ctx = {{ window: {{}}, document: documentTarget }};
        vm.createContext(ctx);
        vm.runInContext(shellSource, ctx);
        vm.runInContext(hintSource, ctx);
        const root = new Node("root");
        body.appendChild(root);
        const {{ elements }} = ctx.window.CodoxearShell.createShellDOM({{
          root,
          el,
          iconSvg: (name) => `<${{name}}>`,
          resolveAppUrl: (value) => value,
          versionedShellAssetPath: (value) => value,
        }});

        const interactive = descendants(root).filter((node) => ["BUTTON", "INPUT", "TEXTAREA", "SELECT"].includes(node.tagName) || (node.tagName === "A" && node.hasAttribute("href")) || node.getAttribute("role") === "button");
        const id = (node) => node.getAttribute("id") || node.getAttribute("aria-label") || node.tagName;
        const missing = interactive.filter((node) => !node.hasAttribute("data-hint") && !node.hasAttribute("data-hint-excluded")).map(id);
        const excluded = interactive.filter((node) => node.hasAttribute("data-hint-excluded")).map((node) => ({{ id: id(node), reason: node.getAttribute("data-hint-excluded") }}));
        const hintNodes = Array.from(documentTarget.querySelectorAll("[data-hint]"));
        const hints = hintNodes.map((node) => ({{ id: id(node), label: node.getAttribute("data-hint") }}));
        const duplicateHints = hints.filter((entry, index) => hints.findIndex((other) => other.label === entry.label) !== index).map((entry) => entry.label);

        for (const node of hintNodes) {{
          node.disabled = false;
          node.style.display = "block";
          node.style.visibility = "visible";
          node.offsetParent = {{}};
        }}
        const spareButton = el("button", {{ id: "spareControl", type: "button", text: "Spare control" }});
        const overflowButton = el("button", {{ id: "overflowControl", type: "button", text: "Overflow control" }});
        body.append(spareButton, overflowButton);
        const controller = ctx.window.CodoxearHintMode.createHintModeController({{
          documentTarget,
          isTextEntryElement: () => false,
          isMobile: () => false,
          modalIsolationTargets: [],
          isModalTargetOpen: () => false,
          addAppEvent: () => {{}},
          shellHints: hints.map((entry) => ({{ label: entry.label, element: hintNodes.find((node) => id(node) === entry.id) }})),
        }});
        const collected = Array.from(controller.collectTargets().entries()).map(([label, node]) => ({{ label, id: id(node) }}));
        controller.enter();
        for (const key of ["f", "a", "a"]) controller.handleKeydown({{
          key,
          preventDefault() {{}},
        }});
        process.stdout.write(JSON.stringify({{
          missing,
          excluded,
          hints,
          duplicateHints,
          collected,
          overflow: {{ label: collected.find((entry) => entry.id === "overflowControl").label, clicked: Boolean(overflowButton.clicked) }},
        }}));
        """
    )
    completed = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={"PATH": os.environ.get("PATH", ""), "TZ": "UTC"},
    )
    return json.loads(completed.stdout)


class TestHintModeCoverage(unittest.TestCase):
    def test_rendered_shell_interactives_are_declared_or_explicitly_excluded(self) -> None:
        result = run_shell_hint_coverage()

        self.assertEqual(result["missing"], [])
        self.assertEqual(result["duplicateHints"], [])
        self.assertEqual(
            result["excluded"],
            [
                {"id": "chatSearchInput", "reason": "native-text-entry"},
                {"id": "imgInput", "reason": "hidden-file-input"},
            ],
        )
        self.assertEqual(
            {entry["id"]: entry["label"] for entry in result["hints"]},
            {
                "helpBtnSide": "h",
                "settingsBtnSide": "w",
                "logoutBtnSide": "l",
                "chatEmptyNewBtn": "m",
                "olderBtn": "o",
                "olderRetryBtn": "r",
                "jumpBtn": "g",
                "chatSearchPrevBtn": "v",
                "chatSearchNextBtn": "k",
                "chatSearchCloseBtn": "x",
                "threadTitle": "t",
                "ctxChip": "y",
                "interruptBtn": "z",
                "toggleSidebarBtn": "s",
                "unattendedBtn": "u",
                "newBtn": "c",
                "diagBtn": "d",
                "prevUserBtn": "p",
                "nextUserBtn": "n",
                "chatSearchBtn": "/",
                "fileBtn": "b",
                "msg": "i",
                "attachBtn": "a",
                "queueBtn": "q",
                "sendBtn": "e",
            },
        )
        self.assertEqual(result["overflow"], {"label": "faa", "clicked": True})
        self.assertEqual(
            {entry["id"]: entry["label"] for entry in result["collected"] if entry["id"] not in {"spareControl", "overflowControl"}},
            {entry["id"]: entry["label"] for entry in result["hints"]},
        )


if __name__ == "__main__":
    unittest.main()
