import json
import os
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_HINT_MODE_JS = ROOT / "codoxear" / "static" / "app_hint_mode.js"
APP_JS = ROOT / "codoxear" / "static" / "app.js"
INDEX_HTML = ROOT / "codoxear" / "static" / "index.html"


def run_node_json(js: str) -> dict:
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={"PATH": os.environ.get("PATH", ""), "TZ": "UTC"},
    )
    return json.loads(proc.stdout)


def eval_hint_mode() -> dict:
    source = APP_HINT_MODE_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const source = {json.dumps(source)};
        const events = {{}};
        const clicks = [];
        let modalOpen = false;
        let mobile = false;

        function makeNode(name, {{ disabled = false, hidden = false, textEntry = false }} = {{}}) {{
          return {{
            name,
            disabled,
            offsetParent: hidden ? null : {{}},
            style: {{ display: hidden ? "none" : "block", visibility: "visible" }},
            isTextEntry: textEntry,
            parentNode: null,
            click() {{ clicks.push(name); }},
            contains(target) {{ return target === this; }},
            getBoundingClientRect() {{ return {{ left: 12, top: 24 }}; }},
          }};
        }}

        const body = {{
          children: [],
          appendChild(node) {{ node.parentNode = this; this.children.push(node); return node; }},
          removeChild(node) {{ this.children = this.children.filter((entry) => entry !== node); node.parentNode = null; }},
        }};
        const hiddenSession = makeNode("hidden-session", {{ hidden: true }});
        const sessionOne = makeNode("session-1");
        const sessionTwo = makeNode("session-2");
        const sidebar = makeNode("sidebar");
        const search = makeNode("search");
        const disabledBrowse = makeNode("browse", {{ disabled: true }});
        const hiddenDetails = makeNode("details", {{ hidden: true }});
        const reserved = makeNode("reserved");
        const textInput = makeNode("input", {{ textEntry: true }});
        const modal = makeNode("modal");
        const documentTarget = {{
          body,
          activeElement: null,
          defaultView: {{ getComputedStyle: (node) => node.style }},
          querySelectorAll(selector) {{
            if (selector === "#sessions .session[data-session-id]") return [hiddenSession, sessionOne, sessionTwo];
            if (selector === ".chat a[data-file-path], .chat a[data-file-picker-query]") return [];
            throw new Error(`unexpected selector: ${{selector}}`);
          }},
          createElement() {{
            const node = {{
              style: {{}},
              parentNode: null,
              className: "",
              textContent: "",
              setAttribute() {{}},
              appendChild(child) {{ child.parentNode = this; }},
              remove() {{ if (this.parentNode) this.parentNode.removeChild(this); }},
            }};
            return node;
          }},
        }};
        const ctx = {{ window: {{}}, document: documentTarget }};
        vm.createContext(ctx);
        vm.runInContext(source, ctx);
        const controller = ctx.window.CodoxearHintMode.createHintModeController({{
          documentTarget,
          isTextEntryElement: (node) => Boolean(node && node.isTextEntry),
          isMobile: () => mobile,
          modalIsolationTargets: [modal],
          isModalTargetOpen: () => modalOpen,
          addAppEvent: (_target, type, handler) => {{ events[type] = handler; }},
          shellHints: [
            {{ label: "s", element: sidebar }},
            {{ label: "/", element: search }},
            {{ label: "b", element: disabledBrowse }},
            {{ label: "d", element: hiddenDetails }},
            {{ label: "f", element: reserved }},
          ],
        }});
        function press(key, target = null) {{
          const event = {{
            key,
            target,
            defaultPrevented: false,
            altKey: false,
            ctrlKey: false,
            metaKey: false,
            shiftKey: false,
            preventDefault() {{ this.defaultPrevented = true; }},
          }};
          events.keydown(event);
          return event;
        }}

        documentTarget.activeElement = textInput;
        const textEntry = press("f", textInput);
        documentTarget.activeElement = null;
        modalOpen = true;
        const modalBlocked = press("f");
        modalOpen = false;
        mobile = true;
        const mobileBlocked = press("f");
        mobile = false;

        const enterSession = press("f");
        const sessionHint = press("2");
        const enterShell = press("f");
        const shellHint = press("s");
        const enterSearch = press("f");
        const searchHint = press("/");
        const enterEscape = press("f");
        const escape = press("Escape");
        const enterReserved = press("f");
        const reservedHint = press("f");
        process.stdout.write(JSON.stringify({{
          frozen: Object.isFrozen(ctx.window.CodoxearHintMode),
          textEntry: {{ active: controller.isActive(), prevented: textEntry.defaultPrevented }},
          modalBlocked: {{ active: controller.isActive(), prevented: modalBlocked.defaultPrevented }},
          mobileBlocked: {{ active: controller.isActive(), prevented: mobileBlocked.defaultPrevented }},
          enterSessionPrevented: enterSession.defaultPrevented,
          sessionHintPrevented: sessionHint.defaultPrevented,
          enterShellPrevented: enterShell.defaultPrevented,
          shellHintPrevented: shellHint.defaultPrevented,
          enterSearchPrevented: enterSearch.defaultPrevented,
          searchHintPrevented: searchHint.defaultPrevented,
          enterEscapePrevented: enterEscape.defaultPrevented,
          escapePrevented: escape.defaultPrevented,
          enterReservedPrevented: enterReserved.defaultPrevented,
          reservedHintPrevented: reservedHint.defaultPrevented,
          active: controller.isActive(),
          badgeContainers: body.children.length,
          clicks,
          labels: Array.from(controller.collectTargets().keys()),
        }}));
        """
    )
    return run_node_json(js)


class TestFrontendHintModeModuleSource(unittest.TestCase):
    def test_hint_mode_activation_filters_targets_and_activates_hints(self) -> None:
        result = eval_hint_mode()
        self.assertTrue(result["frozen"])
        self.assertFalse(result["textEntry"]["prevented"])
        self.assertFalse(result["modalBlocked"]["prevented"])
        self.assertFalse(result["mobileBlocked"]["prevented"])
        self.assertTrue(result["enterSessionPrevented"])
        self.assertTrue(result["sessionHintPrevented"])
        self.assertTrue(result["enterShellPrevented"])
        self.assertTrue(result["shellHintPrevented"])
        self.assertTrue(result["enterSearchPrevented"])
        self.assertTrue(result["searchHintPrevented"])
        self.assertEqual(result["clicks"], ["session-2", "sidebar", "search"])
        self.assertIn("/", result["labels"])
        self.assertNotIn("b", result["labels"])
        self.assertNotIn("d", result["labels"])
        self.assertNotIn("f", result["labels"])

    def test_escape_cleans_up_and_f_never_activates_a_target(self) -> None:
        result = eval_hint_mode()
        self.assertTrue(result["enterEscapePrevented"])
        self.assertTrue(result["escapePrevented"])
        self.assertTrue(result["enterReservedPrevented"])
        self.assertFalse(result["reservedHintPrevented"])
        self.assertFalse(result["active"])
        self.assertEqual(result["badgeContainers"], 0)
        self.assertNotIn("reserved", result["clicks"])

    def test_app_wires_all_locked_shell_hints_before_app_js(self) -> None:
        app_source = APP_JS.read_text(encoding="utf-8")
        expected = {
            "s": "toggleSidebarBtn", "t": "titleLabel", "b": "fileBtn", "d": "diagBtn", "u": "unattendedBtn",
            "z": "interruptBtn", "/": "chatSearchBtn", "p": "prevUserBtn", "n": "nextUserBtn",
            "o": "olderBtn", "g": "jumpBtn", "a": "attachBtn", "q": "queueBtn",
            "e": "sendBtn", "i": "textarea", "c": "$(\"#newBtn\")",
        }
        for label, element in expected.items():
            self.assertIn(f'{{ label: "{label}", element: {element} }}', app_source)
        self.assertNotIn('{ label: "x",', app_source)
        self.assertNotIn('{ label: "r", element: chatSearchBtn }', app_source)
        self.assertNotIn("handleSessionNavigationKeydown", app_source)
        self.assertNotIn("Alt+1", app_source)
        index_source = INDEX_HTML.read_text(encoding="utf-8")
        self.assertLess(index_source.index('src="app_hint_mode.js'), index_source.index('src="app.js?v='))

    def test_app_modal_keys_activate_first_visible_matching_button(self) -> None:
        app_source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('function activateModalButtonForKey(e)', app_source)
        self.assertIn('for (const modal of modalIsolationTargets)', app_source)
        self.assertIn('function modalButtonLabel(button)', app_source)
        self.assertIn('function modalButtonHint(label, labels)', app_source)
        self.assertIn('const buttons = [...modal.querySelectorAll("button")].filter(', app_source)
        self.assertIn('const labels = buttons.map(modalButtonLabel);', app_source)
        self.assertIn('modalButtonHint(labels[index], labels) === key', app_source)
        self.assertIn('labels.filter((other) => other[index] === candidate).length === 1', app_source)
        self.assertIn('button.click();', app_source)
        self.assertIn('addAppEvent(document, "keydown", activateModalButtonForKey);', app_source)


if __name__ == "__main__":
    unittest.main()
