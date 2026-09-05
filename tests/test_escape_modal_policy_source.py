"""Global Escape policy: Esc never closes or cancels a modal dialog.

Behavioral VM suites that execute the real listener bodies. For every surface
that Escape used to dismiss, a test asserts Esc leaves it open. The kept
transient behaviors (the #appConfirm Tab focus trap) are pinned alongside.
"""

from frontend_module_loader import module_path
import json
import os
import subprocess
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EDITOR_OPS = module_path("app_file_editor_ops.js")
NEW_SESSION = module_path("app_new_session.js")
LAUNCH = module_path("app_launch.js")
DISPLAY = module_path("app_display.js")


def run_node(js: str) -> dict:
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={"PATH": os.environ.get("PATH", ""), "TZ": "UTC"},
    )
    return json.loads(proc.stdout)


class TestEscapeNeverDismissesModals(unittest.TestCase):
    def test_editor_ops_keydown_listeners_leave_every_dialog_open_on_escape(self) -> None:
        # Executes createFileEditorOpsController -> bindInteractions with fakes
        # for the interactions the removed Escape dismissal branch used to own,
        # fires Escape at each registered keydown listener, and asserts no
        # dialog-close path ran. The Tab trap on #appConfirm still traps.
        source = EDITOR_OPS.read_text(encoding="utf-8")
        program = textwrap.dedent(
            f"""
            const vm = require("vm");
            const source = {json.dumps(source)};
            const ctx = {{ window: {{}} }};
            vm.createContext(ctx);
            vm.runInContext(source, ctx, {{ filename: "app_file_editor_ops.js" }});
            const listeners = [];
            const addAppEvent = (target, type, handler, capture) => {{
              if (type === "keydown") listeners.push({{ handler, capture: Boolean(capture), target }});
              return handler;
            }};
            const document = {{ activeElement: null }};
            const appConfirm = {{ style: {{ display: "flex" }} }};
            const confirmButtons = [];
            const appConfirmFocusableControls = () => confirmButtons;
            const controller = ctx.window.CodoxearFileEditorOps.createFileEditorOpsController({{
              addAppEvent,
              wiring: {{
                createMonacoLoaderOptions: (options) => options,
                createFileEditorRendererOptions: (options) => options,
              }},
              codoxearFileEditor: {{
                createFileEditorRuntime: () => ({{ runtime: true }}),
                createMonacoLoader: () => ({{ loader: true }}),
                createFileEditorRenderer: () => ({{ renderer: true }}),
              }},
            }});
            const calls = [];
            controller.bindInteractions({{
              addAppEvent,
              document,
              appConfirm,
              appConfirmFocusableControls,
              handleFileEditorSaveShortcut: (event) => {{
                if (event.key === "s" && (event.ctrlKey || event.metaKey)) calls.push(["save-shortcut", event.key]);
              }},
              handleFileEditorDeleteKeydown: (event) => {{
                if (["Backspace", "Delete"].includes(event.key)) calls.push(["delete-keydown", event.key]);
              }},
              suppressFileEditorNativeDelete: (event) => {{
                if (event.type !== "keydown") calls.push(["suppress", event.type]);
              }},
              fileTouchController: {{ handleFileTouchSelectionKeydown: (event) => calls.push(["touch", event.key]) }},
            }});
            const escapeEvent = {{
              key: "Escape", target: document, defaultPrevented: false,
              preventDefault() {{ this.defaultPrevented = true; }},
              stopPropagation() {{}},
            }};
            for (const entry of listeners) entry.handler(escapeEvent);
            // Tab trap: with #appConfirm open, Tab cycles focusable controls.
            const first = {{ focused: 0, focus() {{ this.focused += 1; }} }};
            const second = {{ focused: 0, focus() {{ this.focused += 1; }} }};
            confirmButtons.push(first, second);
            document.activeElement = first;
            const tabEvent = {{
              key: "Tab", shiftKey: false, defaultPrevented: false,
              preventDefault() {{ this.defaultPrevented = true; }},
              stopPropagation() {{}},
            }};
            for (const entry of listeners) entry.handler(tabEvent);
            process.stdout.write(JSON.stringify({{
              listenerCount: listeners.length,
              escapeDialogClosures: calls.filter((call) => call[0] !== "touch"),
              escapeConsumed: escapeEvent.defaultPrevented,
              tabTrap: {{ moved: second.focused === 1, prevented: tabEvent.defaultPrevented }},
            }}));
            """
        )
        result = run_node(program)
        self.assertGreaterEqual(result["listenerCount"], 4)
        # No save-shortcut/delete/suppress path is an Escape dismissal path;
        # every handler must have ignored Escape entirely.
        self.assertEqual(result["escapeDialogClosures"], [])
        self.assertFalse(result["escapeConsumed"])
        self.assertTrue(result["tabTrap"]["moved"])
        self.assertTrue(result["tabTrap"]["prevented"])

    def test_new_session_dialog_stays_open_on_escape(self) -> None:
        sources = [path.read_text(encoding="utf-8") for path in (DISPLAY, LAUNCH, NEW_SESSION)]
        program = textwrap.dedent(
            f"""
            const vm = require("vm");
            const storage = {{
              data: new Map(),
              getItem(key) {{ return this.data.get(key) || null; }},
              setItem(key, value) {{ this.data.set(key, String(value)); }},
              removeItem(key) {{ this.data.delete(key); }},
            }};
            const ctx = {{
              window: {{ CodoxearUrls: {{ resolveAppUrl: (path) => String(path) }}, CodoxearStorage: storage }},
              console,
              HTMLElement: class HTMLElement {{}},
              setTimeout: (fn) => fn(),
              clearTimeout: () => ({{}}),
            }};
            vm.createContext(ctx);
            for (const source of {json.dumps(sources)}) vm.runInContext(source, ctx);
            const doc = {{
              activeElement: null,
              createElement: () => ({{ style: {{}}, appendChild() {{}}, remove() {{}}, setAttribute() {{}} }}),
              querySelectorAll: () => [],
              getElementById: () => null,
              body: {{ appendChild() {{}}, classList: {{ toggle() {{}}, add() {{}}, remove() {{}} }} }},
            }};
            const win = {{ requestAnimationFrame: (fn) => fn(), addEventListener() {{}}, matchMedia: () => ({{ matches: false }}) }};
            const documentKeydown = [];
            const addEvent = (target, type, handler, capture) => {{
              if (type === "keydown") documentKeydown.push(handler);
              return handler;
            }};
            let dialogNode = null;
            const el = (tag, attrs = {{}}, _children = []) => {{
              const node = {{
                tag, children: [], style: {{ display: "block" }}, className: "", attrs: {{}},
                classList: {{ toggle() {{}}, add() {{}}, remove() {{}}, contains: () => false }},
                textContent: "", disabled: false, hidden: false, checked: false, value: "", focused: 0,
                setAttribute(name, value) {{ this.attrs[name] = String(value); if (name === "text") this.textContent = String(value); }},
                getAttribute(name) {{ return this.attrs[name] || ""; }},
                removeAttribute() {{}},
                appendChild(child) {{ this.children.push(child); return child; }},
                removeEventListener() {{}},
                addEventListener() {{}},
                contains(node) {{ return node === this; }},
                querySelector() {{ return el("div"); }},
                querySelectorAll() {{ return []; }},
                focus() {{ this.focused += 1; }},
                getClientRects() {{ return {{ length: 1 }}; }},
              }};
              if (attrs && attrs.id === "newSessionViewer") dialogNode = node;
              for (const [name, value] of Object.entries(attrs || {{}})) {{
                if (name === "class") node.className = value;
                else node.setAttribute(name, value);
              }}
              return node;
            }};
            const iconSvg = () => "";
            const sessionCatalog = {{
              get(key) {{
                if (key === "newSessionDefaults") return {{ models: [], model_providers: [], provider_choices: [], reasoning_efforts: [], reasoning_efforts_by_model: {{}} }};
                if (key === "latestSessions") return [];
                if (key === "sessionIndex") return new Map();
                if (key === "recentCwds") return [];
                return key === "tmuxAvailable" ? false : null;
              }},
              subscribe: () => () => ({{}}),
            }};
            const sessionState = {{ get: () => null, subscribe: () => () => ({{}}) }};
            const controller = ctx.window.CodoxearNewSession.createNewSessionDialogController({{
              root: el("div"), el, iconSvg, document: doc, window: win, addEvent,
              sessionCatalog, sessionState,
              isMobile: () => false,
              prepareModalOpen: () => ({{}}),
              afterModalVisibilityChanged: () => ({{}}),
              isModalTargetOpen: (node) => node === dialogNode && node.style.display !== "none",
              applyDialogMenus: () => ({{}}),
              positionDialogMenu: () => ({{}}),
              setPickerButtonContent: () => ({{}}),
              fetchResumeCandidates: async () => ({{ sessions: [] }}),
              spawnSession: async () => 0,
              storageGetItem: storage.getItem,
              storageSetItem: storage.setItem,
              storageRemoveItem: storage.removeItem,
            }});
            controller.open();
            const wasOpen = controller.isOpen();
            let escapeConsumed = false;
            for (const handler of documentKeydown) {{
              const event = {{ key: "Escape", defaultPrevented: false, preventDefault() {{ this.defaultPrevented = true; }}, stopPropagation() {{}} }};
              handler(event);
              if (event.defaultPrevented) escapeConsumed = true;
            }}
            process.stdout.write(JSON.stringify({{ wasOpen, stillOpen: controller.isOpen(), handlers: documentKeydown.length, escapeConsumed }}));
            """
        )
        result = run_node(program)
        self.assertTrue(result["wasOpen"])
        self.assertTrue(result["stillOpen"])
        self.assertFalse(result["escapeConsumed"])


if __name__ == "__main__":
    unittest.main()
