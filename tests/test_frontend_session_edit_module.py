import json
import os
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SESSION_EDIT_JS = ROOT / "codoxear" / "static" / "app_session_edit.js"


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


def eval_session_edit_lifecycle() -> dict:
    source = SESSION_EDIT_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const source = {json.dumps(source)};

        class ElementStub {{}}
        function makeNode() {{
          const listeners = {{}};
          return Object.assign(new ElementStub(), {{
            style: {{}}, dataset: {{}}, children: [], innerHTML: "", value: "", placeholder: "",
            textContent: "", disabled: false, open: false, className: "",
            classList: {{ toggle() {{}} }},
            setAttribute() {{}}, removeAttribute() {{}},
            appendChild(child) {{ this.children.push(child); return child; }},
            addEventListener(name, handler) {{ listeners[name] = handler; }},
            dispatch(name, event = {{}}) {{ return listeners[name](event); }},
            showModal() {{ this.open = true; }}, close() {{ this.open = false; }},
            closest() {{ return null; }},
          }});
        }}
        function el(_tag, attrs = {{}}, children = []) {{
          const node = makeNode();
          Object.assign(node, attrs);
          if (attrs.class) node.className = attrs.class;
          if (attrs.text !== undefined) node.textContent = attrs.text;
          for (const child of children) node.appendChild(child);
          return node;
        }}
        function makeController({{ alias, selected = "session-1" }}) {{
          const editCloseBtn = makeNode();
          const editStatus = makeNode();
          const editNameInput = makeNode();
          const editPriorityRange = makeNode();
          const editPriorityValue = makeNode();
          const editPriorityResetBtn = makeNode();
          const editSnoozeCustomDate = makeNode();
          const editSnoozeCustomTime = makeNode();
          const editSnoozeCustomRow = makeNode();
          const editDependencyBtn = makeNode();
          const editDependencyMenu = makeNode();
          const editSaveBtn = makeNode();
          const editCancelBtn = makeNode();
          const editViewer = makeNode();
          const snoozeButtons = new Map([["none", makeNode()], ["4h", makeNode()], ["tomorrow", makeNode()], ["custom", makeNode()]]);
          const session = {{ session_id: "session-1", alias, priority_offset: 0, snooze_until: 0, dependency_session_id: "", cwd: "/repo" }};
          const calls = [];
          const unsavedPromptCalls = [];
          let renderedTitle = alias;
          const controller = ctx.window.CodoxearSessionEdit.createSessionEditController({{
            documentTarget: {{ activeElement: makeNode() }}, ElementCtor: ElementStub, el, iconSvg: () => "",
            editCloseBtn, editStatus, editNameInput, editPriorityRange, editPriorityValue,
            editPriorityResetBtn, editSnoozeModeButtons: snoozeButtons, editSnoozeCustomDate,
            editSnoozeCustomTime, editSnoozeCustomRow, editDependencyBtn, editDependencyMenu,
            editSaveBtn, editCancelBtn, editViewer,
            fileUnsavedDialogRuntime: {{
              promptChoice: (activeElement, ctor) => {{ unsavedPromptCalls.push({{ activeMatches: activeElement !== null, ctorMatches: ctor === ElementStub }}); return Promise.resolve("discard"); }},
              hide: () => "cancel",
            }},
            fileViewerController: {{
              maybeHandleUnsavedFileChanges: () => false,
              handleFileUnsavedSaveChoice: () => undefined,
              handleFileUnsavedDiscardChoice: () => undefined,
              handleFileUnsavedCancelChoice: () => undefined,
            }},
            getSessionInfo: (sid) => sid === session.session_id ? session : null,
            getSessions: () => [session], selectedSessionId: () => selected,
            sessionDisplayName: (entry) => entry.alias || "Conversation title", baseName: (value) => value.split("/").pop(),
            formatPriorityOffset: (value) => `+${{Number(value).toFixed(2)}}`, setPickerButtonContent: () => {{}},
            api: async (path, request) => {{ calls.push({{ path, body: request.body }}); session.alias = request.body.name; return {{}}; }},
            refreshSessions: async () => [session], setToast: (value) => calls.push({{ toast: value }}),
            setTitle: (_sid, entry) => {{ renderedTitle = entry.alias; }},
            prepareModalOpen: () => {{}}, afterModalVisibilityChanged: () => {{}}, positionDialogMenu: () => {{}},
            addAppEvent: () => {{}}, now: () => Date.UTC(2026, 0, 1, 12, 0, 0), HTMLElementCtor: ElementStub,
          }});
          return {{ controller, editNameInput, editSaveBtn, editCancelBtn, editViewer, calls, unsavedPromptCalls, title: () => renderedTitle }};
        }}

        const ctx = {{ window: {{}}, ElementStub }};
        vm.createContext(ctx);
        vm.runInContext(source, ctx);

        const save = makeController({{ alias: "Original" }});
        save.controller.openEditSession("session-1");
        save.editNameInput.value = "Renamed";
        save.editSaveBtn.onclick();
        Promise.resolve().then(() => Promise.resolve()).then(() => Promise.resolve()).then(() => {{
          const cancel = makeController({{ alias: "Original" }});
          cancel.controller.openEditSession("session-1");
          cancel.editNameInput.value = "Discarded";
          cancel.editCancelBtn.onclick();
          cancel.controller.promptFileUnsavedChoice().then((unsavedChoice) => {{
            process.stdout.write(JSON.stringify({{
              save: {{ title: save.title(), closed: !save.editViewer.open, calls: save.calls }},
              cancel: {{ title: cancel.title(), closed: !cancel.editViewer.open, calls: cancel.calls }},
              unsaved: {{ choice: unsavedChoice, calls: cancel.unsavedPromptCalls }},
            }}));
          }});
        }});
        """
    )
    return run_node_json(js)


class TestFrontendSessionEditModule(unittest.TestCase):
    def test_save_commits_edited_title_and_updates_title_projection(self) -> None:
        result = eval_session_edit_lifecycle()
        self.assertEqual(result["save"]["title"], "Renamed")
        self.assertTrue(result["save"]["closed"])
        self.assertEqual(
            result["save"]["calls"][0],
            {
                "path": "/api/sessions/session-1/edit",
                "body": {
                    "name": "Renamed",
                    "priority_offset": 0,
                    "snooze_until": None,
                    "dependency_session_id": None,
                },
            },
        )
        self.assertIn({"toast": "conversation updated"}, result["save"]["calls"])

    def test_cancel_discards_draft_without_mutating_title(self) -> None:
        result = eval_session_edit_lifecycle()
        self.assertEqual(result["cancel"]["title"], "Original")
        self.assertTrue(result["cancel"]["closed"])
        self.assertEqual(result["cancel"]["calls"], [])
    def test_unsaved_choice_returns_dialog_action(self) -> None:
        result = eval_session_edit_lifecycle()
        self.assertEqual(result["unsaved"]["choice"], "discard")
        self.assertEqual(result["unsaved"]["calls"], [{"activeMatches": True, "ctorMatches": True}])


if __name__ == "__main__":
    unittest.main()
