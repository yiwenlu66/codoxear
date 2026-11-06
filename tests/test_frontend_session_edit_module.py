from frontend_module_loader import module_path
import json
import os
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SESSION_EDIT_JS = module_path("app_session_edit.js")


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
          const classList = {{
            toggles: [],
            toggle(name, active) {{ this.toggles.push([name, Boolean(active)]); }},
          }};
          return Object.assign(new ElementStub(), {{
            style: {{}}, dataset: {{}}, children: [], innerHTML: "", value: "", placeholder: "",
            textContent: "", disabled: false, open: false, className: "", classList,
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
        function makeSnoozeButton() {{
          const button = makeNode();
          button.appendChild(makeNode());
          button.appendChild(makeNode());
          return button;
        }}
        function makeController({{ alias, snoozeUntil = 0, selected = "session-1", threadId = "uuid-1234" }}) {{
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
          const editUuidValue = makeNode();
          const editUuidCopyBtn = makeNode();
          const snoozeButtons = new Map([
            ["none", makeSnoozeButton()], ["evening", makeSnoozeButton()],
            ["tomorrow", makeSnoozeButton()], ["weekend", makeSnoozeButton()],
            ["nextWeek", makeSnoozeButton()], ["custom", makeSnoozeButton()],
          ]);
          const session = {{ session_id: "session-1", thread_id: threadId, alias, priority_offset: 0, snooze_until: snoozeUntil, dependency_session_id: "", cwd: "/repo" }};
          const calls = [];
          const controller = ctx.window.CodoxearSessionEdit.createSessionEditController({{
            documentTarget: {{ activeElement: makeNode() }}, ElementCtor: ElementStub, el,
            editCloseBtn, editStatus, editNameInput, editPriorityRange, editPriorityValue,
            editPriorityResetBtn, editSnoozeModeButtons: snoozeButtons, editSnoozeCustomDate,
            editSnoozeCustomTime, editSnoozeCustomRow, editDependencyBtn, editDependencyMenu,
            editSaveBtn, editCancelBtn, editViewer,
            editUuidValue, editUuidCopyBtn, copyText: (value) => calls.push({{ copy: value }}),
            getSessionInfo: (sid) => sid === session.session_id ? session : null,
            getSessions: () => [session], sessionState: {{ get: () => selected }},
            sessionDisplayName: (entry) => entry.alias || "Conversation title", baseName: (value) => value.split("/").pop(),
            formatPriorityOffset: (value) => `+${{Number(value).toFixed(2)}}`, setPickerButtonContent: () => {{}},
            api: async (path, request) => {{ calls.push({{ path, body: request.body }}); session.alias = request.body.name; return {{}}; }},
            refreshSessions: async () => [session], setToast: (value) => calls.push({{ toast: value }}),
            prepareModalOpen: () => {{}}, afterModalVisibilityChanged: () => {{}}, positionDialogMenu: () => {{}},
            addAppEvent: () => {{}}, now: () => Date.UTC(2026, 0, 1, 12, 0, 0), HTMLElementCtor: ElementStub,
          }});
          return {{
            controller, editNameInput, editSaveBtn, editCancelBtn, editViewer, calls,
            snoozeButtons, editSnoozeCustomRow, alias: () => session.alias,
            editUuidValue, editUuidCopyBtn,
          }};
        }}
        function active(button) {{
          return button.classList.toggles.some(([name, value]) => name === "active" && value);
        }}

        const ctx = {{ window: {{}}, ElementStub }};
        vm.createContext(ctx);
        vm.runInContext(source, ctx);

        const save = makeController({{ alias: "Original" }});
        save.controller.openEditSession("session-1");
        save.editNameInput.value = "Renamed";
        save.editSaveBtn.onclick();

        const evening = makeController({{ alias: "Original" }});
        evening.controller.openEditSession("session-1");
        evening.snoozeButtons.get("evening").onclick();
        evening.editSaveBtn.onclick();

        const cancel = makeController({{ alias: "Original" }});
        cancel.controller.openEditSession("session-1");
        cancel.editNameInput.value = "Discarded";
        cancel.editCancelBtn.onclick();

        const matching = makeController({{ alias: "Original", snoozeUntil: Date.UTC(2026, 0, 2, 9, 0, 0) / 1000 }});
        matching.controller.openEditSession("session-1");
        const custom = makeController({{ alias: "Original", snoozeUntil: Date.UTC(2026, 0, 3, 11, 0, 0) / 1000 }});
        custom.controller.openEditSession("session-1");

        const uuid = makeController({{ alias: "Original" }});
        uuid.controller.openEditSession("session-1");
        uuid.editUuidCopyBtn.onclick({{ preventDefault() {{}} }});
        const noBind = makeController({{ alias: "Original", threadId: null }});
        noBind.controller.openEditSession("session-1");

        Promise.resolve().then(() => Promise.resolve()).then(() => Promise.resolve()).then(() => {{
          process.stdout.write(JSON.stringify({{
            save: {{ alias: save.alias(), closed: !save.editViewer.open, calls: save.calls }},
            evening: {{ calls: evening.calls }},
            cancel: {{ alias: cancel.alias(), closed: !cancel.editViewer.open, calls: cancel.calls }},
            recognition: {{
              tomorrowActive: active(matching.snoozeButtons.get("tomorrow")),
              matchingCustomRow: matching.editSnoozeCustomRow.style.display,
              customActive: active(custom.snoozeButtons.get("custom")),
              customRow: custom.editSnoozeCustomRow.style.display,
            }},
            chips: {{
              eveningLabel: save.snoozeButtons.get("evening").children[0].textContent,
              eveningTime: save.snoozeButtons.get("evening").children[1].textContent,
            }},
            uuid: {{ text: uuid.editUuidValue.textContent, copyDisabled: uuid.editUuidCopyBtn.disabled, calls: uuid.calls }},
            noBind: {{ text: noBind.editUuidValue.textContent, copyDisabled: noBind.editUuidCopyBtn.disabled }},
          }}));
        }});
        """
    )
    return run_node_json(js)


class TestFrontendSessionEditModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = eval_session_edit_lifecycle()

    def test_save_commits_edited_title_and_refreshes_catalog_data(self) -> None:
        save = self.result["save"]
        self.assertEqual(save["alias"], "Renamed")
        self.assertTrue(save["closed"])
        self.assertEqual(
            save["calls"][0],
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
        self.assertIn({"toast": "conversation updated"}, save["calls"])

    def test_cancel_discards_draft_without_mutating_title(self) -> None:
        cancel = self.result["cancel"]
        self.assertEqual(cancel["alias"], "Original")
        self.assertTrue(cancel["closed"])
        self.assertEqual(cancel["calls"], [])

    def test_evening_preset_saves_its_local_anchor_epoch(self) -> None:
        self.assertEqual(
            self.result["evening"]["calls"][0]["body"]["snooze_until"],
            1767290400,
        )

    def test_reopens_matching_presets_and_uses_custom_for_other_future_times(self) -> None:
        recognition = self.result["recognition"]
        self.assertTrue(recognition["tomorrowActive"])
        self.assertEqual(recognition["matchingCustomRow"], "none")
        self.assertTrue(recognition["customActive"])
        self.assertEqual(recognition["customRow"], "grid")

    def test_open_rewrites_life_anchor_chip_labels_and_times(self) -> None:
        self.assertEqual(
            self.result["chips"],
            {"eveningLabel": "This evening", "eveningTime": "· 6pm"},
        )

    def test_session_uuid_row_shows_thread_id_and_copies(self) -> None:
        uuid = self.result["uuid"]
        self.assertEqual(uuid["text"], "uuid-1234")
        self.assertFalse(uuid["copyDisabled"])
        self.assertIn({"copy": "uuid-1234"}, uuid["calls"])

    def test_session_uuid_row_marks_unbound_sessions(self) -> None:
        no_bind = self.result["noBind"]
        self.assertEqual(no_bind["text"], "not bound yet")
        self.assertTrue(no_bind["copyDisabled"])


if __name__ == "__main__":
    unittest.main()
