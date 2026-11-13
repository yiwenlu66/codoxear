from frontend_module_loader import module_path
import json
import os
import subprocess
import textwrap
import unittest
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
SESSION_EDIT_JS = module_path("app_session_edit.js")


def run_node_json(js: str, tz: str = "UTC") -> dict:
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={"PATH": os.environ.get("PATH", ""), "TZ": tz},
    )
    return json.loads(proc.stdout)


def eval_session_edit_lifecycle(tz: str = "UTC") -> dict:
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
        function makeController({{ alias, snoozeUntil = 0, snoozedIndefinitely = false, selected = "session-1", propagation = null }}) {{
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
          const snoozeButtons = new Map([
            ["none", makeSnoozeButton()], ["evening", makeSnoozeButton()],
            ["tomorrow", makeSnoozeButton()], ["weekend", makeSnoozeButton()],
            ["nextWeek", makeSnoozeButton()], ["indefinitely", makeSnoozeButton()],
            ["custom", makeSnoozeButton()],
          ]);
          const session = {{ session_id: "session-1", alias, priority_offset: 0, snooze_until: snoozeUntil, snoozed_indefinitely: snoozedIndefinitely, dependency_session_id: "", cwd: "/repo" }};
          const calls = [];
          const controller = ctx.window.CodoxearSessionEdit.createSessionEditController({{
            documentTarget: {{ activeElement: makeNode() }}, ElementCtor: ElementStub, el,
            editCloseBtn, editStatus, editNameInput, editPriorityRange, editPriorityValue,
            editPriorityResetBtn, editSnoozeModeButtons: snoozeButtons, editSnoozeCustomDate,
            editSnoozeCustomTime, editSnoozeCustomRow, editDependencyBtn, editDependencyMenu,
            editSaveBtn, editCancelBtn, editViewer,
            getSessionInfo: (sid) => sid === session.session_id ? session : null,
            getSessions: () => [session], sessionState: {{ get: () => selected }},
            sessionDisplayName: (entry) => entry.alias || "Conversation title", baseName: (value) => value.split("/").pop(),
            formatPriorityOffset: (value) => `+${{Number(value).toFixed(2)}}`, setPickerButtonContent: () => {{}},
            api: async (path, request) => {{
              calls.push({{ path, body: request.body }});
              session.alias = request.body.name;
              return propagation ? {{ name_propagation: propagation }} : {{}};
            }},
            refreshSessions: async () => [session], setToast: (value) => calls.push({{ toast: value }}),
            prepareModalOpen: () => {{}}, afterModalVisibilityChanged: () => {{}}, positionDialogMenu: () => {{}},
            addAppEvent: () => {{}}, now: () => Date.UTC(2026, 0, 1, 12, 0, 0), HTMLElementCtor: ElementStub,
          }});
          return {{
            controller, editNameInput, editSaveBtn, editCancelBtn, editViewer, calls,
            snoozeButtons, editSnoozeCustomRow, editSnoozeCustomDate, editSnoozeCustomTime,
            alias: () => session.alias,
          }};
        }}
        function active(button) {{
          const lastToggle = button.classList.toggles.filter(([name]) => name === "active").pop();
          return Boolean(lastToggle && lastToggle[1]);
        }}

        const ctx = {{ window: {{}}, ElementStub }};
        vm.createContext(ctx);
        vm.runInContext(source, ctx);

        const save = makeController({{ alias: "Original" }});
        const projectMiss = makeController({{
          alias: "Original",
          propagation: {{ ok: false, backend: "pi", reason: "unsupported" }},
        }});
        const projectHit = makeController({{
          alias: "Original",
          propagation: {{ ok: true, backend: "pi", reason: "applied", name: "Renamed" }},
        }});
        for (const attempt of [projectMiss, projectHit]) {{
          attempt.controller.openEditSession("session-1");
          attempt.editNameInput.value = "Renamed";
          attempt.editSaveBtn.onclick();
        }}
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
        custom.editSnoozeCustomDate.value = "2026-01-03";
        custom.editSnoozeCustomTime.value = "11:15";
        custom.editSaveBtn.onclick();

        const indefinite = makeController({{
          alias: "Original", snoozedIndefinitely: true,
          snoozeUntil: Date.UTC(2026, 0, 2, 9, 0, 0) / 1000,
        }});
        indefinite.controller.openEditSession("session-1");
        const indefiniteOpening = {{
          active: active(indefinite.snoozeButtons.get("indefinitely")),
          customActive: active(indefinite.snoozeButtons.get("custom")),
          tomorrowActive: active(indefinite.snoozeButtons.get("tomorrow")),
          customRow: indefinite.editSnoozeCustomRow.style.display,
          date: indefinite.editSnoozeCustomDate.value,
          time: indefinite.editSnoozeCustomTime.value,
        }};
        indefinite.editSaveBtn.onclick();

        const selectIndefinite = makeController({{ alias: "Original" }});
        selectIndefinite.controller.openEditSession("session-1");
        selectIndefinite.snoozeButtons.get("indefinitely").onclick();
        selectIndefinite.editSaveBtn.onclick();

        const clear = makeController({{ alias: "Original", snoozedIndefinitely: true }});
        clear.controller.openEditSession("session-1");
        clear.snoozeButtons.get("none").onclick();
        clear.editSaveBtn.onclick();

        const switchCustom = makeController({{ alias: "Original", snoozedIndefinitely: true }});
        switchCustom.controller.openEditSession("session-1");
        switchCustom.snoozeButtons.get("custom").onclick();
        const switched = {{
          active: active(switchCustom.snoozeButtons.get("custom")),
          indefiniteActive: active(switchCustom.snoozeButtons.get("indefinitely")),
          row: switchCustom.editSnoozeCustomRow.style.display,
        }};
        switchCustom.editSaveBtn.onclick();

        Promise.resolve().then(() => Promise.resolve()).then(() => Promise.resolve()).then(() => {{
          process.stdout.write(JSON.stringify({{
            save: {{ alias: save.alias(), closed: !save.editViewer.open, calls: save.calls }},
            projectMiss: {{ calls: projectMiss.calls }},
            projectHit: {{ calls: projectHit.calls }},
            evening: {{ calls: evening.calls }},
            customSave: {{ calls: custom.calls }},
            indefinite: {{ opening: indefiniteOpening, calls: indefinite.calls }},
            selectIndefinite: {{ calls: selectIndefinite.calls }},
            clear: {{ calls: clear.calls }},
            switchCustom: {{ ...switched, calls: switchCustom.calls }},
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
          }}));
        }});
        """
    )
    return run_node_json(js, tz)


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
                    "snoozed_indefinitely": False,
                    "dependency_session_id": None,
                },
            },
        )
        self.assertIn({"toast": "conversation updated"}, save["calls"])

    def test_a_missed_backend_title_is_reported_without_failing_the_rename(self) -> None:
        calls = self.result["projectMiss"]["calls"]
        self.assertEqual(calls[0]["path"], "/api/sessions/session-1/edit")
        self.assertIn(
            {"toast": "renamed — Pi needs /reload before it can adopt the title"},
            calls,
        )
        self.assertNotIn({"toast": "conversation updated"}, calls)

    def test_an_applied_backend_title_keeps_the_plain_confirmation(self) -> None:
        calls = self.result["projectHit"]["calls"]
        self.assertIn({"toast": "conversation updated"}, calls)

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
        self.assertIs(self.result["evening"]["calls"][0]["body"]["snoozed_indefinitely"], False)

    def test_opens_indefinite_snooze_without_selecting_custom_or_a_preset(self) -> None:
        self.assertEqual(
            self.result["indefinite"]["opening"],
            {
                "active": True,
                "customActive": False,
                "tomorrowActive": False,
                "customRow": "none",
                "date": "2026-01-02",
                "time": "09:00",
            },
        )

    def test_saves_existing_and_new_indefinite_snoozes_without_a_timestamp(self) -> None:
        for case in ("indefinite", "selectIndefinite"):
            with self.subTest(case=case):
                request = self.result[case]["calls"][0]
                self.assertEqual(request["path"], "/api/sessions/session-1/edit")
                self.assertIs(request["body"]["snoozed_indefinitely"], True)
                self.assertIsNone(request["body"]["snooze_until"])

    def test_custom_date_saves_local_epoch_and_clears_indefinite_flag(self) -> None:
        for tz in ("UTC", "America/New_York"):
            with self.subTest(tz=tz):
                result = self.result if tz == "UTC" else eval_session_edit_lifecycle(tz)
                body = result["customSave"]["calls"][0]["body"]
                expected = int(datetime(2026, 1, 3, 11, 15, tzinfo=ZoneInfo(tz)).timestamp())
                self.assertEqual(body["snooze_until"], expected)
                self.assertIs(body["snoozed_indefinitely"], False)

    def test_no_snooze_clears_timestamp_and_indefinite_flag(self) -> None:
        body = self.result["clear"]["calls"][0]["body"]
        self.assertIsNone(body["snooze_until"])
        self.assertIs(body["snoozed_indefinitely"], False)

    def test_switching_from_indefinite_to_custom_reveals_and_saves_default_time(self) -> None:
        switched = self.result["switchCustom"]
        self.assertTrue(switched["active"])
        self.assertFalse(switched["indefiniteActive"])
        self.assertEqual(switched["row"], "grid")
        body = switched["calls"][0]["body"]
        self.assertEqual(body["snooze_until"], 1767344400)
        self.assertIs(body["snoozed_indefinitely"], False)

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


if __name__ == "__main__":
    unittest.main()
