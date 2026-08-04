import json
import os
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"


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


def modal_keyboard_program() -> tuple[str, str]:
    source = APP_JS.read_text(encoding="utf-8")
    targets_start = source.index("        const modalIsolationTargets = [")
    targets_end = source.index("        ];", targets_start) + len("        ];")
    keyboard_start = source.index("        function modalButtonLabel(button) {")
    listener = "        addAppEvent(document, \"keydown\", activateModalButtonForKey);"
    keyboard_end = source.index(listener, keyboard_start) + len(listener)
    return source[targets_start:targets_end], source[keyboard_start:keyboard_end]


def eval_modal_keyboard() -> dict:
    target_source, keyboard_source = modal_keyboard_program()
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const targetsSource = {json.dumps(target_source)};
        const keyboardSource = {json.dumps(keyboard_source)};
        const clicks = [];
        const events = {{}};

        function button(name, label, {{ disabled = false, hidden = false, visible = true }} = {{}}) {{
          return {{
            name,
            textContent: label,
            disabled,
            hidden,
            getAttribute(name) {{ return name === "aria-label" ? "" : null; }},
            getClientRects() {{ return visible ? [{{}}] : []; }},
            click() {{ clicks.push(name); }},
          }};
        }}

        function modal(name, buttons) {{
          return {{
            name,
            style: {{ display: "none" }},
            querySelectorAll(selector) {{
              if (selector !== "button") throw new Error(`unexpected selector: ${{selector}}`);
              return buttons;
            }},
          }};
        }}

        const fileViewer = modal("fileViewer", [button("viewer-save", "Save"), button("viewer-delete", "Delete")]);
        const fileUnsavedDialog = modal("fileUnsavedDialog", [button("unsaved-discard", "Discard"), button("unsaved-cancel", "Cancel")]);
        const filePasteDialog = modal("filePasteDialog", [button("paste-replace", "Replace")]);
        const sendChoice = modal("sendChoice", [button("send-now", "Now")]);
        const appConfirm = modal("appConfirm", [button("confirm-apply", "Apply"), button("confirm-cancel", "Cancel")]);
        const queueViewer = modal("queueViewer", [button("queue-close", "Close")]);
        const helpViewer = modal("helpViewer", [button("help-close", "Close")]);
        const diagViewer = modal("diagViewer", [button("diag-copy", "Copy")]);
        const editViewer = modal("editViewer", [button("edit-save", "Save")]);
        const voiceSettingsViewer = modal("voiceSettingsViewer", [button("voice-save", "Save")]);
        const newSessionDialogController = {{ viewer: modal("newSessionViewer", [button("new-start", "Start")]) }};
        const document = {{}};

        function isModalTargetOpen(node) {{ return Boolean(node && node.style.display !== "none"); }}
        function isTextEntryElement(node) {{ return Boolean(node && node.isTextEntry); }}
        function addAppEvent(target, type, handler) {{
          if (target !== document) throw new Error("listener bound to wrong target");
          events[type] = handler;
        }}

        const ctx = {{
          fileViewer,
          fileUnsavedDialog,
          filePasteDialog,
          sendChoice,
          appConfirm,
          queueViewer,
          helpViewer,
          diagViewer,
          editViewer,
          voiceSettingsViewer,
          newSessionDialogController,
          document,
          isModalTargetOpen,
          isTextEntryElement,
          addAppEvent,
        }};
        vm.createContext(ctx);
        vm.runInContext(`${{targetsSource}}\nglobalThis.modalTargetNames = modalIsolationTargets.map((target) => target.name);\nglobalThis.modalIsolationTargetsForTest = modalIsolationTargets;\n${{keyboardSource}}`, ctx);

        function press(key, target = null) {{
          const event = {{
            key,
            target,
            defaultPrevented: false,
            propagationStopped: false,
            altKey: false,
            ctrlKey: false,
            metaKey: false,
            isComposing: false,
            preventDefault() {{ this.defaultPrevented = true; }},
            stopPropagation() {{ this.propagationStopped = true; }},
          }};
          events.keydown(event);
          return event;
        }}

        const noDialog = press("a");

        appConfirm.style.display = "flex";
        const apply = press("a");
        const cancel = press("c");
        appConfirm.style.display = "none";

        fileViewer.style.display = "flex";
        fileUnsavedDialog.style.display = "flex";
        const nestedDiscard = press("d");
        fileUnsavedDialog.style.display = "none";
        fileViewer.style.display = "none";

        const tieModal = modal("tieModal", [button("tie-cancel", "Cancel"), button("tie-continue", "Continue")]);
        ctx.modalIsolationTargetsForTest.unshift(tieModal);
        tieModal.style.display = "flex";
        const tieCancel = press("a");
        const tieContinue = press("o");
        const tiedFirstLetter = press("c");
        tieModal.style.display = "none";

        appConfirm.style.display = "flex";
        const textEntry = press("a", {{ isTextEntry: true }});
        appConfirm.style.display = "none";

        process.stdout.write(JSON.stringify({{
          listenerRegistered: typeof events.keydown === "function",
          modalTargetNames: ctx.modalTargetNames,
          clicks,
          noDialog,
          apply,
          cancel,
          nestedDiscard,
          tieCancel,
          tieContinue,
          tiedFirstLetter,
          textEntry,
        }}));
        """
    )
    return run_node_json(js)


class TestModalKeyboard(unittest.TestCase):
    def test_visible_dialog_buttons_activate_from_real_keydown_events(self) -> None:
        result = eval_modal_keyboard()

        self.assertTrue(result["listenerRegistered"])
        self.assertLess(
            result["modalTargetNames"].index("fileUnsavedDialog"),
            result["modalTargetNames"].index("fileViewer"),
        )
        self.assertEqual(
            result["clicks"],
            ["confirm-apply", "confirm-cancel", "unsaved-discard", "tie-cancel", "tie-continue"],
        )
        for event_name in ("apply", "cancel", "nestedDiscard", "tieCancel", "tieContinue"):
            self.assertTrue(result[event_name]["defaultPrevented"], event_name)
            self.assertTrue(result[event_name]["propagationStopped"], event_name)

    def test_tied_labels_require_later_letter_and_closed_or_text_entry_paths_do_nothing(self) -> None:
        result = eval_modal_keyboard()

        self.assertFalse(result["noDialog"]["defaultPrevented"])
        self.assertFalse(result["noDialog"]["propagationStopped"])
        self.assertFalse(result["tiedFirstLetter"]["defaultPrevented"])
        self.assertFalse(result["tiedFirstLetter"]["propagationStopped"])
        self.assertFalse(result["textEntry"]["defaultPrevented"])
        self.assertFalse(result["textEntry"]["propagationStopped"])


if __name__ == "__main__":
    unittest.main()
