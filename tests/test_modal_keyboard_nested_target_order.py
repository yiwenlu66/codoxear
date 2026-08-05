import json
import os
import subprocess
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_MODAL_JS = ROOT / "codoxear" / "static" / "app_modal.js"


def test_nested_file_dialogs_own_keyboard_targets_before_file_viewer() -> None:
    """Nested file dialogs receive their distinctive keys before the viewer."""
    app_source = APP_JS.read_text(encoding="utf-8")
    targets_start = app_source.index("        const modalIsolationTargets = [")
    targets_end = app_source.index("        ];", targets_start) + len("        ];")
    targets_source = app_source[targets_start:targets_end]
    modal_source = APP_MODAL_JS.read_text(encoding="utf-8")
    program = textwrap.dedent(
        f"""
        const vm = require("vm");
        const modalSource = {json.dumps(modal_source)};
        const targetsSource = {json.dumps(targets_source)};
        const clicks = [];

        function button(name, label) {{
          return {{
            name,
            textContent: label,
            disabled: false,
            hidden: false,
            getAttribute() {{ return null; }},
            getClientRects() {{ return [{{}}]; }},
            click() {{ clicks.push(name); }},
          }};
        }}

        function dialog(name, buttons, open = false) {{
          return {{
            name,
            style: {{ display: open ? "flex" : "none" }},
            querySelectorAll(selector) {{
              if (selector !== "button") throw new Error(`unexpected selector: ${{selector}}`);
              return buttons;
            }},
          }};
        }}

        const fileViewer = dialog("fileViewer", [button("viewer-delete", "Delete"), button("viewer-replace", "Replace")], true);
        const fileUnsavedDialog = dialog("fileUnsavedDialog", [button("unsaved-discard", "Discard"), button("unsaved-cancel", "Cancel")], true);
        const filePasteDialog = dialog("filePasteDialog", [button("paste-replace", "Replace")]);
        const sendChoice = dialog("sendChoice", []);
        const appConfirm = dialog("appConfirm", []);
        const queueViewer = dialog("queueViewer", []);
        const helpViewer = dialog("helpViewer", []);
        const diagViewer = dialog("diagViewer", []);
        const editViewer = dialog("editViewer", []);
        const voiceSettingsViewer = dialog("voiceSettingsViewer", []);
        const newSessionDialogController = {{ viewer: dialog("newSessionViewer", []) }};
        const ctx = {{
          window: {{}},
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
        }};
        vm.createContext(ctx);
        vm.runInContext(modalSource, ctx);
        vm.runInContext(`${{targetsSource}}\nglobalThis.modalIsolationTargetsForTest = modalIsolationTargets;`, ctx);
        const handler = ctx.window.CodoxearModal.createModalKeyboardHandler({{
          modalIsolationTargets: ctx.modalIsolationTargetsForTest,
          isTextEntryElement: () => false,
        }});
        function press(key) {{
          const event = {{
            key,
            defaultPrevented: false,
            propagationStopped: false,
            altKey: false,
            ctrlKey: false,
            metaKey: false,
            isComposing: false,
            preventDefault() {{ this.defaultPrevented = true; }},
            stopPropagation() {{ this.propagationStopped = true; }},
          }};
          const activated = handler(event);
          return {{ activated, defaultPrevented: event.defaultPrevented, propagationStopped: event.propagationStopped }};
        }}
        const unsaved = press("d");
        fileUnsavedDialog.style.display = "none";
        filePasteDialog.style.display = "flex";
        const paste = press("r");
        process.stdout.write(JSON.stringify({{ clicks, unsaved, paste }}));
        """
    )
    proc = subprocess.run(
        ["node", "-e", program],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={"PATH": os.environ.get("PATH", ""), "TZ": "UTC"},
    )
    result = json.loads(proc.stdout)

    assert result["clicks"] == ["unsaved-discard", "paste-replace"]
    for event_name in ("unsaved", "paste"):
        assert result[event_name]["activated"] is True
        assert result[event_name]["defaultPrevented"] is True
        assert result[event_name]["propagationStopped"] is True
