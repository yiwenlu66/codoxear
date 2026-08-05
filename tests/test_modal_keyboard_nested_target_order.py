import json
import os
import subprocess
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_MODAL_JS = ROOT / "codoxear" / "static" / "app_modal.js"


def test_nested_file_dialogs_own_keyboard_targets_before_file_viewer() -> None:
    """Nested file dialogs receive their distinctive keys before the viewer."""
    modal_source = APP_MODAL_JS.read_text(encoding="utf-8")
    program = textwrap.dedent(
        f"""
        const vm = require("vm");
        const modalSource = {json.dumps(modal_source)};
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

        function dialog(name, buttons) {{
          return {{
            name,
            style: {{ display: "flex" }},
            querySelectorAll(selector) {{
              if (selector !== "button") throw new Error(`unexpected selector: ${{selector}}`);
              return buttons;
            }},
          }};
        }}

        const fileViewer = dialog("fileViewer", [button("viewer-delete", "Delete")]);
        const fileUnsavedDialog = dialog("fileUnsavedDialog", [button("unsaved-discard", "Discard"), button("unsaved-cancel", "Cancel")]);
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext(modalSource, ctx);
        const handler = ctx.window.CodoxearModal.createModalKeyboardHandler({{
          modalIsolationTargets: [fileUnsavedDialog, fileViewer],
          isTextEntryElement: () => false,
        }});
        const event = {{
          key: "d",
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
        process.stdout.write(JSON.stringify({{ activated, clicks, event }}));
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

    assert result["activated"] is True
    assert result["event"]["defaultPrevented"] is True
    assert result["event"]["propagationStopped"] is True
    assert result["clicks"] == ["unsaved-discard"]
