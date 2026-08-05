import json
import os
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_MODAL_JS = ROOT / "codoxear" / "static" / "app_modal.js"


def test_nested_file_viewer_precedes_unsaved_dialog_keyboard_targets() -> None:
    """The first open target owns a key, so this mirrors app.js's viewer-first order."""
    source = APP_MODAL_JS.read_text(encoding="utf-8")
    program = textwrap.dedent(
        f"""
        const vm = require("vm");
        const source = {json.dumps(source)};
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
        vm.runInContext(source, ctx);
        const handler = ctx.window.CodoxearModal.createModalKeyboardHandler({{
          modalIsolationTargets: [fileViewer, fileUnsavedDialog],
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
    assert result["clicks"] == ["viewer-delete"]
