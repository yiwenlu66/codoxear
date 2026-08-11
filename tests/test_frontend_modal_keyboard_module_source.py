from frontend_module_loader import module_path
import json
import os
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_MODAL_JS = module_path("app_modal.js")


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


def eval_modal_keyboard() -> dict:
    source = APP_MODAL_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const source = {json.dumps(source)};
        const clicks = [];

        function button(name, label, {{ disabled = false, hidden = false, visible = true }} = {{}}) {{
          return {{
            name,
            textContent: label,
            disabled,
            hidden,
            getAttribute(attribute) {{ return attribute === "aria-label" ? "" : null; }},
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

        const appConfirm = modal("appConfirm", [
          button("confirm-apply", "Apply"),
          button("confirm-cancel", "Cancel"),
          button("confirm-disabled", "Archive", {{ disabled: true }}),
          button("confirm-hidden", "Delete", {{ hidden: true }}),
        ]);
        const tieModal = modal("tieModal", [
          button("tie-cancel", "Cancel"),
          button("tie-continue", "Continue"),
        ]);
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext(source, ctx);
        const handler = ctx.window.CodoxearModal.createModalKeyboardHandler({{
          modalIsolationTargets: [appConfirm, tieModal],
          isTextEntryElement: (node) => Boolean(node && node.isTextEntry),
        }});

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
          const activated = handler(event);
          return {{ activated, defaultPrevented: event.defaultPrevented, propagationStopped: event.propagationStopped }};
        }}

        const noDialog = press("a");
        appConfirm.style.display = "flex";
        const apply = press("a");
        const cancel = press("c");
        const textEntry = press("a", {{ isTextEntry: true }});
        appConfirm.style.display = "none";
        tieModal.style.display = "flex";
        const tieCancel = press("a");
        const tieContinue = press("o");
        const tiedFirstLetter = press("c");
        process.stdout.write(JSON.stringify({{
          frozen: Object.isFrozen(ctx.window.CodoxearModal),
          handler: typeof handler === "function",
          clicks,
          noDialog,
          apply,
          cancel,
          textEntry,
          tieCancel,
          tieContinue,
          tiedFirstLetter,
        }}));
        """
    )
    return run_node_json(js)


class TestFrontendModalKeyboardModule(unittest.TestCase):
    def test_visible_enabled_dialog_buttons_activate_by_distinct_label_letter(self) -> None:
        result = eval_modal_keyboard()

        self.assertTrue(result["frozen"])
        self.assertTrue(result["handler"])
        self.assertEqual(result["clicks"], ["confirm-apply", "confirm-cancel", "tie-cancel", "tie-continue"])
        for event_name in ("apply", "cancel", "tieCancel", "tieContinue"):
            self.assertTrue(result[event_name]["activated"], event_name)
            self.assertTrue(result[event_name]["defaultPrevented"], event_name)
            self.assertTrue(result[event_name]["propagationStopped"], event_name)

    def test_nonmodal_tied_and_text_entry_keys_do_not_activate_buttons(self) -> None:
        result = eval_modal_keyboard()

        for event_name in ("noDialog", "textEntry", "tiedFirstLetter"):
            self.assertFalse(result[event_name]["activated"], event_name)
            self.assertFalse(result[event_name]["defaultPrevented"], event_name)
            self.assertFalse(result[event_name]["propagationStopped"], event_name)


if __name__ == "__main__":
    unittest.main()
