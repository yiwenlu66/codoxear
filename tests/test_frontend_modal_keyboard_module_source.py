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


def eval_modal_controllers() -> dict:
    source = APP_MODAL_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const source = {json.dumps(source)};
        class ElementStub {{}}
        function node(name) {{
          const listeners = {{}};
          return Object.assign(new ElementStub(), {{
            name,
            style: {{ display: "none" }},
            textContent: "",
            innerHTML: "old",
            disabled: false,
            isConnected: true,
            children: [],
            attributes: {{}},
            appendChild(child) {{ this.children.push(child); }},
            setAttribute(key, value) {{ this.attributes[key] = String(value); }},
            removeAttribute(key) {{ delete this.attributes[key]; }},
            toggleAttribute(key, value) {{ if (value) this.attributes[key] = ""; else delete this.attributes[key]; }},
            addEventListener(type, handler) {{ listeners[type] = handler; }},
            emit(type, event = {{}}) {{ return listeners[type](event); }},
            focus() {{ focused.push(name); }},
          }});
        }}
        const focused = [];
        const calls = [];
        const origin = node("origin");
        const backdrop = node("backdrop");
        const viewer = node("viewer");
        const title = node("title");
        const message = node("message");
        const confirmButton = node("confirm");
        const cancelButton = node("cancel");
        const app = node("app");
        const openModal = node("open-modal");
        openModal.style.display = "flex";
        const createElement = (tag, attrs) => {{
          const child = node(`${{tag}}:${{attrs.class || ""}}`);
          child.textContent = attrs.text || "";
          child.innerHTML = attrs.html || "";
          child.className = attrs.class || "";
          return child;
        }};
        const ctx = {{ window: {{}}, requestAnimationFrame: (fn) => fn() }};
        vm.createContext(ctx);
        vm.runInContext(source, ctx);

        const policy = ctx.window.CodoxearModal.createModalPolicyController({{
          app,
          modalTargets: () => [openModal],
          closeUnattended: () => calls.push("unattended"),
          isUnattendedOpen: () => true,
          closeSearch: () => calls.push("search"),
          isSearchOpen: () => true,
          isSidebarOpen: () => true,
          closeSidebar: () => calls.push("sidebar"),
          closeFilePicker: () => calls.push("file-picker"),
          closeNewSessionMenus: () => calls.push("new-session"),
          closeSessionDependencyMenu: () => calls.push("dependency"),
          el: createElement,
          iconSvg: (name) => `<${{name}}>`,
        }});
        policy.prepareModalOpen({{ closeSearch: true }});
        const isolated = policy.afterModalVisibilityChanged();
        const picker = node("picker");
        policy.setPickerButtonContent(picker, "Primary", "Secondary", true);

        const confirmation = ctx.window.CodoxearModal.createConfirmationController({{
          backdrop, viewer, title, message, confirmButton, cancelButton,
          documentTarget: {{ activeElement: origin }},
          ElementCtor: ElementStub,
          requestFrame: (fn) => fn(),
          prepareModalOpen: () => calls.push("prepare-confirm"),
          afterModalVisibilityChanged: () => calls.push("visibility-confirm"),
          addEvent: (target, type, handler) => target.addEventListener(type, handler),
        }});
        const destructivePromise = confirmation.confirm({{
          title: "Delete session", message: "Really?", confirmText: "Delete", cancelText: "Keep", destructive: true,
        }});
        const destructiveSnapshot = {{
          title: title.textContent,
          message: message.textContent,
          confirm: confirmButton.textContent,
          cancel: cancelButton.textContent,
          backdrop: backdrop.style.display,
          viewer: viewer.style.display,
          focusable: confirmation.focusableControls().map((item) => item.name),
        }};
        cancelButton.emit("click");
        destructivePromise.then((destructiveResult) => {{
          const constructivePromise = confirmation.confirm("Continue?");
          confirmButton.emit("click");
          constructivePromise.then((constructiveResult) => {{
            process.stdout.write(JSON.stringify({{
              frozen: Object.isFrozen(policy) && Object.isFrozen(confirmation),
              calls,
              isolated,
              appAttributes: app.attributes,
              picker: picker.children.map((child) => ({{ className: child.className, text: child.textContent, html: child.innerHTML, children: child.children.map((nested) => ({{ className: nested.className, text: nested.textContent }})) }})),
              focused,
              destructiveSnapshot,
              destructiveResult,
              constructiveResult,
              closed: {{ backdrop: backdrop.style.display, viewer: viewer.style.display }},
            }}));
          }});
        }});
        """
    )
    return run_node_json(js)


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


class TestFrontendModalControllers(unittest.TestCase):
    def test_modal_policy_owns_overlay_closure_isolation_and_picker_shape(self) -> None:
        result = eval_modal_controllers()
        self.assertTrue(result["frozen"])
        self.assertEqual(result["calls"][:6], ["unattended", "search", "sidebar", "file-picker", "new-session", "dependency"])
        self.assertTrue(result["isolated"])
        self.assertIn("inert", result["appAttributes"])
        self.assertEqual(result["appAttributes"]["aria-hidden"], "true")
        self.assertEqual(result["picker"][0]["className"], "pickerButtonText placeholder")
        self.assertEqual(result["picker"][0]["children"], [
            {"className": "pickerButtonPrimary", "text": "Primary"},
            {"className": "pickerButtonSecondary", "text": "Secondary"},
        ])
        self.assertEqual(result["picker"][1]["className"], "pickerButtonChevron")
        self.assertEqual(result["picker"][1]["html"], "<chevronDown>")

    def test_confirmation_owns_render_focus_resolution_and_restore(self) -> None:
        result = eval_modal_controllers()
        self.assertEqual(result["destructiveSnapshot"], {
            "title": "Delete session",
            "message": "Really?",
            "confirm": "Delete",
            "cancel": "Keep",
            "backdrop": "block",
            "viewer": "flex",
            "focusable": ["cancel", "confirm"],
        })
        self.assertFalse(result["destructiveResult"])
        self.assertTrue(result["constructiveResult"])
        self.assertEqual(result["closed"], {"backdrop": "none", "viewer": "none"})
        self.assertEqual(result["focused"], ["cancel", "origin", "confirm", "origin"])
        self.assertEqual(result["calls"][6:], [
            "prepare-confirm", "visibility-confirm", "visibility-confirm",
            "prepare-confirm", "visibility-confirm", "visibility-confirm",
        ])


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
