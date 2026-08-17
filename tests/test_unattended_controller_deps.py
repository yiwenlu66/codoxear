from frontend_module_loader import module_path
import json
import os
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_UNATTENDED_JS = module_path("app_unattended.js")
APP_SESSION_HELPERS_JS = module_path("app_session_helpers.js")
APP_MODAL_JS = module_path("app_modal.js")


def run_node_json(program: str) -> dict:
    completed = subprocess.run(
        ["node", "-e", program],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={"PATH": os.environ.get("PATH", ""), "TZ": "UTC"},
    )
    return json.loads(completed.stdout)


class TestUnattendedControllerStorageDependencies(unittest.TestCase):
    def test_app_style_storage_functions_create_and_persist_request_draft(self) -> None:
        """The app shell injects its storage functions; the controller owns the WAL use."""
        program = textwrap.dedent(
            f"""
            const vm = require("vm");
            const modalSource = {json.dumps(APP_MODAL_JS.read_text(encoding="utf-8"))};
            const helpersSource = {json.dumps(APP_SESSION_HELPERS_JS.read_text(encoding="utf-8"))};
            const unattendedSource = {json.dumps(APP_UNATTENDED_JS.read_text(encoding="utf-8"))};

            function node() {{
              return {{
                style: {{}}, value: "", checked: false, disabled: false,
                setAttribute() {{}}, appendChild() {{}}, getBoundingClientRect() {{ return {{}}; }},
                classList: {{ toggle() {{}} }},
              }};
            }}

            const storage = new Map();
            const storageCalls = [];
            // These are the direct function dependencies used by app.js. The
            // controller must call these functions, not a hidden global store.
            function storageGetItem(key) {{
              storageCalls.push(["get", key]);
              return storage.has(key) ? storage.get(key) : null;
            }}
            function storageSetItem(key, value) {{
              storageCalls.push(["set", key, String(value)]);
              storage.set(key, String(value));
            }}
            function storageRemoveItem(key) {{
              storageCalls.push(["remove", key]);
              storage.delete(key);
            }}

            const selected = "session-1";
            const sessions = new Map([[selected, {{ launch_state: "ready" }}]]);
            const requestEl = node();
            const appStyleDeps = {{
              unattendedBtn: node(),
              unattendedMenu: node(),
              enabledEl: node(),
              cooldownEl: node(),
              remainingEl: node(),
              requestEl,
              sessionState: {{ get: () => selected, subscribe: () => () => {{}} }},
              sessionCatalog: {{
                patchSession: (sid, patch) => {{ const session = sessions.get(sid); if (!session) return null; Object.assign(session, patch); return session; }},
                subscribe: () => () => {{}},
              }},
              getSessionInfo: (sid) => sessions.get(sid) || null,
              isAppDisposed: () => false,
              api: async () => ({{ enabled: false, request: "", cooldown_minutes: 5, remaining_injections: 10 }}),
              refreshSessions: async () => {{}},
              handleAppAuthLoss: () => {{}},
              setToast: () => {{}},
              addAppEvent: () => {{}},
              documentTarget: {{}},
              windowTarget: {{}},
              requestFrame: (callback) => callback(),
              setTimeout: () => 1,
              clearTimeout: () => {{}},
              storageGetItem,
              storageSetItem,
              storageRemoveItem,
            }};

            const context = {{ HTMLElement: function HTMLElement() {{}}, window: {{}} }};
            vm.createContext(context);
            vm.runInContext(modalSource, context);
            vm.runInContext(helpersSource, context);
            vm.runInContext(unattendedSource, context);

            let dependencyError = null;
            let controller;
            try {{
              controller = context.window.CodoxearUnattended.createUnattendedController(appStyleDeps);
            }} catch (error) {{
              dependencyError = String(error && error.message || error);
            }}
            if (controller) {{
              requestEl.value = "preserve this request";
              requestEl.oninput({{ target: requestEl }});
            }}

            process.stdout.write(JSON.stringify({{
              dependencyError,
              storageCalls,
              draft: storage.get("codexweb.unattended.pending.v1") || null,
            }}));
            """
        )

        result = run_node_json(program)

        self.assertIsNone(result["dependencyError"])
        self.assertIn(["get", "codexweb.unattended.pending.v1"], result["storageCalls"])
        writes = [call for call in result["storageCalls"] if call[:2] == ["set", "codexweb.unattended.pending.v1"]]
        self.assertEqual(len(writes), 1)
        self.assertEqual(
            json.loads(result["draft"]),
            {
                "version": 1,
                "patches": {
                    "session-1": {
                        "revision": 1,
                        "patch": {"request": "preserve this request"},
                    }
                },
            },
        )


if __name__ == "__main__":
    unittest.main()
