"""Runtime API checks for focused frontend extraction modules."""

from __future__ import annotations
from frontend_module_loader import module_path

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "codoxear" / "static"

EXTRACTED_MODULES = {
    "app_message_flow.js": "CodoxearMessageFlow",
    "app_session_edit.js": "CodoxearSessionEdit",
    "app_unattended.js": "CodoxearUnattended",
}
if (module_path("app_attachments.js")).is_file():
    EXTRACTED_MODULES["app_attachments.js"] = "CodoxearAttachments"


def _load_exports(path: Path, global_name: str) -> dict[str, str]:
    source = json.dumps(path.read_text(encoding="utf-8"))
    program = f"""
        const vm = require("vm");
        const context = {{
          window: {{
            CodoxearPolling: {{
              messagePollDelayMs() {{}}, normalizeMessagePollKickDelay() {{}}, browserOffline() {{}},
            }},
            CodoxearTranscript: {{
              startsTypingCountWindow() {{}}, hasHumanOriginatedUserEvent() {{}},
              thinkingModeForTokens() {{}}, transcriptSnapshotFromData() {{}},
            }},
            CodoxearSessionHelpers: {{ sessionLaunchFailed() {{ return false; }} }},
            CodoxearModal: {{ restoreModalFocus() {{}} }},
          }},
        }};
        vm.createContext(context);
        vm.runInContext({source}, context, {{ filename: {json.dumps(path.name)} }});
        const exported = context.window[{json.dumps(global_name)}];
        const api = Object.fromEntries(
          Object.getOwnPropertyNames(exported).sort().map((name) => [name, typeof exported[name]])
        );
        process.stdout.write(JSON.stringify(api));
    """
    completed = subprocess.run(
        ["node", "-e", program],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_extracted_modules_publish_callable_public_apis() -> None:
    for filename, global_name in EXTRACTED_MODULES.items():
        exports = _load_exports(module_path(filename), global_name)
        assert exports, f"{filename} did not publish {global_name}"
        assert set(exports.values()) == {"function"}, (
            f"{filename} has a non-callable public API: {exports}"
        )
