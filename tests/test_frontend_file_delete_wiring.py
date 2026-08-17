from frontend_module_loader import module_path
import json
import os
import subprocess


def test_file_touch_wiring_preserves_delete_command_dependency() -> None:
    """The touch-editor runtime receives its delete-command resolver through wiring."""
    source = module_path("app_wiring.js").read_text(encoding="utf-8")
    program = """
const vm = require("vm");
const context = { window: {} };
vm.createContext(context);
vm.runInContext(__SOURCE__, context);
const wiring = context.window.CodoxearWiring.createWiring();
const deleteCommand = (key) => key === "backspace" ? "deleteLeft" : "";
const options = wiring.createFileTouchOptions({
  fileEditorDeleteCommandForKey: deleteCommand,
  unrelated: "must not reach viewer options",
});
process.stdout.write(JSON.stringify({
  keys: Object.keys(options).sort(),
  command: options.fileEditorDeleteCommandForKey("backspace"),
  unknown: options.fileEditorDeleteCommandForKey("x"),
}));
""".replace("__SOURCE__", json.dumps(source))
    completed = subprocess.run(
        ["node", "-e", program],
        check=True,
        capture_output=True,
        text=True,
        env={"PATH": os.environ.get("PATH", "")},
    )
    assert json.loads(completed.stdout) == {
        "keys": ["fileEditorDeleteCommandForKey"],
        "command": "deleteLeft",
        "unknown": "",
    }
