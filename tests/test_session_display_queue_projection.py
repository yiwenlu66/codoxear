from frontend_module_loader import module_path
import json
import os
import subprocess


APP_SESSION_DISPLAY_JS = module_path("app_session_display.js")


def run_node_json(script: str) -> dict:
    result = subprocess.run(
        ["node", "-e", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={"PATH": os.environ.get("PATH", ""), "TZ": "UTC"},
    )
    if result.returncode:
        raise AssertionError(result.stderr)
    return json.loads(result.stdout)


def test_status_header_projects_queue_count_while_idle_or_busy() -> None:
    """The header count stays comparable with the sidebar/badge during a busy turn."""
    script = f"""
    const vm = require("vm");
    const ctx = {{ window: {{}}, console }};
    vm.createContext(ctx);
    vm.runInContext({json.dumps(APP_SESSION_DISPLAY_JS.read_text(encoding="utf-8"))}, ctx);
    let running = false;
    let queueLen = 0;
    const statusChip = {{ style: {{}}, textContent: "" }};
    const interruptBtn = {{ style: {{}}, disabled: false }};
    const ctxChip = {{ style: {{}}, disabled: false, textContent: "", title: "" }};
    const controller = ctx.window.CodoxearSessionDisplay.createSessionDisplayController({{
      getSelected: () => "sid",
      getRunning: () => running,
      setRunning: (value) => {{ running = value; }},
      getQueueLen: () => queueLen,
      setQueueLen: (value) => {{ queueLen = value; }},
      getSubagentsRunning: () => 0,
      getAttachmentsController: () => null,
      updateQueueBadge: () => {{}},
      setToast: () => {{}},
      statusChip, interruptBtn, ctxChip, eventBindings: {{ on: () => {{}} }},
    }});
    controller.setStatus({{ running: false, queueLen: 2 }});
    const idle = statusChip.textContent;
    controller.setStatus({{ running: true, queueLen: 2 }});
    const busy = statusChip.textContent;
    controller.setStatus({{ running: true, queueLen: 0 }});
    const empty = statusChip.textContent;
    process.stdout.write(JSON.stringify({{ idle, busy, empty }}));
    """
    assert run_node_json(script) == {
        "idle": "Idle · Queue 2",
        "busy": "Busy · Queue 2",
        "empty": "Busy",
    }
