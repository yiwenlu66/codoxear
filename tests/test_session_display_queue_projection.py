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


def test_status_header_projects_queue_count_only() -> None:
    """The topbar chip shows only the queue payload; busy/idle and ▸N live elsewhere."""
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
      setRunning: (value) => {{ running = value; }},
      getQueueLen: () => queueLen,
      setQueueLen: (value) => {{ queueLen = value; }},
      getAttachmentsController: () => null,
      updateQueueBadge: () => {{}},
      setToast: () => {{}},
      statusChip, interruptBtn, ctxChip, eventBindings: {{ on: () => {{}} }},
    }});
    const snap = () => ({{ text: statusChip.textContent, display: statusChip.style.display }});
    controller.setStatus({{ running: false, queueLen: 2 }});
    const idle = snap();
    controller.setStatus({{ running: true, queueLen: 2 }});
    const busy = snap();
    controller.setStatus({{ running: true, queueLen: 0 }});
    const busyEmpty = snap();
    controller.setStatus({{ running: false, queueLen: 0 }});
    const idleEmpty = snap();
    process.stdout.write(JSON.stringify({{ idle, busy, busyEmpty, idleEmpty }}));
    """
    assert run_node_json(script) == {
        "idle": {"text": "Queue 2", "display": "inline-flex"},
        "busy": {"text": "Queue 2", "display": "inline-flex"},
        "busyEmpty": {"text": "", "display": "none"},
        "idleEmpty": {"text": "", "display": "none"},
    }
