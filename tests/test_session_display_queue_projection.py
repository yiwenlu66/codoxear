from frontend_module_loader import module_path
import json
import os
import subprocess


APP_SESSION_STATE_JS = module_path("app_session_state.js")
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


def test_session_display_subscribes_to_runtime_store_and_disposes() -> None:
    """Store writes synchronously project queue, interrupt, and context DOM state."""
    script = f"""
    const vm = require("vm");
    const ctx = {{ window: {{}}, console }};
    vm.createContext(ctx);
    vm.runInContext({json.dumps(APP_SESSION_STATE_JS.read_text(encoding="utf-8"))}, ctx);
    vm.runInContext({json.dumps(APP_SESSION_DISPLAY_JS.read_text(encoding="utf-8"))}, ctx);
    const sessionState = ctx.window.CodoxearSessionState.createSessionState({{ consoleError: () => {{}} }});
    const statusChip = {{ style: {{}}, textContent: "" }};
    const interruptBtn = {{ style: {{}}, disabled: false }};
    const ctxChip = {{ style: {{}}, disabled: false, textContent: "", title: "" }};
    sessionState.set("selected", "sid");
    const controller = ctx.window.CodoxearSessionDisplay.createSessionDisplayController({{
      sessionState,
      setToast: () => {{}},
      statusChip, interruptBtn, ctxChip, eventBindings: {{ on: () => {{}} }},
    }});
    const snap = () => ({{
      status: {{ text: statusChip.textContent, display: statusChip.style.display }},
      interrupt: {{ display: interruptBtn.style.display, disabled: interruptBtn.disabled }},
      context: {{ text: ctxChip.textContent, display: ctxChip.style.display, disabled: ctxChip.disabled, title: ctxChip.title }},
    }});
    const initial = snap();
    sessionState.applyRuntime({{
      running: true,
      queueLen: 2,
      token: {{ context_window: 100, tokens_in_context: 40, percent_remaining: 60, max_input_tokens: 80, reserved_tokens: 20 }},
    }});
    const active = snap();
    sessionState.applyRuntime({{ running: false, queueLen: 0, token: null }});
    const cleared = snap();
    controller.dispose();
    sessionState.applyRuntime({{ running: true, queueLen: 7 }});
    const disposed = snap();
    process.stdout.write(JSON.stringify({{ initial, active, cleared, disposed }}));
    """
    assert run_node_json(script) == {
        "initial": {
            "status": {"text": "", "display": "none"},
            "interrupt": {"display": "none", "disabled": True},
            "context": {"text": "", "display": "none", "disabled": True, "title": ""},
        },
        "active": {
            "status": {"text": "Queue 2", "display": "inline-flex"},
            "interrupt": {"display": "inline-flex", "disabled": False},
            "context": {
                "text": "Ctx 60%",
                "display": "inline-flex",
                "disabled": False,
                "title": "Context input: 40/80 tokens (20 reserved; window 100).",
            },
        },
        "cleared": {
            "status": {"text": "", "display": "none"},
            "interrupt": {"display": "none", "disabled": True},
            "context": {"text": "", "display": "none", "disabled": True, "title": ""},
        },
        "disposed": {
            "status": {"text": "", "display": "none"},
            "interrupt": {"display": "none", "disabled": True},
            "context": {"text": "", "display": "none", "disabled": True, "title": ""},
        },
    }
