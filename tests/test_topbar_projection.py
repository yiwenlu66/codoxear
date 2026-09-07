from frontend_module_loader import module_path
import json
import os
import subprocess


APP_SESSION_STATE_JS = module_path("app_session_state.js")
APP_TOPBAR_JS = module_path("app_topbar.js")


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


def test_topbar_subscribes_to_runtime_store_and_disposes() -> None:
    """Topbar store subscriptions synchronously project interrupt and context widgets.

    The topbar owns no status chip: queue depth belongs to the composer queue
    button badge and the sidebar session card, so queueLen churn must not reach
    this controller.
    """
    script = f"""
    const vm = require("vm");
    const ctx = {{ window: {{}}, console }};
    vm.createContext(ctx);
    vm.runInContext({json.dumps(APP_SESSION_STATE_JS.read_text(encoding="utf-8"))}, ctx);
    vm.runInContext({json.dumps(APP_TOPBAR_JS.read_text(encoding="utf-8"))}, ctx);
    const sessionState = ctx.window.CodoxearSessionState.createSessionState({{ consoleError: () => {{}} }});
    class Node {{
      constructor(tag, attrs = {{}}) {{ this.tag = tag; this.attrs = attrs; this.children = []; this.style = {{}}; this.textContent = attrs.text || ""; this.disabled = false; }}
      appendChild(child) {{ child.parent = this; this.children.push(child); return child; }}
      append(...children) {{ children.forEach((child) => this.appendChild(child)); }}
      setAttribute(name, value) {{ this.attrs[name] = String(value); }}
    }}
    const el = (tag, attrs = {{}}) => new Node(tag, attrs);
    const topMeta = new Node("div");
    const topActions = new Node("div");
    const events = {{ on: (target, type, handler) => {{ target[type] = handler; }} }};
    const toasts = [];
    let interrupts = 0;
    sessionState.set("selected", "sid");
    const controller = ctx.window.CodoxearTopbar.createTopbarController({{
      el, iconSvg: () => "", sessionState, setToast: (text) => toasts.push(text), onInterrupt: () => {{ interrupts += 1; }},
      topMeta, topActions, eventBindings: events,
    }});
    const {{ interruptBtn, ctxChip }} = controller.elements;
    const snap = () => ({{
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
    ctxChip.click();
    interruptBtn.click({{ preventDefault: () => {{}}, stopPropagation: () => {{}} }});
    const interactions = {{
      toast: toasts[0],
      interrupts,
      mounted: {{
        topMeta: topMeta.children.map((node) => node.attrs.id),
        topActions: topActions.children.map((node) => node.attrs.id),
        contextHint: ctxChip.attrs["data-hint"],
        interruptHint: interruptBtn.attrs["data-hint"],
      }},
    }};
    sessionState.applyRuntime({{ running: false, queueLen: 0, token: null }});
    const cleared = snap();
    controller.dispose();
    sessionState.applyRuntime({{ running: true, queueLen: 7 }});
    const disposed = snap();
    process.stdout.write(JSON.stringify({{ initial, active, interactions, cleared, disposed }}));
    """
    assert run_node_json(script) == {
        "initial": {
            "interrupt": {"display": "none", "disabled": True},
            "context": {"text": "", "display": "none", "disabled": True, "title": ""},
        },
        "active": {
            "interrupt": {"display": "inline-flex", "disabled": False},
            "context": {
                "text": "Ctx 60%",
                "display": "inline-flex",
                "disabled": False,
                "title": "Context input: 40/80 tokens (20 reserved; window 100).",
            },
        },
        "interactions": {
            "toast": "ctx 40/100 (60% left)",
            "interrupts": 1,
            "mounted": {
                "topMeta": ["ctxChip"],
                "topActions": ["interruptBtn"],
                "contextHint": "y",
                "interruptHint": "z",
            },
        },
        "cleared": {
            "interrupt": {"display": "none", "disabled": True},
            "context": {"text": "", "display": "none", "disabled": True, "title": ""},
        },
        "disposed": {
            "interrupt": {"display": "none", "disabled": True},
            "context": {"text": "", "display": "none", "disabled": True, "title": ""},
        },
    }
