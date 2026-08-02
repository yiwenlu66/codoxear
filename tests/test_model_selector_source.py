import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_COMPOSER_JS = ROOT / "codoxear" / "static" / "app_composer.js"


class TestComposerModelPicker(unittest.TestCase):
    def test_model_prefix_filters_and_selects_only_for_pi(self) -> None:
        source = APP_COMPOSER_JS.read_text(encoding="utf-8")
        script = "(async () => {\n" + textwrap.dedent(
            f"""
            const vm = require("vm");
            let scrollCalls = 0;
            class Node {{
              constructor() {{
                this.listeners = {{}};
                this.style = {{}};
                this.classList = {{ toggle: () => {{}} }};
                this.attributes = {{}};
                this.children = [];
                this.value = "";
                this.scrollHeight = 32;
                this.disabled = false;
                this.textContent = "";
              }}
              set innerHTML(value) {{ this.children = []; this._innerHTML = value; }}
              get innerHTML() {{ return this._innerHTML || ""; }}
              addEventListener(type, fn) {{ (this.listeners[type] ||= []).push(fn); }}
              removeEventListener() {{}}
              dispatch(type, event = {{}}) {{ for (const fn of this.listeners[type] || []) fn(event); }}
              setAttribute(name, value) {{ this.attributes[name] = String(value); }}
              removeAttribute(name) {{ delete this.attributes[name]; }}
              appendChild(child) {{ this.children.push(child); return child; }}
              scrollIntoView() {{ scrollCalls += 1; }}
              focus() {{}}
              blur() {{}}
            }}
            const document = {{ createElement: () => new Node(), activeElement: null }};
            const ctx = {{ window: {{}}, document, console, Date, Set, Object, String, Number, Promise }};
            vm.createContext(ctx);
            vm.runInContext({json.dumps(source)}, ctx);
            const nodes = Array.from({{ length: 10 }}, () => new Node());
            const [form, textarea, msgPh, sendBtn, sendChoice, sendChoiceBackdrop, nowBtn, laterBtn, cancelBtn, modelPicker] = nodes;
            form.requestSubmit = () => {{ state.formSubmits += 1; }};
            const state = {{ backend: "pi", thinkingCapability: true, sent: [], sending: false, formSubmits: 0 }};
            const noop = () => {{}};
            const controller = ctx.window.CodoxearComposer.createComposerController({{
              form, textarea, msgPh, sendBtn, sendChoice, sendChoiceBackdrop,
              sendChoiceNowBtn: nowBtn, sendChoiceLaterBtn: laterBtn, sendChoiceCancelBtn: cancelBtn,
              modelPicker,
              getSelected: () => "sid",
              getSessionInfo: () => ({{
                agent_backend: state.backend,
                pi_thinking_command: state.thinkingCapability,
                model_provider: "anthropic",
                model: "claude-sonnet-4",
                reasoning_effort: "high",
              }}),
              getNewSessionDefaults: () => ({{ backends: {{ pi: {{
                provider_models: {{ anthropic: ["claude-sonnet-4"], openai: ["gpt-5"] }},
                reasoning_efforts_by_model: {{ "anthropic/claude-sonnet-4": ["off", "low", "high"] }},
              }} }} }}),
              patchSessionInfo: noop,
              sessionLaunchFailed: () => false,
              getSending: () => state.sending,
              setSending: (value) => {{ state.sending = value; }},
              getCurrentRunning: () => false,
              setCurrentRunning: noop, setTurnOpen: noop, resetTypingStats: noop,
              getStagedAttachments: () => [], normalizedStagedAttachments: () => [],
              setSelectedSessionPendingAttachment: noop, setAttachCount: noop,
              syncAttachButtonState: noop, syncQueueSubmitState: noop,
              syncRecoveryUiForSession: noop, confirmAction: async () => false,
              api: async (_path, options) => {{ state.sent.push(options.body.text); return {{}}; }},
              setToast: noop, handleAppAuthLoss: noop, refreshSessions: async () => [],
              setPollFastUntilMs: noop, kickPoll: noop, isTranscriptRenewalCommand: () => false,
              nextLocalEchoId: () => "local", renderedAtLiveTail: () => true,
              clearTranscriptDom: noop, clearRenderedTranscriptRange: noop, setOlderState: noop,
              getSessionTranscriptSlot: () => ({{ epoch: 0 }}), addPendingUser: noop,
              appendEvent: noop, deleteTailCache: noop, beginTranscriptRenewal: noop,
              clearLiveCursor: noop, invalidateOlderLoad: noop, renderPendingTranscriptSlot: noop,
              dropPendingUser: noop, removePendingUserRow: noop, hasPendingForSession: () => false,
              enqueueComposerText: async () => true, prepareModalOpen: noop,
              afterModalVisibilityChanged: noop, restoreModalFocus: noop,
              storageGetItem: () => "", storageSetItem: noop, storageRemoveItem: noop,
              getComputedStyle: () => ({{ minHeight: "32px" }}), requestFrame: noop,
              activeElement: () => textarea, isHTMLElement: () => true, now: () => 1,
            }});
            textarea.value = "/model";
            textarea.dispatch("input");
            if (modelPicker.style.display !== "block" || modelPicker.children.length !== 2) throw new Error("Pi /model did not open full picker");
            if (textarea.attributes.role !== "combobox" || textarea.attributes["aria-expanded"] !== "true") throw new Error("textarea did not expose expanded combobox semantics");
            if (textarea.attributes["aria-activedescendant"] !== "model-picker-option-0") throw new Error("active descendant belongs on the focused textarea");
            textarea.value = "/model claude";
            textarea.dispatch("input");
            if (modelPicker.children.length !== 1 || modelPicker.children[0].textContent !== "anthropic/claude-sonnet-4") throw new Error("model filter mismatch");
            textarea.dispatch("keydown", {{ key: "Escape", preventDefault() {{}} }});
            if (modelPicker.style.display !== "none") throw new Error("Escape did not dismiss picker");
            textarea.value = "/model";
            textarea.dispatch("input");
            const downEvent = {{ key: "ArrowDown", defaultPrevented: false, preventDefault() {{ this.defaultPrevented = true; }} }};
            textarea.dispatch("keydown", downEvent);
            if (!downEvent.defaultPrevented || scrollCalls !== 1) throw new Error("ArrowDown did not consume the key and reveal the active option");
            if (modelPicker.children[1].attributes["aria-selected"] !== "true" || textarea.attributes["aria-activedescendant"] !== "model-picker-option-1") throw new Error("ArrowDown did not expose the selected option");
            const enterEvent = {{ key: "Enter", defaultPrevented: false, preventDefault() {{ this.defaultPrevented = true; }} }};
            textarea.dispatch("keydown", enterEvent);
            if (!enterEvent.defaultPrevented) throw new Error("picker Enter was not consumed");
            await new Promise((resolve) => setTimeout(resolve, 0));
            if (state.sent[0] !== "/model openai/gpt-5") throw new Error("selected provider/model id was not sent");
            if (state.sent.length !== 1 || state.formSubmits !== 0) throw new Error("picker Enter also submitted the composer form");
            if (textarea.attributes.role || textarea.attributes["aria-expanded"] || textarea.attributes["aria-activedescendant"]) throw new Error("closed picker left stale combobox state");
            textarea.value = "/thinking";
            textarea.dispatch("input");
            if (modelPicker.style.display !== "block" || modelPicker.children.length !== 3) throw new Error("Pi /thinking did not open model-scoped picker");
            if (modelPicker.children[0].textContent !== "high" || modelPicker.attributes["aria-label"] !== "Available Pi thinking levels") throw new Error("current thinking level was not first/highlighted");
            textarea.value = "/thinking lo";
            textarea.dispatch("input");
            if (modelPicker.children.length !== 1 || modelPicker.children[0].textContent !== "low") throw new Error("thinking level filter mismatch");
            const thinkingEnter = {{ key: "Enter", defaultPrevented: false, preventDefault() {{ this.defaultPrevented = true; }} }};
            textarea.dispatch("keydown", thinkingEnter);
            if (!thinkingEnter.defaultPrevented) throw new Error("thinking picker Enter was not consumed");
            await new Promise((resolve) => setTimeout(resolve, 0));
            if (state.sent[1] !== "/thinking low") throw new Error("selected thinking level was not sent through composer path");
            textarea.value = "/thinking";
            textarea.dispatch("input");
            textarea.dispatch("keydown", {{ key: "Escape", preventDefault() {{}} }});
            if (modelPicker.style.display !== "none") throw new Error("thinking picker Escape did not dismiss picker");
            state.thinkingCapability = false;
            textarea.value = "/thinking";
            textarea.dispatch("input");
            if (modelPicker.style.display !== "none") throw new Error("incapable Pi session opened thinking picker");
            state.thinkingCapability = true;
            textarea.value = "/thinking";
            textarea.dispatch("input");
            if (modelPicker.style.display !== "block") throw new Error("capability refresh did not restore thinking picker");
            state.backend = "codex";
            textarea.value = "/model";
            textarea.dispatch("input");
            if (modelPicker.style.display !== "none") throw new Error("non-Pi session opened picker");
            process.stdout.write(JSON.stringify({{ sent: state.sent, selected: modelPicker.style.display }}));
            """
        ) + "\n})();"
        result = subprocess.run(["node", "-e", script], check=False, capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(result.stderr or result.stdout)
        self.assertEqual(json.loads(result.stdout), {"sent": ["/model openai/gpt-5", "/thinking low"], "selected": "none"})


if __name__ == "__main__":
    unittest.main()
