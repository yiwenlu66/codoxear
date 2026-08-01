import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_COMPOSER_JS = ROOT / "codoxear" / "static" / "app_composer.js"


class TestComposerSlashPickers(unittest.TestCase):
    def test_model_and_thinking_prefixes_select_only_for_pi(self) -> None:
        source = APP_COMPOSER_JS.read_text(encoding="utf-8")
        script = "(async () => {\n" + textwrap.dedent(
            f"""
            const vm = require("vm");
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
              focus() {{}}
              blur() {{}}
            }}
            const document = {{ createElement: () => new Node(), activeElement: null }};
            const ctx = {{ window: {{}}, document, console, Date, Set, Object, String, Number, Promise }};
            vm.createContext(ctx);
            vm.runInContext({json.dumps(source)}, ctx);
            const nodes = Array.from({{ length: 10 }}, () => new Node());
            const [form, textarea, msgPh, sendBtn, sendChoice, sendChoiceBackdrop, nowBtn, laterBtn, cancelBtn, modelPicker] = nodes;
            form.requestSubmit = () => {{}};
            const state = {{ backend: "pi", sent: [], keys: [], thinking: "high", sending: false }};
            const noop = () => {{}};
            const controller = ctx.window.CodoxearComposer.createComposerController({{
              form, textarea, msgPh, sendBtn, sendChoice, sendChoiceBackdrop,
              sendChoiceNowBtn: nowBtn, sendChoiceLaterBtn: laterBtn, sendChoiceCancelBtn: cancelBtn,
              modelPicker,
              getSelected: () => "sid",
              getSessionInfo: () => ({{ agent_backend: state.backend, reasoning_effort: state.thinking }}),
              getNewSessionDefaults: () => ({{ backends: {{ pi: {{
                provider_models: {{ anthropic: ["claude-sonnet-4"], openai: ["gpt-5"] }},
                reasoning_efforts: ["off", "minimal", "low", "medium", "high", "xhigh", "max"],
              }} }} }}),
              patchSessionInfo: (_sid, patch) => {{ if (patch.reasoning_effort) state.thinking = patch.reasoning_effort; }},
              sessionLaunchFailed: () => false,
              getSending: () => state.sending,
              setSending: (value) => {{ state.sending = value; }},
              getCurrentRunning: () => false,
              setCurrentRunning: noop, setTurnOpen: noop,
              getStagedAttachments: () => [], normalizedStagedAttachments: () => [],
              setSelectedSessionPendingAttachment: noop, setAttachCount: noop,
              syncAttachButtonState: noop, syncQueueSubmitState: noop,
              syncRecoveryUiForSession: noop, confirmAction: async () => false,
              api: async (_path, options) => {{
                if (options.body.seq) state.keys.push({{ path: _path, ...options.body }});
                else state.sent.push(options.body.text);
                return {{}};
              }},
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
            textarea.value = "/model claude";
            textarea.dispatch("input");
            if (modelPicker.children.length !== 1 || modelPicker.children[0].textContent !== "anthropic/claude-sonnet-4") throw new Error("model filter mismatch");
            textarea.dispatch("keydown", {{ key: "Escape", preventDefault() {{}} }});
            if (modelPicker.style.display !== "none") throw new Error("Escape did not dismiss picker");
            textarea.value = "/model";
            textarea.dispatch("input");
            textarea.dispatch("keydown", {{ key: "ArrowDown", preventDefault() {{}} }});
            textarea.dispatch("keydown", {{ key: "Enter", preventDefault() {{}} }});
            await Promise.resolve();
            if (state.sent[0] !== "/model openai/gpt-5") throw new Error("selected provider/model id was not sent");
            textarea.value = "/thinking";
            textarea.dispatch("input");
            if (modelPicker.style.display !== "block" || modelPicker.children.length !== 7) throw new Error("Pi /thinking did not open all thinking levels");
            textarea.dispatch("keydown", {{ key: "Escape", preventDefault() {{}} }});
            if (modelPicker.style.display !== "none") throw new Error("Escape did not dismiss thinking picker");
            textarea.value = "/thinking";
            textarea.dispatch("input");
            for (let index = 0; index < 6; index += 1) textarea.dispatch("keydown", {{ key: "ArrowDown", preventDefault() {{}} }});
            textarea.dispatch("keydown", {{ key: "Enter", preventDefault() {{}} }});
            await Promise.resolve();
            await new Promise((resolve) => setTimeout(resolve, 0));
            if (state.keys.length !== 1 || state.keys[0].path !== "/api/sessions/sid/keys" || state.keys[0].seq !== "\\\\x1b[Z" || state.keys[0].count !== 2) throw new Error("thinking picker did not inject the exact cycle distance");
            if (state.thinking !== "max") throw new Error("thinking picker did not update the known session level");
            state.backend = "codex";
            textarea.value = "/model";
            textarea.dispatch("input");
            if (modelPicker.style.display !== "none") throw new Error("non-Pi session opened picker");
            process.stdout.write(JSON.stringify({{ sent: state.sent, keys: state.keys, selected: modelPicker.style.display }}));
            """
        ) + "\n})();"
        result = subprocess.run(["node", "-e", script], check=False, capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(result.stderr or result.stdout)
        self.assertEqual(
            json.loads(result.stdout),
            {
                "sent": ["/model openai/gpt-5"],
                "keys": [{"path": "/api/sessions/sid/keys", "seq": "\\x1b[Z", "count": 2}],
                "selected": "none",
            },
        )


if __name__ == "__main__":
    unittest.main()
