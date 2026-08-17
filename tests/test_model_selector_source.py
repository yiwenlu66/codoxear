from frontend_module_loader import module_path
import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_COMPOSER_JS = module_path("app_composer.js")
APP_SESSION_CATALOG_JS = module_path("app_session_catalog.js")
APP_SESSION_STATE_JS = module_path("app_session_state.js")


class TestComposerModelPicker(unittest.TestCase):
    def test_model_and_effort_pickers_follow_backend_command_specs(self) -> None:
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
            vm.runInContext({json.dumps(APP_SESSION_CATALOG_JS.read_text(encoding="utf-8"))}, ctx);
            vm.runInContext({json.dumps(APP_SESSION_STATE_JS.read_text(encoding="utf-8"))}, ctx);
            vm.runInContext({json.dumps(source)}, ctx);
            const nodes = Array.from({{ length: 10 }}, () => new Node());
            const [form, textarea, msgPh, sendBtn, sendChoice, sendChoiceBackdrop, nowBtn, laterBtn, cancelBtn, modelPicker] = nodes;
            form.requestSubmit = () => {{ state.formSubmits += 1; }};
            const state = {{ backend: "pi", thinkingCapability: true, codexCapability: true, sent: [], settingsCalls: [], toasts: [], sending: false, formSubmits: 0, ccModel: "claude-sonnet-4-6", ccEffort: "high" }};
            const noop = () => {{}};
            const sessionState = ctx.window.CodoxearSessionState.createSessionState({{ consoleError: noop }});
            sessionState.set("selected", "sid");
            const sessionCatalog = ctx.window.CodoxearSessionCatalog.createSessionCatalog({{ consoleError: noop }});
            const sessionInfo = {{
              session_id: "sid",
              get agent_backend() {{ return state.backend; }},
              get pi_thinking_command() {{ return state.thinkingCapability; }},
              model_provider: "anthropic",
              get model() {{ return state.backend === "cc" ? state.ccModel : state.backend === "codex" ? "gpt-5.4" : "claude-sonnet-4"; }},
              get reasoning_effort() {{ return state.backend === "cc" ? state.ccEffort : "high"; }},
              get slash_commands() {{ return state.backend === "codex" && state.codexCapability ? [{{ name: "model" }}, {{ name: "effort" }}] : []; }},
            }};
            sessionCatalog.set("latestSessions", [sessionInfo]);
            sessionCatalog.set("newSessionDefaults", {{ backends: {{ pi: {{
              provider_models: {{ anthropic: ["claude-sonnet-4"], openai: ["gpt-5"] }},
              reasoning_efforts_by_model: {{ "anthropic/claude-sonnet-4": ["off", "low", "high"] }},
            }}, cc: {{ models: ["sonnet", "opus", "fable", "haiku", "best", "default", "claude-sonnet-4-6"], reasoning_efforts: ["low", "medium", "high", "xhigh", "max", "auto"], reasoning_efforts_by_model: {{ "claude-sonnet-4-6": ["low", "high", "auto"] }} }}, codex: {{ models: ["gpt-5.4", "gpt-5.4-mini"], reasoning_efforts: ["minimal", "low", "medium", "high", "xhigh", "max"], reasoning_efforts_by_model: {{ "gpt-5.4": ["low", "high", "max"] }} }} }} }});
            const controller = ctx.window.CodoxearComposer.createComposerController({{
              form, textarea, msgPh, sendBtn, sendChoice, sendChoiceBackdrop,
              sendChoiceNowBtn: nowBtn, sendChoiceLaterBtn: laterBtn, sendChoiceCancelBtn: cancelBtn,
              modelPicker,
              sessionState, sessionCatalog,
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
              api: async (path, options) => {{
                if (path.endsWith("/settings")) state.settingsCalls.push({{ path, body: options.body }});
                else state.sent.push(options.body.text);
                return {{}};
              }},
              setToast: (message) => {{ state.toasts.push(message); }}, handleAppAuthLoss: noop, refreshSessions: async () => [],
              sendText: async (text) => {{ state.sent.push(text); return true; }},
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
            if (state.sent[1] !== "/effort low") throw new Error("selected thinking level was not sent through composer path as /effort");
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
            state.thinkingCapability = false;
            textarea.value = "/thinking high";
            await form.onsubmit({{ preventDefault() {{}} }});
            if (state.sent.length !== 2) throw new Error("incapable Pi /thinking was sent");
            if (state.toasts.at(-1) !== "this session runs an older bridge — send /reload to enable /effort") throw new Error("incapable Pi /thinking did not explain how to enable it");
            state.thinkingCapability = true;
            textarea.value = "/thinking high";
            await form.onsubmit({{ preventDefault() {{}} }});
            if (state.sent[2] !== "/thinking high") throw new Error("capable Pi /thinking did not pass through");
            textarea.value = "ordinary text";
            await form.onsubmit({{ preventDefault() {{}} }});
            if (state.sent[3] !== "ordinary text") throw new Error("non-/thinking message behavior changed");
            state.backend = "cc";
            state.thinkingCapability = false;
            textarea.value = "/model hai";
            textarea.dispatch("input");
            if (modelPicker.style.display !== "block" || modelPicker.children.length !== 1 || modelPicker.children[0].textContent !== "haiku") throw new Error("CC /model did not filter aliases from launch defaults");
            const ccModelEnter = {{ key: "Enter", defaultPrevented: false, preventDefault() {{ this.defaultPrevented = true; }} }};
            textarea.dispatch("keydown", ccModelEnter);
            if (!ccModelEnter.defaultPrevented) throw new Error("CC model picker Enter was not consumed");
            await new Promise((resolve) => setTimeout(resolve, 0));
            if (state.sent[4] !== "/model haiku") throw new Error("CC model picker did not send the selected alias");
            textarea.value = "/effort";
            textarea.dispatch("input");
            if (modelPicker.style.display !== "block" || modelPicker.children.length !== 3 || modelPicker.children[0].textContent !== "high") throw new Error("CC /effort did not use the selected-model effort set");
            textarea.value = "/effort au";
            textarea.dispatch("input");
            if (modelPicker.children.length !== 1 || modelPicker.children[0].textContent !== "auto") throw new Error("CC /effort did not include auto");
            const ccEffortEnter = {{ key: "Enter", defaultPrevented: false, preventDefault() {{ this.defaultPrevented = true; }} }};
            textarea.dispatch("keydown", ccEffortEnter);
            if (!ccEffortEnter.defaultPrevented) throw new Error("CC effort picker Enter was not consumed");
            await new Promise((resolve) => setTimeout(resolve, 0));
            if (state.sent[5] !== "/effort auto") throw new Error("CC effort picker did not send /effort auto");
            state.backend = "codex";
            textarea.value = "/model mini";
            textarea.dispatch("input");
            if (modelPicker.style.display !== "block" || modelPicker.children.length !== 2 || modelPicker.children[1].textContent !== "gpt-5.4-mini") throw new Error("Codex session did not expose the active provider and advertised model picker");
            const codexModelEnter = {{ key: "Enter", defaultPrevented: false, preventDefault() {{ this.defaultPrevented = true; }} }};
            textarea.dispatch("keydown", codexModelEnter);
            await new Promise((resolve) => setTimeout(resolve, 0));
            if (!codexModelEnter.defaultPrevented || state.settingsCalls[0].path !== "/api/sessions/sid/settings" || state.settingsCalls[0].body.model !== "gpt-5.4-mini") throw new Error("Codex model picker did not use typed settings route");
            textarea.value = "/effort";
            textarea.dispatch("input");
            if (modelPicker.style.display !== "block" || modelPicker.children.length !== 3 || modelPicker.children[0].textContent !== "high") throw new Error("Codex effort picker did not use model-scoped levels/current-first ordering");
            textarea.value = "/effort max";
            textarea.dispatch("input");
            const codexEffortEnter = {{ key: "Enter", defaultPrevented: false, preventDefault() {{ this.defaultPrevented = true; }} }};
            textarea.dispatch("keydown", codexEffortEnter);
            await new Promise((resolve) => setTimeout(resolve, 0));
            if (state.settingsCalls[1].body.effort !== "max") throw new Error("Codex effort picker did not use typed settings route");
            if (state.sent.length !== 6) throw new Error("Codex typed settings leaked into PTY text send");
            state.codexCapability = false;
            textarea.value = "/model";
            textarea.dispatch("input");
            if (modelPicker.style.display !== "none") throw new Error("Codex session without advertised protocol capability opened picker");
            process.stdout.write(JSON.stringify({{ sent: state.sent, settingsCalls: state.settingsCalls, selected: modelPicker.style.display }}));
            """
        ) + "\n})();"
        result = subprocess.run(["node", "-e", script], check=False, capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(result.stderr or result.stdout)
        self.assertEqual(
            json.loads(result.stdout),
            {
                "sent": ["/model openai/gpt-5", "/effort low", "/thinking high", "ordinary text", "/model haiku", "/effort auto"],
                "settingsCalls": [
                    {"path": "/api/sessions/sid/settings", "body": {"model": "gpt-5.4-mini"}},
                    {"path": "/api/sessions/sid/settings", "body": {"effort": "max"}},
                ],
                "selected": "none",
            },
        )


if __name__ == "__main__":
    unittest.main()
