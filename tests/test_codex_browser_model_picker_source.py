import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_COMPOSER_JS = ROOT / "codoxear" / "static" / "app_composer.js"


class TestCodexBrowserModelPicker(unittest.TestCase):
    def test_codex_picker_projects_read_only_provider_availability_and_active_model(self) -> None:
        script = textwrap.dedent(
            r"""
            (async () => {
              const fs = require("fs");
              const vm = require("vm");
              class Node {
                constructor(tagName = "div") {
                  this.tagName = tagName;
                  this.listeners = {};
                  this.style = {};
                  this.classList = { toggle: () => {} };
                  this.attributes = {};
                  this.children = [];
                  this.value = "";
                  this.scrollHeight = 32;
                  this.disabled = false;
                  this.textContent = "";
                  this.className = "";
                  this.title = "";
                }
                set innerHTML(value) { this.children = []; this._innerHTML = value; }
                get innerHTML() { return this._innerHTML || ""; }
                addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
                removeEventListener() {}
                dispatch(type, event = {}) { for (const fn of this.listeners[type] || []) fn(event); }
                setAttribute(name, value) { this.attributes[name] = String(value); }
                removeAttribute(name) { delete this.attributes[name]; }
                appendChild(child) { this.children.push(child); return child; }
                focus() {}
                blur() {}
              }
              const document = { createElement: (tag) => new Node(tag), activeElement: null };
              const context = { window: {}, document, console, Date, Set, Object, String, Number, Promise };
              vm.createContext(context);
              vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), context);
              const nodes = Array.from({ length: 10 }, () => new Node());
              const [form, textarea, msgPh, sendBtn, sendChoice, sendChoiceBackdrop, nowBtn, laterBtn, cancelBtn, modelPicker] = nodes;
              form.requestSubmit = () => {};
              const settingsCalls = [];
              const noop = () => {};
              context.window.CodoxearComposer.createComposerController({
                form, textarea, msgPh, sendBtn, sendChoice, sendChoiceBackdrop,
                sendChoiceNowBtn: nowBtn, sendChoiceLaterBtn: laterBtn, sendChoiceCancelBtn: cancelBtn, modelPicker,
                getSelected: () => "codex-live",
                getSessionInfo: () => ({
                  agent_backend: "codex", model_provider: "openai", preferred_auth_method: "chatgpt",
                  provider_choice: "chatgpt", model: "gpt-current", reasoning_effort: "high",
                  slash_commands: [{ name: "model" }, { name: "effort" }],
                }),
                getNewSessionDefaults: () => ({ backends: { codex: {
                  provider_choices: ["chatgpt", "openai-api", "custom"],
                  provider_models: {
                    chatgpt: ["gpt-current", "gpt-small"],
                    "openai-api": ["gpt-other"],
                    custom: ["custom-1"],
                  },
                  models: ["gpt-current", "gpt-small"],
                  reasoning_efforts: ["low", "high"], reasoning_efforts_by_model: {},
                } } }),
                patchSessionInfo: noop, sessionLaunchFailed: () => false,
                getSending: () => false, setSending: noop, getCurrentRunning: () => false, setCurrentRunning: noop,
                setTurnOpen: noop, resetTypingStats: noop, getStagedAttachments: () => [], normalizedStagedAttachments: () => [],
                setSelectedSessionPendingAttachment: noop, setAttachCount: noop, syncAttachButtonState: noop,
                syncQueueSubmitState: noop, syncRecoveryUiForSession: noop, confirmAction: async () => false,
                api: async (path, options) => { settingsCalls.push({ path, body: options.body }); return {}; },
                setToast: noop, handleAppAuthLoss: noop, refreshSessions: async () => [], sendText: async () => true,
                setPollFastUntilMs: noop, kickPoll: noop, isTranscriptRenewalCommand: () => false,
                nextLocalEchoId: () => "local", renderedAtLiveTail: () => true, clearTranscriptDom: noop,
                clearRenderedTranscriptRange: noop, setOlderState: noop, getSessionTranscriptSlot: () => ({ epoch: 0 }),
                addPendingUser: noop, appendEvent: noop, deleteTailCache: noop, beginTranscriptRenewal: noop,
                clearLiveCursor: noop, invalidateOlderLoad: noop, renderPendingTranscriptSlot: noop,
                dropPendingUser: noop, removePendingUserRow: noop, hasPendingForSession: () => false,
                enqueueComposerText: async () => true, prepareModalOpen: noop, afterModalVisibilityChanged: noop,
                restoreModalFocus: noop, storageGetItem: () => "", storageSetItem: noop, storageRemoveItem: noop,
                getComputedStyle: () => ({ minHeight: "32px" }), requestFrame: noop,
                activeElement: () => textarea, isHTMLElement: () => true, now: () => 1,
              });
              textarea.value = "/model";
              textarea.dispatch("input");
              const providerRows = modelPicker.children.filter((child) => child.className === "modelPickerProvider");
              const options = modelPicker.children.filter((child) => child.tagName === "button");
              if (providerRows.length !== 3) throw new Error("configured providers were not rendered");
              if (providerRows.map((row) => row.textContent).join("|") !== "chatgpt — active provider|openai-api — unavailable on current provider|custom — unavailable on current provider") throw new Error("provider availability labels are wrong");
              if (providerRows[0].attributes["data-active"] !== "true") throw new Error("active provider was not marked");
              if (providerRows.slice(1).some((row) => row.tagName !== "div" || typeof row.onclick === "function")) throw new Error("provider labels became provider-switching controls");
              if (providerRows[1].title !== "Model provider changes require starting a new Codex session." || providerRows[2].title !== "Model provider changes require starting a new Codex session.") throw new Error("inactive provider models lack the unavailable tooltip");
              if (options.map((option) => option.textContent).join("|") !== "gpt-current — current model|gpt-small") throw new Error("inactive-provider models leaked into picker or active model lacks indicator");
              if (options.some((option) => option.textContent.includes("gpt-other") || option.textContent.includes("custom-1"))) throw new Error("inactive provider model rendered as selectable");
              options[1].onclick();
              await new Promise((resolve) => setTimeout(resolve, 0));
              textarea.value = "/effort low";
              textarea.dispatch("input");
              const effortOptions = modelPicker.children.filter((child) => child.tagName === "button");
              if (effortOptions.length !== 1 || effortOptions[0].textContent !== "low") throw new Error("Codex effort picker did not render the requested level");
              effortOptions[0].onclick();
              await new Promise((resolve) => setTimeout(resolve, 0));
              if (JSON.stringify(settingsCalls) !== JSON.stringify([
                { path: "/api/sessions/codex-live/settings", body: { model: "gpt-small" } },
                { path: "/api/sessions/codex-live/settings", body: { effort: "low" } },
              ])) throw new Error("Codex picker settings did not use typed model- and effort-only payloads");
              process.stdout.write(JSON.stringify({ providers: providerRows.map((row) => row.textContent), settings: settingsCalls }));
            })();
            """
        )
        result = subprocess.run(
            ["node", "-e", script, str(APP_COMPOSER_JS)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise AssertionError(result.stderr or result.stdout)
        self.assertEqual(
            json.loads(result.stdout),
            {
                "providers": [
                    "chatgpt — active provider",
                    "openai-api — unavailable on current provider",
                    "custom — unavailable on current provider",
                ],
                "settings": [
                    {"path": "/api/sessions/codex-live/settings", "body": {"model": "gpt-small"}},
                    {"path": "/api/sessions/codex-live/settings", "body": {"effort": "low"}},
                ],
            },
        )


if __name__ == "__main__":
    unittest.main()
