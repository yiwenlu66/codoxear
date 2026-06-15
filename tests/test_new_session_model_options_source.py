import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_LAUNCH_JS = ROOT / "codoxear" / "static" / "app_launch.js"


def eval_pi_provider_model_runtime(query: str, *, provider_choices: list[str], reasoning_map: dict[str, list[str]], literal_model: str = "", provider_absent: bool = False) -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    launch_source = APP_LAUNCH_JS.read_text(encoding="utf-8")
    choices_start = source.index("function providerChoicesForBackend(backend)")
    choices_end = source.index("function backendSupportsFast(backend)", choices_start)
    dialog_start = source.index("function newSessionProviderChoices()")
    dialog_end = source.index("function newSessionModelOption(model", dialog_start)
    snippet = source[choices_start:choices_end] + "\n" + source[dialog_start:dialog_end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const defaults = {{
          provider_choice: null,
          provider_choices: {json.dumps(provider_choices)},
          model: "claude-haiku-4-5",
          reasoning_efforts: ["off", "minimal", "low", "medium", "high", "xhigh"],
          reasoning_efforts_by_model: {json.dumps(reasoning_map)},
        }};
        const ctx = {{
          URL,
          console,
          location: {{ origin: "http://localhost", href: "http://localhost/" }},
          window: {{
            CodoxearUrls: {{ resolveAppUrl: (path) => new URL(String(path ?? "").replace(/^[/]/, ""), "http://localhost/").toString() }},
            CodoxearStorage: {{
              getItem: () => null,
              setItem: () => true,
              removeItem: () => true,
            }},
          }},
          newSessionBackend: "pi",
          newSessionProvider: "",
          newSessionDefaults: defaults,
          newSessionModelInput: {{ value: {json.dumps(query)} }},
          newSessionLiteralModelInputValue: {json.dumps(literal_model)},
          newSessionLaunchPresetProviderAbsent: {json.dumps(provider_absent)},
          newSessionModelField: {{ classList: {{ toggle() {{}}, remove() {{}} }} }},
          newSessionStatus: {{ textContent: "" }},
          defaultsForAgentBackend: () => defaults,
          loadRememberedProviderChoice: () => "",
          loadRememberedProviderModelChoice: () => "",
          rememberProviderChoice: (_backend, value) => {{ ctx.rememberedProvider = value; }},
          setNewSessionReasoningEffort: (value) => {{ ctx.reasoningSet = value; }},
          renderNewSessionReasoningMenu: () => {{ ctx.reasoningMenuRendered = true; }},
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(launch_source)}, ctx);
        ctx.codoxearLaunch = ctx.window.CodoxearLaunch;
        vm.runInContext({json.dumps(snippet + "\nglobalThis.__test_dialog = { parseNewSessionProviderModelInput, currentReasoningChoices, setNewSessionProvider, newSessionProviderModelDisplay };\n")}, ctx);
        const parsed = ctx.__test_dialog.parseNewSessionProviderModelInput();
        const choices = ctx.__test_dialog.currentReasoningChoices();
        process.stdout.write(JSON.stringify({{
          parsed,
          choices,
          display: ctx.__test_dialog.newSessionProviderModelDisplay(parsed.model, parsed.providerChoice),
        }}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


def eval_new_session_launch_preset(session_info: dict, *, backend: str = "pi", provider_choices: list[str] | None = None) -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("function launchPresetProviderChoice(s) {")
    end = source.index("function openNewSessionDialog", start)
    snippet = source[start:end]
    choices = provider_choices if provider_choices is not None else []
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{
          newSessionBackend: {json.dumps(backend)},
          newSessionProvider: "",
          newSessionModelInput: {{ value: "" }},
          newSessionLiteralModelInputValue: "",
          newSessionLaunchPresetProviderAbsent: false,
          newSessionTmuxToggle: {{ checked: false }},
          tmuxAvailable: true,
          setBackendCalls: [],
          providerCalls: [],
          reasoningCalls: [],
          fastCalls: [],
          sessionAgentBackend: (item) => item && item.agent_backend || "codex",
          setNewSessionBackend: (value, opts) => {{ ctx.setBackendCalls.push([value, opts]); ctx.newSessionBackend = value; }},
          newSessionProviderChoices: () => {json.dumps(choices)},
          newSessionAllowsCustomProvider: () => ctx.newSessionBackend === "pi",
          setNewSessionProvider: (value) => {{ ctx.providerCalls.push(value); ctx.newSessionProvider = value; }},
          newSessionProviderModelDisplay: (model, provider = "") => provider ? `${{provider}}/${{model || "default"}}` : String(model || "default"),
          clearNewSessionProviderModelError: () => {{ ctx.errorCleared = true; }},
          setNewSessionReasoningEffort: (value) => {{ ctx.reasoningCalls.push(value); }},
          defaultsForAgentBackend: () => ({{ supports_fast: true }}),
          setNewSessionFast: (value) => {{ ctx.fastCalls.push(Boolean(value)); }},
          renderNewSessionReasoningMenu: () => {{ ctx.reasoningMenuRendered = true; }},
          renderNewSessionModelMenu: () => {{ ctx.modelMenuRendered = true; }},
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet + "\nglobalThis.__test = { launchPresetProviderChoice, applyNewSessionLaunchPreset };\n")}, ctx);
        const applied = ctx.__test.applyNewSessionLaunchPreset({json.dumps(session_info)});
        process.stdout.write(JSON.stringify({{
          applied,
          provider: ctx.newSessionProvider,
          providerCalls: ctx.providerCalls,
          modelInput: ctx.newSessionModelInput.value,
          reasoningCalls: ctx.reasoningCalls,
          fastCalls: ctx.fastCalls,
          tmuxChecked: ctx.newSessionTmuxToggle.checked,
          setBackendCalls: ctx.setBackendCalls,
          errorCleared: Boolean(ctx.errorCleared),
          reasoningMenuRendered: Boolean(ctx.reasoningMenuRendered),
          modelMenuRendered: Boolean(ctx.modelMenuRendered),
          literalModelInputValue: ctx.newSessionLiteralModelInputValue || "",
          providerAbsent: Boolean(ctx.newSessionLaunchPresetProviderAbsent),
          providerChoice: ctx.__test.launchPresetProviderChoice({json.dumps(session_info)}),
        }}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


def eval_pi_recent_providerless_selection() -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("function newSessionModelOption(model")
    end = source.index("function renderNewSessionModelMenu()", start)
    snippet = source[start:end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{
          newSessionBackend: "pi",
          newSessionProvider: "openrouter",
          newSessionModelInput: {{ value: "", focus() {{}}, setSelectionRange() {{}} }},
          newSessionModelField: {{ classList: {{ remove() {{}} }} }},
          newSessionReasoningEffort: "high",
          newSessionReasoningBtn: {{}},
          newSessionModelMenuOpen: true,
          newSessionModelMenuFocus: 0,
          newSessionLiteralModelInputValue: "",
          newSessionLaunchPresetProviderAbsent: false,
          latestSessions: [{{ agent_backend: "pi", model_provider: null, provider_choice: "openai-api", model: "anthropic/claude-3-5-sonnet" }}],
          remembered: null,
          defaultsForAgentBackend: () => ({{ model: "", models: [] }}),
          newSessionProviderChoices: () => ["openrouter"],
          defaultNewSessionProviderChoice: () => "openrouter",
          newSessionAllowsCustomProvider: () => true,
          newSessionProviderModelDisplay: (model, provider = "") => provider ? `${{provider}}/${{model || "default"}}` : String(model || "default"),
          sessionAgentBackend: (item) => item && item.agent_backend || "codex",
          sessionProviderChoice: (item) => item && item.agent_backend === "pi" ? (typeof item.model_provider === "string" ? item.model_provider.trim() : "") : "chatgpt",
          setNewSessionProvider: (value) => {{ ctx.providerSet = value; ctx.newSessionProvider = value; }},
          rememberProviderModelChoice: (backend, provider, model, opts = {{}}) => {{ ctx.remembered = {{ backend, provider, model, providerAbsent: Boolean(opts.providerAbsent) }}; }},
          currentReasoningChoices: () => ["high", "medium", "low"],
          setPickerButtonContent: () => {{}},
          renderNewSessionReasoningMenu: () => {{ ctx.reasoningMenuRendered = true; }},
          applyDialogMenus: () => {{ ctx.appliedMenus = true; }},
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet + "\nglobalThis.__test = { sessionModelOptions, selectNewSessionModel };\n")}, ctx);
        const options = ctx.__test.sessionModelOptions();
        ctx.__test.selectNewSessionModel(options[0]);
        process.stdout.write(JSON.stringify({{
          option: options[0],
          input: ctx.newSessionModelInput.value,
          remembered: ctx.remembered,
          literalModelInputValue: ctx.newSessionLiteralModelInputValue,
          providerAbsent: ctx.newSessionLaunchPresetProviderAbsent,
          providerSet: ctx.providerSet || "",
        }}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


def eval_new_session_model_options(query: str = "") -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("function newSessionModelOption(model")
    end = source.index("function setNewSessionReasoningEffort(value) {", start)
    snippet = source[start:end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{
          newSessionBackend: "codex",
          newSessionModelInput: {{ value: {json.dumps(query)} }},
          latestSessions: [
            {{ agent_backend: "codex", model: "gpt-5.4", model_provider: "openai", preferred_auth_method: "chatgpt" }},
            {{ agent_backend: "codex", model: "gpt-5.4", model_provider: "crs", preferred_auth_method: "apikey" }},
            {{ agent_backend: "pi", model: "gpt-5.4", model_provider: "macaron" }},
          ],
          defaultsForAgentBackend: () => ({{ model: "gpt-5.4-mini", models: ["gpt-5.4", "o4-mini"] }}),
          providerChoicesForBackend: () => ["chatgpt", "openai-api", "crs"],
          newSessionProviderChoices: () => ["chatgpt", "openai-api", "crs"],
          defaultNewSessionProviderChoice: () => "chatgpt",
          newSessionProviderModelDisplay: (model, providerChoice = "") => providerChoice ? `${{providerChoice}}/${{model || "default"}}` : String(model || "default"),
          sessionAgentBackend: (item) => item.agent_backend || "codex",
          sessionProviderChoice: (item) => item.model_provider === "openai" && item.preferred_auth_method === "chatgpt" ? "chatgpt" : item.model_provider,
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet + "\nglobalThis.__test_model_options = { sessionModelOptions, filteredNewSessionModelOptions };\n")}, ctx);
        process.stdout.write(JSON.stringify({{
          options: ctx.__test_model_options.sessionModelOptions(),
          filtered: ctx.__test_model_options.filteredNewSessionModelOptions(),
        }}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


class TestNewSessionModelOptionsSource(unittest.TestCase):
    def test_recent_model_options_keep_provider_choice(self) -> None:
        result = eval_new_session_model_options()
        options = result["options"]
        self.assertEqual(options[0]["displayText"], "chatgpt/gpt-5.4-mini")
        recent_pairs = {(item["providerChoice"], item["model"]) for item in options if item.get("recent")}
        self.assertIn(("chatgpt", "gpt-5.4"), recent_pairs)
        self.assertIn(("crs", "gpt-5.4"), recent_pairs)
        self.assertNotIn(("macaron", "gpt-5.4"), recent_pairs)
        configured_pairs = {(item["providerChoice"], item["model"]) for item in options if item.get("configured")}
        self.assertIn(("openai-api", "o4-mini"), configured_pairs)

    def test_model_filter_matches_provider_model_text(self) -> None:
        result = eval_new_session_model_options("crs/gpt")
        self.assertEqual(result["filtered"][0]["providerChoice"], "crs")
        self.assertEqual(result["filtered"][0]["model"], "gpt-5.4")
        self.assertEqual(result["filtered"][0]["displayText"], "crs/gpt-5.4")

    def test_source_selecting_recent_model_updates_provider(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("function newSessionModelOption(model", source)
        self.assertIn("displayText,", source)
        self.assertIn("searchText: cleanProvider ? `${cleanProvider}/${cleanModel} ${cleanModel}` : cleanModel", source)
        self.assertIn("setNewSessionProvider(item.providerChoice);", source)
        self.assertIn("const selectedProvider = item.providerAbsent ? \"\" : item.providerChoice || newSessionProvider;", source)
        self.assertIn("newSessionProviderModelDisplay(item.model || \"default\", selectedProvider)", source)
        self.assertIn("rememberProviderModelChoice(newSessionBackend, selectedProvider, item.model || \"default\", { providerAbsent: Boolean(item.providerAbsent) });", source)
        self.assertIn("item.recent ? \"Recent\" : item.configured ? \"Configured\"", source)

    def test_provider_only_selector_is_not_rendered_or_called(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('newSessionModelLabel.textContent = hasProviders ? "Provider / model" : "Model"', source)
        self.assertNotIn('id: "newSessionProviderBtn"', source)
        self.assertNotIn('id: "newSessionProviderMenu"', source)
        self.assertNotIn('text: "Provider"', source)
        self.assertNotIn("renderNewSessionProviderMenu", source)
        refresh_open_start = source.index('if (newSessionViewer.style.display === "flex") {')
        refresh_open_end = source.index("          }", refresh_open_start)
        refresh_open_block = source[refresh_open_start:refresh_open_end]
        self.assertIn("renderNewSessionModelMenu();", refresh_open_block)

    def test_pi_provider_model_input_allows_custom_provider(self) -> None:
        result = eval_pi_provider_model_runtime(
            "anthropic/claude-haiku-4-5",
            provider_choices=[],
            reasoning_map={"claude-haiku-4-5": ["off"], "occ/claude-haiku-4-5": ["off"]},
        )
        self.assertEqual(result["parsed"]["providerChoice"], "anthropic")
        self.assertEqual(result["parsed"]["model"], "claude-haiku-4-5")
        self.assertEqual(result["parsed"]["providerError"], "")
        self.assertEqual(result["display"], "anthropic/claude-haiku-4-5")
        self.assertIn("low", result["choices"])
        self.assertNotEqual(result["choices"], ["off"])

    def test_pi_custom_provider_source_paths_remain_connected(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('function newSessionAllowsCustomProvider() {\n          return newSessionBackend === "pi";\n        }', source)
        self.assertIn('options.includes(next) || (next && newSessionAllowsCustomProvider())', source)
        self.assertIn('const hasProviders = newSessionHasProviderChoices() || newSessionAllowsCustomProvider();', source)
        self.assertIn('providerChoices.includes(selectedPair.providerChoice) || newSessionAllowsCustomProvider()', source)
        self.assertIn('providerChoices.includes(provider) || (provider && newSessionAllowsCustomProvider())', source)

    def test_provider_model_error_clears_when_backend_or_input_changes(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("function clearNewSessionProviderModelError()", source)
        self.assertIn('String(newSessionStatus.textContent || "").startsWith("Provider must be one of ")', source)
        self.assertIn("if (!parsed.providerError) clearNewSessionProviderModelError();", source)
        self.assertIn("clearNewSessionProviderModelError();\n          }\n          const reasoningChoices = currentReasoningChoices();", source)

    def test_degraded_launch_defaults_are_visible_but_nonblocking(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("function newSessionDefaultsWarningText()", source)
        self.assertIn("Launch defaults degraded for ${names.join(\", \")}; using safe defaults.", source)
        self.assertIn("statusText || newSessionDefaultsWarningText()", source)
        self.assertIn('statusText.startsWith("Launch defaults degraded for ")', source)

    def test_new_session_start_button_has_inflight_guard(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("let newSessionStartBusy = false;", source)
        self.assertIn("newSessionStartBtn.onclick = async () => {\n          if (newSessionStartBusy) return;", source)
        self.assertIn("newSessionStartBusy = true;\n          newSessionStartBtn.disabled = true;", source)
        self.assertIn("} finally {\n            newSessionStartBusy = false;\n            newSessionStartBtn.disabled = false;", source)
        start = source.index("newSessionStartBtn.onclick = async () => {")
        end = source.index("        let fileViewMode", start)
        block = source[start:end]
        self.assertLess(block.index("newSessionStartBusy = true;"), block.index("await spawnSessionWithCwd("))

    def test_new_session_initial_backend_prefers_selected_session_when_no_user_choice(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        launch_source = APP_LAUNCH_JS.read_text(encoding="utf-8")
        self.assertIn("function loadRememberedBackendChoice()", source)
        self.assertIn("return codoxearLaunch.loadRememberedBackendChoice();", source)
        self.assertIn('const value = String(storageGetItem(LAST_BACKEND_KEY) || "").trim();', launch_source)
        self.assertIn('return value ? normalizeAgentBackendName(value) : "";', launch_source)
        start = source.index("function openNewSessionDialog")
        end = source.index("editPriorityRange.oninput", start)
        block = source[start:end]
        self.assertIn("const rememberedBackend = loadRememberedBackendChoice();", block)
        self.assertIn('const like = likeSession && typeof likeSession === "object" ? likeSession : null;', block)
        self.assertIn('const currentBackend = like ? sessionAgentBackend(like) : cur ? sessionAgentBackend(cur) : "";', block)
        self.assertIn("const defaultBackend = normalizeAgentBackendName(newSessionDefaults && newSessionDefaults.default_backend);", block)
        self.assertIn("const initialBackend = currentBackend || rememberedBackend || defaultBackend;", block)
        self.assertNotIn("rememberedBackend || (cur ? sessionAgentBackend(cur)", block)

    def test_new_like_this_preset_preserves_pi_custom_provider(self) -> None:
        result = eval_new_session_launch_preset(
            {
                "agent_backend": "pi",
                "cwd": "/repo",
                "model_provider": "anthropic",
                "model": "claude-haiku-4-5",
                "reasoning_effort": "low",
                "service_tier": "fast",
                "transport": "tmux",
            },
            backend="pi",
            provider_choices=[],
        )
        self.assertTrue(result["applied"])
        self.assertEqual(result["providerChoice"], "anthropic")
        self.assertEqual(result["provider"], "anthropic")
        self.assertEqual(result["providerCalls"], ["anthropic"])
        self.assertEqual(result["modelInput"], "anthropic/claude-haiku-4-5")
        self.assertEqual(result["reasoningCalls"], ["low"])
        self.assertEqual(result["fastCalls"], [True])
        self.assertTrue(result["tmuxChecked"])

    def test_new_like_this_preset_does_not_invent_pi_provider(self) -> None:
        result = eval_new_session_launch_preset(
            {
                "agent_backend": "pi",
                "cwd": "/repo",
                "model_provider": None,
                "provider_choice": "openai-api",
                "model": "claude-haiku-4-5",
                "reasoning_effort": "low",
            },
            backend="pi",
            provider_choices=[],
        )
        self.assertTrue(result["applied"])
        self.assertEqual(result["providerChoice"], "")
        self.assertEqual(result["provider"], "")
        self.assertEqual(result["providerCalls"], [])
        self.assertEqual(result["modelInput"], "claude-haiku-4-5")
        self.assertEqual(result["literalModelInputValue"], "claude-haiku-4-5")

    def test_new_like_this_preset_keeps_pi_slash_model_literal_without_provider(self) -> None:
        result = eval_new_session_launch_preset(
            {
                "agent_backend": "pi",
                "cwd": "/repo",
                "model_provider": None,
                "provider_choice": "openai-api",
                "model": "anthropic/claude-3-5-sonnet",
                "reasoning_effort": "low",
            },
            backend="pi",
            provider_choices=[],
        )
        self.assertTrue(result["applied"])
        self.assertEqual(result["providerChoice"], "")
        self.assertEqual(result["providerCalls"], [])
        self.assertEqual(result["modelInput"], "anthropic/claude-3-5-sonnet")
        self.assertEqual(result["literalModelInputValue"], "anthropic/claude-3-5-sonnet")
        self.assertTrue(result["providerAbsent"])

    def test_new_like_this_pi_absent_provider_stays_absent_with_provider_defaults(self) -> None:
        result = eval_new_session_launch_preset(
            {
                "agent_backend": "pi",
                "cwd": "/repo",
                "model_provider": None,
                "provider_choice": "openai-api",
                "model": "anthropic/claude-3-5-sonnet",
                "reasoning_effort": "low",
            },
            backend="pi",
            provider_choices=["openrouter"],
        )
        self.assertTrue(result["applied"])
        self.assertEqual(result["providerChoice"], "")
        self.assertEqual(result["providerCalls"], [])
        self.assertEqual(result["modelInput"], "anthropic/claude-3-5-sonnet")
        self.assertEqual(result["literalModelInputValue"], "anthropic/claude-3-5-sonnet")
        self.assertTrue(result["providerAbsent"])

    def test_new_like_this_pi_absent_provider_without_model_resets_prefilled_default(self) -> None:
        result = eval_new_session_launch_preset(
            {
                "agent_backend": "pi",
                "cwd": "/repo",
                "model_provider": None,
                "provider_choice": "openai-api",
                "model": None,
            },
            backend="pi",
            provider_choices=["openrouter"],
        )
        self.assertTrue(result["applied"])
        self.assertEqual(result["providerChoice"], "")
        self.assertEqual(result["providerCalls"], [])
        self.assertEqual(result["modelInput"], "default")
        self.assertEqual(result["literalModelInputValue"], "default")
        self.assertTrue(result["providerAbsent"])

    def test_new_like_this_pi_present_provider_without_model_resets_prefilled_default(self) -> None:
        result = eval_new_session_launch_preset(
            {
                "agent_backend": "pi",
                "cwd": "/repo",
                "model_provider": "anthropic",
                "provider_choice": "openai-api",
                "model": None,
            },
            backend="pi",
            provider_choices=["openrouter", "anthropic"],
        )
        self.assertTrue(result["applied"])
        self.assertEqual(result["providerChoice"], "anthropic")
        self.assertEqual(result["providerCalls"], ["anthropic"])
        self.assertEqual(result["modelInput"], "anthropic/default")
        self.assertEqual(result["literalModelInputValue"], "")
        self.assertFalse(result["providerAbsent"])

    def test_new_like_this_pi_literal_absent_provider_parses_without_default_provider(self) -> None:
        result = eval_pi_provider_model_runtime(
            "anthropic/claude-3-5-sonnet",
            provider_choices=["openrouter"],
            reasoning_map={},
            literal_model="anthropic/claude-3-5-sonnet",
            provider_absent=True,
        )
        self.assertEqual(result["parsed"]["providerChoice"], "")
        self.assertEqual(result["parsed"]["model"], "anthropic/claude-3-5-sonnet")
        self.assertTrue(result["parsed"]["providerAbsent"])

    def test_providerless_pi_recent_selection_stays_providerless(self) -> None:
        result = eval_pi_recent_providerless_selection()
        self.assertTrue(result["option"]["providerAbsent"])
        self.assertEqual(result["option"]["providerChoice"], "")
        self.assertEqual(result["option"]["displayText"], "anthropic/claude-3-5-sonnet")
        self.assertEqual(result["input"], "anthropic/claude-3-5-sonnet")
        self.assertEqual(result["remembered"], {"backend": "pi", "provider": "", "model": "anthropic/claude-3-5-sonnet", "providerAbsent": True})
        self.assertEqual(result["literalModelInputValue"], "anthropic/claude-3-5-sonnet")
        self.assertTrue(result["providerAbsent"])
        self.assertEqual(result["providerSet"], "")

    def test_new_like_this_preset_maps_codex_chatgpt_provider(self) -> None:
        result = eval_new_session_launch_preset(
            {
                "agent_backend": "codex",
                "cwd": "/repo",
                "model_provider": "openai",
                "preferred_auth_method": "chatgpt",
                "model": "gpt-5.4",
                "reasoning_effort": "high",
            },
            backend="codex",
            provider_choices=["chatgpt", "openai-api"],
        )
        self.assertEqual(result["providerChoice"], "chatgpt")
        self.assertEqual(result["provider"], "chatgpt")
        self.assertEqual(result["modelInput"], "chatgpt/gpt-5.4")

    def test_new_like_this_source_is_reviewable_and_allowlisted(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('id: "diagNewLikeBtn"', source)
        self.assertIn('text: "New like this"', source)
        self.assertIn('hideDiagViewer({ restoreFocus: false });', source)
        self.assertIn('openNewSessionDialog({ likeSession: preset, statusText: "Review copied launch settings before starting.", returnFocusEl });', source)
        self.assertIn('function openNewSessionDialog({ cwd = null, statusText = "", likeSession = null, returnFocusEl = null } = {})', source)
        self.assertIn('if (like) applyNewSessionLaunchPreset(like);', source)
        self.assertIn('function launchPresetProviderChoice(s)', source)
        self.assertIn('function applyNewSessionLaunchPreset(sessionInfo)', source)
        self.assertIn('if (backend === "pi") return provider;', source)
        self.assertIn('const providerAbsent = Boolean(newSessionLaunchPresetProviderAbsent && raw && raw === newSessionLiteralModelInputValue);', source)
        self.assertIn('if (providerAbsent) providerChoice = "";', source)
        self.assertIn('if (hasProviders && raw.includes("/") && raw !== newSessionLiteralModelInputValue)', source)
        self.assertIn('newSessionLiteralModelInputValue = newSessionModelInput.value;', source)
        self.assertIn('const providerAbsent = backend === "pi" && !provider;', source)
        self.assertIn('if (model || providerAbsent || acceptsProvider) {', source)
        self.assertIn('newSessionModelInput.value = newSessionProviderModelDisplay(model || "default", acceptsProvider ? provider : "");', source)
        self.assertIn('const providerChoice = String(parsedProviderModel.providerAbsent ? "" : parsedProviderModel.providerChoice || newSessionProvider || "").trim();', source)
        self.assertIn('diagNewLikeSession = d && typeof d === "object" ? {', source)
        self.assertIn('preferred_auth_method: d.preferred_auth_method,', source)
        self.assertIn('tmux_window: d.tmux_window,', source)
        self.assertNotIn('diagNewLikeSession = d;', source)

    def test_provider_model_pair_is_remembered_per_backend(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        launch_source = APP_LAUNCH_JS.read_text(encoding="utf-8")
        self.assertIn("const codoxearLaunch = window.CodoxearLaunch;", source)
        self.assertIn('throw new Error("Codoxear launch helpers failed to load")', source)
        self.assertIn("function lastProviderModelKey(backend)", source)
        self.assertIn("return codoxearLaunch.lastProviderModelKey(backend);", source)
        self.assertIn("function loadRememberedProviderModelChoice(backend)", source)
        self.assertIn("return codoxearLaunch.loadRememberedProviderModelChoice(backend);", source)
        self.assertIn("function rememberedProviderModelAbsentChoice(value)", source)
        self.assertIn("return codoxearLaunch.rememberedProviderModelAbsentChoice(value);", source)
        self.assertIn("function rememberProviderModelChoice(backend, provider, model, options = {})", source)
        self.assertIn("return codoxearLaunch.rememberProviderModelChoice(backend, provider, model, options);", source)
        self.assertIn("codoxear.newSessionProviderModel.${normalizeAgentBackendName(backend)}", launch_source)
        self.assertIn("NO_PROVIDER_MODEL_PREFIX", launch_source)
        self.assertIn("function rememberProviderModelChoice(backend, provider, model, { providerAbsent = false } = {})", launch_source)
        self.assertIn("rememberProviderModelChoice(newSessionBackend, selectedProvider, item.model || \"default\", { providerAbsent: Boolean(item.providerAbsent) });", source)
        self.assertIn("rememberProviderModelChoice(agentBackend, providerChoice, model, { providerAbsent: Boolean(parsedProviderModel.providerAbsent) });", source)
        self.assertIn("function rememberedNewSessionProviderModelChoice()", source)
        self.assertIn("const absent = rememberedProviderModelAbsentChoice(remembered);", source)
        self.assertIn("if (absent) return absent;", source)
        self.assertIn("const rememberedPair = rememberedNewSessionProviderModelChoice();", source)
        self.assertIn("last provider/model pair for each backend", source)


if __name__ == "__main__":
    unittest.main()
