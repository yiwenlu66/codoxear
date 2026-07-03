import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_LAUNCH_JS = ROOT / "codoxear" / "static" / "app_launch.js"
APP_NEW_SESSION_JS = ROOT / "codoxear" / "static" / "app_new_session.js"
APP_DISPLAY_JS = ROOT / "codoxear" / "static" / "app_display.js"
APP_DIAGNOSTICS_JS = ROOT / "codoxear" / "static" / "app_diagnostics.js"


def _run_node(js: str) -> dict:
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(proc.stdout)


def _new_session_module_js() -> str:
    launch_source = APP_LAUNCH_JS.read_text(encoding="utf-8")
    display_source = APP_DISPLAY_JS.read_text(encoding="utf-8")
    new_session_source = APP_NEW_SESSION_JS.read_text(encoding="utf-8")
    return textwrap.dedent(
        f"""
        const vm = require("vm");
        const storageWrites = [];
        const localStorage = {{
          data: new Map(),
          getItem(key) {{ return this.data.has(String(key)) ? this.data.get(String(key)) : null; }},
          setItem(key, value) {{ this.data.set(String(key), String(value)); storageWrites.push([String(key), String(value)]); }},
          removeItem(key) {{ this.data.delete(String(key)); }},
        }};
        const ctx = {{
          URL,
          console,
          window: {{
            CodoxearUrls: {{ resolveAppUrl: (path) => String(path || "") }},
            CodoxearStorage: localStorage,
          }},
          storageWrites,
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(launch_source)}, ctx, {{ filename: "app_launch.js" }});
        vm.runInContext({json.dumps(display_source)}, ctx, {{ filename: "app_display.js" }});
        vm.runInContext({json.dumps(new_session_source)}, ctx, {{ filename: "app_new_session.js" }});
        """
    )


def _basic_controller_options(js: str) -> str:
    """Extra option stubs every controller instantiation needs."""
    return js


# Stubs for the pass-2 controller dependencies (cwd/resume/worktree/tmux UI).
# Injected into every test fixture so existing provider/model assertions keep
# working without repeating the full DOM surface at each call site.
_NEW_SESSION_CONTROLLER_DEP_STUBS = (
    "cwdInput: { value: \"\" },\n"
    "          cwdMenu: { innerHTML: \"\" },\n"
    "          cwdField: { classList: { toggle() {}, remove() {} } },\n"
    "          cwdHint: { classList: { toggle() {} } },\n"
    "          nameInput: { value: \"\" },\n"
    "          recentCwds: () => [],\n"
    "          cwdMenuFocus: () => -1,\n"
    "          assignCwdMenuFocus: () => {},\n"
    "          closeCwdMenu: () => {},\n"
    "          el: () => ({ appendChild() {} }),\n"
    "          resumeMenu: { innerHTML: \"\" },\n"
    "          resumeBtn: {},\n"
    "          closeResumeMenu: () => {},\n"
    "          fetchResumeCandidates: async () => ({ sessions: [] }),\n"
    "          tmuxToggle: {},\n"
    "          tmuxField: { style: {} },\n"
    "          worktreeToggle: {},\n"
    "          worktreeInput: { value: \"\" },\n"
    "          worktreeField: { style: {} },\n"
    "          startBtn: {},\n"
)


def eval_pi_provider_model_runtime(query: str, *, provider_choices: list[str], reasoning_map: dict[str, list[str]], literal_model: str = "", provider_absent: bool = False) -> dict:
    js = _new_session_module_js() + textwrap.dedent(
        f"""
        const defaults = {{
          provider_choice: null,
          provider_choices: {json.dumps(provider_choices)},
          model: "claude-haiku-4-5",
          reasoning_efforts: ["off", "minimal", "low", "medium", "high", "xhigh"],
          reasoning_efforts_by_model: {json.dumps(reasoning_map)},
        }};
        let backend = "pi";
        let provider = "";
        let literalModel = {json.dumps(literal_model)};
        let presetAbsent = {json.dumps(provider_absent)};
        const modelInput = {{ value: {json.dumps(query)} }};
        const controller = ctx.window.CodoxearNewSession.createNewSessionController({{
          backend: () => backend,
          provider: () => provider,
          reasoningEffort: () => "high",
          literalModelInputValue: () => literalModel,
          launchPresetProviderAbsent: () => presetAbsent,
          defaultsSource: () => defaults,
          latestSessions: () => [],
          tmuxAvailable: () => true,
          assignProvider: (v) => {{ provider = v; }},
          assignReasoningEffort: () => {{}},
          assignLiteralModelInputValue: (v) => {{ literalModel = v; }},
          assignLaunchPresetProviderAbsent: (v) => {{ presetAbsent = Boolean(v); }},
          modelInput,
          modelField: {{ classList: {{ toggle() {{}}, remove() {{}} }} }},
          status: {{ textContent: "" }},
          reasoningBtn: {{ innerHTML: "", appendChild() {{}} }},
          setPickerButtonContent: () => {{}},
          renderReasoningMenu: () => {{}},
          renderModelMenu: () => {{}},
          setFast: () => {{}},
          setBackend: () => {{}},
          setTmuxChecked: () => {{}},
          applyDialogMenus: () => {{}},
          closeModelMenu: () => {{}},
          {_NEW_SESSION_CONTROLLER_DEP_STUBS}
        }});
        const parsed = controller.parseNewSessionProviderModelInput();
        const choices = controller.currentReasoningChoices();
        process.stdout.write(JSON.stringify({{
          parsed,
          choices,
          display: controller.newSessionProviderModelDisplay(parsed.model, parsed.providerChoice),
        }}));
        """
    )
    return _run_node(js)


def eval_new_session_launch_preset(session_info: dict, *, backend: str = "pi", provider_choices: list[str] | None = None) -> dict:
    choices = provider_choices if provider_choices is not None else []
    js = _new_session_module_js() + textwrap.dedent(
        f"""
        const defaults = {{
          provider_choice: null,
          provider_choices: {json.dumps(choices)},
          model: null,
          models: [],
          reasoning_efforts: ["off", "minimal", "low", "medium", "high", "xhigh"],
          reasoning_efforts_by_model: {{}},
          supports_fast: true,
        }};
        let currentBackend = {json.dumps(backend)};
        let provider = "";
        let literalModel = "";
        let presetAbsent = false;
        let reasoningValue = "";
        const modelInput = {{ value: "" }};
        const setBackendCalls = [];
        const providerCalls = [];
        const reasoningCalls = [];
        const fastCalls = [];
        const classRemoveCalls = [];
        let reasoningMenuRendered = 0;
        let modelMenuRendered = 0;
        let tmuxChecked = null;
        const controller = ctx.window.CodoxearNewSession.createNewSessionController({{
          backend: () => currentBackend,
          provider: () => provider,
          reasoningEffort: () => reasoningValue,
          literalModelInputValue: () => literalModel,
          launchPresetProviderAbsent: () => presetAbsent,
          defaultsSource: () => defaults,
          latestSessions: () => [],
          tmuxAvailable: () => true,
          assignProvider: (v) => {{ providerCalls.push(v); provider = v; }},
          assignReasoningEffort: (v) => {{ reasoningValue = v; reasoningCalls.push(v); }},
          assignLiteralModelInputValue: (v) => {{ literalModel = v; }},
          assignLaunchPresetProviderAbsent: (v) => {{ presetAbsent = Boolean(v); }},
          modelInput,
          modelField: {{ classList: {{ toggle() {{}}, remove(name) {{ classRemoveCalls.push(name); }} }} }},
          status: {{ textContent: "" }},
          reasoningBtn: {{ innerHTML: "", appendChild() {{}} }},
          setPickerButtonContent: () => {{}},
          renderReasoningMenu: () => {{ reasoningMenuRendered += 1; }},
          renderModelMenu: () => {{ modelMenuRendered += 1; }},
          setFast: (v) => {{ fastCalls.push(Boolean(v)); }},
          setBackend: (v, opts) => {{ setBackendCalls.push([v, opts]); currentBackend = v; }},
          setTmuxChecked: (v) => {{ tmuxChecked = v; }},
          applyDialogMenus: () => {{}},
          closeModelMenu: () => {{}},
          {_NEW_SESSION_CONTROLLER_DEP_STUBS}
        }});
        const applied = controller.applyNewSessionLaunchPreset({json.dumps(session_info)});
        process.stdout.write(JSON.stringify({{
          applied,
          provider,
          providerCalls,
          modelInput: modelInput.value,
          reasoningCalls,
          finalReasoning: reasoningValue,
          fastCalls,
          tmuxChecked,
          setBackendCalls,
          errorCleared: classRemoveCalls.includes("error"),
          reasoningMenuRendered: reasoningMenuRendered > 0,
          modelMenuRendered: modelMenuRendered > 0,
          literalModelInputValue: literalModel,
          providerAbsent: presetAbsent,
          providerChoice: controller.launchPresetProviderChoice({json.dumps(session_info)}),
        }}));
        """
    )
    return _run_node(js)


def eval_pi_recent_providerless_selection() -> dict:
    js = _new_session_module_js() + textwrap.dedent(
        """
        const defaults = { provider_choice: null, provider_choices: ["openrouter"], model: "", models: [] };
        let backend = "pi";
        let provider = "openrouter";
        let literalModel = "";
        let presetAbsent = false;
        let reasoningValue = "high";
        const modelInput = { value: "", focus() {}, setSelectionRange() {} };
        const providerCalls = [];
        const controller = ctx.window.CodoxearNewSession.createNewSessionController({
          backend: () => backend,
          provider: () => provider,
          reasoningEffort: () => reasoningValue,
          literalModelInputValue: () => literalModel,
          launchPresetProviderAbsent: () => presetAbsent,
          defaultsSource: () => defaults,
          latestSessions: () => [
            { agent_backend: "pi", model_provider: null, provider_choice: "openai-api", model: "anthropic/claude-3-5-sonnet" },
          ],
          tmuxAvailable: () => true,
          assignProvider: (v) => { providerCalls.push(v); provider = v; },
          assignReasoningEffort: (v) => { reasoningValue = v; },
          assignLiteralModelInputValue: (v) => { literalModel = v; },
          assignLaunchPresetProviderAbsent: (v) => { presetAbsent = Boolean(v); },
          modelInput,
          modelField: { classList: { toggle() {}, remove() {} } },
          status: { textContent: "" },
          reasoningBtn: { innerHTML: "", appendChild() {} },
          setPickerButtonContent: () => {},
          renderReasoningMenu: () => {},
          renderModelMenu: () => {},
          setFast: () => {},
          setBackend: () => {},
          setTmuxChecked: () => {},
          applyDialogMenus: () => {},
          closeModelMenu: () => {},
__DEP_STUBS__        });
        const options = controller.sessionModelOptions();
        controller.selectNewSessionModel(options[0]);
        const rememberedKey = ctx.storageWrites.find(([key]) => key.endsWith(".newSessionProviderModel.pi")) || [null, null];
        process.stdout.write(JSON.stringify({
          option: options[0],
          input: modelInput.value,
          rememberedValue: rememberedKey[1],
          literalModelInputValue: literalModel,
          providerAbsent: presetAbsent,
          providerSet: providerCalls.length ? providerCalls[providerCalls.length - 1] : "",
        }));
        """
    ).replace("__DEP_STUBS__", _NEW_SESSION_CONTROLLER_DEP_STUBS)
    return _run_node(js)


def eval_new_session_model_options(query: str = "") -> dict:
    js = _new_session_module_js() + textwrap.dedent(
        f"""
        const defaults = {{ model: "gpt-5.4-mini", models: ["gpt-5.4", "o4-mini"], model_providers: ["chatgpt", "openai-api", "crs"] }};
        let backend = "codex";
        let provider = "chatgpt";
        const modelInput = {{ value: {json.dumps(query)} }};
        const controller = ctx.window.CodoxearNewSession.createNewSessionController({{
          backend: () => backend,
          provider: () => provider,
          reasoningEffort: () => "high",
          literalModelInputValue: () => "",
          launchPresetProviderAbsent: () => false,
          defaultsSource: () => defaults,
          latestSessions: () => [
            {{ agent_backend: "codex", model: "gpt-5.4", model_provider: "openai", preferred_auth_method: "chatgpt" }},
            {{ agent_backend: "codex", model: "gpt-5.4", model_provider: "crs", preferred_auth_method: "apikey" }},
            {{ agent_backend: "pi", model: "gpt-5.4", model_provider: "macaron" }},
          ],
          tmuxAvailable: () => true,
          assignProvider: (v) => {{ provider = v; }},
          assignReasoningEffort: () => {{}},
          assignLiteralModelInputValue: () => {{}},
          assignLaunchPresetProviderAbsent: () => {{}},
          modelInput,
          modelField: {{ classList: {{ toggle() {{}}, remove() {{}} }} }},
          status: {{ textContent: "" }},
          reasoningBtn: {{ innerHTML: "", appendChild() {{}} }},
          setPickerButtonContent: () => {{}},
          renderReasoningMenu: () => {{}},
          renderModelMenu: () => {{}},
          setFast: () => {{}},
          setBackend: () => {{}},
          setTmuxChecked: () => {{}},
          applyDialogMenus: () => {{}},
          closeModelMenu: () => {{}},
          {_NEW_SESSION_CONTROLLER_DEP_STUBS}
        }});
        process.stdout.write(JSON.stringify({{
          options: controller.sessionModelOptions(),
          filtered: controller.filteredNewSessionModelOptions(),
        }}));
        """
    )
    return _run_node(js)


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
        source = APP_NEW_SESSION_JS.read_text(encoding="utf-8")
        self.assertIn("function newSessionModelOption(model", source)
        self.assertIn("displayText,", source)
        self.assertIn("searchText: cleanProvider ? `${cleanProvider}/${cleanModel} ${cleanModel}` : cleanModel", source)
        self.assertIn("setNewSessionProvider(item.providerChoice);", source)
        self.assertIn('const selectedProvider = item.providerAbsent ? "" : item.providerChoice || provider();', source)
        self.assertIn("newSessionProviderModelDisplay(item.model || \"default\", selectedProvider)", source)
        self.assertIn("return codoxearLaunch.providerModelDisplay(model, providerChoice, {", source)
        self.assertIn("codoxearLaunch.rememberProviderModelChoice(backend(), selectedProvider, item.model || \"default\", { providerAbsent: Boolean(item.providerAbsent) });", source)
        app_source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('item.recent ? "Recent" : item.configured ? "Configured"', app_source)

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
        module_source = APP_NEW_SESSION_JS.read_text(encoding="utf-8")
        app_source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('function newSessionAllowsCustomProvider() {\n      return backend() === "pi";\n    }', module_source)
        self.assertIn('options.includes(next) || (next && newSessionAllowsCustomProvider())', module_source)
        self.assertIn('const hasProviders = choices.length > 0 || allowCustomProvider;', module_source)
        self.assertIn('providerChoices.includes(prov) || (prov && newSessionAllowsCustomProvider())', module_source)
        self.assertIn('const hasProviders = newSessionHasProviderChoices() || newSessionAllowsCustomProvider();', app_source)
        self.assertIn('providerChoices.includes(selectedPair.providerChoice) || newSessionAllowsCustomProvider()', app_source)

    def test_provider_model_error_clears_when_backend_or_input_changes(self) -> None:
        source = APP_NEW_SESSION_JS.read_text(encoding="utf-8")
        self.assertIn("function clearNewSessionProviderModelError()", source)
        self.assertIn('String(status.textContent || "").startsWith("Provider must be one of ")', source)
        self.assertIn("if (!parsed.providerError) clearNewSessionProviderModelError();", source)
        app_source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("clearNewSessionProviderModelError();\n          }\n          const reasoningChoices = currentReasoningChoices();", app_source)

    def test_degraded_launch_defaults_are_visible_but_nonblocking(self) -> None:
        source = APP_NEW_SESSION_JS.read_text(encoding="utf-8")
        self.assertIn("function newSessionDefaultsWarningText()", source)
        self.assertIn("Launch defaults degraded for ${names.join(\", \")}; using safe defaults.", source)
        self.assertIn('statusText || newSessionDefaultsWarningText()', APP_JS.read_text(encoding="utf-8"))
        self.assertIn('statusText.startsWith("Launch defaults degraded for ")', APP_JS.read_text(encoding="utf-8"))

    def test_new_session_start_button_has_inflight_guard(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("let newSessionStartBusy = false;", source)
        self.assertIn("newSessionStartBtn.onclick = async () => {\n          if (newSessionStartBusy) return;", source)
        self.assertIn("newSessionStartBusy = true;\n          newSessionStartBtn.disabled = true;", source)
        self.assertIn("} finally {\n            newSessionStartBusy = false;\n            newSessionStartBtn.disabled = false;", source)
        start = source.index("newSessionStartBtn.onclick = async () => {")
        end = source.index("        const FILE_CANDIDATE_CACHE_TTL_MS", start)
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
        self.assertEqual(result["finalReasoning"], "low")
        self.assertEqual(result["fastCalls"], [])
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
        launch = APP_LAUNCH_JS.read_text(encoding="utf-8")
        self.assertIn("NO_PROVIDER_MODEL_PREFIX", launch)
        self.assertEqual(result["rememberedValue"], "__codoxear_no_provider__:anthropic/claude-3-5-sonnet")
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
                "service_tier": "fast",
            },
            backend="codex",
            provider_choices=["chatgpt", "openai-api"],
        )
        self.assertEqual(result["providerChoice"], "chatgpt")
        self.assertEqual(result["provider"], "chatgpt")
        self.assertEqual(result["modelInput"], "chatgpt/gpt-5.4")
        self.assertEqual(result["fastCalls"], [True])

    def test_new_like_this_source_is_reviewable_and_allowlisted(self) -> None:
        app_source = APP_JS.read_text(encoding="utf-8")
        module_source = APP_NEW_SESSION_JS.read_text(encoding="utf-8")
        diag_source = APP_DIAGNOSTICS_JS.read_text(encoding="utf-8")
        # The diag New-like-this button DOM stays in app.js; the click behavior
        # (hide without focus restore + open New Session with the preset) and the
        # preset construction from diagnostics live in the controller module.
        self.assertIn('id: "diagNewLikeBtn"', app_source)
        self.assertIn('text: "New like this"', app_source)
        self.assertIn('diagNewLikeBtn.onclick = (e) => diagController.onNewLikeClick(e);', app_source)
        self.assertIn('hide({ restoreFocus: false });', diag_source)
        self.assertIn('openNewSessionDialog({ likeSession: preset, statusText: "Review copied launch settings before starting.", returnFocusEl });', diag_source)
        self.assertIn('function openNewSessionDialog({ cwd = null, statusText = "", likeSession = null, returnFocusEl = null } = {})', app_source)
        self.assertIn('if (like) applyNewSessionLaunchPreset(like);', app_source)
        self.assertIn("function applyNewSessionLaunchPreset(sessionInfo)", module_source)
        self.assertIn("function launchPresetProviderChoice(s)", module_source)
        self.assertIn('if (backendValue === "pi") return prov;', module_source)
        self.assertIn('const providerAbsent = Boolean(launchPresetProviderAbsent() && raw && raw === literalModelInputValue());', module_source)
        self.assertIn("assignLaunchPresetProviderAbsent(true);", module_source)
        self.assertIn('if (hasProviders && raw.includes("/") && raw !== literalModelInputValue())', module_source)
        self.assertIn('assignLiteralModelInputValue(modelInput.value);', module_source)
        self.assertIn("const providerAbsent = backendValue === \"pi\" && !prov;", module_source)
        self.assertIn("if (model || providerAbsent || acceptsProvider) {", module_source)
        self.assertIn('modelInput.value = newSessionProviderModelDisplay(model || "default", acceptsProvider ? prov : "");', module_source)
        self.assertIn('diagNewLikeSession =\n        d && typeof d === "object"\n          ? {', diag_source)
        self.assertIn("preferred_auth_method: d.preferred_auth_method,", diag_source)
        self.assertIn("tmux_window: d.tmux_window,", diag_source)
        self.assertNotIn("diagNewLikeSession = d;", diag_source)
        self.assertNotIn("diagNewLikeSession = d;", app_source)

    def test_provider_model_pair_is_remembered_per_backend(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        launch_source = APP_LAUNCH_JS.read_text(encoding="utf-8")
        module_source = APP_NEW_SESSION_JS.read_text(encoding="utf-8")
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
        self.assertIn("codoxearLaunch.rememberProviderModelChoice(backend(), selectedProvider, item.model || \"default\", { providerAbsent: Boolean(item.providerAbsent) });", module_source)
        self.assertIn("function rememberedNewSessionProviderModelChoice()", source)
        module_source = APP_NEW_SESSION_JS.read_text(encoding="utf-8")
        self.assertIn("const absent = codoxearLaunch.rememberedProviderModelAbsentChoice(remembered);", module_source)
        self.assertIn("if (absent) return absent;", module_source)
        self.assertIn("const rememberedPair = rememberedNewSessionProviderModelChoice();", source)
        self.assertIn("last provider/model pair for each backend", source)


if __name__ == "__main__":
    unittest.main()
