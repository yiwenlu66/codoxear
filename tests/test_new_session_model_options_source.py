import json
import subprocess
import textwrap
import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


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
        self.assertIn("const selectedProvider = item.providerChoice || newSessionProvider;", source)
        self.assertIn("newSessionProviderModelDisplay(item.model || \"default\", selectedProvider)", source)
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

    def test_provider_model_error_clears_when_backend_or_input_changes(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("function clearNewSessionProviderModelError()", source)
        self.assertIn('String(newSessionStatus.textContent || "").startsWith("Provider must be one of ")', source)
        self.assertIn("if (!parsed.providerError) clearNewSessionProviderModelError();", source)
        self.assertIn("clearNewSessionProviderModelError();\n          }\n          const reasoningChoices = currentReasoningChoices();", source)

    def test_provider_model_pair_is_remembered_per_backend(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("function lastProviderModelKey(backend)", source)
        self.assertIn("codoxear.newSessionProviderModel.${normalizeAgentBackendName(backend)}", source)
        self.assertIn("function loadRememberedProviderModelChoice(backend)", source)
        self.assertIn("function rememberProviderModelChoice(backend, provider, model)", source)
        self.assertIn("rememberProviderModelChoice(newSessionBackend, selectedProvider, item.model || \"default\");", source)
        self.assertIn("rememberProviderModelChoice(agentBackend, providerChoice, model);", source)
        self.assertIn("function rememberedNewSessionProviderModelChoice()", source)
        self.assertIn("const rememberedPair = rememberedNewSessionProviderModelChoice();", source)
        self.assertIn("last provider/model pair for each backend", source)


if __name__ == "__main__":
    unittest.main()
