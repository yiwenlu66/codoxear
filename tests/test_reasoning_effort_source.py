import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"
SERVER_PY = Path(__file__).resolve().parents[1] / "codoxear" / "server.py"
LAUNCH_CONFIG_PY = Path(__file__).resolve().parents[1] / "codoxear" / "launch_config.py"


class TestReasoningEffortSource(unittest.TestCase):
    def test_frontend_uses_model_specific_reasoning_effort_map(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("reasoning_efforts_by_model", source)
        self.assertIn("function reasoningChoicesForBackend(backend, { provider = null, model = null } = {})", source)
        self.assertIn("const providerKey = providerName ? `${providerName}/${modelName}` : \"\";", source)
        self.assertIn("if (providerKey && Array.isArray(map[providerKey])) rawChoices = map[providerKey];", source)
        self.assertIn("else if (!providerName && Array.isArray(map[modelName])) rawChoices = map[modelName];", source)
        self.assertIn("function currentNewSessionModelForCapabilities()", source)
        self.assertIn("function currentReasoningChoices()", source)

    def test_frontend_revalidates_reasoning_when_model_changes(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("setNewSessionReasoningEffort(newSessionReasoningEffort);\n          renderNewSessionReasoningMenu();\n          applyDialogMenus();\n          newSessionModelInput.focus();", source)
        self.assertIn("newSessionModelInput.oninput = () =>", source)
        self.assertIn("renderNewSessionModelMenu();\n          setNewSessionReasoningEffort(newSessionReasoningEffort);\n          renderNewSessionReasoningMenu();", source)

    def test_server_validates_pi_reasoning_against_model_capabilities(self) -> None:
        server_source = SERVER_PY.read_text(encoding="utf-8")
        source = LAUNCH_CONFIG_PY.read_text(encoding="utf-8")
        self.assertIn("def _read_pi_reasoning_efforts_by_model()", server_source)
        self.assertIn("def read_pi_reasoning_efforts_by_model(paths: LaunchConfigPaths)", source)
        self.assertIn('if reasoning is False:\n        return ["off"]', source)
        self.assertIn("def normalize_requested_pi_reasoning_effort(\n    value: Any,", source)
        self.assertIn("reasoning_efforts_by_model: Mapping[str, list[str]] | None = None", source)
        self.assertIn("reasoning_effort must be one of {', '.join(allowed)} for Pi model", source)
        self.assertIn("def parse_new_session_launch_request(", source)
        self.assertIn("model_provider=model_provider,\n            model=model,\n            reasoning_efforts_by_model=pi_launch_defaults.get", source)


if __name__ == "__main__":
    unittest.main()
