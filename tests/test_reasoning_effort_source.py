import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"
SERVER_PY = Path(__file__).resolve().parents[1] / "codoxear" / "server.py"


class TestReasoningEffortSource(unittest.TestCase):
    def test_frontend_uses_model_specific_reasoning_effort_map(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("reasoning_efforts_by_model", source)
        self.assertIn("function reasoningChoicesForBackend(backend, { provider = null, model = null } = {})", source)
        self.assertIn("const providerKey = providerName ? `${providerName}/${modelName}` : \"\";", source)
        self.assertIn("if (providerKey && Array.isArray(map[providerKey])) rawChoices = map[providerKey];", source)
        self.assertIn("else if (Array.isArray(map[modelName])) rawChoices = map[modelName];", source)
        self.assertIn("function currentNewSessionModelForCapabilities()", source)
        self.assertIn("function currentReasoningChoices()", source)

    def test_frontend_revalidates_reasoning_when_model_changes(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("setNewSessionReasoningEffort(newSessionReasoningEffort);\n          renderNewSessionReasoningMenu();\n          applyDialogMenus();\n          newSessionModelInput.focus();", source)
        self.assertIn("newSessionModelInput.oninput = () =>", source)
        self.assertIn("renderNewSessionModelMenu();\n          setNewSessionReasoningEffort(newSessionReasoningEffort);\n          renderNewSessionReasoningMenu();", source)

    def test_server_validates_pi_reasoning_against_model_capabilities(self) -> None:
        source = SERVER_PY.read_text(encoding="utf-8")
        self.assertIn("def _read_pi_reasoning_efforts_by_model()", source)
        self.assertIn('if reasoning is False:\n        return ["off"]', source)
        self.assertIn("def _normalize_requested_pi_reasoning_effort(value: Any, *, model_provider: str | None = None, model: str | None = None)", source)
        self.assertIn("reasoning_effort must be one of {', '.join(allowed)} for Pi model", source)
        self.assertIn("def _parse_new_session_launch_request(obj: dict[str, Any]) -> NewSessionLaunchRequest", source)
        self.assertIn("model_provider=model_provider,\n            model=model,", source)


if __name__ == "__main__":
    unittest.main()
