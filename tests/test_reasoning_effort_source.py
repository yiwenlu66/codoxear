import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_LAUNCH_JS = ROOT / "codoxear" / "static" / "app_launch.js"
APP_NEW_SESSION_JS = ROOT / "codoxear" / "static" / "app_new_session.js"
APP_CSS = ROOT / "codoxear" / "static" / "app.css"
SERVER_PY = ROOT / "codoxear" / "server.py"
LAUNCH_CONFIG_PY = ROOT / "codoxear" / "launch_config.py"
AGENT_BACKEND_PY = ROOT / "codoxear" / "agent_backend.py"


class TestReasoningEffortSource(unittest.TestCase):
    def test_frontend_uses_model_specific_reasoning_effort_map(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        launch_source = APP_LAUNCH_JS.read_text(encoding="utf-8")
        self.assertIn("reasoning_efforts_by_model", launch_source)
        self.assertIn("function reasoningChoicesForBackend(backend, defaultsSource = null, { provider = null, model = null } = {})", launch_source)
        self.assertIn("const providerKey = providerName ? `${providerName}/${modelName}` : \"\";", launch_source)
        self.assertIn("if (providerKey && Array.isArray(map[providerKey])) rawChoices = map[providerKey];", launch_source)
        self.assertIn("else if (!providerName && Array.isArray(map[modelName])) rawChoices = map[modelName];", launch_source)
        self.assertIn("function reasoningChoicesForBackend(backend, options = {})", source)
        self.assertIn("return codoxearLaunch.reasoningChoicesForBackend(backend, newSessionDefaults, options);", source)
        self.assertIn("function currentNewSessionModelForCapabilities()", source)
        self.assertIn("function currentReasoningChoices()", source)

    def test_frontend_revalidates_reasoning_when_model_changes(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        module_source = APP_NEW_SESSION_JS.read_text(encoding="utf-8")
        self.assertIn("setNewSessionReasoningEffort(reasoningEffort());\n      renderReasoningMenu();\n      applyDialogMenus();\n      modelInput.focus();", module_source)
        self.assertIn("newSessionModelInput.oninput = () =>", source)
        self.assertIn("renderNewSessionModelMenu();\n          setNewSessionReasoningEffort(newSessionReasoningEffort);\n          renderNewSessionReasoningMenu();", source)

    def test_sidebar_reasoning_effort_markers_cover_supported_values(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        match = re.search(
            r"const REASONING_EFFORT_MARKERS = Object\.freeze\(\{(?P<body>.*?)\n      \}\);",
            source,
            re.S,
        )
        self.assertIsNotNone(match)
        body = match.group("body") if match else ""
        markers = dict(re.findall(r'\n\s+([a-z]+): "([^"]*)",', body))
        self.assertEqual(
            markers,
            {
                "xhigh": "X",
                "high": "H",
                "medium": "M",
                "low": "L",
                "max": "M+",
                "minimal": "m",
                "off": "–",
            },
        )
        self.assertIn('return REASONING_EFFORT_MARKERS[effortTxt] || "";', source)
        self.assertIn("const effortMark = reasoningEffortMarker(effortTxt);", source)
        self.assertIn("class: `effortMark effort-${effortTxt}`", source)
        self.assertIn("title: `reasoning effort ${effortTxt}`", source)

    def test_sidebar_reasoning_effort_css_covers_new_values(self) -> None:
        css = APP_CSS.read_text(encoding="utf-8")
        for effort in ("max", "minimal", "off"):
            self.assertRegex(css, rf"\.effortMark\.effort-{effort}\s*\{{\s*color: #[0-9a-fA-F]{{6}};\s*\}}")

    def test_server_validates_pi_reasoning_against_model_capabilities(self) -> None:
        server_source = SERVER_PY.read_text(encoding="utf-8")
        source = LAUNCH_CONFIG_PY.read_text(encoding="utf-8")
        backend_source = AGENT_BACKEND_PY.read_text(encoding="utf-8")
        self.assertIn("def _read_pi_reasoning_efforts_by_model()", server_source)
        self.assertIn("def read_pi_reasoning_efforts_by_model(paths: LaunchConfigPaths)", source)
        self.assertIn('if reasoning is False:\n        return ["off"]', source)
        self.assertIn("def normalize_requested_pi_reasoning_effort(\n    value: Any,", source)
        self.assertIn("reasoning_efforts_by_model: Mapping[str, list[str]] | None = None", source)
        self.assertIn("reasoning_effort must be one of {', '.join(allowed)} for Pi model", source)
        self.assertIn("def parse_new_session_launch_request(", source)
        self.assertIn("get_agent_backend(agent_backend).normalize_launch_request_options(", source)
        self.assertIn("model_provider=model_provider,\n                model=model,\n                reasoning_efforts_by_model=pi_launch_defaults.get", backend_source)


if __name__ == "__main__":
    unittest.main()
