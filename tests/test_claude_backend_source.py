import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"
SERVER_PY = Path(__file__).resolve().parents[1] / "codoxear" / "server.py"
BACKEND_LAUNCH_PY = Path(__file__).resolve().parents[1] / "codoxear" / "backend_launch.py"
LAUNCH_CONFIG_PY = Path(__file__).resolve().parents[1] / "codoxear" / "launch_config.py"


class TestClaudeBackendSource(unittest.TestCase):
    def test_frontend_exposes_cc_backend(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('if (raw === "cc" || raw === "claude" || raw === "claude-code") return "cc";', source)
        self.assertIn('if (backend === "cc") return "Claude";', source)
        self.assertIn('function emptyCcLaunchDefaults(seed = {})', source)
        self.assertIn('["low", "medium", "high", "xhigh", "max"]', source)
        self.assertIn('for (const backend of ["codex", "pi", "cc"])', source)
        self.assertIn('if (backend === "cc") return { model_provider: null, preferred_auth_method: null };', source)
        self.assertIn('newSessionModelLabel.textContent = hasProviders ? "Provider / model" : "Model";', source)
        self.assertNotIn('id: "newSessionProviderBtn"', source)

    def test_server_exposes_cc_launch_contract(self) -> None:
        server_source = SERVER_PY.read_text(encoding="utf-8")
        launch_source = BACKEND_LAUNCH_PY.read_text(encoding="utf-8")
        config_source = LAUNCH_CONFIG_PY.read_text(encoding="utf-8")
        self.assertIn('CC_SETTINGS_PATH = CC_HOME / "settings.json"', server_source)
        self.assertIn('def _read_cc_launch_defaults()', server_source)
        self.assertIn('"cc": cc', config_source)
        self.assertIn('args = ["--dangerously-skip-permissions"]', launch_source)
        self.assertIn('args.extend(["--effort", reasoning_effort])', launch_source)
        self.assertIn('return ["--resume", resume_id]', launch_source)
        self.assertIn('"cc": "CLAUDE_CONFIG_DIR"', launch_source)


if __name__ == "__main__":
    unittest.main()
