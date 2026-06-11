import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"
SERVER_PY = Path(__file__).resolve().parents[1] / "codoxear" / "server.py"


class TestClaudeBackendSource(unittest.TestCase):
    def test_frontend_exposes_cc_backend(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('if (raw === "cc" || raw === "claude" || raw === "claude-code") return "cc";', source)
        self.assertIn('if (backend === "cc") return "Claude";', source)
        self.assertIn('function emptyCcLaunchDefaults(seed = {})', source)
        self.assertIn('["low", "medium", "high", "xhigh", "max"]', source)
        self.assertIn('for (const backend of ["codex", "pi", "cc"])', source)
        self.assertIn('if (backend === "cc") return { model_provider: null, preferred_auth_method: null };', source)
        self.assertIn('newSessionProviderField.style.display = hasProviders ? "" : "none";', source)

    def test_server_exposes_cc_launch_contract(self) -> None:
        source = SERVER_PY.read_text(encoding="utf-8")
        self.assertIn('CC_SETTINGS_PATH = CC_HOME / "settings.json"', source)
        self.assertIn('def _read_cc_launch_defaults()', source)
        self.assertIn('"cc": cc', source)
        self.assertIn('codex_args.extend(["--dangerously-skip-permissions"])', source)
        self.assertIn('codex_args.extend(["--effort", reasoning_effort])', source)
        self.assertIn('codex_args.extend(["--resume", resume_id])', source)
        self.assertIn('env.setdefault("CLAUDE_CONFIG_DIR", str(CC_HOME))', source)


if __name__ == "__main__":
    unittest.main()
