import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_LAUNCH_JS = ROOT / "codoxear" / "static" / "app_launch.js"
SERVER_PY = ROOT / "codoxear" / "server.py"
BROKER_PY = ROOT / "codoxear" / "broker.py"
BACKEND_LAUNCH_PY = ROOT / "codoxear" / "backend_launch.py"
AGENT_BACKEND_PY = ROOT / "codoxear" / "agent_backend.py"
LAUNCH_CONFIG_PY = ROOT / "codoxear" / "launch_config.py"


class TestClaudeBackendSource(unittest.TestCase):
    def test_frontend_exposes_cc_backend(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        launch_source = APP_LAUNCH_JS.read_text(encoding="utf-8")
        self.assertIn('if (raw === "cc" || raw === "claude" || raw === "claude-code") return "cc";', launch_source)
        self.assertIn('if (backend === "cc") return "Claude";', launch_source)
        self.assertIn('function emptyCcLaunchDefaults(seed = {})', launch_source)
        self.assertIn('["low", "medium", "high", "xhigh", "max"]', launch_source)
        self.assertIn('if (backend === "cc") return { model_provider: null, preferred_auth_method: null };', launch_source)
        self.assertIn('function normalizeAgentBackendName(value)', source)
        self.assertIn('return codoxearLaunch.normalizeAgentBackendName(value);', source)
        self.assertIn('for (const backend of ["codex", "pi", "cc"])', source)
        self.assertIn('newSessionModelLabel.textContent = hasProviders ? "Provider / model" : "Model";', source)
        self.assertNotIn('id: "newSessionProviderBtn"', source)

    def test_server_exposes_cc_launch_contract(self) -> None:
        server_source = SERVER_PY.read_text(encoding="utf-8")
        server_config_source = (SERVER_PY.parent / "server_config.py").read_text(encoding="utf-8")
        launch_source = BACKEND_LAUNCH_PY.read_text(encoding="utf-8")
        backend_source = AGENT_BACKEND_PY.read_text(encoding="utf-8")
        config_source = LAUNCH_CONFIG_PY.read_text(encoding="utf-8")
        self.assertIn('CC_SETTINGS_PATH=cc_home / "settings.json"', server_config_source)
        self.assertIn("_export_server_config(globals(), _SERVER_CONFIG)", server_source)
        self.assertIn('def _read_cc_launch_defaults()', server_source)
        self.assertIn('"cc": cc', config_source)
        self.assertIn('return get_agent_backend(agent_backend).build_launch_args(', launch_source)
        self.assertIn('class ClaudeCodeBackend(AgentBackend):', backend_source)
        self.assertIn('args = ["--dangerously-skip-permissions"]', backend_source)
        self.assertIn('args.extend(["--effort", reasoning_effort])', backend_source)
        self.assertIn('return ["--resume", resume_id]', backend_source)
        self.assertIn('home_env_var="CLAUDE_CONFIG_DIR"', backend_source)

    def test_broker_has_cc_closed_log_discovery_fallback(self) -> None:
        source = BROKER_PY.read_text(encoding="utf-8")
        self.assertIn('if AGENT_BACKEND == "cc" and current_log_path is None:', source)
        self.assertIn('found = _find_new_session_log(', source)
        self.assertIn('agent_backend=AGENT_BACKEND', source)
        self.assertIn('preexisting=st.known_rollout_paths', source)
        self.assertIn('prelaunch_rollout_paths = set(_iter_session_logs(self.sessions_dir, agent_backend=AGENT_BACKEND))', source)
        prelaunch_idx = source.index('prelaunch_rollout_paths = set(_iter_session_logs(self.sessions_dir, agent_backend=AGENT_BACKEND))')
        self.assertLess(prelaunch_idx, source.index('os.fork()'))
        self.assertLess(prelaunch_idx, source.index('pty.fork()'))
        self.assertIn('st.known_rollout_paths = set(prelaunch_rollout_paths)', source)


if __name__ == "__main__":
    unittest.main()
