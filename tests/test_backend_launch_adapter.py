import unittest
from pathlib import Path

from codoxear.backend_launch import apply_backend_environment
from codoxear.backend_launch import build_backend_args
from codoxear.backend_launch import build_backend_resume_args
from codoxear.backend_launch import build_tmux_inline_env
from codoxear.backend_launch import tmux_unset_vars
from codoxear.agent_backend import ClaudeCodeBackend
from codoxear.agent_backend import CodexBackend
from codoxear.agent_backend import PiBackend
from codoxear.agent_backend import get_agent_backend
from codoxear.launch_config import LaunchRequestValidationError
from codoxear.launch_config import normalize_requested_cc_reasoning_effort
from codoxear.launch_config import normalize_requested_model_provider
from codoxear.launch_config import normalize_requested_pi_reasoning_effort
from codoxear.launch_config import normalize_requested_preferred_auth_method
from codoxear.launch_config import normalize_requested_reasoning_effort
from codoxear.launch_config import normalize_requested_service_tier


class TestBackendLaunchAdapter(unittest.TestCase):
    def test_backend_registry_returns_explicit_adapter_objects(self) -> None:
        self.assertIsInstance(get_agent_backend("codex"), CodexBackend)
        self.assertIsInstance(get_agent_backend("pi"), PiBackend)
        self.assertIsInstance(get_agent_backend("cc"), ClaudeCodeBackend)
        self.assertEqual(get_agent_backend("pi").build_resume_args(resume_id="sid", resume_row={"log_path": "/tmp/pi.jsonl"}), ["--session", "/tmp/pi.jsonl"])
        self.assertEqual(get_agent_backend("cc").build_launch_args(spawn_cwd=Path("/repo"), codex_trust_override="", model="sonnet"), ["--dangerously-skip-permissions", "--model", "sonnet"])

    def test_backend_launch_module_is_compatibility_facade(self) -> None:
        source = Path("codoxear/backend_launch.py").read_text(encoding="utf-8")
        self.assertNotIn('backend_name == "codex"', source)
        self.assertNotIn('backend_name == "pi"', source)
        self.assertNotIn('backend_name == "cc"', source)
        self.assertIn("get_agent_backend(agent_backend).build_launch_args(", source)
        self.assertIn("get_agent_backend(agent_backend).apply_launch_environment(", source)

    def test_backend_adapters_normalize_launch_request_options(self) -> None:
        kwargs = {
            "validation_error_type": LaunchRequestValidationError,
            "normalize_model_provider": normalize_requested_model_provider,
            "normalize_preferred_auth_method": normalize_requested_preferred_auth_method,
            "normalize_reasoning_effort": normalize_requested_reasoning_effort,
            "normalize_pi_reasoning_effort": normalize_requested_pi_reasoning_effort,
            "normalize_cc_reasoning_effort": normalize_requested_cc_reasoning_effort,
            "normalize_service_tier": normalize_requested_service_tier,
            "codex_launch_defaults_provider": lambda: {"model_providers": ["openai", "custom"]},
            "pi_launch_defaults_provider": lambda: {"reasoning_efforts_by_model": {"macaron/gpt-pi": ["off", "high"]}},
        }
        codex = get_agent_backend("codex").normalize_launch_request_options(
            {"model_provider": "custom", "preferred_auth_method": "apikey", "reasoning_effort": "high", "service_tier": "fast"},
            model="gpt-codex",
            **kwargs,
        )
        self.assertEqual(codex, {"model_provider": "custom", "preferred_auth_method": "apikey", "reasoning_effort": "high", "service_tier": "fast"})
        pi = get_agent_backend("pi").normalize_launch_request_options(
            {"model_provider": "macaron", "reasoning_effort": "high"},
            model="gpt-pi",
            **kwargs,
        )
        self.assertEqual(pi, {"model_provider": "macaron", "preferred_auth_method": None, "reasoning_effort": "high", "service_tier": None})
        cc = get_agent_backend("cc").normalize_launch_request_options(
            {"reasoning_effort": "max"},
            model="sonnet",
            **kwargs,
        )
        self.assertEqual(cc, {"model_provider": None, "preferred_auth_method": None, "reasoning_effort": "max", "service_tier": None})
        with self.assertRaises(LaunchRequestValidationError):
            get_agent_backend("cc").normalize_launch_request_options({"model_provider": "openai"}, model="sonnet", **kwargs)

    def test_backend_adapters_project_launch_default_metadata(self) -> None:
        codex = get_agent_backend("codex").project_launch_defaults(
            {"model_providers": ["chatgpt", "openai-api", "custom"], "reasoning_effort": None},
            reasoning_efforts=("xhigh", "high"),
        )
        self.assertEqual(codex["agent_backend"], "codex")
        self.assertEqual(codex["provider_choices"], ["chatgpt", "openai-api", "custom"])
        self.assertEqual(codex["reasoning_efforts"], ["xhigh", "high"])
        self.assertTrue(codex["supports_fast"])
        pi = get_agent_backend("pi").project_launch_defaults(
            {"provider_choices": ["macaron"], "reasoning_efforts": ["off", "high"]},
            reasoning_efforts=("off", "minimal", "high"),
        )
        self.assertEqual(pi["agent_backend"], "pi")
        self.assertEqual(pi["provider_choices"], ["macaron"])
        self.assertEqual(pi["reasoning_efforts"], ["off", "high"])
        self.assertFalse(pi["supports_fast"])

    def test_codex_args_match_web_owned_launch_contract(self) -> None:
        args = build_backend_args(
            agent_backend="codex",
            spawn_cwd=Path("/repo"),
            codex_trust_override="projects={}",
            model_provider="bytecat",
            preferred_auth_method="apikey",
            model="gpt-5.4",
            reasoning_effort="xhigh",
            service_tier="fast",
        )
        self.assertEqual(
            args,
            [
                "-c",
                "projects={}",
                "-c",
                "check_for_update_on_startup=false",
                "--disable",
                "goals",
                "--dangerously-bypass-approvals-and-sandbox",
                "--model",
                "gpt-5.4",
                "-c",
                'model_reasoning_effort="xhigh"',
                "-c",
                'model_provider="bytecat"',
                "-c",
                'preferred_auth_method="apikey"',
                "-c",
                'service_tier="fast"',
            ],
        )

    def test_backend_specific_args_and_resume_args(self) -> None:
        self.assertEqual(
            build_backend_args(agent_backend="pi", spawn_cwd=Path("/repo"), codex_trust_override="", model_provider="macaron", model="gpt-5.4", reasoning_effort="medium"),
            ["--provider", "macaron", "--model", "gpt-5.4", "--thinking", "medium"],
        )
        self.assertEqual(
            build_backend_args(agent_backend="cc", spawn_cwd=Path("/repo"), codex_trust_override="", model="sonnet", reasoning_effort="max"),
            ["--dangerously-skip-permissions", "--model", "sonnet", "--effort", "max"],
        )
        self.assertEqual(build_backend_resume_args(agent_backend="codex", resume_id="resume-a"), ["resume", "resume-a"])
        self.assertEqual(
            build_backend_resume_args(agent_backend="pi", resume_id="resume-a", resume_row={"log_path": "/tmp/pi.jsonl"}),
            ["--session", "/tmp/pi.jsonl"],
        )
        self.assertEqual(build_backend_resume_args(agent_backend="cc", resume_id="resume-a"), ["--resume", "resume-a"])

    def test_backend_environment_owns_home_and_request_vars(self) -> None:
        env = {
            "CODEX_HOME": "/old-codex",
            "PI_HOME": "/old-pi",
            "CLAUDE_CONFIG_DIR": "/old-claude",
            "CODEX_WEB_MODEL": "stale",
            "CODEX_WEB_TRANSPORT": "tmux",
        }
        apply_backend_environment(
            env,
            agent_backend="pi",
            homes={"codex": "/codex", "pi": "/pi", "cc": "/claude"},
            model_provider="macaron",
            model="gpt-5.4",
            reasoning_effort="high",
            resume_session_id="resume-a",
        )
        self.assertEqual(env["CODEX_WEB_OWNER"], "web")
        self.assertEqual(env["CODEX_WEB_AGENT_BACKEND"], "pi")
        self.assertEqual(env["PI_HOME"], "/old-pi")
        self.assertNotIn("CODEX_HOME", env)
        self.assertNotIn("CLAUDE_CONFIG_DIR", env)
        self.assertNotIn("CODEX_WEB_TRANSPORT", env)
        self.assertEqual(env["CODEX_WEB_MODEL_PROVIDER"], "macaron")
        self.assertEqual(env["CODEX_WEB_MODEL"], "gpt-5.4")
        self.assertEqual(env["CODEX_WEB_REASONING_EFFORT"], "high")
        self.assertEqual(env["CODEX_WEB_RESUME_SESSION_ID"], "resume-a")

    def test_tmux_inline_env_uses_backend_home_and_unset_contract(self) -> None:
        inline = build_tmux_inline_env(
            {"CLAUDE_CONFIG_DIR": "/claude"},
            agent_backend="cc",
            tmux_session="codoxear",
            tmux_window="work-abc123",
            launch_id="launch-1",
            spawn_nonce="nonce",
            model="sonnet",
            inherited_backend_bin="/bin/claude",
        )
        self.assertEqual(inline["CODEX_WEB_AGENT_BACKEND"], "cc")
        self.assertEqual(inline["CODEX_WEB_TRANSPORT"], "tmux")
        self.assertEqual(inline["CLAUDE_CONFIG_DIR"], "/claude")
        self.assertEqual(inline["CODEX_WEB_MODEL"], "sonnet")
        self.assertEqual(inline["CLAUDE_BIN"], "/bin/claude")
        unset = tmux_unset_vars()
        self.assertIn("CODEX_WEB_RESUME_LOG_PATH", unset)
        self.assertLess(unset.index("CODEX_HOME"), unset.index("CODEX_WEB_OWNER"))


if __name__ == "__main__":
    unittest.main()
