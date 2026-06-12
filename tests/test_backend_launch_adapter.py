import unittest
from pathlib import Path

from codoxear.backend_launch import apply_backend_environment
from codoxear.backend_launch import build_backend_args
from codoxear.backend_launch import build_backend_resume_args
from codoxear.backend_launch import build_tmux_inline_env
from codoxear.backend_launch import tmux_unset_vars


class TestBackendLaunchAdapter(unittest.TestCase):
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
