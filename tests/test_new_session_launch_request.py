import unittest
from unittest.mock import patch

from codoxear.server import LaunchRequestValidationError
from codoxear.server import _parse_new_session_launch_request


class TestNewSessionLaunchRequest(unittest.TestCase):
    def test_parses_codex_launch_fields(self) -> None:
        with patch("codoxear.server._read_codex_launch_defaults", return_value={"model_providers": ["chatgpt", "openai-api", "crs"]}):
            req = _parse_new_session_launch_request(
                {
                    "agent_backend": "codex",
                    "cwd": "/repo",
                    "model_provider": "crs",
                    "preferred_auth_method": "apikey",
                    "model": "gpt-5.4",
                    "reasoning_effort": "high",
                    "service_tier": "fast",
                    "create_in_tmux": True,
                    "resume_session_id": "  resume-me  ",
                    "worktree_branch": "  feature/x  ",
                    "args": ["--flag", ""],
                }
            )

        self.assertEqual(req.agent_backend, "codex")
        self.assertEqual(req.cwd, "/repo")
        self.assertEqual(req.model_provider, "crs")
        self.assertEqual(req.preferred_auth_method, "apikey")
        self.assertEqual(req.model, "gpt-5.4")
        self.assertEqual(req.reasoning_effort, "high")
        self.assertEqual(req.service_tier, "fast")
        self.assertTrue(req.create_in_tmux)
        self.assertEqual(req.resume_session_id, "resume-me")
        self.assertEqual(req.worktree_branch, "feature/x")
        self.assertEqual(req.args, ["--flag"])

    def test_parses_providerless_pi_without_codex_default(self) -> None:
        with patch("codoxear.server._read_pi_launch_defaults", return_value={"provider_choices": []}):
            req = _parse_new_session_launch_request({"agent_backend": "pi", "cwd": "/repo", "model": "default", "reasoning_effort": "high"})

        self.assertEqual(req.agent_backend, "pi")
        self.assertIsNone(req.model_provider)
        self.assertIsNone(req.preferred_auth_method)
        self.assertIsNone(req.service_tier)
        self.assertIsNone(req.model)
        self.assertEqual(req.reasoning_effort, "high")

    def test_pi_provider_is_passed_to_cli_when_defaults_are_incomplete(self) -> None:
        with patch(
            "codoxear.server._read_pi_launch_defaults",
            return_value={"provider_choices": ["occ", "occ-claude"], "reasoning_efforts_by_model": {}},
        ):
            req = _parse_new_session_launch_request({"agent_backend": "pi", "cwd": "/repo", "model_provider": "anthropic", "model": "claude-haiku-4-5"})

        self.assertEqual(req.model_provider, "anthropic")
        self.assertEqual(req.model, "claude-haiku-4-5")

    def test_pi_provider_specific_request_ignores_stale_bare_model_reasoning(self) -> None:
        with patch(
            "codoxear.server._read_pi_launch_defaults",
            return_value={"provider_choices": ["occ"], "reasoning_efforts_by_model": {"claude-haiku-4-5": ["off"], "occ/claude-haiku-4-5": ["off"]}},
        ):
            req = _parse_new_session_launch_request(
                {
                    "agent_backend": "pi",
                    "cwd": "/repo",
                    "model_provider": "anthropic",
                    "model": "claude-haiku-4-5",
                    "reasoning_effort": "low",
                }
            )

        self.assertEqual(req.model_provider, "anthropic")
        self.assertEqual(req.reasoning_effort, "low")

    def test_rejects_unsupported_backend_specific_fields(self) -> None:
        with self.assertRaisesRegex(LaunchRequestValidationError, "model_provider is not supported for cc"):
            _parse_new_session_launch_request({"agent_backend": "cc", "cwd": "/repo", "model_provider": "openai"})
        with patch("codoxear.server._read_pi_launch_defaults", return_value={"provider_choices": []}):
            with self.assertRaisesRegex(LaunchRequestValidationError, "service_tier is not supported for pi"):
                _parse_new_session_launch_request({"agent_backend": "pi", "cwd": "/repo", "service_tier": "fast"})

    def test_cwd_error_preserves_field_marker(self) -> None:
        with self.assertRaises(LaunchRequestValidationError) as ctx:
            _parse_new_session_launch_request({"agent_backend": "codex", "cwd": ""})

        self.assertEqual(str(ctx.exception), "cwd required")
        self.assertEqual(ctx.exception.field, "cwd")

    def test_parser_uses_safe_defaults_when_backend_config_is_malformed(self) -> None:
        with patch("codoxear.server._read_codex_launch_defaults", side_effect=RuntimeError("bad codex config")):
            codex_req = _parse_new_session_launch_request({"agent_backend": "codex", "cwd": "/repo", "model_provider": "openai"})
        with patch("codoxear.server._read_pi_launch_defaults", side_effect=RuntimeError("bad pi config")):
            pi_req = _parse_new_session_launch_request({"agent_backend": "pi", "cwd": "/repo", "model_provider": "macaron"})

        self.assertEqual(codex_req.model_provider, "openai")
        self.assertEqual(codex_req.preferred_auth_method, None)
        self.assertEqual(pi_req.model_provider, "macaron")
        self.assertEqual(pi_req.agent_backend, "pi")

    def test_pi_parser_safe_defaults_cover_reasoning_effort(self) -> None:
        with patch("codoxear.server._read_pi_launch_defaults", side_effect=RuntimeError("bad pi config")):
            with patch("codoxear.server._read_pi_reasoning_efforts_by_model", side_effect=AssertionError("must not reread malformed Pi models")):
                req = _parse_new_session_launch_request({"agent_backend": "pi", "cwd": "/repo", "reasoning_effort": "high"})

        self.assertEqual(req.reasoning_effort, "high")


if __name__ == "__main__":
    unittest.main()
