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


if __name__ == "__main__":
    unittest.main()
