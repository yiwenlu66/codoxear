import os
import unittest

from codoxear.agent_backend import infer_agent_backend_from_cli_bin
from codoxear.agent_backend import resolve_agent_backend


class TestAgentBackendResolution(unittest.TestCase):
    def test_infer_agent_backend_from_cli_bin_accepts_pi_names(self) -> None:
        self.assertEqual(infer_agent_backend_from_cli_bin("pi"), "pi")
        self.assertEqual(infer_agent_backend_from_cli_bin("/usr/local/bin/piox"), "pi")

    def test_resolve_agent_backend_prefers_explicit_env(self) -> None:
        env = {"CODEX_BIN": "pi"}
        self.assertEqual(resolve_agent_backend("codex", env=env), "codex")

    def test_resolve_agent_backend_infers_pi_from_codex_bin_override(self) -> None:
        env = {"CODEX_BIN": "pi"}
        self.assertEqual(resolve_agent_backend(None, env=env), "pi")


if __name__ == "__main__":
    unittest.main()
