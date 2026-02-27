from __future__ import annotations

import unittest
from unittest.mock import patch

from codoxear import broker


class TestBrokerSpawnEnv(unittest.TestCase):
    def test_exec_via_login_shell_unsets_auth_token_when_requested(self) -> None:
        with patch.dict(
            "codoxear.broker.os.environ",
            {"CODEX_WEB_UNSET_ANTHROPIC_AUTH_TOKEN": "1"},
            clear=False,
        ), patch("codoxear.broker.CODEX_BIN", "claude"), patch(
            "codoxear.broker._shell_argv_for_command",
            side_effect=lambda cmd: ["/bin/bash", "-lc", cmd],
        ) as shell_mock, patch("codoxear.broker.os.chdir"), patch(
            "codoxear.broker.os.execvpe", side_effect=RuntimeError("stop")
        ):
            with self.assertRaisesRegex(RuntimeError, "stop"):
                broker._exec_codex_via_login_shell(cwd="/tmp", codex_args=["--print", "ok"])

        cmd = shell_mock.call_args.args[0]
        self.assertIn("unset ANTHROPIC_AUTH_TOKEN; ", cmd)
        self.assertIn("exec claude --print ok", cmd)

    def test_exec_via_login_shell_without_unset_flag(self) -> None:
        with patch.dict(
            "codoxear.broker.os.environ",
            {"CODEX_WEB_UNSET_ANTHROPIC_AUTH_TOKEN": "0"},
            clear=False,
        ), patch("codoxear.broker.CODEX_BIN", "claude"), patch(
            "codoxear.broker._shell_argv_for_command",
            side_effect=lambda cmd: ["/bin/bash", "-lc", cmd],
        ) as shell_mock, patch("codoxear.broker.os.chdir"), patch(
            "codoxear.broker.os.execvpe", side_effect=RuntimeError("stop")
        ):
            with self.assertRaisesRegex(RuntimeError, "stop"):
                broker._exec_codex_via_login_shell(cwd="/tmp", codex_args=["--print", "ok"])

        cmd = shell_mock.call_args.args[0]
        self.assertTrue(cmd.startswith("exec "))
        self.assertNotIn("unset ANTHROPIC_AUTH_TOKEN;", cmd)


if __name__ == "__main__":
    unittest.main()
