from __future__ import annotations

import unittest
from unittest.mock import patch

from codoxear.server import SessionManager


class _DummyProc:
    def __init__(self, pid: int) -> None:
        self.pid = int(pid)
        self.stderr = None

    def wait(self) -> int:
        return 0


class TestServerSpawnCli(unittest.TestCase):
    def _mgr(self) -> SessionManager:
        return SessionManager.__new__(SessionManager)

    def test_spawn_web_session_sets_claude_env(self) -> None:
        mgr = self._mgr()
        with patch("codoxear.server._env_flag", return_value=False), patch(
            "codoxear.server._wait_or_raise", return_value=None
        ), patch("codoxear.server.subprocess.Popen", return_value=_DummyProc(4321)) as popen:
            res = mgr.spawn_web_session(cwd="/tmp", cli="claude")

        self.assertEqual(res.get("broker_pid"), 4321)
        self.assertEqual(res.get("cli"), "claude")
        env = popen.call_args.kwargs["env"]
        self.assertEqual(env.get("CODEX_WEB_CLI"), "claude")
        self.assertTrue(bool(env.get("CLAUDE_HOME")))
        self.assertTrue(bool(env.get("CLAUDE_BIN")))

    def test_spawn_web_session_sets_codex_env(self) -> None:
        mgr = self._mgr()
        with patch("codoxear.server._env_flag", return_value=False), patch(
            "codoxear.server._wait_or_raise", return_value=None
        ), patch("codoxear.server.subprocess.Popen", return_value=_DummyProc(5432)) as popen:
            res = mgr.spawn_web_session(cwd="/tmp", cli="codex")

        self.assertEqual(res.get("broker_pid"), 5432)
        self.assertEqual(res.get("cli"), "codex")
        env = popen.call_args.kwargs["env"]
        self.assertEqual(env.get("CODEX_WEB_CLI"), "codex")
        self.assertTrue(bool(env.get("CODEX_HOME")))
        self.assertTrue(bool(env.get("CODEX_BIN")))

    def test_spawn_web_session_rejects_unknown_cli(self) -> None:
        mgr = self._mgr()
        with self.assertRaises(ValueError):
            mgr.spawn_web_session(cwd="/tmp", cli="unknown")


if __name__ == "__main__":
    unittest.main()
