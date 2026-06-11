import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from codoxear.agent_backend import get_agent_backend
from codoxear.agent_backend import infer_agent_backend_from_log_path
from codoxear.agent_backend import normalize_agent_backend


class TestCcBackendRegistration(unittest.TestCase):
    def test_cc_backend_registered(self) -> None:
        backend = get_agent_backend("cc")
        self.assertEqual(normalize_agent_backend("cc"), "cc")
        self.assertEqual(backend.default_bin, "claude")
        self.assertEqual(backend.bin_env_var, "CLAUDE_BIN")
        self.assertEqual(backend.home_env_var, "CLAUDE_CONFIG_DIR")
        self.assertEqual(backend.sessions_relpath, ("projects",))

    def test_infer_cc_log_path_excludes_subagent_logs(self) -> None:
        main = Path("/home/alice/.claude/projects/-repo/11111111-2222-3333-4444-555555555555.jsonl")
        sub = Path("/home/alice/.claude/projects/-repo/11111111-2222-3333-4444-555555555555/subagents/agent-abc.jsonl")
        self.assertEqual(infer_agent_backend_from_log_path(main), "cc")
        self.assertIsNone(infer_agent_backend_from_log_path(sub))

    def test_infer_cc_log_path_respects_custom_config_dir(self) -> None:
        with TemporaryDirectory() as td, patch.dict("os.environ", {"CLAUDE_CONFIG_DIR": td}):
            path = Path(td) / "projects" / "-repo" / "11111111-2222-3333-4444-555555555555.jsonl"
            self.assertEqual(infer_agent_backend_from_log_path(path), "cc")


if __name__ == "__main__":
    unittest.main()
