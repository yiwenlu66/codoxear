import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UTIL = ROOT / "codoxear" / "util.py"
PROCESS_LOG_PATHS = ROOT / "codoxear" / "process_log_paths.py"


class TestProcessLogPathsSource(unittest.TestCase):
    def test_open_log_enumeration_has_dedicated_owner_with_util_facade(self) -> None:
        util_source = UTIL.read_text(encoding="utf-8")
        process_source = PROCESS_LOG_PATHS.read_text(encoding="utf-8")

        moved_names = (
            "_macos_children",
            "_macos_descendants",
            "_macos_open_rollout_logs",
            "_proc_pid_uid",
            "_proc_children",
            "_proc_descendants",
            "_proc_fd_flags",
            "_fd_has_write_intent",
            "proc_open_rollout_logs",
            "proc_open_rollout_logs_for_backend",
            "proc_open_writable_rollout_logs",
            "proc_open_writable_rollout_logs_for_backend",
        )
        for name in moved_names:
            self.assertIn(f"from .process_log_paths import {name}", util_source)
            self.assertIn(f"def {name}(", process_source)
            self.assertNotIn(f"def {name}(", util_source)

        self.assertIn("def proc_find_open_rollout_log(", util_source)
        self.assertIn("return _proc_find_open_rollout_log_impl(", util_source)
        self.assertIn("proc_open_writable_rollout_logs_for_backend_func=proc_open_writable_rollout_logs_for_backend", util_source)
        self.assertIn("read_session_meta_payload_func=read_session_meta_payload", util_source)
        self.assertNotIn("cands = list(proc_open_writable_rollout_logs_for_backend(", util_source)
        self.assertNotIn("matches: list[Path] = []", util_source)
        self.assertIn("from .agent_backend import get_agent_backend", process_source)
        self.assertIn("if backend.is_session_log_path(path, sessions_dir=sessions_dir):", process_source)
        self.assertNotIn("_is_codex_rollout_log_path", process_source)
        self.assertNotIn("normalize_agent_backend", process_source)
        self.assertIn('if sys.platform == "darwin":', process_source)
        self.assertIn("_proc_fd_flags(proc_root, pid, ent.name)", process_source)
        self.assertIn("_fd_has_write_intent(flags)", process_source)


if __name__ == "__main__":
    unittest.main()
