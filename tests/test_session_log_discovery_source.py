import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UTIL = ROOT / "codoxear" / "util.py"
DISCOVERY = ROOT / "codoxear" / "session_log_discovery.py"


class TestSessionLogDiscoverySource(unittest.TestCase):
    def test_session_log_discovery_has_dedicated_owner_with_util_facade(self) -> None:
        util_source = UTIL.read_text(encoding="utf-8")
        discovery_source = DISCOVERY.read_text(encoding="utf-8")

        self.assertIn("from .session_log_discovery import read_session_meta_payload as _read_session_meta_payload_impl", util_source)
        self.assertIn("from .session_log_discovery import find_new_session_log as _find_new_session_log_impl", util_source)
        self.assertIn("def read_session_meta_payload(", util_source)
        self.assertIn("return _read_session_meta_payload_impl(", util_source)
        self.assertIn("now_func=now", util_source)
        self.assertIn("sleep_func=time.sleep", util_source)
        self.assertIn("log_exception=_log_exception", util_source)
        self.assertIn("read_session_meta_payload_func=read_session_meta_payload", util_source)
        self.assertIn("iter_session_logs_func=iter_session_logs", util_source)
        self.assertIn("is_subagent_session_meta_func=is_subagent_session_meta", util_source)

        facade_start = util_source.index("def _read_session_meta_payload_once(")
        facade_end = util_source.index("def proc_find_open_rollout_log(")
        facade_block = util_source[facade_start:facade_end]
        self.assertNotIn("with log_path.open", facade_block)
        self.assertNotIn("for p in sessions_dir.rglob", facade_block)
        self.assertNotIn("while True:\n        matches", facade_block)
        self.assertNotIn("read_pi_session_id(p)", facade_block)
        self.assertNotIn("read_cc_session_id(p)", facade_block)

        self.assertIn("def _read_session_meta_payload_once(log_path: Path, *, max_bytes: int, log_exception: LogException)", discovery_source)
        self.assertIn("with log_path.open(\"rb\") as f:", discovery_source)
        self.assertIn("def read_session_meta_payload(", discovery_source)
        self.assertIn("now_func: NowFunc", discovery_source)
        self.assertIn("sleep_func: SleepFunc = time.sleep", discovery_source)
        self.assertIn("log_exception: LogException", discovery_source)
        self.assertIn("read_pi_session_header(log_path)", discovery_source)
        self.assertIn("read_cc_session_header(log_path)", discovery_source)
        self.assertIn("def iter_session_logs(", discovery_source)
        self.assertIn("for p in sessions_dir.rglob(pattern):", discovery_source)
        self.assertIn("def find_session_log_for_session_id(", discovery_source)
        self.assertIn("read_pi_session_id(p)", discovery_source)
        self.assertIn("read_cc_session_id(p)", discovery_source)
        self.assertIn("def find_new_session_log(", discovery_source)
        self.assertIn("_payload_cwd_matches(payload.get(\"cwd\"), cwd)", discovery_source)
        self.assertIn("sleep_func(0.2)", discovery_source)
        self.assertNotIn("from .util", discovery_source)


if __name__ == "__main__":
    unittest.main()
