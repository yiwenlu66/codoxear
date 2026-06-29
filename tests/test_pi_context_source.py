import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PI_LOG = ROOT / "codoxear" / "pi_log.py"
PI_CONTEXT = ROOT / "codoxear" / "pi_context.py"


class TestPiContextSource(unittest.TestCase):
    def test_pi_context_owns_context_window_and_settings_helpers(self) -> None:
        pi_log = PI_LOG.read_text(encoding="utf-8")
        pi_context = PI_CONTEXT.read_text(encoding="utf-8")

        self.assertIn("PI_DEFAULT_RESERVED_TOKENS = 16384", pi_context)
        self.assertIn('PI_MODEL_QUERY_ID = "codoxear-models"', pi_context)
        self.assertIn("def _context_windows_from_model_rows(", pi_context)
        self.assertIn("def _context_windows_from_models_file(", pi_context)
        self.assertIn("def _pi_rpc_context_windows(", pi_context)
        self.assertIn("def _query_pi_context_windows(", pi_context)
        self.assertIn("def _pi_reserved_tokens(", pi_context)
        self.assertIn("def pi_reserved_tokens(", pi_context)
        self.assertIn('env["PI_OFFLINE"] = "1"', pi_context)
        self.assertIn('"--offline",', pi_context)
        self.assertIn("timeout=PI_MODEL_QUERY_TIMEOUT_SECONDS", pi_context)
        self.assertIn("return PI_DEFAULT_RESERVED_TOKENS", pi_context)

        self.assertIn("from .pi_context import _query_pi_context_windows as _query_pi_context_windows_impl", pi_log)
        self.assertIn("def _query_pi_context_windows(models_path: Path)", pi_log)
        self.assertIn("return _query_pi_context_windows_impl(models_path)", pi_log)
        self.assertIn("def pi_model_context_window(", pi_log)
        self.assertIn("queried_context_window = _query_pi_context_windows(path).get(key)", pi_log)
        self.assertIn("def pi_context_token_update(", pi_log)
        self.assertIn("reserve = pi_reserved_tokens(settings_path=settings_path) if reserved_tokens is None else reserved_tokens", pi_log)
        self.assertNotIn("import functools", pi_log)
        self.assertNotIn("import shutil", pi_log)
        self.assertNotIn("import subprocess", pi_log)
        self.assertNotIn("from .agent_backend import get_agent_backend", pi_log)
        self.assertNotIn("subprocess.run(", pi_log)

    def test_pi_log_facade_preserves_patchable_private_cache_helpers(self) -> None:
        pi_log = PI_LOG.read_text(encoding="utf-8")
        for name in (
            "PI_DEFAULT_RESERVED_TOKENS",
            "PI_MODEL_QUERY_TIMEOUT_SECONDS",
            "PI_MODEL_QUERY_ID",
            "_context_percent_remaining",
            "_context_token_update",
            "_context_windows_from_model_rows",
            "_context_windows_from_models_file",
            "_default_pi_models_path",
            "_default_pi_settings_path",
            "_file_mtime_ns",
            "_pi_context_windows",
            "_pi_reserved_tokens",
            "_pi_rpc_context_windows",
            "_query_pi_context_windows",
            "pi_model_context_window",
            "pi_reserved_tokens",
            "pi_context_token_update",
        ):
            self.assertIn(name, pi_log)


if __name__ == "__main__":
    unittest.main()
