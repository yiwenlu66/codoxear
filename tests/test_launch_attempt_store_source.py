import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from codoxear import launch_attempt_store
from codoxear import util


ROOT = Path(__file__).resolve().parents[1]
UTIL = ROOT / "codoxear" / "util.py"
LAUNCH_STORE = ROOT / "codoxear" / "launch_attempt_store.py"


class TestLaunchAttemptStoreSource(unittest.TestCase):
    def test_launch_attempt_store_owns_redaction_and_log_persistence(self) -> None:
        util_source = UTIL.read_text(encoding="utf-8")
        store_source = LAUNCH_STORE.read_text(encoding="utf-8")

        self.assertIn("def _jsonable(value: Any) -> Any:", store_source)
        self.assertIn("def redact_launch_failure_text(", store_source)
        self.assertIn("def redact_launch_failure_value(", store_source)
        self.assertIn("def redacted_launch_attempt_persist_record(", store_source)
        self.assertIn("def redacted_launch_attempt_response_record(", store_source)
        self.assertIn("def append_launch_attempt(", store_source)
        self.assertIn("def read_launch_attempts(", store_source)
        self.assertIn("f.write(json.dumps(out, sort_keys=True) + \"\\n\")", store_source)
        self.assertIn("latest[launch_id] = obj", store_source)

        self.assertNotIn("def _jsonable(value: Any) -> Any:", util_source)
        self.assertNotIn("def redact_launch_failure_text(", util_source)
        self.assertNotIn("def redact_launch_failure_value(", util_source)
        self.assertNotIn("def redacted_launch_attempt_persist_record(", util_source)
        self.assertNotIn("def redacted_launch_attempt_response_record(", util_source)

    def test_util_preserves_launch_attempt_facade_and_time_patch_seam(self) -> None:
        util_source = UTIL.read_text(encoding="utf-8")
        self.assertIn("from .launch_attempt_store import append_launch_attempt as _append_launch_attempt_impl", util_source)
        self.assertIn("from .launch_attempt_store import read_launch_attempts as _read_launch_attempts_impl", util_source)
        self.assertIn("from .launch_attempt_store import redact_launch_failure_text", util_source)
        self.assertIn("def launch_attempts_path(app_dir: Path | None = None) -> Path:", util_source)
        self.assertIn("def append_launch_attempt(record: dict[str, Any], *, path: Path | None = None)", util_source)
        self.assertIn("now_ts=now()", util_source)
        self.assertIn("now_ts=now() if now_ts is None else float(now_ts)", util_source)

        self.assertIs(util.redact_launch_failure_text, launch_attempt_store.redact_launch_failure_text)
        self.assertIs(util.redact_launch_failure_value, launch_attempt_store.redact_launch_failure_value)
        self.assertIs(util.redacted_launch_attempt_persist_record, launch_attempt_store.redacted_launch_attempt_persist_record)
        self.assertIs(util.redacted_launch_attempt_response_record, launch_attempt_store.redacted_launch_attempt_response_record)

    def test_util_wrappers_use_util_now_for_append_and_read_defaults(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "launches.jsonl"
            with patch("codoxear.util.now", return_value=123.456):
                rec = util.append_launch_attempt({"agent_backend": "codex"}, path=path)
            self.assertEqual(rec["created_ts"], 123.456)
            self.assertEqual(rec["updated_ts"], 123.456)
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(raw["created_ts"], 123.456)

            with patch("codoxear.util.now", return_value=123.456 + 10.0):
                self.assertEqual(len(util.read_launch_attempts(path=path, max_age_s=20.0)), 1)
            with patch("codoxear.util.now", return_value=123.456 + 30.0):
                self.assertEqual(util.read_launch_attempts(path=path, max_age_s=20.0), [])


if __name__ == "__main__":
    unittest.main()
