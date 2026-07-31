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
