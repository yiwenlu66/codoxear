import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestPathHelpersSource(unittest.TestCase):
    def test_path_matching_and_rollout_session_id_are_centralized(self) -> None:
        sources = {path: path.read_text(encoding="utf-8") for path in (ROOT / "codoxear").glob("*.py")}
        util_source = sources[ROOT / "codoxear" / "util.py"]
        self.assertIn("def _paths_match(a: Path, b: Path) -> bool:", util_source)
        self.assertIn("def session_id_from_rollout_path(log_path: Path) -> str | None:", util_source)
        self.assertEqual(sum(src.count("def _paths_match(") for src in sources.values()), 1)
        self.assertEqual(sum(src.count("def session_id_from_rollout_path(") for src in sources.values()), 1)
        self.assertIn("from codoxear.util import _paths_match as _paths_match", sources[ROOT / "codoxear" / "broker.py"])
        self.assertIn("from .util import session_id_from_rollout_path as _session_id_from_rollout_path", sources[ROOT / "codoxear" / "server.py"])
        self.assertIn("from codoxear.util import session_id_from_rollout_path as _session_id_from_rollout_path", sources[ROOT / "codoxear" / "broker.py"])


if __name__ == "__main__":
    unittest.main()
