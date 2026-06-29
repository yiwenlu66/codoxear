import unittest
from pathlib import Path

from codoxear import json_state
from codoxear import util


ROOT = Path(__file__).resolve().parents[1]
UTIL_PY = ROOT / "codoxear" / "util.py"
JSON_STATE_PY = ROOT / "codoxear" / "json_state.py"


class TestJsonStateSource(unittest.TestCase):
    def test_json_state_helpers_have_dedicated_owner_with_util_reexports(self) -> None:
        util_source = UTIL_PY.read_text(encoding="utf-8")
        json_state_source = JSON_STATE_PY.read_text(encoding="utf-8")

        self.assertIn("from .json_state import atomic_write_json", util_source)
        self.assertIn("from .json_state import load_json_file", util_source)
        self.assertNotIn("def load_json_file(", util_source)
        self.assertNotIn("def atomic_write_json(", util_source)

        self.assertIn("def load_json_file(path: Path, default: Any = None) -> Any:", json_state_source)
        self.assertIn("except FileNotFoundError:", json_state_source)
        self.assertIn("return default", json_state_source)
        self.assertIn("def atomic_write_json(path: Path, obj: Any", json_state_source)
        self.assertIn("path.parent.mkdir(parents=True, exist_ok=True)", json_state_source)
        self.assertIn("os.replace(tmp, path)", json_state_source)
        self.assertIn("tmp.unlink()", json_state_source)

        self.assertIs(util.load_json_file, json_state.load_json_file)
        self.assertIs(util.atomic_write_json, json_state.atomic_write_json)


if __name__ == "__main__":
    unittest.main()
