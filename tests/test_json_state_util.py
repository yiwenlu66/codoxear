import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from codoxear.util import atomic_write_json
from codoxear.util import load_json_file


class TestJsonStateUtil(unittest.TestCase):
    def test_load_json_file_returns_default_for_missing_path(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "missing.json"

            self.assertEqual(load_json_file(path, default={"ok": True}), {"ok": True})

    def test_atomic_write_json_creates_parent_and_cleans_temp(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "nested" / "state.json"

            atomic_write_json(path, {"b": 2, "a": 1})

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"a": 1, "b": 2})
            self.assertEqual(list(path.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
