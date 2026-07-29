import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from codoxear import json_state
from codoxear import util
from codoxear.util import atomic_write_json
from codoxear.util import load_json_file


class TestJsonStateUtil(unittest.TestCase):
    def test_load_json_file_returns_default_for_missing_path(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "missing.json"

            self.assertEqual(load_json_file(path, default={"ok": True}), {"ok": True})

    def test_util_reexports_json_state_helpers(self) -> None:
        self.assertIs(util.load_json_file, json_state.load_json_file)
        self.assertIs(util.atomic_write_json, json_state.atomic_write_json)

if __name__ == "__main__":
    unittest.main()
