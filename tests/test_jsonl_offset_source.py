import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UTIL = ROOT / "codoxear" / "util.py"
JSONL_OFFSET = ROOT / "codoxear" / "jsonl_offset.py"


class TestJsonlOffsetSource(unittest.TestCase):
    def test_jsonl_offset_reader_has_dedicated_owner_with_util_facade(self) -> None:
        util_source = UTIL.read_text(encoding="utf-8")
        offset_source = JSONL_OFFSET.read_text(encoding="utf-8")

        self.assertIn("from .jsonl_offset import read_jsonl_from_offset as _read_jsonl_from_offset_impl", util_source)
        self.assertIn("def read_jsonl_from_offset(path: Path, offset: int, *, max_bytes: int, advance_on_oversized_unterminated: bool = True)", util_source)
        self.assertIn("return _read_jsonl_from_offset_impl(", util_source)
        self.assertIn("log_exception=_log_exception", util_source)
        self.assertNotIn("data = f.read(target)", util_source)
        self.assertNotIn("last_nl = data.rfind", util_source)

        self.assertIn("def read_jsonl_from_offset(", offset_source)
        self.assertIn("log_exception: LogException", offset_source)
        self.assertIn("data = f.read(target)", offset_source)
        self.assertIn("advance_on_oversized_unterminated", offset_source)
        self.assertIn("while True:", offset_source)
        self.assertIn("last_nl = data.rfind", offset_source)
        self.assertIn("except (json.JSONDecodeError, UnicodeDecodeError):", offset_source)
        self.assertIn("if isinstance(obj, dict):", offset_source)
        self.assertNotIn("from .util", offset_source)


if __name__ == "__main__":
    unittest.main()
