import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestPtyUtilSource(unittest.TestCase):
    def test_bracketed_paste_injection_is_centralized(self) -> None:
        sources = {path: path.read_text(encoding="utf-8") for path in (ROOT / "codoxear").glob("*.py")}
        util_source = sources[ROOT / "codoxear" / "pty_util.py"]
        self.assertIn("def write_all(fd: int, data: bytes) -> None:", util_source)
        self.assertIn("def inject_bracketed_paste(fd: int, *, text: str, suffix: bytes, delay_s: float = 0.05) -> None:", util_source)
        self.assertIn("_BRACKETED_PASTE_START", util_source)
        self.assertIn("_BRACKETED_PASTE_END", util_source)
        self.assertEqual(sum(src.count("_BRACKETED_PASTE_START =") for src in sources.values()), 1)
        self.assertIn("_pty_util.inject_bracketed_paste(fd, text=text, suffix=suffix, delay_s=delay_s)", sources[ROOT / "codoxear" / "broker.py"])
        self.assertIn("_pty_util.inject_bracketed_paste(fd, text=text, suffix=suffix, delay_s=delay_s)", sources[ROOT / "codoxear" / "sessiond.py"])


if __name__ == "__main__":
    unittest.main()
