import os
import unittest
from pathlib import Path
from unittest.mock import patch

from codoxear import pty_util


ROOT = Path(__file__).resolve().parents[1]


class TestPtyUtilSource(unittest.TestCase):
    def test_bracketed_paste_injection_is_centralized(self) -> None:
        sources = {path: path.read_text(encoding="utf-8") for path in (ROOT / "codoxear").glob("*.py")}
        util_source = sources[ROOT / "codoxear" / "pty_util.py"]
        self.assertIn("def write_all(fd: int, data: bytes) -> None:", util_source)
        self.assertIn("def inject_bracketed_paste(fd: int, *, text: str, suffix: bytes, delay_s: float = 0.05) -> None:", util_source)
        self.assertIn("def term_size(stdin) -> tuple[int, int]:", util_source)
        self.assertIn("_BRACKETED_PASTE_START", util_source)
        self.assertIn("_BRACKETED_PASTE_END", util_source)
        self.assertEqual(sum(src.count("_BRACKETED_PASTE_START =") for src in sources.values()), 1)
        broker_source = sources[ROOT / "codoxear" / "broker.py"]
        self.assertIn("_pty_util.inject_bracketed_paste(fd, text=text, suffix=suffix, delay_s=delay_s)", broker_source)
        self.assertIn("_pty_util.term_size(sys.stdin)", broker_source)
        self.assertNotIn("os.get_terminal_size(sys.stdin.fileno())", broker_source)
        self.assertIn("_pty_util.inject_bracketed_paste(fd, text=text, suffix=suffix, delay_s=delay_s)", sources[ROOT / "codoxear" / "sessiond.py"])

    def test_term_size_reads_stdin_dimensions_and_falls_back(self) -> None:
        class FakeStdin:
            def fileno(self) -> int:
                return 7

        with patch("codoxear.pty_util.os.get_terminal_size", return_value=os.terminal_size((132, 43))) as get_size:
            self.assertEqual(pty_util.term_size(FakeStdin()), (43, 132))
        get_size.assert_called_once_with(7)

        with patch("codoxear.pty_util.os.get_terminal_size", side_effect=OSError("no tty")):
            self.assertEqual(pty_util.term_size(FakeStdin()), (40, 120))


if __name__ == "__main__":
    unittest.main()
