import os
import unittest
from unittest.mock import patch

from codoxear import pty_util


class TestPtyUtil(unittest.TestCase):
    def test_term_size_reads_stdin_dimensions_and_falls_back(self) -> None:
        class FakeStdin:
            def fileno(self) -> int:
                return 7

        with patch("codoxear.pty_util.os.get_terminal_size", return_value=os.terminal_size((132, 43))) as get_size:
            self.assertEqual(pty_util.term_size(FakeStdin()), (43, 132))
        get_size.assert_called_once_with(7)

        with patch("codoxear.pty_util.os.get_terminal_size", side_effect=OSError("no tty")):
            self.assertEqual(pty_util.term_size(FakeStdin()), (40, 120))

    def test_inject_bracketed_paste_writes_wrapped_text_and_suffix(self) -> None:
        read_fd, write_fd = os.pipe()
        try:
            pty_util.inject_bracketed_paste(write_fd, text="hello", suffix=b"\r", delay_s=0.0)
            data = os.read(read_fd, 4096)
        finally:
            os.close(read_fd)
            os.close(write_fd)

        self.assertEqual(data, b"\x1b[200~hello\x1b[201~\r")


if __name__ == "__main__":
    unittest.main()
