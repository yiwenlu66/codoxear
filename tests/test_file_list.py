import tempfile
import unittest
from pathlib import Path

from codoxear.server import _list_session_relative_files


class TestSessionFileList(unittest.TestCase):
    def test_lists_files_relative_to_session_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "z.txt").write_text("z\n", encoding="utf-8")
            nested = root / "src" / "app.py"
            nested.parent.mkdir(parents=True, exist_ok=True)
            nested.write_text("print('ok')\n", encoding="utf-8")

            self.assertEqual(_list_session_relative_files(root), ["src/app.py", "z.txt"])

    def test_ignores_git_directory(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / ".git").mkdir(parents=True, exist_ok=True)
            (root / ".git" / "config").write_text("[core]\n", encoding="utf-8")
            (root / "README.md").write_text("# repo\n", encoding="utf-8")

            self.assertEqual(_list_session_relative_files(root), ["README.md"])


if __name__ == "__main__":
    unittest.main()
