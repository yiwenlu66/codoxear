import subprocess
import tempfile
import unittest
from pathlib import Path

from codoxear.server import _file_search_score
from codoxear.server import _list_session_relative_files
from codoxear.server import _search_session_relative_files


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

    def test_search_score_prefers_closer_basename_matches(self) -> None:
        best = _file_search_score("src/app.py", "app")
        worse = _file_search_score("docs/reference/application-notes.md", "app")

        self.assertGreater(best, worse)

    def test_search_walk_mode_returns_ranked_matches(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "src").mkdir(parents=True, exist_ok=True)
            (root / "docs").mkdir(parents=True, exist_ok=True)
            (root / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")
            (root / "docs" / "app-notes.md").write_text("# app\n", encoding="utf-8")
            (root / "docs" / "misc.txt").write_text("misc\n", encoding="utf-8")

            result = _search_session_relative_files(root, query="app")

            self.assertEqual(result["mode"], "walk")
            self.assertFalse(result["truncated"])
            self.assertEqual([item["path"] for item in result["matches"]], ["src/app.py", "docs/app-notes.md"])

    def test_search_walk_mode_ignores_git_directory_contents(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / ".git").mkdir(parents=True, exist_ok=True)
            (root / ".git" / "config").write_text("[core]\n", encoding="utf-8")
            (root / "README.md").write_text("# repo\n", encoding="utf-8")

            result = _search_session_relative_files(root, query="config")

            self.assertEqual(result["mode"], "walk")
            self.assertEqual(result["matches"], [])

    def test_search_git_mode_uses_git_tracked_and_other_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            subprocess.run(["git", "init"], cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (root / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
            (root / "src").mkdir(parents=True, exist_ok=True)
            (root / "notes").mkdir(parents=True, exist_ok=True)
            (root / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")
            (root / "notes" / "app.txt").write_text("notes\n", encoding="utf-8")
            (root / "ignored.txt").write_text("ignored\n", encoding="utf-8")
            subprocess.run(["git", "add", ".gitignore", "src/app.py"], cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            result = _search_session_relative_files(root, query="app")

            self.assertEqual(result["mode"], "git")
            self.assertFalse(result["truncated"])
            self.assertEqual([item["path"] for item in result["matches"]], ["src/app.py", "notes/app.txt"])


if __name__ == "__main__":
    unittest.main()
