import subprocess
import tempfile
import unittest
from pathlib import Path

from codoxear import server


class TestSessionFileList(unittest.TestCase):
    def test_lists_only_direct_children_for_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "README.md").write_text("# repo\n", encoding="utf-8")
            nested = root / "src"
            nested.mkdir(parents=True, exist_ok=True)
            (nested / "app.py").write_text("print('ok')\n", encoding="utf-8")

            self.assertEqual(
                server._list_session_directory_entries(root, ""),
                [
                    {"name": "src", "path": "src", "kind": "dir"},
                    {"name": "README.md", "path": "README.md", "kind": "file"},
                ],
            )

    def test_lists_only_direct_children_for_nested_directory(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "src" / "components").mkdir(parents=True, exist_ok=True)
            (root / "src" / "main.tsx").write_text("export {};\n", encoding="utf-8")

            self.assertEqual(
                server._list_session_directory_entries(root, "src"),
                [
                    {"name": "components", "path": "src/components", "kind": "dir"},
                    {"name": "main.tsx", "path": "src/main.tsx", "kind": "file"},
                ],
            )

    def test_directories_sort_before_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "b-dir").mkdir(parents=True, exist_ok=True)
            (root / "a-dir").mkdir(parents=True, exist_ok=True)
            (root / "b.txt").write_text("b\n", encoding="utf-8")
            (root / "a.txt").write_text("a\n", encoding="utf-8")

            self.assertEqual(
                [
                    item["path"]
                    for item in server._list_session_directory_entries(root, "")
                ],
                ["a-dir", "b-dir", "a.txt", "b.txt"],
            )

    def test_ignores_builtin_filtered_directories(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / ".git").mkdir(parents=True, exist_ok=True)
            (root / ".git" / "config").write_text("[core]\n", encoding="utf-8")
            (root / "README.md").write_text("# repo\n", encoding="utf-8")

            self.assertEqual(
                server._list_session_directory_entries(root, ""),
                [{"name": "README.md", "path": "README.md", "kind": "file"}],
            )

    def test_root_gitignore_hides_ignored_files_and_directories(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / ".gitignore").write_text("ignored.txt\ncache/\n", encoding="utf-8")
            (root / "ignored.txt").write_text("ignored\n", encoding="utf-8")
            (root / "cache").mkdir(parents=True, exist_ok=True)
            (root / "visible.txt").write_text("visible\n", encoding="utf-8")

            self.assertEqual(
                server._list_session_directory_entries(root, ""),
                [
                    {"name": ".gitignore", "path": ".gitignore", "kind": "file"},
                    {"name": "visible.txt", "path": "visible.txt", "kind": "file"},
                ],
            )

    def test_rejects_escaping_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            with self.assertRaisesRegex(ValueError, "escapes session cwd"):
                server._list_session_directory_entries(root, "../outside")

    def test_rejects_non_directory_paths(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "README.md").write_text("# repo\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "path is not a directory"):
                server._list_session_directory_entries(root, "README.md")

    def test_rejects_missing_paths(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            with self.assertRaisesRegex(FileNotFoundError, "path not found"):
                server._list_session_directory_entries(root, "missing")

    def test_search_score_prefers_closer_basename_matches(self) -> None:
        best = server._file_search_score("src/app.py", "app")
        worse = server._file_search_score("docs/reference/application-notes.md", "app")

        self.assertGreater(best, worse)

    def test_search_walk_mode_returns_ranked_matches(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "src").mkdir(parents=True, exist_ok=True)
            (root / "docs").mkdir(parents=True, exist_ok=True)
            (root / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")
            (root / "docs" / "app-notes.md").write_text("# app\n", encoding="utf-8")
            (root / "docs" / "misc.txt").write_text("misc\n", encoding="utf-8")

            result = server._search_session_relative_files(root, query="app")

            self.assertEqual(result["mode"], "walk")
            self.assertFalse(result["truncated"])
            self.assertEqual(
                [item["path"] for item in result["matches"]],
                ["src/app.py", "docs/app-notes.md"],
            )

    def test_search_walk_mode_ignores_git_directory_contents(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / ".git").mkdir(parents=True, exist_ok=True)
            (root / ".git" / "config").write_text("[core]\n", encoding="utf-8")
            (root / "README.md").write_text("# repo\n", encoding="utf-8")

            result = server._search_session_relative_files(root, query="config")

            self.assertEqual(result["mode"], "walk")
            self.assertEqual(result["matches"], [])

    def test_search_git_mode_uses_git_tracked_and_other_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            subprocess.run(
                ["git", "init"],
                cwd=root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            (root / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
            (root / "src").mkdir(parents=True, exist_ok=True)
            (root / "notes").mkdir(parents=True, exist_ok=True)
            (root / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")
            (root / "notes" / "app.txt").write_text("notes\n", encoding="utf-8")
            (root / "ignored.txt").write_text("ignored\n", encoding="utf-8")
            subprocess.run(
                ["git", "add", ".gitignore", "src/app.py"],
                cwd=root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            result = server._search_session_relative_files(root, query="app")

            self.assertEqual(result["mode"], "git")
            self.assertFalse(result["truncated"])
            self.assertEqual(
                [item["path"] for item in result["matches"]],
                ["src/app.py", "notes/app.txt"],
            )


if __name__ == "__main__":
    unittest.main()
