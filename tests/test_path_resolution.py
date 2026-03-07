import subprocess
import tempfile
import unittest
from pathlib import Path

from codoxear.server import _resolve_git_path
from codoxear.server import _resolve_unique_bare_filename
from codoxear.server import _resolve_session_path


class TestPathResolution(unittest.TestCase):
    def test_resolve_session_path_allows_absolute(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            target = base / "a" / "b.txt"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("x", encoding="utf-8")
            resolved = _resolve_session_path(base, str(target))
            self.assertEqual(resolved, target.resolve())

    def test_resolve_session_path_resolves_relative_against_session_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            target = base / "a" / "b.txt"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("x", encoding="utf-8")
            resolved = _resolve_session_path(base, "a/b.txt")
            self.assertEqual(resolved, target.resolve())

    def test_resolve_git_path_allows_absolute_inside_repo(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            subprocess.run(["git", "init"], cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            target = root / "dir" / "file.txt"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("x", encoding="utf-8")
            resolved, repo_root, rel = _resolve_git_path(root, str(target))
            self.assertEqual(resolved, target.resolve())
            self.assertEqual(repo_root, root.resolve())
            self.assertEqual(rel, "dir/file.txt")

    def test_resolve_git_path_rejects_absolute_outside_repo(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "repo"
            other = Path(td) / "other.txt"
            root.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init"], cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            other.write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside git repo"):
                _resolve_git_path(root, str(other))

    def test_resolve_unique_bare_filename_returns_unique_match(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            target = root / "codoxear" / "server.py"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("x", encoding="utf-8")
            resolved = _resolve_unique_bare_filename(root, "server.py")
            self.assertEqual(resolved, target.resolve())

    def test_resolve_unique_bare_filename_ignores_build_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            target = root / "codoxear" / "server.py"
            dup = root / "build" / "lib" / "codoxear" / "server.py"
            target.parent.mkdir(parents=True, exist_ok=True)
            dup.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("x", encoding="utf-8")
            dup.write_text("x", encoding="utf-8")
            resolved = _resolve_unique_bare_filename(root, "server.py")
            self.assertEqual(resolved, target.resolve())

    def test_resolve_unique_bare_filename_rejects_ambiguous_match(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            a = root / "a" / "app.js"
            b = root / "b" / "app.js"
            a.parent.mkdir(parents=True, exist_ok=True)
            b.parent.mkdir(parents=True, exist_ok=True)
            a.write_text("a", encoding="utf-8")
            b.write_text("b", encoding="utf-8")
            self.assertIsNone(_resolve_unique_bare_filename(root, "app.js"))
