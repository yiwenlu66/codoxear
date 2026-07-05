import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from codoxear import file_search as file_search_module
from codoxear.file_search import file_search_score
from codoxear.file_search import search_session_relative_files
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

    def test_search_score_prefers_closer_basename_matches(self) -> None:
        best = file_search_score("src/app.py", "app")
        worse = file_search_score("docs/reference/application-notes.md", "app")

        self.assertGreater(best, worse)

    def test_search_walk_mode_returns_ranked_matches(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "src").mkdir(parents=True, exist_ok=True)
            (root / "docs").mkdir(parents=True, exist_ok=True)
            (root / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")
            (root / "docs" / "app-notes.md").write_text("# app\n", encoding="utf-8")
            (root / "docs" / "misc.txt").write_text("misc\n", encoding="utf-8")

            result = search_session_relative_files(root, query="app")

            self.assertEqual(result["mode"], "walk")
            self.assertFalse(result["truncated"])
            self.assertEqual([item["path"] for item in result["matches"]], ["src/app.py", "docs/app-notes.md"])

    def test_search_walk_mode_ignores_git_directory_contents(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / ".git").mkdir(parents=True, exist_ok=True)
            (root / ".git" / "config").write_text("[core]\n", encoding="utf-8")
            (root / "README.md").write_text("# repo\n", encoding="utf-8")

            result = search_session_relative_files(root, query="config")

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

            result = search_session_relative_files(root, query="app")

            self.assertEqual(result["mode"], "git")
            self.assertFalse(result["truncated"])
            self.assertEqual([item["path"] for item in result["matches"]], ["src/app.py", "notes/app.txt"])

    def test_search_git_mode_cap_parity_with_walk_mode_scanned_count(self) -> None:
        # Under a candidate cap, git and walk modes must report the same
        # ``scanned`` count for the same number of on-disk files. Previously git
        # mode returned scanned=cap+1 while walk returned scanned=cap because the
        # cap-exceeding candidate was counted but not processed. Both modes now
        # report the number of candidates actually admitted under the cap.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "a.py").write_text("a\n", encoding="utf-8")
            (root / "b.py").write_text("b\n", encoding="utf-8")
            (root / "c.py").write_text("c\n", encoding="utf-8")

            with patch.object(file_search_module, "FILE_SEARCH_MAX_CANDIDATES", 2):
                walk_result = search_session_relative_files(root, query="py")
                self.assertEqual(walk_result["mode"], "walk")
                self.assertTrue(walk_result["truncated"])
                self.assertEqual(walk_result["scanned"], 2)

            subprocess.run(["git", "init"], cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with patch.object(file_search_module, "FILE_SEARCH_MAX_CANDIDATES", 2):
                git_result = search_session_relative_files(root, query="py")
                self.assertEqual(git_result["mode"], "git")
                self.assertTrue(git_result["truncated"])
                self.assertEqual(git_result["scanned"], 2)
                # Non-UTF token emission is preserved under truncation.
                for match in git_result["matches"]:
                    self.assertIn("path", match)

    def test_list_serializes_non_utf8_filename_without_surrogates(self) -> None:
        # A raw 0xff byte in a filename is undecodable as UTF-8; os.walk surfaces
        # it as a lone surrogate (\udcff). The list must return a JSON/UTF-8 safe
        # display string instead of crashing on response encoding.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "ok.txt").write_text("ok\n", encoding="utf-8")
            bad_name = os.fsencode(root) + b"/bad\xffname.bin"
            with open(bad_name, "wb") as fh:
                fh.write(b"\x00\x01")

            files = _list_session_relative_files(root)

            # The lone surrogate must not be present; the path is JSON-encodable
            # and uses the same backslashreplace display convention as git paths.
            self.assertIn(r"bad\xffname.bin", files)
            for value in files:
                self.assertNotIn("\udcff", value)
                self.assertNotIn("\udc80", value)
            body = json.dumps({"ok": True, "files": files}, ensure_ascii=False).encode("utf-8")
            self.assertIn(r"bad\\xffname.bin".encode("utf-8"), body)

    def test_search_walk_mode_serializes_non_utf8_filename(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "src").mkdir(parents=True, exist_ok=True)
            (root / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")
            bad_name = os.fsencode(root) + b"/src/bad\xffname.bin"
            with open(bad_name, "wb") as fh:
                fh.write(b"\x00\x01")

            result = search_session_relative_files(root, query="name")

            self.assertEqual(result["mode"], "walk")
            paths = [item["path"] for item in result["matches"]]
            # The surrogate-named file is found and its path is JSON/UTF-8 safe
            # with the same backslashreplace display convention as git paths.
            self.assertIn(r"src/bad\xffname.bin", paths)
            for value in paths:
                self.assertNotIn("\udcff", value)
            body = json.dumps(result, ensure_ascii=False).encode("utf-8")
            self.assertIn(r"src/bad\\xffname.bin".encode("utf-8"), body)


if __name__ == "__main__":
    unittest.main()
