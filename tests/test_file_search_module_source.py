import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER_PY = ROOT / "codoxear" / "server.py"
FILE_SEARCH_PY = ROOT / "codoxear" / "file_search.py"


class TestFileSearchModuleSource(unittest.TestCase):
    def test_file_search_implementation_lives_outside_server(self) -> None:
        server_source = SERVER_PY.read_text(encoding="utf-8")
        module_source = FILE_SEARCH_PY.read_text(encoding="utf-8")

        self.assertIn("from .file_search import file_search_score as _file_search_score", server_source)
        self.assertIn("from .file_search import search_session_relative_files as _search_session_relative_files_impl", server_source)
        self.assertIn("def _search_session_relative_files(", server_source)
        self.assertIn("git_root_func=_git_repo_root", server_source)
        self.assertNotIn("def _search_git_relative_files(", server_source)
        self.assertNotIn("def _search_walk_relative_files(", server_source)
        self.assertNotIn("def _push_file_search_match(", server_source)
        self.assertNotIn("def _finish_file_search(", server_source)

        self.assertIn("def file_search_score(", module_source)
        self.assertIn("def search_session_relative_files(", module_source)
        self.assertIn("def search_git_relative_files(", module_source)
        self.assertIn("def search_walk_relative_files(", module_source)


if __name__ == "__main__":
    unittest.main()
