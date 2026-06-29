import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
GIT_OPS_PY = ROOT / "codoxear" / "git_ops.py"


class TestNewSessionWorktreeSource(unittest.TestCase):
    def test_frontend_does_not_own_worktree_path_slug_policy(self) -> None:
        app_source = APP_JS.read_text(encoding="utf-8")
        git_ops_source = GIT_OPS_PY.read_text(encoding="utf-8")

        self.assertNotIn("function worktreePathSlug(", app_source)
        self.assertIn("const worktreeBranch = !resumeSessionId && newSessionWorktreeToggle.checked ? String(newSessionWorktreeInput.value || \"\").trim() : null;", app_source)
        self.assertIn("if (worktreeBranch) body.worktree_branch = String(worktreeBranch);", app_source)
        self.assertIn("def worktree_path_slug(branch: str) -> str:", git_ops_source)
        self.assertIn("target = default_worktree_path(source_cwd, branch)", git_ops_source)


if __name__ == "__main__":
    unittest.main()
