from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from codoxear import git_ops
from codoxear import server


class TestGitOps(unittest.TestCase):
    def test_parse_git_numstat_handles_renames_and_binary_totals(self) -> None:
        text = "\0".join(["1\t2\ta.txt", "-\t-\tb.bin", "4\t5\t", "old name.txt", "new name.txt", ""])
        self.assertEqual(
            git_ops.parse_git_numstat(text),
            {
                "a.txt": {"additions": 1, "deletions": 2},
                "b.bin": {"additions": None, "deletions": None},
                "new name.txt": {"additions": 4, "deletions": 5},
            },
        )

    def test_resolve_git_path_uses_injected_run_git_and_preserves_backslash_literal(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()

            def fake_run_git(_cwd: Path, args: list[str], **_kwargs: object) -> str:
                self.assertEqual(args, ["rev-parse", "--show-toplevel"])
                return str(root)

            target, repo_root, rel = git_ops.resolve_git_path(root, "back\\slash.md", run_git_func=fake_run_git, timeout_s=1.0)
            self.assertEqual(repo_root, root)
            self.assertEqual(rel, "back\\slash.md")
            self.assertEqual(target, root / "back\\slash.md")

    def test_resolve_git_path_rejects_absolute_path_outside_repo(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td, "repo").resolve()
            outside = Path(td, "outside.txt").resolve()

            def fake_run_git(_cwd: Path, _args: list[str], **_kwargs: object) -> str:
                return str(root)

            with self.assertRaisesRegex(ValueError, "path is outside git repo"):
                git_ops.resolve_git_path(root, str(outside), run_git_func=fake_run_git, timeout_s=1.0)

    def test_current_git_branch_preserves_git_output_except_empty(self) -> None:
        self.assertEqual(git_ops.current_git_branch(Path("."), run_git_func=lambda *_args, **_kwargs: "HEAD\n", timeout_s=1.0), "HEAD")
        self.assertEqual(git_ops.current_git_branch(Path("."), run_git_func=lambda *_args, **_kwargs: "main\n", timeout_s=1.0), "main")
        self.assertIsNone(git_ops.current_git_branch(Path("."), run_git_func=lambda *_args, **_kwargs: "\n", timeout_s=1.0))

    def test_run_git_literal_pathspecs_sets_literal_environment(self) -> None:
        captured: dict[str, object] = {}

        class Proc:
            returncode = 0
            stdout = b"ok\n"
            stderr = b""

        def fake_run(cmd: list[str], **kwargs: object) -> Proc:
            captured["cmd"] = cmd
            captured["env"] = kwargs.get("env")
            return Proc()

        with patch.object(subprocess, "run", side_effect=fake_run):
            out = git_ops.run_git(Path("."), ["status"], timeout_s=1.0, max_bytes=4096, literal_pathspecs=True)
        self.assertEqual(out, "ok\n")
        env = captured["env"]
        self.assertIsInstance(env, dict)
        self.assertEqual(env["GIT_LITERAL_PATHSPECS"], "1")
        self.assertNotIn("GIT_GLOB_PATHSPECS", env)

    def test_server_wrappers_preserve_run_git_patch_seam(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            calls: list[list[str]] = []

            def fake_run_git(_cwd: Path, args: list[str], **_kwargs: object) -> str:
                calls.append(args)
                return str(root)

            with patch("codoxear.server._run_git", side_effect=fake_run_git):
                target, repo_root, rel = server._resolve_git_path(root, "src/app.py")
            self.assertEqual(target, root / "src/app.py")
            self.assertEqual(repo_root, root)
            self.assertEqual(rel, "src/app.py")
            self.assertEqual(calls, [["rev-parse", "--show-toplevel"]])


if __name__ == "__main__":
    unittest.main()
