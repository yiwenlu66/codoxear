import unittest
from unittest.mock import patch

import codoxear.server as server


class TestUpdateHelpers(unittest.TestCase):
    def test_parse_upstream_ref(self) -> None:
        self.assertEqual(server._parse_upstream_ref("origin/main"), ("origin", "main"))
        self.assertEqual(server._parse_upstream_ref("upstream/feature/x"), ("upstream", "feature/x"))
        self.assertIsNone(server._parse_upstream_ref(""))
        self.assertIsNone(server._parse_upstream_ref("HEAD"))
        self.assertIsNone(server._parse_upstream_ref("main"))

    def test_github_repo_base(self) -> None:
        self.assertEqual(
            server._github_repo_base("https://github.com/san-tian/codoxear.git"),
            "https://github.com/san-tian/codoxear",
        )
        self.assertEqual(
            server._github_repo_base("git@github.com:yiwenlu66/codoxear.git"),
            "https://github.com/yiwenlu66/codoxear",
        )
        self.assertIsNone(server._github_repo_base("https://example.com/repo.git"))

    def test_parse_ls_remote_head(self) -> None:
        raw = "4f8da3701653982d3c32e635713ec19cadce9ab4\trefs/heads/main\n"
        self.assertEqual(
            server._parse_ls_remote_head(raw, branch="main", remote="origin"),
            "4f8da3701653982d3c32e635713ec19cadce9ab4",
        )


class TestUpdateStatus(unittest.TestCase):
    def setUp(self) -> None:
        self._prev_cache = server._UPDATE_CHECK_CACHE
        server._UPDATE_CHECK_CACHE = None

    def tearDown(self) -> None:
        server._UPDATE_CHECK_CACHE = self._prev_cache

    def test_check_update_status_success(self) -> None:
        local = "1111111111111111111111111111111111111111"
        remote = "2222222222222222222222222222222222222222"

        def fake_git_run(args, *, required=True, timeout_s=None):
            mapping = {
                ("rev-parse", "--is-inside-work-tree"): "true",
                ("rev-parse", "HEAD"): local,
                ("rev-parse", "--abbrev-ref", "HEAD"): "feature/claude-code-support",
                ("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"): "origin/main",
                ("ls-remote", "--heads", "origin", "main"): f"{remote}\trefs/heads/main\n",
                ("rev-list", "--left-right", "--count", f"{local}...{remote}"): "0\t3",
                ("remote", "get-url", "origin"): "https://github.com/san-tian/codoxear.git",
            }
            key = tuple(args)
            if key not in mapping:
                raise AssertionError(f"unexpected git args: {args}")
            return mapping[key]

        with patch("codoxear.server._git_run", side_effect=fake_git_run), patch(
            "codoxear.server.UPDATE_CHECK_REMOTE", ""
        ), patch("codoxear.server.UPDATE_CHECK_BRANCH", ""), patch("codoxear.server.time.time", return_value=1000.0):
            data = server._check_update_status_now()

        self.assertTrue(data["ok"])
        self.assertTrue(data["update_available"])
        self.assertEqual(data["remote"], "origin")
        self.assertEqual(data["branch"], "main")
        self.assertEqual(data["local_commit"], local)
        self.assertEqual(data["remote_commit"], remote)
        self.assertEqual(data["local_only_commits"], 0)
        self.assertEqual(data["remote_only_commits"], 3)
        self.assertEqual(
            data["compare_url"],
            f"https://github.com/san-tian/codoxear/compare/{local}...{remote}",
        )

    def test_check_update_status_handles_git_error(self) -> None:
        with patch("codoxear.server._git_run", side_effect=RuntimeError("boom")), patch(
            "codoxear.server.time.time", return_value=1000.0
        ):
            data = server._check_update_status_now()

        self.assertFalse(data["ok"])
        self.assertFalse(data["update_available"])
        self.assertIn("boom", data["error"])

    def test_check_update_status_marks_update_when_divergence_unavailable(self) -> None:
        local = "1111111111111111111111111111111111111111"
        remote = "2222222222222222222222222222222222222222"

        def fake_git_run(args, *, required=True, timeout_s=None):
            mapping = {
                ("rev-parse", "--is-inside-work-tree"): "true",
                ("rev-parse", "HEAD"): local,
                ("rev-parse", "--abbrev-ref", "HEAD"): "main",
                ("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"): "origin/main",
                ("ls-remote", "--heads", "origin", "main"): f"{remote}\trefs/heads/main\n",
                ("remote", "get-url", "origin"): "https://github.com/san-tian/codoxear.git",
            }
            key = tuple(args)
            if key == ("rev-list", "--left-right", "--count", f"{local}...{remote}"):
                raise RuntimeError("bad object")
            if key not in mapping:
                raise AssertionError(f"unexpected git args: {args}")
            return mapping[key]

        with patch("codoxear.server._git_run", side_effect=fake_git_run), patch(
            "codoxear.server.UPDATE_CHECK_REMOTE", ""
        ), patch("codoxear.server.UPDATE_CHECK_BRANCH", ""), patch("codoxear.server.time.time", return_value=1000.0):
            data = server._check_update_status_now()

        self.assertTrue(data["ok"])
        self.assertTrue(data["update_available"])
        self.assertEqual(data["remote_only_commits"], 1)

    def test_update_status_uses_cache(self) -> None:
        result = {
            "ok": True,
            "checked_at": 1000,
            "update_available": False,
            "remote": "origin",
            "branch": "main",
            "local_commit": "1111111111111111111111111111111111111111",
            "remote_commit": "1111111111111111111111111111111111111111",
            "local_only_commits": 0,
            "remote_only_commits": 0,
        }

        with patch("codoxear.server.UPDATE_CHECK_TTL_SECONDS", 600.0), patch(
            "codoxear.server.time.time", side_effect=[100.0, 101.0]
        ), patch("codoxear.server._check_update_status_now", return_value=result) as run_check:
            first = server._update_status()
            second = server._update_status()

        self.assertEqual(first, second)
        self.assertEqual(run_check.call_count, 1)


if __name__ == "__main__":
    unittest.main()
