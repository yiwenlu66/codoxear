"""File resolver outcome tests for `_resolve_client_file_path_typed`."""

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from codoxear.server import _resolve_client_file_path_typed


class FileResolutionOutcomeTests(unittest.TestCase):
    def test_ok_for_existing_absolute_path(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "real.txt"
            p.write_text("hi")
            res = _resolve_client_file_path_typed(session_id="", raw_path=str(p))
            self.assertEqual(res.status, "ok")
            self.assertIsNotNone(res.path)
            if res.path:
                self.assertTrue(res.path.exists())

    def test_not_found_for_missing_absolute_path(self):
        with TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "nope.txt")
            res = _resolve_client_file_path_typed(session_id="", raw_path=missing)
        self.assertEqual(res.status, "not_found")
        self.assertIsNone(res.path)

    def test_dead_symlink_classified_correctly(self):
        with TemporaryDirectory() as tmp:
            link = Path(tmp) / "link"
            link.symlink_to(Path(tmp) / "absent_target")
            res = _resolve_client_file_path_typed(session_id="", raw_path=str(link))
        self.assertEqual(res.status, "dead_symlink")
        self.assertIsNone(res.path)
        self.assertIn("absent_target", res.target)

    def test_permission_denied_when_parent_unreadable(self):
        if os.geteuid() == 0:
            self.skipTest("root bypasses permission bits")
        with TemporaryDirectory() as tmp:
            sub = Path(tmp) / "locked"
            sub.mkdir()
            inner = sub / "secret.txt"
            inner.write_text("x")
            try:
                os.chmod(sub, 0o000)
                res = _resolve_client_file_path_typed(session_id="", raw_path=str(inner))
            finally:
                os.chmod(sub, 0o700)
        self.assertEqual(res.status, "permission_denied")

    def test_relative_path_without_session_uses_cwd(self):
        # When no session id is passed and the path is relative, the resolver
        # falls back to current process cwd. Just confirm it does not crash
        # and returns a typed result rather than a bare Path.
        res = _resolve_client_file_path_typed(session_id="", raw_path="definitely_missing_xyz_abc.txt")
        self.assertIn(res.status, {"not_found", "ok"})

    def test_absolute_path_outside_session_root_is_blocked(self):
        # With a session whose cwd is a tmp dir, an absolute path pointing
        # outside that tree must be reported as outside_allowed_root rather than
        # served (prevents reading e.g. /etc/hostname from a session view).
        from unittest.mock import patch

        class _S:
            def __init__(self, cwd):
                self.cwd = cwd

        with TemporaryDirectory() as tmp:
            inside = Path(tmp) / "inside.txt"
            inside.write_text("ok")
            with patch("codoxear.server.MANAGER") as mgr:
                mgr.get_session.return_value = _S(tmp)
                mgr.refresh_session_meta.return_value = None
                res_out = _resolve_client_file_path_typed(session_id="sid", raw_path="/etc/hostname")
                res_in = _resolve_client_file_path_typed(session_id="sid", raw_path=str(inside))
        self.assertEqual(res_out.status, "outside_allowed_root")
        self.assertIsNone(res_out.path)
        self.assertEqual(res_in.status, "ok")


if __name__ == "__main__":
    unittest.main()
