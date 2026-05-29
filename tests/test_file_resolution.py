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
                mgr.files_get.return_value = []
                res_out = _resolve_client_file_path_typed(session_id="sid", raw_path="/etc/hostname")
                res_in = _resolve_client_file_path_typed(session_id="sid", raw_path=str(inside))
        self.assertEqual(res_out.status, "outside_allowed_root")
        self.assertIsNone(res_out.path)
        self.assertEqual(res_in.status, "ok")

    def test_relative_dotdot_traversal_is_blocked(self):
        # A relative path that climbs out of the session cwd via `..` must be
        # blocked even though the joined path points at a real file on disk.
        from unittest.mock import patch

        class _S:
            def __init__(self, cwd):
                self.cwd = cwd

        with TemporaryDirectory() as tmp:
            cwd = Path(tmp) / "cwd"
            cwd.mkdir()
            secret = Path(tmp) / "secret.txt"
            secret.write_text("classified")
            with patch("codoxear.server.MANAGER") as mgr:
                mgr.get_session.return_value = _S(str(cwd))
                mgr.refresh_session_meta.return_value = None
                mgr.files_get.return_value = []
                res = _resolve_client_file_path_typed(session_id="sid", raw_path="../secret.txt")
        self.assertEqual(res.status, "outside_allowed_root")
        self.assertIsNone(res.path)

    def test_inside_cwd_symlink_to_outside_is_blocked(self):
        # A live symlink that lives inside the cwd but resolves to a target
        # outside it must be blocked (its resolved target escapes the root).
        from unittest.mock import patch

        class _S:
            def __init__(self, cwd):
                self.cwd = cwd

        with TemporaryDirectory() as tmp:
            cwd = Path(tmp) / "cwd"
            cwd.mkdir()
            secret = Path(tmp) / "secret.txt"
            secret.write_text("classified")
            link = cwd / "link"
            link.symlink_to(secret)
            with patch("codoxear.server.MANAGER") as mgr:
                mgr.get_session.return_value = _S(str(cwd))
                mgr.refresh_session_meta.return_value = None
                mgr.files_get.return_value = []
                res_rel = _resolve_client_file_path_typed(session_id="sid", raw_path="link")
                res_abs = _resolve_client_file_path_typed(session_id="sid", raw_path=str(link))
        self.assertEqual(res_rel.status, "outside_allowed_root")
        self.assertEqual(res_abs.status, "outside_allowed_root")

    def test_inside_cwd_dead_symlink_still_reports_dead(self):
        # A dead symlink inside the cwd (target absent) must report dead_symlink,
        # not outside_allowed_root, even when the target text points outside.
        from unittest.mock import patch

        class _S:
            def __init__(self, cwd):
                self.cwd = cwd

        with TemporaryDirectory() as tmp:
            cwd = Path(tmp) / "cwd"
            cwd.mkdir()
            link = cwd / "deadlink"
            link.symlink_to(Path(tmp) / "absent_target")
            with patch("codoxear.server.MANAGER") as mgr:
                mgr.get_session.return_value = _S(str(cwd))
                mgr.refresh_session_meta.return_value = None
                mgr.files_get.return_value = []
                res = _resolve_client_file_path_typed(session_id="sid", raw_path="deadlink")
        self.assertEqual(res.status, "dead_symlink")
        self.assertIn("absent_target", res.target)

    def test_inside_cwd_relative_file_is_ok(self):
        # Sanity: an ordinary in-cwd relative file still resolves ok after the
        # containment checks are in place (no false positives).
        from unittest.mock import patch

        class _S:
            def __init__(self, cwd):
                self.cwd = cwd

        with TemporaryDirectory() as tmp:
            cwd = Path(tmp) / "cwd"
            cwd.mkdir()
            real = cwd / "real.txt"
            real.write_text("hi")
            with patch("codoxear.server.MANAGER") as mgr:
                mgr.get_session.return_value = _S(str(cwd))
                mgr.refresh_session_meta.return_value = None
                mgr.files_get.return_value = []
                res = _resolve_client_file_path_typed(session_id="sid", raw_path="real.txt")
        self.assertEqual(res.status, "ok")
        self.assertIsNotNone(res.path)

    def test_symlinked_cwd_component_does_not_false_reject(self):
        # The session cwd is reached through a symlinked parent component (real
        # tree: tmp/real/proj; the session reports its cwd as tmp/link/proj where
        # tmp/link -> tmp/real). allowed_root resolves symlinks; the containment
        # check must resolve the request's parent the same way, otherwise every
        # in-cwd file is falsely rejected as outside_allowed_root. This is the
        # macOS /tmp -> /private/tmp and bind-mount-home case.
        from unittest.mock import patch

        class _S:
            def __init__(self, cwd):
                self.cwd = cwd

        with TemporaryDirectory() as tmp:
            real = Path(tmp) / "real"
            proj = real / "proj"
            proj.mkdir(parents=True)
            (proj / "real.txt").write_text("hi")
            link = Path(tmp) / "link"
            link.symlink_to(real)
            session_cwd = link / "proj"  # symlinked spelling of proj
            with patch("codoxear.server.MANAGER") as mgr:
                mgr.get_session.return_value = _S(str(session_cwd))
                mgr.refresh_session_meta.return_value = None
                mgr.files_get.return_value = []
                res_rel = _resolve_client_file_path_typed(session_id="sid", raw_path="real.txt")
                res_abs = _resolve_client_file_path_typed(session_id="sid", raw_path=str(session_cwd / "real.txt"))
                res_escape = _resolve_client_file_path_typed(session_id="sid", raw_path="../secret.txt")
        self.assertEqual(res_rel.status, "ok")
        self.assertEqual(res_abs.status, "ok")
        # `..` still escapes even through the symlinked spelling.
        self.assertEqual(res_escape.status, "outside_allowed_root")

    def test_symlinked_dir_with_missing_intermediate_does_not_resolve_ok_outside(self):
        # The write-overwrite escape: an in-cwd symlink points to a directory
        # outside the root (cwd/sl -> outside), and the request walks through it
        # with a nonexistent intermediate segment and a `..`
        # ("sl/nope/../secret.txt"). The strict resolve raises (nope absent), so
        # the resolver must NOT return an `ok` FileResolution whose path lands
        # outside the root. It must be not_found or outside_allowed_root; never
        # `ok` pointing at outside/secret.txt. (The handler then refuses to fall
        # back to a weaker resolver, so no write escapes.)
        from unittest.mock import patch

        class _S:
            def __init__(self, cwd):
                self.cwd = cwd

        with TemporaryDirectory() as tmp:
            cwd = Path(tmp) / "cwd"
            cwd.mkdir()
            outside = Path(tmp) / "outside"
            outside.mkdir()
            secret = outside / "secret.txt"
            secret.write_text("classified")
            (cwd / "sl").symlink_to(outside)
            with patch("codoxear.server.MANAGER") as mgr:
                mgr.get_session.return_value = _S(str(cwd))
                mgr.refresh_session_meta.return_value = None
                mgr.files_get.return_value = []
                res = _resolve_client_file_path_typed(session_id="sid", raw_path="sl/nope/../secret.txt")
        # Must not be served as a contained file pointing outside the root.
        self.assertNotEqual(res.status, "ok")
        if res.path is not None:
            real_root = Path(str(cwd)).resolve()
            self.assertTrue(
                str(res.path.resolve()).startswith(str(real_root) + os.sep) or res.path.resolve() == real_root,
                f"resolver returned a path outside the root: {res.path}",
            )

    def test_absolute_path_with_unresolvable_session_root_fails_closed(self):
        # If a session id is supplied but its root cannot be resolved (eviction
        # race / MANAGER error -> allowed_root None), an absolute path must NOT be
        # served. Fail closed instead of disabling containment.
        from unittest.mock import patch

        with patch("codoxear.server.MANAGER") as mgr:
            mgr.get_session.return_value = None
            mgr.refresh_session_meta.return_value = None
            mgr.files_get.return_value = []
            res = _resolve_client_file_path_typed(session_id="sid", raw_path="/etc/hostname")
        self.assertEqual(res.status, "not_found")
        self.assertIsNone(res.path)

    def test_relative_traversal_with_unresolvable_session_fails_closed(self):
        # A non-empty but unknown session id with a relative `..` path must fail
        # closed, NOT fall through to resolving against the server process cwd
        # (which would serve /etc/passwd via ../../.. with no containment).
        from unittest.mock import patch

        with patch("codoxear.server.MANAGER") as mgr:
            mgr.get_session.return_value = None
            mgr.refresh_session_meta.return_value = None
            mgr.files_get.return_value = []
            res = _resolve_client_file_path_typed(
                session_id="nonexistent", raw_path="../../../../../../etc/passwd"
            )
        self.assertEqual(res.status, "not_found")
        self.assertIsNone(res.path)

    def test_sessionless_relative_path_still_resolves(self):
        # The sessionless global case (session_id="") is intentionally not
        # cwd-scoped; a relative path still resolves against the process cwd and
        # is not broken by the named-session fail-closed guard.
        res = _resolve_client_file_path_typed(session_id="", raw_path="definitely_missing_xyz_abc.txt")
        self.assertIn(res.status, {"not_found", "ok"})


if __name__ == "__main__":
    unittest.main()
