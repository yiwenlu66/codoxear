import threading
import unittest
from pathlib import Path

from codoxear.server import Session
from codoxear.server import SessionManager


def _make_manager() -> SessionManager:
    mgr = SessionManager.__new__(SessionManager)
    mgr._lock = threading.Lock()
    mgr._sessions = {}
    mgr._files = {}
    mgr._save_files = lambda: None  # type: ignore[assignment]
    return mgr


def _make_session(*, sid: str, cwd: str) -> Session:
    p = Path("/tmp") / f"{sid}.jsonl"
    return Session(
        session_id=sid,
        thread_id="thread-1",
        broker_pid=1,
        codex_pid=1,
        cli="codex",
        owned=False,
        start_ts=0.0,
        cwd=cwd,
        log_path=p,
        sock_path=p.with_suffix(".sock"),
    )


class TestFileHistory(unittest.TestCase):
    def test_remove_file_from_workspace(self) -> None:
        mgr = _make_manager()
        sess = _make_session(sid="sid-a", cwd="/tmp/workspace")
        mgr._sessions[sess.session_id] = sess

        mgr.files_add(sess.session_id, "/tmp/workspace/a.txt")
        mgr.files_add(sess.session_id, "/tmp/workspace/b.txt")

        remaining = mgr.files_remove(sess.session_id, "/tmp/workspace/a.txt")
        self.assertEqual(remaining, ["/tmp/workspace/b.txt"])
        self.assertEqual(mgr.files_get(sess.session_id), ["/tmp/workspace/b.txt"])

    def test_clear_workspace_clears_shared_key(self) -> None:
        mgr = _make_manager()
        s1 = _make_session(sid="sid-a", cwd="/tmp/workspace")
        s2 = _make_session(sid="sid-b", cwd="/tmp/workspace")
        mgr._sessions[s1.session_id] = s1
        mgr._sessions[s2.session_id] = s2

        mgr.files_add(s1.session_id, "/tmp/workspace/a.txt")
        mgr.files_clear_scope(s2.session_id)

        self.assertEqual(mgr.files_get(s1.session_id), [])
        self.assertEqual(mgr.files_get(s2.session_id), [])

    def test_remove_file_by_cwd(self) -> None:
        mgr = _make_manager()
        s1 = _make_session(sid="sid-a", cwd="/tmp/workspace")
        s2 = _make_session(sid="sid-b", cwd="/tmp/workspace")
        mgr._sessions[s1.session_id] = s1
        mgr._sessions[s2.session_id] = s2

        mgr.files_add(s1.session_id, "/tmp/workspace/a.txt")
        mgr.files_add(s2.session_id, "/tmp/workspace/b.txt")

        mgr.files_remove_cwd("/tmp/workspace", "/tmp/workspace/a.txt")

        self.assertEqual(mgr.files_get(s1.session_id), ["/tmp/workspace/b.txt"])
        self.assertEqual(mgr.files_get(s2.session_id), ["/tmp/workspace/b.txt"])

    def test_clear_files_by_cwd(self) -> None:
        mgr = _make_manager()
        s1 = _make_session(sid="sid-a", cwd="/tmp/workspace")
        s2 = _make_session(sid="sid-b", cwd="/tmp/workspace")
        mgr._sessions[s1.session_id] = s1
        mgr._sessions[s2.session_id] = s2

        mgr.files_add(s1.session_id, "/tmp/workspace/a.txt")
        mgr.files_add(s2.session_id, "/tmp/workspace/b.txt")

        mgr.files_clear_cwd("/tmp/workspace")

        self.assertEqual(mgr.files_get(s1.session_id), [])
        self.assertEqual(mgr.files_get(s2.session_id), [])

    def test_remove_file_from_all_keys(self) -> None:
        mgr = _make_manager()
        s1 = _make_session(sid="sid-a", cwd="/tmp/workspace-a")
        s2 = _make_session(sid="sid-b", cwd="/tmp/workspace-b")
        mgr._sessions[s1.session_id] = s1
        mgr._sessions[s2.session_id] = s2

        mgr.files_add(s1.session_id, "/tmp/workspace-a/a.txt")
        mgr.files_add(s2.session_id, "/tmp/workspace-a/a.txt")

        removed = mgr.files_remove_all("/tmp/workspace-a/a.txt")
        self.assertEqual(removed, 2)
        self.assertEqual(mgr.files_get(s1.session_id), [])
        self.assertEqual(mgr.files_get(s2.session_id), [])


if __name__ == "__main__":
    unittest.main()
