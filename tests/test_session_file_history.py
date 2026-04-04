import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import codoxear.server as server
from codoxear.server import Session
from codoxear.server import SessionManager


class _FakeVoicePushCoordinator:
    def __init__(self, *args, **kwargs) -> None:
        pass


def _make_session(session_id: str, cwd: str) -> Session:
    return Session(
        session_id=session_id,
        thread_id=f"thread-{session_id}",
        broker_pid=1,
        codex_pid=2,
        agent_backend="codex",
        owned=False,
        start_ts=0.0,
        cwd=cwd,
        log_path=None,
        sock_path=Path(f"/tmp/{session_id}.sock"),
    )


class TestSessionFileHistory(unittest.TestCase):
    def _build_manager(self) -> SessionManager:
        with patch.object(SessionManager, "_load_harness", lambda self: None), \
            patch.object(SessionManager, "_load_aliases", lambda self: None), \
            patch.object(SessionManager, "_load_sidebar_meta", lambda self: None), \
            patch.object(SessionManager, "_load_hidden_sessions", lambda self: None), \
            patch.object(SessionManager, "_load_files", lambda self: None), \
            patch.object(SessionManager, "_load_queues", lambda self: None), \
            patch.object(SessionManager, "_load_recent_cwds", lambda self: None), \
            patch.object(SessionManager, "_backfill_recent_cwds_from_logs", lambda self: None), \
            patch.object(SessionManager, "_discover_existing", lambda self, *args, **kwargs: None), \
            patch.object(server, "VoicePushCoordinator", _FakeVoicePushCoordinator), \
            patch("threading.Thread.start", lambda self: None):
            return SessionManager()

    def test_history_is_scoped_per_session_even_with_same_cwd(self) -> None:
        mgr = self._build_manager()
        mgr._sessions = {
            "session-a": _make_session("session-a", "/tmp/shared-project"),
            "session-b": _make_session("session-b", "/tmp/shared-project"),
        }

        mgr.files_add("session-a", "/tmp/shared-project/file-a.py")

        self.assertEqual(mgr.files_get("session-a"), ["/tmp/shared-project/file-a.py"])
        self.assertEqual(mgr.files_get("session-b"), [])

    def test_delete_session_keeps_other_same_cwd_history(self) -> None:
        mgr = self._build_manager()
        mgr._sessions = {
            "session-a": _make_session("session-a", "/tmp/shared-project"),
            "session-b": _make_session("session-b", "/tmp/shared-project"),
        }
        mgr._files = {}
        mgr._queues = {}
        mgr._aliases = {}
        mgr._sidebar_meta = {}
        mgr._harness = {}

        mgr.files_add("session-a", "/tmp/shared-project/file-a.py")
        mgr.files_add("session-b", "/tmp/shared-project/file-b.py")
        mgr._sock_call = lambda *_args, **_kwargs: {"ok": True}  # type: ignore[method-assign]

        self.assertTrue(mgr.delete_session("session-a"))
        self.assertEqual(mgr.files_get("session-b"), ["/tmp/shared-project/file-b.py"])

    def test_load_files_discards_legacy_cwd_buckets(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "session_files.json"
            path.write_text(
                json.dumps(
                    {
                        "cwd:/tmp/shared-project": ["/tmp/shared-project/file-a.py"],
                        "session-a": ["/tmp/project-a/legacy.py"],
                        "sid:session-b": ["/tmp/project-b/current.py"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            mgr = SessionManager.__new__(SessionManager)
            mgr._lock = threading.Lock()
            mgr._files = {}

            with patch.object(server, "FILE_HISTORY_PATH", path):
                SessionManager._load_files(mgr)

        self.assertEqual(
            mgr._files,
            {
                "sid:session-a": ["/tmp/project-a/legacy.py"],
                "sid:session-b": ["/tmp/project-b/current.py"],
            },
        )


if __name__ == "__main__":
    unittest.main()
