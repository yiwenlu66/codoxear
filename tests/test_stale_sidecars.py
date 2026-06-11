import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from codoxear import server
from codoxear.server import Session
from codoxear.server import SessionManager


def _make_manager() -> SessionManager:
    mgr = SessionManager.__new__(SessionManager)
    mgr._lock = threading.Lock()
    mgr._sessions = {}
    mgr._harness = {}
    mgr._aliases = {}
    mgr._sidebar_meta = {}
    mgr._hidden_sessions = set()
    mgr._files = {}
    mgr._queues = {}
    mgr._recent_cwds = {}
    mgr._include_launch_attempts = True
    mgr._last_discover_ts = 0.0
    mgr._save_hidden_sessions = lambda *args, **kwargs: None  # type: ignore[method-assign]
    mgr._save_aliases = lambda *args, **kwargs: None  # type: ignore[method-assign]
    mgr._save_sidebar_meta = lambda *args, **kwargs: None  # type: ignore[method-assign]
    mgr._save_harness = lambda *args, **kwargs: None  # type: ignore[method-assign]
    mgr._save_files = lambda *args, **kwargs: None  # type: ignore[method-assign]
    mgr._save_queues = lambda *args, **kwargs: None  # type: ignore[method-assign]
    mgr._save_recent_cwds = lambda *args, **kwargs: None  # type: ignore[method-assign]
    return mgr


def _session(session_id: str, sock: Path) -> Session:
    return Session(
        session_id=session_id,
        thread_id=session_id,
        broker_pid=0,
        codex_pid=0,
        agent_backend="codex",
        owned=False,
        transport="pty",
        start_ts=time.time(),
        cwd="/tmp",
        log_path=None,
        sock_path=sock,
    )


class TestStaleSidecars(unittest.TestCase):
    def test_discovery_prunes_sock_without_metadata_sidecar(self) -> None:
        with TemporaryDirectory() as td:
            sock_dir = Path(td)
            sock = sock_dir / "stale.sock"
            sock.touch()
            mgr = _make_manager()
            mgr._sessions["stale"] = _session("stale", sock)
            mgr._hidden_sessions.add("stale")
            mgr._aliases["stale"] = "Stale"
            mgr._queues["stale"] = [{"id": "q", "text": "later"}]

            with patch.object(server, "SOCK_DIR", sock_dir):
                SessionManager._discover_existing(mgr, force=True)

            self.assertFalse(sock.exists())
            self.assertNotIn("stale", mgr._sessions)
            self.assertNotIn("stale", mgr._hidden_sessions)
            self.assertNotIn("stale", mgr._aliases)
            self.assertNotIn("stale", mgr._queues)
            self.assertGreater(mgr._last_discover_ts, 0)

    def test_refresh_prunes_existing_session_when_sidecar_disappears(self) -> None:
        with TemporaryDirectory() as td:
            sock = Path(td) / "gone.sock"
            sock.touch()
            mgr = _make_manager()
            mgr._sessions["gone"] = _session("gone", sock)

            SessionManager.refresh_session_meta(mgr, "gone")

            self.assertFalse(sock.exists())
            self.assertNotIn("gone", mgr._sessions)


if __name__ == "__main__":
    unittest.main()
