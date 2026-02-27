import threading
import unittest
from pathlib import Path

from codoxear.server import Session
from codoxear.server import SessionManager


def _make_manager() -> SessionManager:
    mgr = SessionManager.__new__(SessionManager)
    mgr._lock = threading.Lock()
    mgr._sessions = {}
    mgr._aliases = {}
    mgr._files = {}
    mgr._harness = {}
    mgr._last_discover_ts = 0.0
    mgr._stop = threading.Event()
    mgr._discover_existing_if_stale = lambda *args, **kwargs: None  # type: ignore[assignment]
    mgr._prune_dead_sessions = lambda *args, **kwargs: None  # type: ignore[assignment]
    mgr._update_meta_counters = lambda *args, **kwargs: None  # type: ignore[assignment]
    mgr._save_files = lambda: None  # type: ignore[assignment]
    return mgr


class TestSessionTmux(unittest.TestCase):
    def test_list_sessions_includes_tmux_name(self) -> None:
        mgr = _make_manager()
        sock = Path("/tmp/fake.sock")
        sess = Session(
            session_id="sid-a",
            thread_id="thread-1",
            broker_pid=123,
            codex_pid=456,
            cli="codex",
            owned=True,
            start_ts=0.0,
            cwd="/tmp/workspace",
            log_path=None,
            sock_path=sock,
            tmux_name="codoxear-web-abc123",
        )
        mgr._sessions[sess.session_id] = sess

        items = mgr.list_sessions()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].get("cli"), "codex")
        self.assertEqual(items[0].get("tmux_name"), "codoxear-web-abc123")


if __name__ == "__main__":
    unittest.main()
