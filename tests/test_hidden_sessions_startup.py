import threading
import unittest
from unittest.mock import patch

from codoxear.server import SessionManager


class TestHiddenSessionsStartup(unittest.TestCase):
    def test_hidden_sessions_load_before_discovery(self) -> None:
        order: list[str] = []

        def record(name):
            def _fn(self, *args, **kwargs):
                order.append(name)
                if name == "_load_hidden_sessions":
                    self._hidden_sessions = {"terminal-hidden"}
                return None

            return _fn

        with patch.object(SessionManager, "_load_harness", record("_load_harness")), \
            patch.object(SessionManager, "_load_aliases", record("_load_aliases")), \
            patch.object(SessionManager, "_load_sidebar_meta", record("_load_sidebar_meta")), \
            patch.object(SessionManager, "_load_hidden_sessions", record("_load_hidden_sessions")), \
            patch.object(SessionManager, "_load_files", record("_load_files")), \
            patch.object(SessionManager, "_load_queues", record("_load_queues")), \
            patch.object(SessionManager, "_discover_existing", record("_discover_existing")), \
            patch("threading.Thread.start", lambda self: None):
            mgr = SessionManager()

        self.assertEqual(mgr._hidden_sessions, {"terminal-hidden"})
        self.assertLess(order.index("_load_hidden_sessions"), order.index("_discover_existing"))


if __name__ == "__main__":
    unittest.main()
