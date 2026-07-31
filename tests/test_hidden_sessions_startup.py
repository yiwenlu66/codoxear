import threading
import unittest
from unittest.mock import patch

from codoxear.server import SessionManager
from codoxear.session_store import SessionStore


class TestHiddenSessionsStartup(unittest.TestCase):
    def test_hidden_sessions_load_before_discovery(self) -> None:
        order: list[str] = []

        def load_state(store):
            order.append("load_persistent_state")
            store.hidden_sessions = {"terminal-hidden"}

        def discover(self, *args, **kwargs):
            order.append("_discover_existing")

        with patch.object(SessionStore, "load_persistent_state", load_state), \
            patch.object(SessionManager, "_discover_existing", discover), \
            patch("threading.Thread.start", lambda self: None):
            mgr = SessionManager()

        self.assertEqual(mgr._hidden_sessions, {"terminal-hidden"})
        self.assertLess(order.index("load_persistent_state"), order.index("_discover_existing"))


if __name__ == "__main__":
    unittest.main()
