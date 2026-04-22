import unittest
from pathlib import Path


SERVER_PY = Path(__file__).resolve().parents[1] / "codoxear" / "server.py"


class TestMessageRouteSource(unittest.TestCase):
    def test_server_has_no_legacy_messages_route(self) -> None:
        source = SERVER_PY.read_text(encoding="utf-8")
        self.assertNotIn('path.endswith("/messages")', source)
        self.assertNotIn('"offset": int(new_off)', source)
        self.assertNotIn('"next_before": int(next_before)', source)

    def test_tail_live_history_routes_use_opaque_cursors(self) -> None:
        source = SERVER_PY.read_text(encoding="utf-8")
        self.assertIn('"live_cursor": live_cursor', source)
        self.assertIn('"history_cursor": history_cursor', source)
        self.assertIn('{"error": "cursor required"}', source)
        self.assertIn('_decode_message_cursor(cursor_q[0], kind="live", session=s)', source)
        self.assertIn('_decode_message_cursor(cursor_q[0], kind="history", session=s)', source)


if __name__ == "__main__":
    unittest.main()
