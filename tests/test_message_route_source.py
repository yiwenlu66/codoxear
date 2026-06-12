import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER_PY = ROOT / "codoxear" / "server.py"
MESSAGE_CURSOR_PY = ROOT / "codoxear" / "message_cursor.py"


class TestMessageRouteSource(unittest.TestCase):
    def test_server_has_no_legacy_messages_route(self) -> None:
        source = SERVER_PY.read_text(encoding="utf-8")
        self.assertNotIn('path.endswith("/messages")', source)
        self.assertNotIn('"offset": int(new_off)', source)
        self.assertNotIn('"next_before": int(next_before)', source)

    def test_tail_live_history_routes_use_opaque_cursors(self) -> None:
        source = SERVER_PY.read_text(encoding="utf-8")
        cursor_source = MESSAGE_CURSOR_PY.read_text(encoding="utf-8")
        self.assertIn('"live_cursor": live_cursor', source)
        self.assertIn('"history_cursor": history_cursor', source)
        self.assertIn('ev2["history_cursor"] = encode_cursor(kind="history", session=session, pos=pos)', cursor_source)
        self.assertIn('events = _attach_history_cursors(events, session=s)', source)
        self.assertIn('{"error": "cursor required"}', source)
        self.assertIn('_decode_message_cursor(cursor_q[0], kind="live", session=s)', source)
        self.assertIn('_decode_message_cursor(cursor_q[0], kind="history", session=s)', source)
        self.assertIn('initial_cc_pending = _rollout_log._cc_pending_tool_ids_before(s.log_path, after_byte) if after_byte > 0 else set()', source)
        self.assertIn('events, meta_delta, flags, diag = _extract_chat_events(objs, initial_cc_pending_tool_ids=initial_cc_pending)', source)
        self.assertIn('events = _extract_positioned_chat_events(records, initial_cc_pending_tool_ids=initial_cc_pending)', source)


if __name__ == "__main__":
    unittest.main()
