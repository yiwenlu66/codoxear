import unittest
from pathlib import Path


ROLLOUT_LOG = Path(__file__).resolve().parents[1] / "codoxear" / "rollout_log.py"


class TestRolloutLogHelpersSource(unittest.TestCase):
    def test_chat_event_timestamp_and_message_id_helpers_are_not_redeclared(self) -> None:
        source = ROLLOUT_LOG.read_text(encoding="utf-8")
        self.assertEqual(source.count("def _event_ts("), 1)
        self.assertEqual(source.count("def _text_message_id("), 1)
        extract_start = source.index("def _extract_chat_events(")
        extract_end = source.index("def _extract_delivery_messages", extract_start)
        extract_block = source[extract_start:extract_end]
        self.assertNotIn("def event_ts(", extract_block)
        self.assertNotIn("def text_message_id(", extract_block)
        self.assertIn("ets = _event_ts(obj)", extract_block)
        self.assertIn('"message_id": _text_message_id(', extract_block)


if __name__ == "__main__":
    unittest.main()
