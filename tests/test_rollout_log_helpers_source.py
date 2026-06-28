import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROLLOUT_LOG = ROOT / "codoxear" / "rollout_log.py"
ROLLOUT_JSONL = ROOT / "codoxear" / "rollout_jsonl.py"
ROLLOUT_EVENTS = ROOT / "codoxear" / "rollout_events.py"
ROLLOUT_CHAT_EVENTS = ROOT / "codoxear" / "rollout_chat_events.py"
ROLLOUT_TOKENS = ROOT / "codoxear" / "rollout_tokens.py"
ROLLOUT_DELIVERY = ROOT / "codoxear" / "rollout_delivery.py"


class TestRolloutLogHelpersSource(unittest.TestCase):
    def test_delivery_message_record_import_is_lightweight(self) -> None:
        source = ROLLOUT_LOG.read_text(encoding="utf-8")
        self.assertIn("from .voice_push_state import ClassifiedAssistantMessage", source)
        self.assertNotIn("from .voice_push import ClassifiedAssistantMessage", source)

    def test_jsonl_reader_primitives_have_dedicated_owner(self) -> None:
        source = ROLLOUT_LOG.read_text(encoding="utf-8")
        jsonl_source = ROLLOUT_JSONL.read_text(encoding="utf-8")
        self.assertIn("from .rollout_jsonl import JsonlRecord", source)
        self.assertNotIn("class JsonlRecord", source)
        self.assertNotIn("def _read_jsonl_records_from_offset", source)
        self.assertIn("class JsonlRecord", jsonl_source)
        self.assertIn("def _read_jsonl_records_from_offset", jsonl_source)

    def test_single_row_chat_event_policy_has_dedicated_owner(self) -> None:
        source = ROLLOUT_LOG.read_text(encoding="utf-8")
        chat_source = ROLLOUT_CHAT_EVENTS.read_text(encoding="utf-8")
        self.assertIn("from .rollout_chat_events import _single_chat_event", source)
        self.assertIn("from .rollout_chat_events import _dedupe_assistant_chat_events", source)
        self.assertNotIn("def _single_chat_event", source)
        self.assertNotIn("def _dedupe_assistant_chat_events", source)
        self.assertIn("def _single_chat_event", chat_source)
        self.assertIn("def _dedupe_assistant_chat_events", chat_source)

    def test_token_context_scanners_have_dedicated_owner(self) -> None:
        source = ROLLOUT_LOG.read_text(encoding="utf-8")
        token_source = ROLLOUT_TOKENS.read_text(encoding="utf-8")
        self.assertIn("from .rollout_tokens import _extract_token_update", source)
        self.assertNotIn("def _extract_token_update", source)
        self.assertIn("def _extract_token_update", token_source)
        self.assertIn("def _find_latest_turn_context", token_source)

    def test_delivery_message_extraction_has_dedicated_owner(self) -> None:
        source = ROLLOUT_LOG.read_text(encoding="utf-8")
        delivery_source = ROLLOUT_DELIVERY.read_text(encoding="utf-8")
        self.assertIn("from .rollout_delivery import _extract_delivery_messages", source)
        self.assertNotIn("def _extract_delivery_messages", source)
        self.assertIn("def _extract_delivery_messages", delivery_source)
        self.assertIn("ClassifiedAssistantMessage", delivery_source)

    def test_chat_event_timestamp_and_message_id_helpers_are_not_redeclared(self) -> None:
        source = ROLLOUT_LOG.read_text(encoding="utf-8")
        event_source = ROLLOUT_EVENTS.read_text(encoding="utf-8")
        self.assertIn("from .rollout_events import _event_ts", source)
        self.assertIn("from .rollout_events import _text_message_id", source)
        self.assertNotIn("def _event_ts(", source)
        self.assertNotIn("def _text_message_id(", source)
        self.assertEqual(event_source.count("def _event_ts("), 1)
        self.assertEqual(event_source.count("def _text_message_id("), 1)
        extract_start = source.index("def _extract_chat_events(")
        extract_end = source.index("def _read_chat_tail_snapshot", extract_start)
        extract_block = source[extract_start:extract_end]
        self.assertNotIn("def event_ts(", extract_block)
        self.assertNotIn("def text_message_id(", extract_block)
        self.assertIn("event = _single_chat_event(obj, cc_pending_tool_ids=cc_pending_tool_ids)", extract_block)
        self.assertIn("events.append(event)", extract_block)


if __name__ == "__main__":
    unittest.main()
