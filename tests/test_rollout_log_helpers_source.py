import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROLLOUT_LOG = ROOT / "codoxear" / "rollout_log.py"
ROLLOUT_JSONL = ROOT / "codoxear" / "rollout_jsonl.py"
ROLLOUT_EVENTS = ROOT / "codoxear" / "rollout_events.py"
ROLLOUT_CHAT_EVENTS = ROOT / "codoxear" / "rollout_chat_events.py"
ROLLOUT_CHAT_BATCH = ROOT / "codoxear" / "rollout_chat_batch.py"
ROLLOUT_TOKENS = ROOT / "codoxear" / "rollout_tokens.py"
ROLLOUT_DELIVERY = ROOT / "codoxear" / "rollout_delivery.py"
ROLLOUT_IDLE = ROOT / "codoxear" / "rollout_idle.py"


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

    def test_chat_batch_extraction_has_dedicated_owner(self) -> None:
        source = ROLLOUT_LOG.read_text(encoding="utf-8")
        batch_source = ROLLOUT_CHAT_BATCH.read_text(encoding="utf-8")
        self.assertIn("from .rollout_chat_batch import _extract_chat_events", source)
        self.assertNotIn("def _extract_chat_events", source)
        self.assertIn("def _extract_chat_events", batch_source)
        self.assertIn("event = _single_chat_event(obj, cc_pending_tool_ids=cc_pending_tool_ids)", batch_source)
        self.assertIn("events.append(event)", batch_source)

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

    def test_idle_analysis_has_dedicated_owner(self) -> None:
        source = ROLLOUT_LOG.read_text(encoding="utf-8")
        idle_source = ROLLOUT_IDLE.read_text(encoding="utf-8")
        self.assertIn("from .rollout_idle import _compute_idle_from_log", source)
        self.assertIn("from .rollout_idle import _analyze_log_chunk", source)
        self.assertNotIn("def _compute_idle_from_log", source)
        self.assertNotIn("def _analyze_log_chunk", source)
        self.assertIn("def _compute_idle_from_log", idle_source)
        self.assertIn("def _analyze_log_chunk", idle_source)

    def test_chat_event_timestamp_and_message_id_helpers_are_not_redeclared(self) -> None:
        source = ROLLOUT_LOG.read_text(encoding="utf-8")
        event_source = ROLLOUT_EVENTS.read_text(encoding="utf-8")
        self.assertIn("from .rollout_events import _event_ts", source)
        self.assertIn("from .rollout_events import _text_message_id", source)
        self.assertNotIn("def _event_ts(", source)
        self.assertNotIn("def _text_message_id(", source)
        self.assertEqual(event_source.count("def _event_ts("), 1)
        self.assertEqual(event_source.count("def _text_message_id("), 1)
        extract_start = event_source.index("def _event_ts(")
        extract_end = event_source.index("def _strip_oai_mem_citation_tail", extract_start)
        extract_block = event_source[extract_start:extract_end]
        self.assertNotIn("def event_ts(", extract_block)
        self.assertNotIn("def text_message_id(", extract_block)


if __name__ == "__main__":
    unittest.main()
