from __future__ import annotations

from codoxear.agent_backend import get_agent_backend
from codoxear.rollout_chat_batch import _extract_chat_events
from codoxear.rollout_jsonl import JsonlRecord
from codoxear.rollout_log import _extract_positioned_chat_events


def _record(row: dict, start: int) -> JsonlRecord:
    return JsonlRecord(start=start, end=start + 1, obj=row)


def test_pi_backend_normalizes_subagent_notices_but_excludes_generic_coordination() -> None:
    pi = get_agent_backend("pi")

    assert pi.chat_event_from_log_row({"type": "active_long_running", "id": "active-1"}) is None

    notice = {
        "type": "custom_message",
        "customType": "subagent_control_notice",
        "id": "notice-1",
        "timestamp": "2026-08-05T00:00:00Z",
        "content": "Subagent progress update\nRun: abcdef12-3456\nUPDATE: executor finished the inspection",
    }
    assert pi.chat_event_from_log_row(notice) == {
        "role": "assistant",
        "text": "Subagent progress update (run abcdef12): executor finished the inspection",
        "message_class": "narration",
        "message_id": "pi-subagent:notice-1",
        "ts": 1_785_888_000.0,
    }

    intercom = {
        "type": "custom_message",
        "customType": "intercom_message",
        "id": "intercom-1",
        "timestamp": "2026-08-05T00:00:01Z",
        "content": "Subagent needs attention: executor is waiting for a supervisor reply",
    }
    assert pi.chat_event_from_log_row(intercom) == {
        "role": "assistant",
        "text": "Subagent needs attention",
        "message_class": "narration",
        "message_id": "pi-subagent:intercom-1",
        "ts": 1_785_888_001.0,
    }

    assert pi.chat_event_from_log_row({"type": "custom_message", "customType": "other", "content": "x"}) is None


def test_pi_subagent_events_dedupe_in_batch_and_positioned_transcript() -> None:
    notice = {
        "type": "custom_message",
        "customType": "subagent_control_notice",
        "id": "notice-1",
        "timestamp": "2026-08-05T00:00:00Z",
        "content": "Subagent progress update\nRun: abcdef12-3456\nUPDATE: executor finished the inspection",
    }

    batch_events, _meta, _flags, _diag = _extract_chat_events([notice, notice])
    assert [event["message_id"] for event in batch_events] == ["pi-subagent:notice-1"]

    positioned_events = _extract_positioned_chat_events([_record(notice, 10), _record(notice, 20)])
    assert [event["message_id"] for event in positioned_events] == ["pi-subagent:notice-1"]
