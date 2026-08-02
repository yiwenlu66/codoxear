from __future__ import annotations

from codoxear.agent_backend import get_agent_backend
from codoxear.rollout_chat_batch import _extract_chat_events
from codoxear.rollout_jsonl import JsonlRecord
from codoxear.rollout_log import _extract_positioned_chat_events


def _record(row: dict, start: int) -> JsonlRecord:
    return JsonlRecord(start=start, end=start + 1, obj=row)


def test_pi_backend_excludes_harness_coordination_traffic_from_transcript() -> None:
    """Harness<->agent coordination rows are not agent->user communication and
    must not surface as transcript rows. Awareness of background work lives in
    the ambient subagent indicator (subagents_running), not in narration."""
    pi = get_agent_backend("pi")

    assert (
        pi.chat_event_from_log_row(
            {"type": "active_long_running", "id": "active-1", "timestamp": "2026-08-01T20:46:10.000Z"}
        )
        is None
    )

    notice = {
        "type": "custom_message",
        "customType": "subagent_control_notice",
        "id": "notice-1",
        "timestamp": "2026-08-01T20:46:20.000Z",
        "content": "Subagent progress update\nRun: abcdef12-3456\nUPDATE: executor finished the inspection",
    }
    assert pi.chat_event_from_log_row(notice) is None

    intercom = {
        "type": "custom_message",
        "customType": "intercom_message",
        "id": "intercom-1",
        "timestamp": "2026-08-01T20:46:30.000Z",
        "content": "Subagent needs attention: executor is waiting for a supervisor reply",
    }
    assert pi.chat_event_from_log_row(intercom) is None

    # Other custom_message rows are likewise excluded as a class.
    assert pi.chat_event_from_log_row({"type": "custom_message", "customType": "other", "content": "x"}) is None


def test_pi_harness_rows_produce_no_batch_or_positioned_events() -> None:
    active = {"type": "active_long_running", "id": "active-1", "timestamp": "2026-08-01T20:46:10.000Z"}
    notice = {
        "type": "custom_message",
        "customType": "subagent_control_notice",
        "id": "notice-1",
        "timestamp": "2026-08-01T20:46:20.000Z",
        "content": "Subagent progress update\nRun: abcdef12-3456\nUPDATE: executor finished the inspection",
    }
    rows = [active, notice, active]

    batch_events, _meta, _flags, _diag = _extract_chat_events(rows)
    assert batch_events == []

    positioned_events = _extract_positioned_chat_events(
        [_record(active, 10), _record(notice, 20), _record(active, 30)]
    )
    assert positioned_events == []
