from __future__ import annotations

from codoxear.agent_backend import get_agent_backend
from codoxear.rollout_chat_batch import _extract_chat_events
from codoxear.rollout_jsonl import JsonlRecord
from codoxear.rollout_log import _extract_positioned_chat_events


def _record(row: dict, start: int) -> JsonlRecord:
    return JsonlRecord(start=start, end=start + 1, obj=row)


def test_pi_backend_projects_compact_subagent_activity_narrations() -> None:
    pi = get_agent_backend("pi")

    active = pi.chat_event_from_log_row(
        {"type": "active_long_running", "id": "active-1", "timestamp": "2026-08-01T20:46:10.000Z"}
    )
    assert active == {
        "role": "assistant",
        "text": "Background task in progress...",
        "message_class": "narration",
        "message_id": "pi-subagent:active-1",
        "ts": 1785617170.0,
    }

    notice = pi.chat_event_from_log_row(
        {
            "type": "custom_message",
            "customType": "subagent_control_notice",
            "id": "notice-1",
            "timestamp": "2026-08-01T20:46:20.000Z",
            "content": (
                "Subagent needs attention: executor\n"
                "Run: 2a127391-d423-4cc4-9136-63b614cbbc56 step 1\n"
                "Signal: executor is waiting for a supervisor reply\n"
                "Hint: This long operational detail is intentionally omitted."
            ),
        }
    )
    assert notice is not None
    assert notice["role"] == "assistant"
    assert notice["message_class"] == "narration"
    assert notice["message_id"] == "pi-subagent:notice-1"
    assert notice["text"] == "Subagent needs attention — executor (run 2a127391): executor is waiting for a supervisor reply"
    assert "\n" not in notice["text"]

    result = pi.chat_event_from_log_row(
        {
            "type": "custom_message",
            "customType": "intercom_message",
            "id": "result-1",
            "timestamp": "2026-08-01T20:46:30.000Z",
            "content": (
                "**📨 From subagent-result** (/repo)\n\n"
                "subagent results\n\n"
                "Run: 2a127391-d423-4cc4-9136-63b614cbbc56\n"
                "Status: completed\n"
                "Children: 1 completed\n\n"
                "Summary:\n"
                "Implemented the requested parser projection and tests with a deliberately long description."
            ),
        }
    )
    assert result is not None
    assert result["role"] == "assistant"
    assert result["message_class"] == "narration"
    assert result["message_id"] == "pi-subagent:result-1"
    assert result["text"].startswith("Subagent result — completed (run 2a127391); 1 completed:")
    assert "Implemented the requested parser projection" in result["text"]
    assert len(result["text"]) <= 200

    intercom_notice = pi.chat_event_from_log_row(
        {
            "type": "custom_message",
            "customType": "intercom_message",
            "id": "intercom-notice-1",
            "content": (
                "**📨 From subagent-control** (/repo)\n\n"
                "subagent needs attention\n\n"
                "executor needs attention in run 2a127391-d423-4cc4-9136-63b614cbbc56.\n\n"
                "Status: subagent({ action: 'status' })"
            ),
        }
    )
    assert intercom_notice is not None
    assert intercom_notice["text"] == "Subagent needs attention — executor (run 2a127391)"


def test_pi_subagent_events_reach_batch_and_positioned_transcripts_once_per_row_id() -> None:
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
    assert [event["message_id"] for event in batch_events] == ["pi-subagent:active-1", "pi-subagent:notice-1"]

    positioned_events = _extract_positioned_chat_events(
        [_record(active, 10), _record(notice, 20), _record(active, 30)]
    )
    assert [event["message_id"] for event in positioned_events] == ["pi-subagent:active-1", "pi-subagent:notice-1"]
    assert [event["_before_byte"] for event in positioned_events] == [10, 20]
