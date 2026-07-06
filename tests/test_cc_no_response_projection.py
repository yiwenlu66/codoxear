"""Claude Code transcript outcome projection tests.

Claude Code closes some turns with ``system`` rows instead of an assistant
message. A selectable backend must still obey the visible-result rule: a user
turn that reaches a terminal close without assistant text must render either a
backend error row (when the close carries a terminal API error) or the shared
no-response event.
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from codoxear.rollout_chat_events import _NO_RESPONSE_TEXT
from codoxear.rollout_log import _extract_positioned_chat_events
from codoxear.rollout_log import _read_chat_live_delta
from codoxear.rollout_log import _read_chat_tail_page
from codoxear.rollout_jsonl import JsonlRecord


SESSION_ID = "11111111-2222-3333-4444-555555555555"
TERMINAL_API_ERROR_TEXT = "API Error: 503 Service Unavailable"
TRANSIENT_API_ERROR_TEXT = "API Error: retrying after overload"


def _rec(obj: dict, start: int = 0) -> JsonlRecord:
    return JsonlRecord(start=start, end=start + 1, obj=obj)


def _write_log(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _cc_user(text: str = "hello", *, ts: str = "2026-07-04T00:00:00.000Z") -> dict:
    return {
        "type": "user",
        "sessionId": SESSION_ID,
        "timestamp": ts,
        "cwd": "/repo",
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
    }


def _cc_assistant_text(text: str = "done", *, ts: str = "2026-07-04T00:00:01.000Z") -> dict:
    return {
        "type": "assistant",
        "sessionId": SESSION_ID,
        "timestamp": ts,
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
        },
    }


def _cc_tool_use(*, ts: str = "2026-07-04T00:00:01.000Z") -> dict:
    return {
        "type": "assistant",
        "sessionId": SESSION_ID,
        "timestamp": ts,
        "message": {
            "role": "assistant",
            "content": [{"type": "tool_use", "name": "Bash", "id": "toolu_1", "input": {}}],
            "stop_reason": "tool_use",
        },
    }


def _cc_tool_result(*, ts: str = "2026-07-04T00:00:02.000Z") -> dict:
    return {
        "type": "user",
        "sessionId": SESSION_ID,
        "timestamp": ts,
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
        },
    }


def _cc_turn_duration(*, ts: str = "2026-07-04T00:00:03.000Z") -> dict:
    return {
        "type": "system",
        "subtype": "turn_duration",
        "sessionId": SESSION_ID,
        "timestamp": ts,
        "durationMs": 1234,
    }


def _cc_system_api_error(
    text: str,
    *,
    retry_attempt: int,
    max_retries: int,
    ts: str = "2026-07-04T00:00:03.000Z",
) -> dict:
    return {
        "type": "system",
        "subtype": "api_error",
        "sessionId": SESSION_ID,
        "timestamp": ts,
        "error": text,
        "retryAttempt": retry_attempt,
        "maxRetries": max_retries,
    }


def test_cc_user_then_turn_duration_no_response_emits_error_event() -> None:
    records = [
        _rec(_cc_user("silent"), start=0),
        _rec(_cc_turn_duration(), start=100),
    ]

    events = _extract_positioned_chat_events(records)

    assert [ev["role"] for ev in events] == ["user", "assistant"]
    assert events[1]["text"] == _NO_RESPONSE_TEXT
    assert events[1]["message_class"] == "error"
    assert events[1]["_before_byte"] == 100


def test_cc_tools_only_turn_duration_no_response_emits_error_event() -> None:
    records = [
        _rec(_cc_user("run a tool"), start=0),
        _rec(_cc_tool_use(), start=50),
        _rec(_cc_tool_result(), start=100),
        _rec(_cc_turn_duration(), start=150),
    ]

    events = _extract_positioned_chat_events(records)

    assert [ev["role"] for ev in events] == ["user", "assistant"]
    assert events[1]["text"] == _NO_RESPONSE_TEXT
    assert events[1]["message_class"] == "error"
    assert events[1]["_before_byte"] == 150


def test_cc_terminal_system_api_error_projects_backend_error_text() -> None:
    records = [
        _rec(_cc_user("fail"), start=0),
        _rec(_cc_system_api_error(TERMINAL_API_ERROR_TEXT, retry_attempt=3, max_retries=3), start=100),
    ]

    events = _extract_positioned_chat_events(records)

    assert [ev["role"] for ev in events] == ["user", "assistant"]
    assert events[1]["text"] == TERMINAL_API_ERROR_TEXT
    assert events[1]["message_class"] == "error"
    assert events[1]["text"] != _NO_RESPONSE_TEXT
    assert events[1]["_before_byte"] == 100


def test_cc_transient_system_api_error_does_not_project_transcript_error_by_itself() -> None:
    records = [
        _rec(_cc_user("retry"), start=0),
        _rec(_cc_system_api_error(TRANSIENT_API_ERROR_TEXT, retry_attempt=1, max_retries=3), start=100),
    ]

    events = _extract_positioned_chat_events(records)

    assert [ev["role"] for ev in events] == ["user"]
    assert all(ev.get("text") != TRANSIENT_API_ERROR_TEXT for ev in events)
    assert all(ev.get("text") != _NO_RESPONSE_TEXT for ev in events)


def test_cc_answered_turn_duration_remains_answered_without_no_response() -> None:
    records = [
        _rec(_cc_user("answer me"), start=0),
        _rec(_cc_assistant_text("answered"), start=100),
        _rec(_cc_turn_duration(), start=200),
    ]

    events = _extract_positioned_chat_events(records)

    assert [ev["role"] for ev in events] == ["user", "assistant"]
    assert events[1]["text"] == "answered"
    assert events[1]["message_class"] == "final_response"
    assert all(ev["text"] != _NO_RESPONSE_TEXT for ev in events)


def test_cc_tail_page_surfaces_turn_duration_no_response(tmp_path: Path) -> None:
    path = tmp_path / "cc-session.jsonl"
    _write_log(path, [_cc_user("silent"), _cc_turn_duration()])

    events, _before, _after, _has_older = _read_chat_tail_page(path, limit=20)

    assert [ev["role"] for ev in events] == ["user", "assistant"]
    assert events[1]["text"] == _NO_RESPONSE_TEXT
    assert events[1]["message_class"] == "error"


def test_cc_live_split_turn_duration_after_user_emits_no_response() -> None:
    with TemporaryDirectory() as d:
        path = Path(d) / "cc-session.jsonl"
        _write_log(path, [_cc_user("silent")])

        events1, cursor, _meta, _flags, _diag, _token = _read_chat_live_delta(path, after_byte=0)
        assert [ev["role"] for ev in events1] == ["user"]

        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_cc_turn_duration()) + "\n")

        events2, _next, _meta, _flags, _diag, _token = _read_chat_live_delta(path, after_byte=cursor)
        assert [ev["role"] for ev in events2] == ["assistant"]
        assert events2[0]["text"] == _NO_RESPONSE_TEXT
        assert events2[0]["message_class"] == "error"


def test_cc_live_split_prior_answer_suppresses_late_turn_duration_no_response() -> None:
    with TemporaryDirectory() as d:
        path = Path(d) / "cc-session.jsonl"
        _write_log(path, [_cc_user("answer"), _cc_assistant_text("already answered")])
        cursor = path.stat().st_size

        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_cc_turn_duration()) + "\n")

        events, _next, _meta, _flags, _diag, _token = _read_chat_live_delta(path, after_byte=cursor)
        assert events == []


def test_cc_live_split_terminal_api_error_after_user_projects_error() -> None:
    with TemporaryDirectory() as d:
        path = Path(d) / "cc-session.jsonl"
        _write_log(path, [_cc_user("fail later")])
        _events1, cursor, _meta, _flags, _diag, _token = _read_chat_live_delta(path, after_byte=0)

        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_cc_system_api_error(TERMINAL_API_ERROR_TEXT, retry_attempt=3, max_retries=3)) + "\n")

        events2, _next, _meta, _flags, _diag, _token = _read_chat_live_delta(path, after_byte=cursor)
        assert [ev["role"] for ev in events2] == ["assistant"]
        assert events2[0]["text"] == TERMINAL_API_ERROR_TEXT
        assert events2[0]["message_class"] == "error"
        assert events2[0]["text"] != _NO_RESPONSE_TEXT
