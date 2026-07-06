"""Tests for the Codex no-response projection fix.

When a Codex turn closes (task_complete / turn_complete) after a user message
with no assistant output and no explicit error, the normalizer must emit an
explicit no-response event into the transcript so the user understands the turn
produced no answer.  These tests exercise the source-of-truth boundary
(``_extract_positioned_chat_events`` and the message tail/live routes) and
verify the exact event shape.
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from codoxear.rollout_chat_events import _NO_RESPONSE_TEXT
from codoxear.rollout_chat_events import _build_no_response_event
from codoxear.rollout_chat_events import _detect_codex_no_response_closes
from codoxear.rollout_chat_events import _inject_no_response_events
from codoxear.rollout_events import _INTERRUPTED_TEXT
from codoxear.rollout_jsonl import JsonlRecord
from codoxear.rollout_log import _extract_positioned_chat_events
from codoxear.rollout_log import _codex_prior_open_turn_context
from codoxear.rollout_log import _read_chat_live_delta
from codoxear.rollout_log import _read_chat_tail_page


def _rec(obj: dict, start: int = 0) -> JsonlRecord:
    return JsonlRecord(start=start, end=start + 1, obj=obj)


def _write_log(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Fixture: user message + task_complete with no assistant output → explicit
#    no-response event in the extracted transcript.
# ---------------------------------------------------------------------------
def test_user_then_task_complete_no_response_emits_no_response_event() -> None:
    records = [
        _rec({"type": "event_msg", "ts": 1.0, "payload": {"type": "user_message", "message": "hello"}}, start=0),
        _rec({"type": "event_msg", "ts": 2.0, "payload": {"type": "task_complete", "turn_id": "t1", "last_agent_message": None}}, start=100),
    ]
    events = _extract_positioned_chat_events(records)

    roles = [ev["role"] for ev in events]
    assert roles == ["user", "assistant"]
    no_response = events[-1]
    assert no_response["message_class"] == "error"
    assert no_response["text"] == _NO_RESPONSE_TEXT
    assert isinstance(no_response["message_id"], str)
    assert no_response["ts"] == 2.0


def test_turn_complete_variant_also_emits_no_response_event() -> None:
    records = [
        _rec({"type": "event_msg", "ts": 10.0, "payload": {"type": "user_message", "message": "do something"}}, start=0),
        _rec({"type": "event_msg", "ts": 11.0, "payload": {"type": "turn_complete", "turn_id": "t2"}}, start=50),
    ]
    events = _extract_positioned_chat_events(records)
    assert events[-1]["text"] == _NO_RESPONSE_TEXT
    assert events[-1]["message_class"] == "error"


def test_no_response_event_is_backend_truthful() -> None:
    """The event text must not invent assistant content or leak debug info."""
    records = [
        _rec({"type": "event_msg", "ts": 1.0, "payload": {"type": "user_message", "message": "hello"}}, start=0),
        _rec({"type": "event_msg", "ts": 2.0, "payload": {"type": "task_complete", "turn_id": "t1"}}, start=10),
    ]
    events = _extract_positioned_chat_events(records)
    text = events[-1]["text"]
    # No stack traces, API keys, or internal codex_error_info leakage.
    for needle in ("traceback", "api_key", "token", "authorization", "codex_error_info"):
        assert needle not in text.lower()


# ---------------------------------------------------------------------------
# 2. Normal successful assistant final response remains unchanged.
# ---------------------------------------------------------------------------
def test_normal_assistant_response_is_unchanged() -> None:
    records = [
        _rec({"type": "event_msg", "ts": 1.0, "payload": {"type": "user_message", "message": "hello"}}, start=0),
        _rec(
            {
                "type": "response_item",
                "ts": 2.0,
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "phase": "final_answer",
                    "content": [{"type": "output_text", "text": "world"}],
                },
            },
            start=100,
        ),
        _rec({"type": "event_msg", "ts": 3.0, "payload": {"type": "task_complete", "turn_id": "t1"}}, start=200),
    ]
    events = _extract_positioned_chat_events(records)
    roles = [ev["role"] for ev in events]
    assert roles == ["user", "assistant"]
    assert events[1]["text"] == "world"
    assert events[1]["message_class"] == "final_response"
    # No phantom no-response event appended.
    assert all(ev["text"] != _NO_RESPONSE_TEXT for ev in events)


# ---------------------------------------------------------------------------
# 3. Backend error events remain unchanged / no duplicate no-response event
#    if an explicit error is already present.
# ---------------------------------------------------------------------------
def test_explicit_error_suppresses_no_response_event() -> None:
    records = [
        _rec({"type": "event_msg", "ts": 1.0, "payload": {"type": "user_message", "message": "hello"}}, start=0),
        _rec(
            {
                "type": "event_msg",
                "ts": 2.0,
                "payload": {
                    "type": "error",
                    "message": "unexpected status 400 Bad Request: invalid model",
                    "codex_error_info": "bad_request",
                },
            },
            start=100,
        ),
        _rec({"type": "event_msg", "ts": 3.0, "payload": {"type": "task_complete", "turn_id": "t1"}}, start=200),
    ]
    events = _extract_positioned_chat_events(records)
    assert events[-1]["message_class"] == "error"
    assert events[-1]["text"] == "unexpected status 400 Bad Request: invalid model"
    assert all(ev["text"] != _NO_RESPONSE_TEXT for ev in events)


def test_stream_error_suppresses_no_response_event() -> None:
    records = [
        _rec({"type": "event_msg", "ts": 1.0, "payload": {"type": "user_message", "message": "hello"}}, start=0),
        _rec(
            {
                "type": "event_msg",
                "ts": 2.0,
                "payload": {"type": "stream_error", "message": "stream disconnected"},
            },
            start=100,
        ),
        _rec({"type": "event_msg", "ts": 3.0, "payload": {"type": "task_complete", "turn_id": "t1"}}, start=200),
    ]
    events = _extract_positioned_chat_events(records)
    assert all(ev["text"] != _NO_RESPONSE_TEXT for ev in events)


# ---------------------------------------------------------------------------
# 4. A log with no user message should not emit a no-response failure.
# ---------------------------------------------------------------------------
def test_task_complete_without_preceding_user_does_not_emit() -> None:
    records = [
        _rec({"type": "event_msg", "ts": 1.0, "payload": {"type": "task_complete", "turn_id": "t1"}}, start=0),
    ]
    events = _extract_positioned_chat_events(records)
    assert events == []


def test_empty_log_does_not_emit() -> None:
    events = _extract_positioned_chat_events([])
    assert events == []


# ---------------------------------------------------------------------------
# 5. Pi composed certification assumptions are not broken: Pi message-type
#    turns must not trigger the Codex-only no-response detection.
# ---------------------------------------------------------------------------
def test_pi_final_response_turn_is_unaffected() -> None:
    records = [
        _rec(
            {
                "type": "message",
                "ts": 1.0,
                "message": {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            },
            start=0,
        ),
        _rec(
            {
                "type": "message",
                "ts": 2.0,
                "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
            },
            start=100,
        ),
    ]
    events = _extract_positioned_chat_events(records)
    roles = [ev["role"] for ev in events]
    assert roles == ["user", "assistant"]
    assert all(ev["text"] != _NO_RESPONSE_TEXT for ev in events)


def test_pi_stop_empty_projects_no_response_event() -> None:
    records = [
        _rec(
            {
                "type": "message",
                "ts": 1.0,
                "message": {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            },
            start=0,
        ),
        _rec(
            {
                "type": "message",
                "ts": 2.0,
                "message": {"role": "assistant", "stopReason": "stop", "content": []},
            },
            start=100,
        ),
    ]
    events = _extract_positioned_chat_events(records)
    assert [ev["role"] for ev in events] == ["user", "assistant"]
    assert events[-1]["text"] == _NO_RESPONSE_TEXT
    assert events[-1]["message_class"] == "error"
    assert events[-1]["ts"] == 2.0
    assert events[-1]["_before_byte"] == 100


def test_pi_end_turn_empty_projects_no_response_event() -> None:
    records = [
        _rec(
            {
                "type": "message",
                "ts": 1.0,
                "message": {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            },
            start=0,
        ),
        _rec(
            {
                "type": "message",
                "ts": 2.0,
                "message": {"role": "assistant", "stopReason": "end_turn", "content": []},
            },
            start=100,
        ),
    ]
    events = _extract_positioned_chat_events(records)
    assert [ev["role"] for ev in events] == ["user", "assistant"]
    assert events[-1]["text"] == _NO_RESPONSE_TEXT
    assert events[-1]["message_class"] == "error"


def test_pi_stop_thinking_only_projects_no_response_event() -> None:
    records = [
        _rec(
            {
                "type": "message",
                "ts": 1.0,
                "message": {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            },
            start=0,
        ),
        _rec(
            {
                "type": "message",
                "ts": 2.0,
                "message": {
                    "role": "assistant",
                    "stopReason": "stop",
                    "content": [{"type": "thinking", "thinking": "internal"}],
                },
            },
            start=100,
        ),
    ]
    events = _extract_positioned_chat_events(records)
    assert [ev["role"] for ev in events] == ["user", "assistant"]
    assert events[-1]["text"] == _NO_RESPONSE_TEXT
    assert events[-1]["message_class"] == "error"


def test_pi_nonterminal_thinking_does_not_project_no_response() -> None:
    records = [
        _rec(
            {
                "type": "message",
                "ts": 1.0,
                "message": {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            },
            start=0,
        ),
        _rec(
            {
                "type": "message",
                "ts": 2.0,
                "message": {"role": "assistant", "content": [{"type": "thinking", "thinking": "internal"}]},
            },
            start=100,
        ),
    ]
    events = _extract_positioned_chat_events(records)
    assert [ev["role"] for ev in events] == ["user"]
    assert all(ev["text"] != _NO_RESPONSE_TEXT for ev in events)


def test_pi_length_thinking_only_does_not_project_no_response() -> None:
    records = [
        _rec(
            {
                "type": "message",
                "ts": 1.0,
                "message": {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            },
            start=0,
        ),
        _rec(
            {
                "type": "message",
                "ts": 2.0,
                "message": {
                    "role": "assistant",
                    "stopReason": "length",
                    "content": [{"type": "thinking", "thinking": "internal"}],
                },
            },
            start=100,
        ),
    ]
    events = _extract_positioned_chat_events(records)
    assert [ev["role"] for ev in events] == ["user"]
    assert all(ev["text"] != _NO_RESPONSE_TEXT for ev in events)


def test_pi_length_compaction_continuation_does_not_insert_false_no_response() -> None:
    records = [
        _rec(
            {
                "type": "message",
                "ts": 1.0,
                "message": {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            },
            start=0,
        ),
        _rec(
            {
                "type": "message",
                "ts": 2.0,
                "message": {
                    "role": "assistant",
                    "stopReason": "length",
                    "content": [{"type": "thinking", "thinking": "internal before compaction"}],
                },
            },
            start=100,
        ),
        _rec({"type": "compaction", "ts": 3.0, "message": "compacting context"}, start=200),
        _rec({"type": "custom_message", "ts": 4.0, "message": "continuing after compaction"}, start=300),
        _rec(
            {
                "type": "message",
                "ts": 5.0,
                "message": {
                    "role": "assistant",
                    "stopReason": "toolUse",
                    "content": [
                        {"type": "text", "text": "continuing with a tool"},
                        {"type": "toolCall", "id": "tool-1", "name": "bash", "arguments": {"command": "pwd"}},
                    ],
                },
            },
            start=400,
        ),
    ]
    events = _extract_positioned_chat_events(records)
    assert [ev["role"] for ev in events] == ["user", "assistant"]
    assert events[-1]["text"] == "continuing with a tool"
    assert events[-1]["message_class"] == "narration"
    assert all(ev["text"] != _NO_RESPONSE_TEXT for ev in events)


def test_pi_tool_use_message_does_not_project_no_response() -> None:
    records = [
        _rec(
            {
                "type": "message",
                "ts": 1.0,
                "message": {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            },
            start=0,
        ),
        _rec(
            {
                "type": "message",
                "ts": 2.0,
                "message": {
                    "role": "assistant",
                    "stopReason": "toolUse",
                    "content": [{"type": "toolCall", "id": "tool-1", "name": "bash", "arguments": {}}],
                },
            },
            start=100,
        ),
    ]
    events = _extract_positioned_chat_events(records)
    assert [ev["role"] for ev in events] == ["user"]
    assert all(ev["text"] != _NO_RESPONSE_TEXT for ev in events)


def test_pi_tail_and_search_surface_no_response_event(tmp_path: Path) -> None:
    from codoxear.transcript_search import search_chat_log_bounded

    log_path = tmp_path / "pi.jsonl"
    _write_log(
        log_path,
        [
            {
                "type": "message",
                "ts": 1.0,
                "message": {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            },
            {
                "type": "message",
                "ts": 2.0,
                "message": {"role": "assistant", "stopReason": "stop", "content": []},
            },
        ],
    )
    events, _before, _after, _has_older = _read_chat_tail_page(log_path, limit=20)
    assert [ev["role"] for ev in events] == ["user", "assistant"]
    assert events[-1]["text"] == _NO_RESPONSE_TEXT
    count, matches, _ = search_chat_log_bounded(log_path, "completed this turn", limit=5)
    assert count == 1
    assert matches[0]["text"] == _NO_RESPONSE_TEXT
    assert matches[0]["message_class"] == "error"


def test_pi_aborted_turn_does_not_emit_no_response() -> None:
    records = [
        _rec(
            {
                "type": "message",
                "ts": 1.0,
                "message": {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            },
            start=0,
        ),
        _rec(
            {
                "type": "message",
                "ts": 2.0,
                "message": {"role": "assistant", "stopReason": "aborted", "content": []},
            },
            start=100,
        ),
    ]
    events = _extract_positioned_chat_events(records)
    # An aborted Pi turn now renders a persistent assistant interruption
    # outcome row (distinct from a generic no-response completion), so the
    # user turn is never left dangling as user-only.
    assert [ev["role"] for ev in events] == ["user", "assistant"]
    interruption = events[-1]
    assert interruption["message_class"] == "error"
    assert interruption["text"] == _INTERRUPTED_TEXT
    assert "interrupted" in interruption["text"]
    assert interruption["text"] != _NO_RESPONSE_TEXT
    assert isinstance(interruption["message_id"], str)
    assert interruption["ts"] == 2.0


def test_pi_aborted_partial_turn_preserves_partial_text() -> None:
    """A Pi aborted turn that already streamed partial text must render the
    interruption outcome AND keep the partial text visible/searchable in a
    single assistant row (partial output is not discarded)."""
    records = [
        _rec(
            {
                "type": "message",
                "ts": 1.0,
                "message": {"role": "user", "content": [{"type": "text", "text": "hello partial"}]},
            },
            start=0,
        ),
        _rec(
            {
                "type": "message",
                "ts": 2.0,
                "message": {
                    "role": "assistant",
                    "stopReason": "aborted",
                    "content": [{"type": "text", "text": "I was halfway through"}],
                },
            },
            start=100,
        ),
    ]
    events = _extract_positioned_chat_events(records)
    assert [ev["role"] for ev in events] == ["user", "assistant"]
    interruption = events[-1]
    assert interruption["message_class"] == "error"
    assert "interrupted" in interruption["text"]
    assert "Partial output before interruption:" in interruption["text"]
    assert "I was halfway through" in interruption["text"]
    assert interruption["text"] != _NO_RESPONSE_TEXT


def test_pi_aborted_partial_text_is_searchable(tmp_path: Path) -> None:
    """The partial text must be findable via the disk-backed search path."""
    from codoxear.transcript_search import search_chat_log_bounded

    log_path = tmp_path / "pi.jsonl"
    _write_log(
        log_path,
        [
            {
                "type": "message",
                "ts": 1.0,
                "message": {"role": "user", "content": [{"type": "text", "text": "hello partial"}]},
            },
            {
                "type": "message",
                "ts": 2.0,
                "message": {
                    "role": "assistant",
                    "stopReason": "aborted",
                    "content": [{"type": "text", "text": "I was halfway through"}],
                },
            },
        ],
    )
    # The partial text itself is searchable.
    count_partial, matches_partial, _ = search_chat_log_bounded(
        log_path, "I was halfway through", limit=5, max_line_bytes=4096
    )
    assert count_partial == 1
    assert matches_partial[0]["message_class"] == "error"
    assert "I was halfway through" in matches_partial[0]["text"]
    # The interruption-word is also searchable and resolves to the same row.
    count_intr, _matches_intr, _ = search_chat_log_bounded(
        log_path, "interrupted", limit=5, max_line_bytes=4096
    )
    assert count_intr == 1


def test_codex_turn_aborted_emits_interruption_event() -> None:
    """A Codex event_msg turn_aborted renders a persistent assistant
    interruption outcome row, not a generic no-response row."""
    records = [
        _rec({"type": "event_msg", "ts": 1.0, "payload": {"type": "user_message", "message": "hello"}}, start=0),
        _rec({"type": "event_msg", "ts": 2.0, "payload": {"type": "turn_aborted"}}, start=100),
    ]
    events = _extract_positioned_chat_events(records)
    assert [ev["role"] for ev in events] == ["user", "assistant"]
    interruption = events[-1]
    assert interruption["message_class"] == "error"
    assert interruption["text"] == _INTERRUPTED_TEXT
    assert "interrupted" in interruption["text"]
    assert interruption["text"] != _NO_RESPONSE_TEXT
    assert isinstance(interruption["message_id"], str)
    assert interruption["ts"] == 2.0


def test_codex_turn_aborted_does_not_trigger_no_response_injection() -> None:
    """A turn_aborted row produces its own interruption event, so a following
    task_complete must NOT additionally inject a generic no-response row."""
    records = [
        _rec({"type": "event_msg", "ts": 1.0, "payload": {"type": "user_message", "message": "hello"}}, start=0),
        _rec({"type": "event_msg", "ts": 2.0, "payload": {"type": "turn_aborted"}}, start=100),
        _rec({"type": "event_msg", "ts": 3.0, "payload": {"type": "task_complete", "turn_id": "t1"}}, start=200),
    ]
    events = _extract_positioned_chat_events(records)
    assert [ev["role"] for ev in events] == ["user", "assistant"]
    assert events[-1]["text"] == _INTERRUPTED_TEXT
    assert all(ev["text"] != _NO_RESPONSE_TEXT for ev in events)


# ---------------------------------------------------------------------------
# Detection primitive tests (byte-offset and event construction).
# ---------------------------------------------------------------------------
def test_detect_returns_close_byte_offset() -> None:
    records = [
        _rec({"type": "event_msg", "ts": 1.0, "payload": {"type": "user_message", "message": "hi"}}, start=0),
        _rec({"type": "event_msg", "ts": 2.0, "payload": {"type": "task_complete", "turn_id": "t1"}}, start=42),
    ]
    closes = _detect_codex_no_response_closes(records)
    assert len(closes) == 1
    user_byte, close_byte, _close_obj = closes[0]
    assert user_byte == 0
    assert close_byte == 42


def test_build_no_response_event_shape() -> None:
    obj = {"type": "event_msg", "ts": 99.0, "payload": {"type": "task_complete", "turn_id": "t1"}}
    event = _build_no_response_event(obj)
    assert event["role"] == "assistant"
    assert event["message_class"] == "error"
    assert event["text"] == _NO_RESPONSE_TEXT
    assert event["ts"] == 99.0
    assert isinstance(event["message_id"], str)


def test_inject_preserves_ordering_by_byte() -> None:
    records = [
        _rec({"type": "event_msg", "ts": 1.0, "payload": {"type": "user_message", "message": "a"}}, start=0),
        _rec({"type": "event_msg", "ts": 2.0, "payload": {"type": "task_complete", "turn_id": "t1"}}, start=10),
        _rec({"type": "event_msg", "ts": 3.0, "payload": {"type": "user_message", "message": "b"}}, start=20),
        _rec(
            {
                "type": "response_item",
                "ts": 4.0,
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "phase": "final_answer",
                    "content": [{"type": "output_text", "text": "answer"}],
                },
            },
            start=30,
        ),
        _rec({"type": "event_msg", "ts": 5.0, "payload": {"type": "task_complete", "turn_id": "t2"}}, start=40),
    ]
    events = _extract_positioned_chat_events(records)
    # Expect: user(a), no-response, user(b), assistant(answer) — second turn had output.
    assert [ev["role"] for ev in events] == ["user", "assistant", "user", "assistant"]
    assert events[1]["text"] == _NO_RESPONSE_TEXT
    assert events[3]["text"] == "answer"


# ---------------------------------------------------------------------------
# Source-of-truth: the injector must consult positioned extracted events, not
# re-scan raw Codex row shapes. A synthetic visible assistant event
# (representing any future row form the normalizer learns to extract) passed
# straight to ``_inject_no_response_events`` must suppress injection.
# ---------------------------------------------------------------------------
def test_inject_uses_positioned_events_synthetic_assistant_suppresses() -> None:
    # Records describe a degraded turn (user + close, no assistant row).
    records = [
        _rec({"type": "event_msg", "ts": 1.0, "payload": {"type": "user_message", "message": "hi"}}, start=0),
        _rec({"type": "event_msg", "ts": 2.0, "payload": {"type": "task_complete", "turn_id": "t1"}}, start=10),
    ]
    # But the normalizer already produced a visible assistant event between
    # the user and the close — e.g. from a row form the detector does not know
    # about. The injector must trust that existing event and not inject.
    synthetic = {
        "role": "assistant",
        "text": "delivered via some future row shape",
        "message_class": "final_response",
        "message_id": "synthetic",
        "_before_byte": 5,
    }
    events = _inject_no_response_events(records, [synthetic])
    assert len(events) == 1
    assert events[0] is synthetic
    assert all(ev["text"] != _NO_RESPONSE_TEXT for ev in events)


def test_inject_uses_positioned_events_no_assistant_injects() -> None:
    # Same records, but only a user event is positioned — no assistant event
    # exists between user and close, so injection must happen.
    records = [
        _rec({"type": "event_msg", "ts": 1.0, "payload": {"type": "user_message", "message": "hi"}}, start=0),
        _rec({"type": "event_msg", "ts": 2.0, "payload": {"type": "task_complete", "turn_id": "t1"}}, start=10),
    ]
    user_event = {"role": "user", "text": "hi", "_before_byte": 0}
    events = _inject_no_response_events(records, [user_event])
    assert [ev["role"] for ev in events] == ["user", "assistant"]
    assert events[1]["text"] == _NO_RESPONSE_TEXT
    assert events[1]["message_class"] == "error"
    assert events[1]["_before_byte"] == 10


def test_inject_error_event_outside_range_does_not_suppress() -> None:
    # An assistant error event that belongs to a *later* turn (byte beyond the
    # close) must not suppress injection for this close.
    records = [
        _rec({"type": "event_msg", "ts": 1.0, "payload": {"type": "user_message", "message": "hi"}}, start=0),
        _rec({"type": "event_msg", "ts": 2.0, "payload": {"type": "task_complete", "turn_id": "t1"}}, start=10),
    ]
    later_error = {
        "role": "assistant",
        "text": "later error",
        "message_class": "error",
        "message_id": "late",
        "_before_byte": 20,
    }
    events = _inject_no_response_events(records, [later_error])
    # The later error sorts after the close, so a no-response is still injected
    # at byte 10, before the later error at byte 20.
    assert [ev["role"] for ev in events] == ["assistant", "assistant"]
    assert events[0]["text"] == _NO_RESPONSE_TEXT
    assert events[0]["_before_byte"] == 10
    assert events[1]["text"] == "later error"


# ---------------------------------------------------------------------------
# End-to-end through the tail page reader (the production message route path).
# ---------------------------------------------------------------------------
def test_tail_page_surfaces_no_response_event(tmp_path: Path) -> None:
    log_path = tmp_path / "rollout.jsonl"
    _write_log(
        log_path,
        [
            {"type": "event_msg", "ts": 1.0, "payload": {"type": "user_message", "message": "hello"}},
            {"type": "event_msg", "ts": 2.0, "payload": {"type": "task_complete", "turn_id": "t1", "last_agent_message": None}},
        ],
    )
    events, _before, _after, _has_older = _read_chat_tail_page(log_path, limit=20)
    assert [ev["role"] for ev in events] == ["user", "assistant"]
    assert events[1]["message_class"] == "error"
    assert events[1]["text"] == _NO_RESPONSE_TEXT


def test_tail_page_normal_response_unchanged(tmp_path: Path) -> None:
    log_path = tmp_path / "rollout.jsonl"
    _write_log(
        log_path,
        [
            {"type": "event_msg", "ts": 1.0, "payload": {"type": "user_message", "message": "hello"}},
            {
                "type": "response_item",
                "ts": 2.0,
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "phase": "final_answer",
                    "content": [{"type": "output_text", "text": "world"}],
                },
            },
            {"type": "event_msg", "ts": 3.0, "payload": {"type": "task_complete", "turn_id": "t1"}},
        ],
    )
    events, _before, _after, _has_older = _read_chat_tail_page(log_path, limit=20)
    assert [ev["role"] for ev in events] == ["user", "assistant"]
    assert events[1]["text"] == "world"


# ---------------------------------------------------------------------------
# Live-poll split: the user_message is delivered in one poll, the close in the
# next. The close-poll delta must still surface a no-response event because the
# prior open turn context is carried across the boundary.
# ---------------------------------------------------------------------------
def test_codex_prior_open_turn_context_finds_open_user() -> None:
    """Reverse scan from the close byte must find the open user_message byte."""
    with TemporaryDirectory() as d:
        log_path = Path(d) / "rollout.jsonl"
        _write_log(
            log_path,
            [
                {"type": "event_msg", "ts": 1.0, "payload": {"type": "user_message", "message": "hi"}},
                {"type": "event_msg", "ts": 2.0, "payload": {"type": "task_complete", "turn_id": "t1"}},
                {"type": "event_msg", "ts": 3.0, "payload": {"type": "user_message", "message": "hi2"}},
            ],
        )
        size = log_path.stat().st_size
        # `before` points at end-of-file: the most recent boundary is the open
        # user_message "hi2" (no close after it).
        user_byte, has_asst = _codex_prior_open_turn_context(log_path, size)
        assert user_byte is not None
        assert user_byte > 0
        assert has_asst is False


def test_codex_prior_open_turn_context_closed_turn_returns_none() -> None:
    with TemporaryDirectory() as d:
        log_path = Path(d) / "rollout.jsonl"
        _write_log(
            log_path,
            [
                {"type": "event_msg", "ts": 1.0, "payload": {"type": "user_message", "message": "hi"}},
                {"type": "event_msg", "ts": 2.0, "payload": {"type": "task_complete", "turn_id": "t1"}},
            ],
        )
        size = log_path.stat().st_size
        user_byte, _has_asst = _codex_prior_open_turn_context(log_path, size)
        # The last boundary is a close → no open turn.
        assert user_byte is None


def test_live_split_user_poll_then_close_poll_emits_no_response() -> None:
    """The reported production bug: user delivered in poll 1, close in poll 2.

    The log grows between polls: poll 1 sees only the user_message; the close
    is appended; poll 2 sees only the task_complete. The close-poll delta must
    inject a no-response event because the prior open turn is carried across
    the split.
    """
    with TemporaryDirectory() as d:
        log_path = Path(d) / "rollout.jsonl"
        user_row = {"type": "event_msg", "ts": 1.0, "payload": {"type": "user_message", "message": "hello"}}
        _write_log(log_path, [user_row])

        # Poll 1: only the user_message exists.
        events1, next_after, _meta, _flags, _diag, _token = _read_chat_live_delta(log_path, after_byte=0)
        assert [ev["role"] for ev in events1] == ["user"]
        assert all(ev["text"] != _NO_RESPONSE_TEXT for ev in events1)
        cursor = next_after

        # Append the close and poll again from the advanced cursor.
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"type": "event_msg", "ts": 2.0, "payload": {"type": "task_complete", "turn_id": "t1", "last_agent_message": None}}) + "\n")

        events2, _next, _meta, _flags, _diag, _token = _read_chat_live_delta(log_path, after_byte=cursor)
        assert [ev["role"] for ev in events2] == ["assistant"]
        assert events2[0]["text"] == _NO_RESPONSE_TEXT
        assert events2[0]["message_class"] == "error"


def test_live_split_with_prior_assistant_does_not_emit_no_response() -> None:
    """If a visible assistant event (a projected response_item assistant
    message) appeared between the prior user_message and the close-poll window,
    no no-response must be injected."""
    with TemporaryDirectory() as d:
        log_path = Path(d) / "rollout.jsonl"
        _write_log(
            log_path,
            [
                {"type": "event_msg", "ts": 1.0, "payload": {"type": "user_message", "message": "hello"}},
                {
                    "type": "response_item",
                    "ts": 2.0,
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "phase": "final_answer",
                        "content": [{"type": "output_text", "text": "world"}],
                    },
                },
            ],
        )
        lines = log_path.read_text(encoding="utf-8").splitlines(keepends=True)
        # Split so poll 1 saw user + assistant (cursor advanced past both);
        # poll 2 sees only the appended close.
        cursor = len(lines[0]) + len(lines[1])
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"type": "event_msg", "ts": 3.0, "payload": {"type": "task_complete", "turn_id": "t1"}}) + "\n")
        events, _next, _meta, _flags, _diag, _token = _read_chat_live_delta(log_path, after_byte=cursor)
        assert events == []


def test_synthetic_positioned_assistant_suppresses_with_prior_context() -> None:
    """A visible assistant event of any future row form (here a synthetic
    positioned event representing a form the normalizer may later project)
    suppresses no-response even when the user row came from prior context."""
    records = [
        _rec({"type": "event_msg", "ts": 2.0, "payload": {"type": "task_complete", "turn_id": "t1"}}, start=100),
    ]
    synthetic = {
        "role": "assistant",
        "text": "delivered via some future row shape",
        "message_class": "final_response",
        "message_id": "synthetic",
        "_before_byte": 50,
    }
    events = _inject_no_response_events(
        records,
        [synthetic],
        prior_user_byte=10,
        prior_turn_has_assistant=False,
    )
    assert events == [synthetic]


def test_in_window_user_message_overrides_prior_context() -> None:
    """If the window contains a new user_message after the prior open turn,
    the prior context is superseded: the new turn governs injection."""
    records = [
        # Prior context describes an open user turn at byte 0 with no assistant.
        _rec({"type": "event_msg", "ts": 5.0, "payload": {"type": "user_message", "message": "new"}}, start=100),
        _rec({"type": "event_msg", "ts": 6.0, "payload": {"type": "task_complete", "turn_id": "t2"}}, start=200),
    ]
    events = _inject_no_response_events(
        records,
        [],
        prior_user_byte=0,
        prior_turn_has_assistant=False,
    )
    # The new in-window user_message supersedes prior; one no-response injected.
    assert [ev["role"] for ev in events] == ["assistant"]
    assert events[0]["text"] == _NO_RESPONSE_TEXT
    assert events[0]["_before_byte"] == 200


def test_prior_turn_has_assistant_suppresses_split_close() -> None:
    """prior_turn_has_assistant=True with prior_user_byte set must suppress
    injection for a close arriving alone in the window."""
    records = [
        _rec({"type": "event_msg", "ts": 2.0, "payload": {"type": "task_complete", "turn_id": "t1"}}, start=100),
    ]
    events = _inject_no_response_events(
        records,
        [],
        prior_user_byte=10,
        prior_turn_has_assistant=True,
    )
    assert events == []


# ---------------------------------------------------------------------------
# Pi / Claude Code false-positive guard: their logs must never trigger the
# Codex-only no-response path through the live delta route.
# ---------------------------------------------------------------------------
def test_pi_live_delta_does_not_emit_no_response() -> None:
    with TemporaryDirectory() as d:
        log_path = Path(d) / "rollout.jsonl"
        _write_log(
            log_path,
            [
                {
                    "type": "message",
                    "ts": 1.0,
                    "message": {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                },
                {
                    "type": "message",
                    "ts": 2.0,
                    "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
                },
            ],
        )
        events, _next, _meta, _flags, _diag, _token = _read_chat_live_delta(log_path, after_byte=0)
        assert all(ev["text"] != _NO_RESPONSE_TEXT for ev in events)


def test_cc_live_delta_does_not_emit_no_response() -> None:
    with TemporaryDirectory() as d:
        log_path = Path(d) / "rollout.jsonl"
        _write_log(
            log_path,
            [
                {"type": "user", "timestamp": "2026-01-01T00:00:00Z", "message": {"role": "user", "content": [{"type": "text", "text": "hi"}]}},
                {"type": "assistant", "timestamp": "2026-01-01T00:00:01Z", "message": {"role": "assistant", "content": [{"type": "text", "text": "hello back"}]}},
            ],
        )
        events, _next, _meta, _flags, _diag, _token = _read_chat_live_delta(log_path, after_byte=0)
        assert all(ev["text"] != _NO_RESPONSE_TEXT for ev in events)


def test_prior_context_none_when_window_at_start() -> None:
    """after_byte=0 (first poll) must yield no prior context."""
    with TemporaryDirectory() as d:
        log_path = Path(d) / "rollout.jsonl"
        _write_log(log_path, [{"type": "event_msg", "ts": 1.0, "payload": {"type": "user_message", "message": "hi"}}])
        user_byte, _has_asst = _codex_prior_open_turn_context(log_path, 0)
        assert user_byte is None


# ---------------------------------------------------------------------------
# Codex event_msg assistant row forms must render as transcript messages.
# These are the same row forms idle/sidebar already treat as assistant output;
# projecting them here lets the no-response injector suppress itself via the
# existing source-of-truth mechanism.
# ---------------------------------------------------------------------------
def test_event_msg_agent_message_final_answer_projects_as_final_response() -> None:
    """event_msg agent_message with phase=final_answer renders as an assistant
    final_response transcript message and suppresses no-response."""
    records = [
        _rec({"type": "event_msg", "ts": 1.0, "payload": {"type": "user_message", "message": "hi"}}, start=0),
        _rec({"type": "event_msg", "ts": 2.0, "payload": {"type": "agent_message", "phase": "final_answer", "message": "the answer"}}, start=10),
        _rec({"type": "event_msg", "ts": 3.0, "payload": {"type": "task_complete", "turn_id": "t1"}}, start=20),
    ]
    events = _extract_positioned_chat_events(records)
    assert [ev["role"] for ev in events] == ["user", "assistant"]
    assert events[1]["text"] == "the answer"
    assert events[1]["message_class"] == "final_response"
    assert all(ev["text"] != _NO_RESPONSE_TEXT for ev in events)


def test_event_msg_agent_message_non_final_projects_as_narration() -> None:
    """event_msg agent_message without phase=final_answer renders as narration."""
    records = [
        _rec({"type": "event_msg", "ts": 1.0, "payload": {"type": "user_message", "message": "hi"}}, start=0),
        _rec({"type": "event_msg", "ts": 2.0, "payload": {"type": "agent_message", "phase": "reasoning", "message": "thinking aloud"}}, start=10),
        _rec({"type": "event_msg", "ts": 3.0, "payload": {"type": "task_complete", "turn_id": "t1"}}, start=20),
    ]
    events = _extract_positioned_chat_events(records)
    # narration is still a visible assistant event → suppresses no-response.
    assert [ev["role"] for ev in events] == ["user", "assistant"]
    assert events[1]["text"] == "thinking aloud"
    assert events[1]["message_class"] == "narration"
    assert all(ev["text"] != _NO_RESPONSE_TEXT for ev in events)


def test_event_msg_agent_message_empty_does_not_project() -> None:
    """An empty/whitespace agent_message must not project (no phantom event,
    no spurious suppression)."""
    records = [
        _rec({"type": "event_msg", "ts": 1.0, "payload": {"type": "user_message", "message": "hi"}}, start=0),
        _rec({"type": "event_msg", "ts": 2.0, "payload": {"type": "agent_message", "phase": "final_answer", "message": "   "}}, start=10),
        _rec({"type": "event_msg", "ts": 3.0, "payload": {"type": "task_complete", "turn_id": "t1"}}, start=20),
    ]
    events = _extract_positioned_chat_events(records)
    # Empty agent_message does not suppress → no-response still fires.
    assert events[-1]["text"] == _NO_RESPONSE_TEXT


def test_task_complete_last_agent_message_projects_as_final_response() -> None:
    """task_complete with non-empty last_agent_message renders as an assistant
    final_response transcript message and suppresses no-response."""
    records = [
        _rec({"type": "event_msg", "ts": 1.0, "payload": {"type": "user_message", "message": "hi"}}, start=0),
        _rec({"type": "event_msg", "ts": 2.0, "payload": {"type": "task_complete", "turn_id": "t1", "last_agent_message": "final via last_agent"}}, start=20),
    ]
    events = _extract_positioned_chat_events(records)
    assert [ev["role"] for ev in events] == ["user", "assistant"]
    assert events[1]["text"] == "final via last_agent"
    assert events[1]["message_class"] == "final_response"
    assert all(ev["text"] != _NO_RESPONSE_TEXT for ev in events)


def test_turn_complete_last_agent_message_projects_as_final_response() -> None:
    """turn_complete variant with last_agent_message also renders."""
    records = [
        _rec({"type": "event_msg", "ts": 1.0, "payload": {"type": "user_message", "message": "hi"}}, start=0),
        _rec({"type": "event_msg", "ts": 2.0, "payload": {"type": "turn_complete", "turn_id": "t1", "last_agent_message": "final"}}, start=20),
    ]
    events = _extract_positioned_chat_events(records)
    assert events[-1]["text"] == "final"
    assert events[-1]["message_class"] == "final_response"


def test_mixed_agent_message_and_last_agent_message_both_render() -> None:
    """The reported discriminator: agent_message final_answer and
    task_complete.last_agent_message with DIFFERENT text both render (no
    no-response); the existing dedupe does not collapse them because the text
    differs."""
    records = [
        _rec({"type": "event_msg", "ts": 1.0, "payload": {"type": "user_message", "message": "hi"}}, start=0),
        _rec({"type": "event_msg", "ts": 2.0, "payload": {"type": "agent_message", "phase": "final_answer", "message": "done via agent_message"}}, start=10),
        _rec({"type": "event_msg", "ts": 3.0, "payload": {"type": "task_complete", "turn_id": "t1", "last_agent_message": "done via last_agent"}}, start=20),
    ]
    events = _extract_positioned_chat_events(records)
    assert [ev["role"] for ev in events] == ["user", "assistant", "assistant"]
    assert events[1]["text"] == "done via agent_message"
    assert events[1]["message_class"] == "final_response"
    assert events[2]["text"] == "done via last_agent"
    assert events[2]["message_class"] == "final_response"
    assert all(ev["text"] != _NO_RESPONSE_TEXT for ev in events)


def test_same_text_agent_message_and_last_agent_message_deduped() -> None:
    """When agent_message final_answer and the closing row carry the SAME
    text, the existing assistant dedupe (keyed on message_class + normalized
    text) collapses them into a single transcript message."""
    records = [
        _rec({"type": "event_msg", "ts": 1.0, "payload": {"type": "user_message", "message": "hi"}}, start=0),
        _rec({"type": "event_msg", "ts": 2.0, "payload": {"type": "agent_message", "phase": "final_answer", "message": "same answer"}}, start=10),
        _rec({"type": "event_msg", "ts": 3.0, "payload": {"type": "task_complete", "turn_id": "t1", "last_agent_message": "same answer"}}, start=20),
    ]
    events = _extract_positioned_chat_events(records)
    assert [ev["role"] for ev in events] == ["user", "assistant"]
    assert events[1]["text"] == "same answer"
    assert all(ev["text"] != _NO_RESPONSE_TEXT for ev in events)


def test_live_split_agent_message_suppresses_close_no_response() -> None:
    """Live split where the agent_message is delivered in poll 1 and only the
    close arrives in poll 2: prior_turn_has_assistant must be True (because
    the extractor now projects agent_message), so no no-response is injected."""
    with TemporaryDirectory() as d:
        log_path = Path(d) / "rollout.jsonl"
        _write_log(
            log_path,
            [
                {"type": "event_msg", "ts": 1.0, "payload": {"type": "user_message", "message": "hi"}},
                {"type": "event_msg", "ts": 2.0, "payload": {"type": "agent_message", "phase": "final_answer", "message": "done"}},
            ],
        )
        cursor = log_path.stat().st_size
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"type": "event_msg", "ts": 3.0, "payload": {"type": "task_complete", "turn_id": "t1"}}) + "\n")
        events, _next, _meta, _flags, _diag, _token = _read_chat_live_delta(log_path, after_byte=cursor)
        assert events == []


def test_live_split_last_agent_message_close_projects_in_close_delta() -> None:
    """Live split where only the close (carrying last_agent_message) arrives
    in poll 2: the close-poll delta projects the last_agent_message as a
    final_response assistant event."""
    with TemporaryDirectory() as d:
        log_path = Path(d) / "rollout.jsonl"
        _write_log(log_path, [{"type": "event_msg", "ts": 1.0, "payload": {"type": "user_message", "message": "hi"}}])
        cursor = log_path.stat().st_size
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"type": "event_msg", "ts": 2.0, "payload": {"type": "task_complete", "turn_id": "t1", "last_agent_message": "delivered on close"}}) + "\n")
        events, _next, _meta, _flags, _diag, _token = _read_chat_live_delta(log_path, after_byte=cursor)
        assert [ev["role"] for ev in events] == ["assistant"]
        assert events[0]["text"] == "delivered on close"
        assert events[0]["message_class"] == "final_response"
        assert all(ev["text"] != _NO_RESPONSE_TEXT for ev in events)
