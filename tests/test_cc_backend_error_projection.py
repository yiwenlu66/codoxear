"""Tests for Claude Code backend/transport error projection.

Claude Code logs API/gateway/provider failures as synthetic assistant rows
with ``isApiErrorMessage: true``. Before the fix, these rows were extracted
as normal assistant text and classified as ``narration`` — the error was
visible in the transcript but without error styling, without closing the
turn, and without clearing the busy indicator. These tests verify the
correct projection: ``message_class == "error"``, ``turn_end == True``,
and the broker turn state closes.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from codoxear.broker_turn_state import State
from codoxear.broker_turn_state import _apply_rollout_obj_to_state
from codoxear.cc_log import cc_assistant_is_api_error
from codoxear.cc_log import cc_current_turn_state_before
from codoxear.rollout_log import _compute_idle_from_log
from codoxear.rollout_log import _extract_chat_events
from codoxear.rollout_log import _read_chat_events_from_tail
from codoxear.rollout_log import _read_chat_tail_page


SESSION_ID = "11111111-2222-3333-4444-555555555555"


def _user(text: str = "hello") -> dict:
    return {
        "type": "user",
        "sessionId": SESSION_ID,
        "timestamp": "2026-07-04T00:00:00.000Z",
        "cwd": "/repo",
        "message": {"role": "user", "content": text},
    }


def _cc_api_error(text: str = "API Error: 529 {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}") -> dict:
    """A Claude Code API error row, matching the shape Claude Code writes to
    its JSONL transcript (isApiErrorMessage + synthetic model)."""
    return {
        "type": "assistant",
        "sessionId": SESSION_ID,
        "timestamp": "2026-07-04T00:00:01.000Z",
        "isApiErrorMessage": True,
        "message": {
            "role": "assistant",
            "model": "<synthetic>",
            "stop_reason": "stop_sequence",
            "stop_sequence": "",
            "type": "message",
            "content": [{"type": "text", "text": text}],
        },
    }


def _cc_system_api_error(
    text: str,
    *,
    retry_attempt: int,
    max_retries: int,
    ts: str = "2026-07-04T00:00:03.000Z",
) -> dict:
    """A Claude Code ``system/api_error`` retry/outcome row."""
    return {
        "type": "system",
        "subtype": "api_error",
        "sessionId": SESSION_ID,
        "timestamp": ts,
        "error": text,
        "retryAttempt": retry_attempt,
        "maxRetries": max_retries,
    }


def _state() -> State:
    return State(
        codex_pid=1,
        pty_master_fd=1,
        cwd="/tmp",
        start_ts=0.0,
        codex_home=Path("/tmp"),
        sessions_dir=Path("/tmp"),
    )


def _write_log(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class TestCcApiErrorDetection(unittest.TestCase):
    def test_cc_assistant_is_api_error_detects_flag(self) -> None:
        self.assertTrue(cc_assistant_is_api_error(_cc_api_error()))

    def test_cc_assistant_is_api_error_rejects_normal_assistant(self) -> None:
        normal = {
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn"},
        }
        self.assertFalse(cc_assistant_is_api_error(normal))

    def test_cc_assistant_is_api_error_rejects_non_assistant(self) -> None:
        self.assertFalse(cc_assistant_is_api_error({"type": "user"}))
        self.assertFalse(cc_assistant_is_api_error({"type": "system"}))


class TestCcApiErrorProjection(unittest.TestCase):
    def test_cc_api_error_is_visible_as_error_class(self) -> None:
        events, _meta, flags, _diag = _extract_chat_events([
            _user(),
            _cc_api_error("API Error: 401 Unauthorized"),
        ])
        self.assertEqual(events[-1]["role"], "assistant")
        self.assertEqual(events[-1]["message_class"], "error")
        self.assertEqual(events[-1]["text"], "API Error: 401 Unauthorized")
        self.assertIsInstance(events[-1]["message_id"], str)

    def test_cc_api_error_sets_turn_end(self) -> None:
        _events, _meta, flags, _diag = _extract_chat_events([
            _user(),
            _cc_api_error("API Error: 529 Overloaded"),
        ])
        self.assertTrue(flags["turn_end"])
        self.assertTrue(flags["turn_start"])
        self.assertFalse(flags["turn_aborted"])

    def test_cc_api_error_carries_timestamp(self) -> None:
        events, _meta, _flags, _diag = _extract_chat_events([
            _user(),
            _cc_api_error(),
        ])
        self.assertEqual(events[-1]["ts"], 1783123201.0)

    def test_cc_api_error_text_does_not_invent_cause(self) -> None:
        """The projected text must be the backend's own error text — no
        invented cause, no diagnostic leakage beyond what Claude Code wrote."""
        events, _meta, _flags, _diag = _extract_chat_events([
            _user(),
            _cc_api_error("API Error: 500 Internal Server Error"),
        ])
        text = events[-1]["text"]
        self.assertEqual(text, "API Error: 500 Internal Server Error")
        # Must not contain fabricated content beyond the backend's own text.
        for needle in ("traceback", "stack", "codex_error_info"):
            self.assertNotIn(needle, text.lower())

    def test_cc_api_error_persists_through_tail_read(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "cc-session.jsonl"
            _write_log(path, [_user(), _cc_api_error("API Error: 429 Rate limit")])
            events = _read_chat_events_from_tail(path, min_events=1, max_scan_bytes=256 * 1024)
            self.assertEqual(events[-1]["message_class"], "error")
            self.assertEqual(events[-1]["text"], "API Error: 429 Rate limit")

    def test_cc_api_error_persists_through_page_read(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "cc-session.jsonl"
            _write_log(path, [_user(), _cc_api_error("API Error: 503 Service Unavailable")])
            events, _before, _size, _has_older = _read_chat_tail_page(path, limit=10)
            self.assertEqual(events[-1]["message_class"], "error")
            self.assertEqual(events[-1]["text"], "API Error: 503 Service Unavailable")


class TestCcApiErrorNormalUnaffected(unittest.TestCase):
    def test_normal_cc_final_response_still_final_response(self) -> None:
        events, _meta, flags, _diag = _extract_chat_events([
            _user(),
            {
                "type": "assistant",
                "sessionId": SESSION_ID,
                "timestamp": "2026-07-04T00:00:01.000Z",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn"},
            },
        ])
        self.assertEqual(events[-1]["message_class"], "final_response")
        self.assertTrue(flags["turn_end"])

    def test_normal_cc_narration_still_narration(self) -> None:
        events, _meta, flags, _diag = _extract_chat_events([
            _user(),
            {
                "type": "assistant",
                "sessionId": SESSION_ID,
                "timestamp": "2026-07-04T00:00:01.000Z",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "working"}], "stop_reason": "tool_use"},
            },
        ])
        self.assertEqual(events[-1]["message_class"], "narration")
        self.assertFalse(flags["turn_end"])


class TestCcApiErrorBrokerTurnState(unittest.TestCase):
    def test_cc_api_error_closes_busy_turn(self) -> None:
        st = _state()
        _apply_rollout_obj_to_state(st, _user(), now_ts=1.0)
        self.assertTrue(st.busy)
        self.assertTrue(st.turn_open)
        _apply_rollout_obj_to_state(st, _cc_api_error("API Error: 401 Unauthorized"), now_ts=2.0)
        self.assertFalse(st.busy)
        self.assertFalse(st.turn_open)

    def test_cc_api_error_clears_pending_tool_calls(self) -> None:
        st = _state()
        _apply_rollout_obj_to_state(st, _user(), now_ts=1.0)
        # Start a tool call (busy with pending)
        _apply_rollout_obj_to_state(
            st,
            {
                "type": "assistant",
                "sessionId": SESSION_ID,
                "timestamp": "2026-07-04T00:00:00.500Z",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "name": "Bash", "id": "toolu_1", "input": {}}],
                    "stop_reason": "tool_use",
                },
            },
            now_ts=1.5,
        )
        self.assertEqual(st.pending_calls, {"toolu_1"})
        _apply_rollout_obj_to_state(st, _cc_api_error("API Error: 500"), now_ts=2.0)
        self.assertEqual(st.pending_calls, set())
        self.assertFalse(st.busy)


class TestCcApiErrorLogIdleReducer(unittest.TestCase):
    """The log-idle reducer drives session busy/idle from the JSONL log used
    by queue sweeps, unattended sweeps, and readiness projection. A CC API
    error row closes the backend turn, so the session must be idle afterwards.
    Before the fix, ``stop_reason == "stop_sequence"`` kept the reducer
    classifying the row as busy."""

    def test_cc_api_error_is_idle_via_log_reducer(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "cc-session.jsonl"
            _write_log(path, [_user(), _cc_api_error("API Error: 529 Overloaded")])
            self.assertIs(_compute_idle_from_log(path), True)

    def test_cc_api_error_idle_after_pending_tool_use(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "cc-session.jsonl"
            rows = [
                _user(),
                {
                    "type": "assistant",
                    "sessionId": SESSION_ID,
                    "timestamp": "2026-07-04T00:00:00.500Z",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "tool_use", "name": "Bash", "id": "toolu_1", "input": {}},
                            {"type": "text", "text": "running"},
                        ],
                        "stop_reason": "tool_use",
                    },
                },
                {
                    "type": "user",
                    "sessionId": SESSION_ID,
                    "timestamp": "2026-07-04T00:00:00.750Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"},
                        ],
                    },
                },
                _cc_api_error("API Error: 500"),
            ]
            _write_log(path, rows)
            self.assertIs(_compute_idle_from_log(path), True)

    def test_normal_cc_final_response_still_idle_via_reducer(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "cc-session.jsonl"
            _write_log(path, [
                _user(),
                {
                    "type": "assistant",
                    "sessionId": SESSION_ID,
                    "timestamp": "2026-07-04T00:00:01.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "done"}],
                        "stop_reason": "end_turn",
                    },
                },
            ])
            self.assertIs(_compute_idle_from_log(path), True)

    def test_normal_cc_narration_still_busy_via_reducer(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "cc-session.jsonl"
            _write_log(path, [
                _user(),
                {
                    "type": "assistant",
                    "sessionId": SESSION_ID,
                    "timestamp": "2026-07-04T00:00:01.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "working"}],
                        "stop_reason": "tool_use",
                    },
                },
            ])
            self.assertIs(_compute_idle_from_log(path), False)


class TestCcSystemApiErrorIdle(unittest.TestCase):
    """A terminal Claude Code ``system/api_error`` (retries exhausted) projects
    a visible error transcript event. The log-idle reducer and
    ``cc_current_turn_state_before`` must treat it as a turn-closing idle/error
    outcome so visible outcome and busy/idle truth stay aligned. A transient
    retry (``retryAttempt < maxRetries``) must not close idle by itself."""

    def test_terminal_system_api_error_is_idle_via_current_turn_state(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "cc-session.jsonl"
            _write_log(path, [_user(), _cc_system_api_error("API Error: 503", retry_attempt=3, max_retries=3)])
            _pending, idle = cc_current_turn_state_before(path, path.stat().st_size)
            self.assertEqual(_pending, set())
            self.assertIs(idle, True)

    def test_terminal_system_api_error_is_idle_via_log_reducer(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "cc-session.jsonl"
            _write_log(path, [_user(), _cc_system_api_error("API Error: 503", retry_attempt=3, max_retries=3)])
            self.assertIs(_compute_idle_from_log(path), True)

    def test_terminal_system_api_error_idle_after_pending_tool_use(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "cc-session.jsonl"
            rows = [
                _user(),
                {
                    "type": "assistant",
                    "sessionId": SESSION_ID,
                    "timestamp": "2026-07-04T00:00:00.500Z",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "tool_use", "name": "Bash", "id": "toolu_1", "input": {}}],
                        "stop_reason": "tool_use",
                    },
                },
                {
                    "type": "user",
                    "sessionId": SESSION_ID,
                    "timestamp": "2026-07-04T00:00:00.750Z",
                    "message": {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
                    },
                },
                _cc_system_api_error("API Error: 500", retry_attempt=2, max_retries=2),
            ]
            _write_log(path, rows)
            self.assertIs(_compute_idle_from_log(path), True)

    def test_transient_system_api_error_does_not_close_idle_by_itself(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "cc-session.jsonl"
            _write_log(path, [_user(), _cc_system_api_error("API Error: retrying", retry_attempt=1, max_retries=3)])
            _pending, idle = cc_current_turn_state_before(path, path.stat().st_size)
            self.assertIs(idle, False)
            self.assertIs(_compute_idle_from_log(path), False)

    def test_terminal_system_api_error_clears_pending_after_unmatched_tool_use(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "cc-session.jsonl"
            rows = [
                _user(),
                {
                    "type": "assistant",
                    "sessionId": SESSION_ID,
                    "timestamp": "2026-07-04T00:00:00.500Z",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "tool_use", "name": "Bash", "id": "toolu_1", "input": {}}],
                        "stop_reason": "tool_use",
                    },
                },
                _cc_system_api_error("API Error: 500", retry_attempt=2, max_retries=2),
            ]
            _write_log(path, rows)
            _pending, idle = cc_current_turn_state_before(path, path.stat().st_size)
            self.assertEqual(_pending, set())
            self.assertIs(idle, True)


if __name__ == "__main__":
    unittest.main()
