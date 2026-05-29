"""Tests for agent error event surfacing across Codex / Pi / Claude backends.

These cover both the parser layer (`_extract_chat_events`,
`_chat_events_for_record`, cursor APIs) and the broker busy-state machine
(`_apply_rollout_obj_to_state`).
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from codoxear.broker import State
from codoxear.broker import _apply_rollout_obj_to_state
from codoxear.rollout_log import _chat_events_for_record
from codoxear.rollout_log import _extract_chat_events
from codoxear.rollout_log import _read_chat_history_page
from codoxear.rollout_log import _read_chat_live_delta
from codoxear.rollout_log import _read_chat_tail_page


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def _busy_state() -> State:
    s = State(
        codex_pid=1,
        pty_master_fd=-1,
        cwd="/tmp",
        start_ts=0.0,
        codex_home=Path("/tmp"),
        sessions_dir=Path("/tmp"),
    )
    s.busy = True
    s.turn_open = True
    s.last_turn_activity_ts = 100.0
    s.pending_calls.add("callA")
    return s


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    chunks = [(json.dumps(r) + "\n").encode("utf-8") for r in rows]
    path.write_bytes(b"".join(chunks))


class ParserAgentErrorTests(unittest.TestCase):
    def test_real_claude_api_error_yields_agent_error_event(self):
        with (FIXTURE_DIR / "claude_api_error_sample.jsonl").open() as f:
            obj = json.loads(f.readline())

        events, _meta, flags, _diag = _extract_chat_events([obj])

        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev["class"], "agent_error")
        self.assertEqual(ev["source"], "claude")
        self.assertEqual(ev["type"], "api_error_502")
        self.assertEqual(ev["message"], "Upstream service temporarily unavailable")
        self.assertIn("ts", ev)
        self.assertTrue(flags["turn_end"])

    def test_codex_stream_error_yields_agent_error_event(self):
        obj = {
            "type": "event_msg",
            "timestamp": "2026-05-29T01:00:00Z",
            "payload": {"type": "stream_error", "message": "upstream timed out"},
        }
        events, _m, flags, _d = _extract_chat_events([obj])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["class"], "agent_error")
        self.assertEqual(events[0]["source"], "codex")
        self.assertEqual(events[0]["type"], "stream_error")
        self.assertEqual(events[0]["message"], "upstream timed out")
        self.assertTrue(flags["turn_end"])

    def test_codex_generic_error_payload_recognized(self):
        obj = {
            "type": "event_msg",
            "payload": {"type": "rate_limit_error", "error": {"message": "429 too many"}},
        }
        events, _m, _f, _d = _extract_chat_events([obj])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "rate_limit_error")
        self.assertEqual(events[0]["message"], "429 too many")

    def test_pi_tool_result_iserror_yields_agent_error_event(self):
        obj = {
            "type": "message",
            "message": {
                "role": "toolResult",
                "content": [{"type": "text", "text": "permission denied", "isError": True}],
            },
        }
        events, _m, flags, _d = _extract_chat_events([obj])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["source"], "pi")
        self.assertEqual(events[0]["type"], "tool_error")
        self.assertEqual(events[0]["message"], "permission denied")
        self.assertTrue(flags["turn_end"])

    def test_pi_normal_tool_result_unaffected(self):
        obj = {
            "type": "message",
            "message": {
                "role": "toolResult",
                "content": [{"type": "text", "text": "file contents"}],
            },
        }
        events, _m, _f, _d = _extract_chat_events([obj])
        self.assertEqual(events, [])  # toolResult without isError emits nothing

    def test_chat_events_for_record_routes_through_extractor(self):
        with (FIXTURE_DIR / "claude_api_error_sample.jsonl").open() as f:
            obj = json.loads(f.readline())
        cursor_events = _chat_events_for_record(obj)
        self.assertEqual(len(cursor_events), 1)
        self.assertEqual(cursor_events[0]["class"], "agent_error")


class CursorApiAgentErrorTests(unittest.TestCase):
    """Confirm the byte-cursor APIs include agent_error events."""

    def test_tail_page_includes_agent_error(self):
        with (FIXTURE_DIR / "claude_api_error_sample.jsonl").open() as f:
            err_obj = json.loads(f.readline())
        text_obj = {
            "type": "user",
            "timestamp": "2026-05-27T12:31:55.000Z",
            "message": {"role": "user", "content": "hello"},
        }

        with TemporaryDirectory() as tmp:
            log = Path(tmp) / "rollout.jsonl"
            _write_jsonl(log, [text_obj, err_obj])

            events, _before, _after, _has_older = _read_chat_tail_page(log, limit=10)

        kinds = [(e.get("class"), e.get("source"), e.get("type"), e.get("role")) for e in events]
        self.assertIn(("agent_error", "claude", "api_error_502", "system"), kinds)

    def test_live_delta_picks_up_appended_error(self):
        text_obj = {
            "type": "user",
            "timestamp": "2026-05-27T12:31:50.000Z",
            "message": {"role": "user", "content": "hi"},
        }
        with (FIXTURE_DIR / "claude_api_error_sample.jsonl").open() as f:
            err_obj = json.loads(f.readline())

        with TemporaryDirectory() as tmp:
            log = Path(tmp) / "rollout.jsonl"
            _write_jsonl(log, [text_obj])

            tail_events, _before, after, _has_older = _read_chat_tail_page(log, limit=10)
            self.assertEqual(len(tail_events), 1)

            # Append the error and read incrementally from the cursor.
            with log.open("ab") as f:
                f.write((json.dumps(err_obj) + "\n").encode("utf-8"))

            new_events, _next_after, _meta, _flags, _diag, _tok = _read_chat_live_delta(
                log, after_byte=after
            )

        self.assertTrue(any(e.get("class") == "agent_error" for e in new_events))

    def test_history_page_paginates_back_through_errors(self):
        with (FIXTURE_DIR / "claude_api_error_sample.jsonl").open() as f:
            err_obj = json.loads(f.readline())
        many = [
            {
                "type": "user",
                "timestamp": f"2026-05-27T12:30:{i:02d}.000Z",
                "message": {"role": "user", "content": f"m{i}"},
            }
            for i in range(5)
        ]

        with TemporaryDirectory() as tmp:
            log = Path(tmp) / "rollout.jsonl"
            _write_jsonl(log, many + [err_obj])

            events, before, _has_older = _read_chat_history_page(log, before_byte=log.stat().st_size, limit=10)

        # The error must show up at the tail position regardless of paging.
        self.assertTrue(any(e.get("class") == "agent_error" for e in events))


class BrokerBusyStateAgentErrorTests(unittest.TestCase):
    def test_claude_api_error_with_pending_retry_keeps_busy(self):
        # The sample fixture is retryAttempt=1/maxRetries=10 with retryInMs set:
        # the Claude CLI will auto-retry, so the turn is NOT terminal and busy
        # must stay true (clearing it makes the UI flap between working/idle).
        with (FIXTURE_DIR / "claude_api_error_sample.jsonl").open() as f:
            obj = json.loads(f.readline())
        st = _busy_state()
        _apply_rollout_obj_to_state(st, obj, now_ts=200.0)
        self.assertTrue(st.busy)
        self.assertTrue(st.turn_open)
        self.assertEqual(st.last_turn_activity_ts, 200.0)

    def test_claude_api_error_no_retry_clears_busy(self):
        # No retryInMs scheduled -> terminal failure -> close the turn.
        obj = {
            "type": "system",
            "subtype": "api_error",
            "level": "error",
            "error": {"status": 500, "error": {"error": {"message": "boom", "type": "x"}}},
        }
        st = _busy_state()
        _apply_rollout_obj_to_state(st, obj, now_ts=200.0)
        self.assertFalse(st.busy)
        self.assertFalse(st.turn_open)
        self.assertFalse(st.pending_calls)

    def test_claude_api_error_retries_exhausted_clears_busy(self):
        # retryAttempt >= maxRetries -> retries exhausted -> terminal.
        obj = {
            "type": "system",
            "subtype": "api_error",
            "level": "error",
            "retryInMs": 500,
            "retryAttempt": 10,
            "maxRetries": 10,
            "error": {"status": 529, "error": {"error": {"message": "overloaded", "type": "x"}}},
        }
        st = _busy_state()
        _apply_rollout_obj_to_state(st, obj, now_ts=200.0)
        self.assertFalse(st.busy)
        self.assertFalse(st.turn_open)

    def test_claude_api_error_malformed_retry_counters_clears_busy(self):
        # retryInMs is set but the attempt/maxRetries counters are missing. We
        # cannot confirm a retry is pending, so the error is treated as terminal
        # to avoid a stuck-busy UI (no follow-up event would ever clear it).
        obj = {
            "type": "system",
            "subtype": "api_error",
            "level": "error",
            "retryInMs": 500,
            "error": {"status": 529, "error": {"error": {"message": "overloaded", "type": "x"}}},
        }
        st = _busy_state()
        _apply_rollout_obj_to_state(st, obj, now_ts=200.0)
        self.assertFalse(st.busy)
        self.assertFalse(st.turn_open)

    def test_claude_api_error_noninteger_retry_counters_clears_busy(self):
        # Non-integer counters (string shapes) cannot be compared as attempt <
        # maxRetries, so the error is terminal rather than silently recoverable.
        obj = {
            "type": "system",
            "subtype": "api_error",
            "level": "error",
            "retryInMs": 500,
            "retryAttempt": "1",
            "maxRetries": "10",
            "error": {"status": 529, "error": {"error": {"message": "overloaded", "type": "x"}}},
        }
        st = _busy_state()
        _apply_rollout_obj_to_state(st, obj, now_ts=200.0)
        self.assertFalse(st.busy)
        self.assertFalse(st.turn_open)

    def test_claude_api_error_bool_retry_in_ms_clears_busy(self):
        # `retryInMs: true` must not be read as a numeric 1ms delay (bool is an
        # int subclass in Python). A boolean retry flag is not a scheduled retry,
        # so the error is terminal.
        obj = {
            "type": "system",
            "subtype": "api_error",
            "level": "error",
            "retryInMs": True,
            "retryAttempt": 1,
            "maxRetries": 10,
            "error": {"status": 529, "error": {"error": {"message": "overloaded", "type": "x"}}},
        }
        st = _busy_state()
        _apply_rollout_obj_to_state(st, obj, now_ts=200.0)
        self.assertFalse(st.busy)
        self.assertFalse(st.turn_open)

    def test_claude_api_error_nan_retry_in_ms_clears_busy(self):
        # `retryInMs: NaN` (json.loads accepts NaN/Infinity) is a float, and
        # `NaN <= 0` is False, so without an isfinite guard it would slip through
        # as a "pending retry" and strand busy forever. It must be terminal.
        nan = float("nan")
        for bad in (nan, float("inf")):
            obj = {
                "type": "system",
                "subtype": "api_error",
                "level": "error",
                "retryInMs": bad,
                "retryAttempt": 1,
                "maxRetries": 10,
                "error": {"status": 529, "error": {"error": {"message": "overloaded", "type": "x"}}},
            }
            st = _busy_state()
            _apply_rollout_obj_to_state(st, obj, now_ts=200.0)
            self.assertFalse(st.busy, f"retryInMs={bad} should be terminal")
            self.assertFalse(st.turn_open)

    def test_claude_api_error_valid_pending_retry_keeps_busy(self):
        # Positive evidence of a pending retry (retryInMs>0, attempt<maxRetries)
        # -> non-terminal, busy stays set and the retry counts as activity.
        obj = {
            "type": "system",
            "subtype": "api_error",
            "level": "error",
            "retryInMs": 500,
            "retryAttempt": 1,
            "maxRetries": 10,
            "error": {"status": 529, "error": {"error": {"message": "overloaded", "type": "x"}}},
        }
        st = _busy_state()
        _apply_rollout_obj_to_state(st, obj, now_ts=200.0)
        self.assertTrue(st.busy)
        self.assertTrue(st.turn_open)
        self.assertEqual(st.last_turn_activity_ts, 200.0)

    def test_pi_tool_error_clears_busy(self):
        # Pi error records carry no retry semantics, so a toolResult isError is
        # turn-terminal: it closes the turn and clears busy. Leaving the turn
        # open would disable the idle-fallback (it only fires once a completion
        # candidate exists) and strand the spinner on "working" indefinitely.
        obj = {
            "type": "message",
            "message": {
                "role": "toolResult",
                "content": [{"type": "text", "text": "denied", "isError": True}],
            },
        }
        st = _busy_state()
        _apply_rollout_obj_to_state(st, obj, now_ts=200.0)
        self.assertFalse(st.busy)
        self.assertFalse(st.turn_open)

    def test_codex_stream_error_clears_busy(self):
        # Codex error records carry no retry semantics, so a stream_error is
        # turn-terminal (same rationale as the Pi case above).
        obj = {"type": "event_msg", "payload": {"type": "stream_error", "message": "503"}}
        st = _busy_state()
        _apply_rollout_obj_to_state(st, obj, now_ts=200.0)
        self.assertFalse(st.busy)
        self.assertFalse(st.turn_open)

    def test_normal_user_message_keeps_busy(self):
        obj = {"type": "event_msg", "payload": {"type": "user_message", "message": "hi"}}
        st = _busy_state()
        _apply_rollout_obj_to_state(st, obj, now_ts=200.0)
        self.assertTrue(st.busy)
        self.assertTrue(st.turn_open)


if __name__ == "__main__":
    unittest.main()
