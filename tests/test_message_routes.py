from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from codoxear.launch_ledger import POST_LOG_RECOVERY_TRANSCRIPT_MAX_BYTES
from codoxear.launch_ledger import launch_attempt_transcript_payload
from codoxear.message_cursor import MessageCursorError
from codoxear.message_cursor import decode_message_cursor
from codoxear.message_cursor import encode_message_cursor
from codoxear.message_routes import MessageRouteDeps
from codoxear.message_routes import handle_messages_export
from codoxear.message_routes import handle_messages_history
from codoxear.message_routes import handle_messages_live
from codoxear.message_routes import handle_messages_search
from codoxear.message_routes import handle_messages_tail
from codoxear.post_log_recovery import POST_LOG_BOUND_BACKEND_STOPPED_TEXT
from codoxear.session_model import Session
from codoxear.token_signal import TOKEN_CLEAR

# Fixed HMAC secret so encode/decode are exercised against the real signing
# implementation, preserving the signed-cursor public contract under test.
_SECRET = b"test-message-route-secret"


class _FakeHandler:
    def __init__(self) -> None:
        self.unauthorized = False

    def _unauthorized(self) -> None:
        self.unauthorized = True


def _session(td: str, log_path: Path | None) -> Session:
    return Session(
        session_id="s1",
        thread_id="thread-1",
        broker_pid=1,
        codex_pid=1,
        agent_backend="codex",
        owned=False,
        start_ts=0.0,
        cwd=td,
        log_path=log_path,
        sock_path=Path(td) / "s1.sock",
    )


class _TailManager:
    """Manager fake for the tail route: only the methods the handler calls."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def refresh_session_meta(self, _sid: str) -> None:
        return None

    def get_session(self, _sid: str) -> Session:
        return self._session

    def _attach_notification_texts(self, events):
        return events


class _LiveManager:
    """Manager fake for the live route: records mark_log_delta calls."""

    def __init__(self, session: Session) -> None:
        self._session = session
        self.marked: list[tuple[tuple, dict]] = []

    def refresh_session_meta(self, _sid: str) -> None:
        return None

    def get_session(self, _sid: str) -> Session:
        return self._session

    def mark_log_delta(self, *args, **kwargs) -> None:
        self.marked.append((args, kwargs))

    def _attach_notification_texts(self, events):
        return events


def _deps(**overrides):
    responses: list[tuple[int, dict[str, object]]] = []
    metrics: list[tuple[str, float]] = []

    def json_response(_handler, status: int, payload: dict[str, object]) -> None:
        responses.append((status, payload))

    def encode_cursor(*, kind: str, session, pos: int) -> str:
        return encode_message_cursor(kind=kind, session=session, pos=pos, secret=_SECRET)

    def decode_cursor(token: str, *, kind: str, session) -> int:
        return decode_message_cursor(token, kind=kind, session=session, secret=_SECRET)

    def runtime_snapshot(_sid: str, _session, **_kw):
        # state, busy, queue_len, token
        return {}, False, 0, None

    deps = MessageRouteDeps(
        require_auth=lambda _handler: True,
        json_response=json_response,
        launch_attempt_transcript_for_session_id=lambda _sid: None,
        transcript_export_max_bytes=50 * 1024 * 1024,
        transcript_search_max_line_bytes=64 * 1024,
        decode_message_cursor=decode_cursor,
        encode_message_cursor=encode_cursor,
        record_metric=lambda name, value: metrics.append((name, value)),
        message_runtime_snapshot=runtime_snapshot,
    )
    for name, value in overrides.items():
        object.__setattr__(deps, name, value)
    return deps, responses, metrics


def test_messages_tail_returns_signed_live_and_history_cursors() -> None:
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "rollout.jsonl"
        rows = [
            {"type": "event_msg", "payload": {"type": "user_message", "message": "hello"}, "ts": 1.0},
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "world"}],
                    "phase": "final_answer",
                },
                "ts": 2.0,
            },
        ]
        log_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        session = _session(td, log_path)

        deps, responses, metrics = _deps()
        handle_messages_tail(
            _FakeHandler(),
            session_id="s1",
            query="limit=20",
            manager=_TailManager(session),
            deps=deps,
        )

    assert len(responses) == 1
    status, body = responses[0]
    assert status == 200
    # Only two rows and limit=20 -> no older page, so no history cursor on the page.
    assert body["history_cursor"] is None
    live_cursor = body["live_cursor"]
    assert isinstance(live_cursor, str)
    # Strengthen: the live cursor is a real signed cursor that decodes for this session.
    assert decode_message_cursor(live_cursor, kind="live", session=session, secret=_SECRET) >= 0
    assert [event["role"] for event in body["events"]] == ["user", "assistant"]
    assert body["events"][0]["text"] == "hello"
    assert body["events"][1]["text"] == "world"
    # The first event carries a per-event signed history cursor.
    ev0_cursor = body["events"][0].get("history_cursor")
    assert isinstance(ev0_cursor, str)
    assert decode_message_cursor(ev0_cursor, kind="history", session=session, secret=_SECRET) >= 0
    assert body["busy"] is False
    assert body["queue_len"] == 0
    assert body["token"] is None
    assert "diag" not in body
    assert metrics and metrics[0][0] == "api_messages_init_ms"


def test_messages_live_rejects_bad_cursor_without_mutating_log_state() -> None:
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "rollout.jsonl"
        log_path.write_text("", encoding="utf-8")
        session = _session(td, log_path)

        deps, responses, _metrics = _deps()
        manager = _LiveManager(session)
        handle_messages_live(
            _FakeHandler(),
            session_id="s1",
            query="cursor=not-a-valid-cursor",
            manager=manager,
            deps=deps,
        )

    assert responses == [(409, {"error": "cursor_invalid"})]
    # The bad cursor must short-circuit before any log mutation.
    assert manager.marked == []


def test_messages_live_streams_new_events_after_valid_cursor() -> None:
    # Regression: the live route must pass an explicit read bound to the JSONL
    # reader. A dropped max_bytes broke every live poll on a bound cursor
    # (TypeError) while tail-only tests stayed green.
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "rollout.jsonl"
        rows = [
            {"type": "event_msg", "payload": {"type": "user_message", "message": "hello"}, "ts": 1.0},
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "world"}],
                    "phase": "final_answer",
                },
                "ts": 2.0,
            },
        ]
        log_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        session = _session(td, log_path)

        deps, responses, metrics = _deps()
        manager = _LiveManager(session)
        cursor = encode_message_cursor(kind="live", session=session, pos=0, secret=_SECRET)
        handle_messages_live(
            _FakeHandler(),
            session_id="s1",
            query=f"cursor={cursor}",
            manager=manager,
            deps=deps,
        )

    assert len(responses) == 1
    status, body = responses[0]
    assert status == 200
    assert [event["role"] for event in body["events"]] == ["user", "assistant"]
    assert body["events"][1]["text"] == "world"
    assert "diag" not in body
    next_cursor = body["live_cursor"]
    assert isinstance(next_cursor, str)
    assert decode_message_cursor(next_cursor, kind="live", session=session, secret=_SECRET) > 0
    # The consumed delta must be marked so busy/idle caches advance.
    assert manager.marked and manager.marked[0][1]["new_off"] > 0
    assert metrics and metrics[0][0] == "api_messages_poll_ms"


def test_messages_live_unknown_cc_usage_clears_stale_session_token() -> None:
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "cc.jsonl"
        known = {
            "type": "assistant",
            "sessionId": "thread-1",
            "timestamp": "2026-07-07T00:00:00.000Z",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "older"}],
                "stop_reason": "end_turn",
                "model": "claude-sonnet-4-5",
                "usage": {"input_tokens": 512},
            },
        }
        unknown = {
            "type": "assistant",
            "sessionId": "thread-1",
            "timestamp": "2026-07-07T00:01:00.000Z",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "newer"}],
                "stop_reason": "end_turn",
                "model": "claude-unmapped-9",
                "usage": {"input_tokens": 12000},
            },
        }
        log_path.write_text(json.dumps(known) + "\n", encoding="utf-8")
        cursor_pos = log_path.stat().st_size
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(unknown) + "\n")
        session = _session(td, log_path)
        session.agent_backend = "cc"
        session.token = {"tokens_in_context": 512}
        seen_token_updates = []

        def runtime_snapshot(_sid: str, live_session, **kw):
            seen_token_updates.append(kw.get("token_update"))
            return {}, False, 0, live_session.token

        deps, responses, _metrics = _deps(message_runtime_snapshot=runtime_snapshot)
        manager = _LiveManager(session)
        cursor = encode_message_cursor(kind="live", session=session, pos=cursor_pos, secret=_SECRET)
        handle_messages_live(
            _FakeHandler(),
            session_id="s1",
            query=f"cursor={cursor}",
            manager=manager,
            deps=deps,
        )

    assert len(responses) == 1
    status, body = responses[0]
    assert status == 200
    assert seen_token_updates == [TOKEN_CLEAR]
    assert session.token is None
    assert body["token"] is None


def test_messages_tail_missing_session_returns_launch_payload_or_404() -> None:
    # Strengthens coverage of the unknown-session branch in the tail route.
    deps, responses, _metrics = _deps(
        launch_attempt_transcript_for_session_id=lambda _sid: {"transcript_state": "pending_bind", "events": []},
    )

    class _MissingManager:
        def refresh_session_meta(self, _sid: str) -> None:
            return None

        def get_session(self, _sid: str):
            return None

    handle_messages_tail(
        _FakeHandler(),
        session_id="nope",
        query="limit=20",
        manager=_MissingManager(),
        deps=deps,
    )
    assert responses == [(200, {"transcript_state": "pending_bind", "events": []})]


def _post_log_recovery_payload(log_path: Path) -> dict[str, object]:
    return {
        "transcript_state": "failed",
        "session_id": "broker-123",
        "thread_id": "thread-recovered",
        "log_path": str(log_path),
        "live_cursor": None,
        "history_cursor": None,
        "events": [
            {"role": "user", "text": "POST_LOG_BOUND_DEATH_SENTINEL", "ts": 1.0, "_before_byte": 0},
            {
                "role": "assistant",
                "text": POST_LOG_BOUND_BACKEND_STOPPED_TEXT,
                "ts": 2.0,
                "message_class": "error",
                "codoxear_lifecycle": "backend_stopped_after_log_bind",
            },
        ],
        "has_older": False,
        "busy": False,
        "queue_len": 0,
        "token": None,
    }


class _MissingManager:
    def refresh_session_meta(self, _sid: str) -> None:
        return None

    def get_session(self, _sid: str):
        return None

    def _attach_notification_texts(self, events):
        return events

    def mark_log_delta(self, *args, **kwargs) -> None:
        raise AssertionError("missing-session recovery route must not mutate live log state")


def test_post_log_recovery_missing_session_routes_share_payload() -> None:
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "post-log.jsonl"
        rows = [
            {"type": "session", "id": "thread-recovered"},
            {"type": "event_msg", "payload": {"type": "user_message", "message": "POST_LOG_BOUND_DEATH_SENTINEL"}, "ts": 1.0},
        ]
        log_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        payload = _post_log_recovery_payload(log_path)
        deps, responses, metrics = _deps(launch_attempt_transcript_for_session_id=lambda _sid: payload)
        manager = _MissingManager()

        handle_messages_tail(_FakeHandler(), session_id="broker-123", query="limit=20", manager=manager, deps=deps)
        tail_status, tail_body = responses.pop()
        assert tail_status == 200
        assert [ev["role"] for ev in tail_body["events"]] == ["user", "assistant"]
        assert tail_body["events"][0]["text"] == "POST_LOG_BOUND_DEATH_SENTINEL"
        assert "_before_byte" not in tail_body["events"][0]
        assert isinstance(tail_body["events"][0].get("history_cursor"), str)
        assert tail_body["events"][1]["message_class"] == "error"
        assert tail_body["busy"] is False
        assert metrics and metrics[-1][0] == "api_messages_init_ms"

        handle_messages_search(_FakeHandler(), session_id="broker-123", query="q=POST_LOG_BOUND_DEATH_SENTINEL&limit=20", manager=manager, deps=deps)
        search_status, search_body = responses.pop()
        assert search_status == 200
        assert search_body["match_count"] == 1
        assert search_body["matches"][0]["text"] == "POST_LOG_BOUND_DEATH_SENTINEL"
        assert isinstance(search_body["matches"][0].get("history_cursor"), str)
        assert isinstance(search_body["matches"][0].get("load_cursor"), str)

        handle_messages_search(_FakeHandler(), session_id="broker-123", query="q=stopped before completing&limit=20", manager=manager, deps=deps)
        error_search_status, error_search_body = responses.pop()
        assert error_search_status == 200
        assert error_search_body["match_count"] == 1
        assert error_search_body["matches"][0]["message_class"] == "error"

        handle_messages_export(_FakeHandler(), session_id="broker-123", manager=manager, deps=deps)
        export_status, export_body = responses.pop()
        assert export_status == 200
        assert export_body["event_count"] == 2
        assert [ev["role"] for ev in export_body["events"]] == ["user", "assistant"]

        handle_messages_history(_FakeHandler(), session_id="broker-123", query="cursor=unused", manager=manager, deps=deps)
        history_status, history_body = responses.pop()
        assert history_status == 200
        assert history_body["has_older"] is False
        assert [ev["role"] for ev in history_body["events"]] == ["user", "assistant"]

        handle_messages_live(_FakeHandler(), session_id="broker-123", query="cursor=unused", manager=manager, deps=deps)
        live_status, live_body = responses.pop()
        assert live_status == 200
        assert live_body["live_cursor"] is None
        assert live_body["turn_end"] is True
        assert live_body["busy"] is False
        assert [ev["role"] for ev in live_body["events"]] == ["user", "assistant"]


def test_pre_log_failed_launch_missing_session_search_remains_payload_only() -> None:
    payload = {
        "transcript_state": "failed",
        "thread_id": "launch-pre-log",
        "log_path": None,
        "live_cursor": None,
        "history_cursor": None,
        "events": [
            {"role": "user", "text": "submitted before launch", "ts": 1.0},
            {"role": "assistant", "text": "Session launch failed before a transcript log was created. PRE_LOG_ERROR_SENTINEL", "ts": 2.0, "message_class": "error"},
        ],
        "has_older": False,
        "busy": False,
        "queue_len": 0,
        "token": None,
    }
    deps, responses, _metrics = _deps(launch_attempt_transcript_for_session_id=lambda _sid: payload)

    handle_messages_search(
        _FakeHandler(),
        session_id="launch-pre-log",
        query="q=PRE_LOG_ERROR_SENTINEL&limit=20",
        manager=_MissingManager(),
        deps=deps,
    )

    assert responses
    status, body = responses[-1]
    assert status == 200
    assert body["match_count"] == 1
    assert body["match_count_truncated"] is False
    assert body["matches"][0]["message_class"] == "error"
    assert body["matches"][0]["text"].endswith("PRE_LOG_ERROR_SENTINEL")


def test_large_post_log_recovery_tail_exposes_usable_history_cursor() -> None:
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "large-post-log.jsonl"
        first = {"type": "event_msg", "ts": 1.0, "payload": {"type": "user_message", "message": "FIRST_EVENT_SENTINEL"}}
        filler = {"type": "debug", "payload": "x" * 4096}
        with log_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(first, separators=(",", ":")) + "\n")
            filler_line = json.dumps(filler, separators=(",", ":")) + "\n"
            while f.tell() <= POST_LOG_RECOVERY_TRANSCRIPT_MAX_BYTES + 4096:
                f.write(filler_line)

        rec = {
            "launch_id": "launch-large-post-log",
            "session_id": "broker-large",
            "thread_id": "thread-large",
            "state": "failed",
            "stage": "broker_exit_after_log_bind",
            "error": "backend stopped after binding a transcript log",
            "agent_backend": "codex",
            "cwd": td,
            "created_ts": 1.0,
            "updated_ts": 3.0,
            "log_path": str(log_path),
        }
        payload = launch_attempt_transcript_payload(rec)

        payload_texts = [ev.get("text") for ev in payload["events"]]
        assert payload_texts == [POST_LOG_BOUND_BACKEND_STOPPED_TEXT]
        assert "FIRST_EVENT_SENTINEL" not in payload_texts
        assert payload["has_older"] is True
        assert payload["history_cursor"] is None
        assert isinstance(payload.get("_history_before_byte"), int)

        deps, responses, _metrics = _deps(launch_attempt_transcript_for_session_id=lambda _sid: payload)
        manager = _MissingManager()
        handle_messages_tail(_FakeHandler(), session_id="broker-large", query="limit=20", manager=manager, deps=deps)
        tail_status, tail_body = responses.pop()
        assert tail_status == 200
        assert "_history_before_byte" not in tail_body
        assert tail_body["has_older"] is True
        assert isinstance(tail_body["history_cursor"], str) and tail_body["history_cursor"]
        cursor_session = SimpleNamespace(thread_id="thread-large", log_path=log_path)
        assert decode_message_cursor(tail_body["history_cursor"], kind="history", session=cursor_session, secret=_SECRET) == payload["_history_before_byte"]
        assert [ev.get("text") for ev in tail_body["events"]] == [POST_LOG_BOUND_BACKEND_STOPPED_TEXT]
        lifecycle_event = tail_body["events"][0]
        assert lifecycle_event.get("codoxear_lifecycle") == "backend_stopped_after_log_bind"
        assert isinstance(lifecycle_event.get("history_cursor"), str) and lifecycle_event["history_cursor"]
        assert decode_message_cursor(lifecycle_event["history_cursor"], kind="history", session=cursor_session, secret=_SECRET) == payload["_history_before_byte"]

        handle_messages_search(
            _FakeHandler(),
            session_id="broker-large",
            query="q=FIRST_EVENT_SENTINEL&limit=20",
            manager=manager,
            deps=deps,
        )
        search_status, search_body = responses.pop()
        assert search_status == 200
        assert search_body["match_count"] == 1
        assert search_body["match_count_truncated"] is False
        assert [match.get("text") for match in search_body["matches"]] == ["FIRST_EVENT_SENTINEL"]
        head_match = search_body["matches"][0]
        assert isinstance(head_match.get("history_cursor"), str) and head_match["history_cursor"]
        assert decode_message_cursor(head_match["history_cursor"], kind="history", session=cursor_session, secret=_SECRET) == 0
        assert isinstance(head_match.get("load_cursor"), str) and head_match["load_cursor"]
        assert decode_message_cursor(head_match["load_cursor"], kind="history", session=cursor_session, secret=_SECRET) > 0

        handle_messages_search(
            _FakeHandler(),
            session_id="broker-large",
            query=f"q=FIRST_EVENT_SENTINEL&limit=20&before={tail_body['history_cursor']}",
            manager=manager,
            deps=deps,
        )
        before_search_status, before_search_body = responses.pop()
        assert before_search_status == 200
        assert before_search_body["match_count"] == 1
        assert [match.get("text") for match in before_search_body["matches"]] == ["FIRST_EVENT_SENTINEL"]
        assert isinstance(before_search_body["matches"][0].get("load_cursor"), str)

        handle_messages_search(
            _FakeHandler(),
            session_id="broker-large",
            query="q=stopped before completing&limit=20",
            manager=manager,
            deps=deps,
        )
        lifecycle_search_status, lifecycle_search_body = responses.pop()
        assert lifecycle_search_status == 200
        assert lifecycle_search_body["match_count"] == 1
        assert [match.get("text") for match in lifecycle_search_body["matches"]] == [POST_LOG_BOUND_BACKEND_STOPPED_TEXT]
        assert lifecycle_search_body["matches"][0]["message_class"] == "error"

        handle_messages_history(
            _FakeHandler(),
            session_id="broker-large",
            query=f"cursor={lifecycle_event['history_cursor']}&limit=20",
            manager=manager,
            deps=deps,
        )
        history_status, history_body = responses.pop()
        assert history_status == 200
        assert "_history_before_byte" not in history_body
        history_texts = [ev.get("text") for ev in history_body["events"]]
        assert "FIRST_EVENT_SENTINEL" in history_texts
        assert POST_LOG_BOUND_BACKEND_STOPPED_TEXT not in history_texts


def test_messages_export_active_session_retains_size_cap() -> None:
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "oversize.jsonl"
        log_path.write_text("x" * 32, encoding="utf-8")
        session = _session(td, log_path)
        deps, responses, _metrics = _deps(transcript_export_max_bytes=8)
        manager = _TailManager(session)

        handle_messages_export(_FakeHandler(), session_id="s1", manager=manager, deps=deps)

    assert responses
    status, body = responses[-1]
    assert status == 413
    assert body["max_bytes"] == 8
    assert "too large to export" in str(body["error"])


def test_messages_export_missing_recovery_session_retains_size_cap() -> None:
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "oversize.jsonl"
        log_path.write_text("x" * 32, encoding="utf-8")
        payload = _post_log_recovery_payload(log_path)
        deps, responses, _metrics = _deps(
            launch_attempt_transcript_for_session_id=lambda _sid, **_kw: payload,
            transcript_export_max_bytes=8,
        )

        handle_messages_export(_FakeHandler(), session_id="broker-123", manager=_MissingManager(), deps=deps)

    assert responses
    status, body = responses[-1]
    assert status == 413
    assert body["max_bytes"] == 8


def test_messages_live_requires_cursor() -> None:
    # Strengthens coverage: missing cursor -> 400 before any log mutation.
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "rollout.jsonl"
        log_path.write_text("", encoding="utf-8")
        session = _session(td, log_path)

        deps, responses, _metrics = _deps()
        manager = _LiveManager(session)
        handle_messages_live(
            _FakeHandler(),
            session_id="s1",
            query="",
            manager=manager,
            deps=deps,
        )

    assert responses == [(400, {"error": "cursor required"})]
    assert manager.marked == []


def _cc_user_row(text: str = "silent", *, ts: str = "2026-07-04T00:00:00.000Z") -> dict:
    return {
        "type": "user",
        "sessionId": "s1",
        "timestamp": ts,
        "cwd": "/repo",
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
    }


def _cc_turn_duration_row(*, ts: str = "2026-07-04T00:00:03.000Z") -> dict:
    return {
        "type": "system",
        "subtype": "turn_duration",
        "sessionId": "s1",
        "timestamp": ts,
        "durationMs": 1234,
    }


def test_messages_live_cc_split_turn_duration_emits_no_response() -> None:
    # Route-level regression for the live polling split close that previously
    # left Claude Code turns with no visible result. The user row is delivered
    # in one poll; the system/turn_duration close arrives in the next poll.
    # Before the fix the public live route used the Codex-only prior-turn
    # context helper, which yields (None, False) for CC user rows, so the
    # close-only delta injected no no-response assistant error and the browser
    # went idle with no result. The route must use the CC-aware prior-turn
    # context and emit the no-response event on the second poll.
    from codoxear.rollout_chat_events import _NO_RESPONSE_TEXT

    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "cc-session.jsonl"
        log_path.write_text(json.dumps(_cc_user_row("silent")) + "\n", encoding="utf-8")
        session = _session(td, log_path)
        session.agent_backend = "cc"

        deps, responses, _metrics = _deps()
        manager = _LiveManager(session)
        cursor0 = encode_message_cursor(kind="live", session=session, pos=0, secret=_SECRET)
        handle_messages_live(
            _FakeHandler(),
            session_id="s1",
            query=f"cursor={cursor0}",
            manager=manager,
            deps=deps,
        )
        assert len(responses) == 1
        _status1, body1 = responses[0]
        assert [ev["role"] for ev in body1["events"]] == ["user"]
        cursor1 = body1["live_cursor"]
        assert isinstance(cursor1, str)
        responses.clear()

        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_cc_turn_duration_row()) + "\n")

        handle_messages_live(
            _FakeHandler(),
            session_id="s1",
            query=f"cursor={cursor1}",
            manager=manager,
            deps=deps,
        )
        assert len(responses) == 1
        status2, body2 = responses[0]
        assert status2 == 200
        roles = [ev["role"] for ev in body2["events"]]
        assert roles == ["assistant"], roles
        assert body2["events"][0]["text"] == _NO_RESPONSE_TEXT
        assert body2["events"][0]["message_class"] == "error"


def test_messages_live_cc_split_prior_answer_suppresses_no_response() -> None:
    # Same split, but the assistant answered in the first poll. The CC-aware
    # prior-turn context must record the visible assistant event so the later
    # close does not false-inject a no-response row. Preserves existing
    # answered-turn behavior.
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "cc-session.jsonl"
        rows = [
            {
                "type": "user",
                "sessionId": "s1",
                "timestamp": "2026-07-04T00:00:00.000Z",
                "cwd": td,
                "message": {"role": "user", "content": [{"type": "text", "text": "answer"}]},
            },
            {
                "type": "assistant",
                "sessionId": "s1",
                "timestamp": "2026-07-04T00:00:01.000Z",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "answered"}],
                    "stop_reason": "end_turn",
                },
            },
        ]
        log_path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        session = _session(td, log_path)
        session.agent_backend = "cc"

        deps, responses, _metrics = _deps()
        manager = _LiveManager(session)
        cursor0 = encode_message_cursor(kind="live", session=session, pos=0, secret=_SECRET)
        handle_messages_live(
            _FakeHandler(),
            session_id="s1",
            query=f"cursor={cursor0}",
            manager=manager,
            deps=deps,
        )
        assert len(responses) == 1
        _status1, body1 = responses[0]
        assert [ev["role"] for ev in body1["events"]] == ["user", "assistant"]
        cursor1 = body1["live_cursor"]
        responses.clear()

        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_cc_turn_duration_row()) + "\n")

        handle_messages_live(
            _FakeHandler(),
            session_id="s1",
            query=f"cursor={cursor1}",
            manager=manager,
            deps=deps,
        )
        assert len(responses) == 1
        _status2, body2 = responses[0]
        assert body2["events"] == []


def test_messages_unauthorized_short_circuits() -> None:
    # Strengthens coverage: auth gate precedes all manager work.
    deps, responses, _metrics = _deps(require_auth=lambda _handler: False)
    handler = _FakeHandler()

    class _TouchManager:
        def __init__(self) -> None:
            self.touched = False

        def refresh_session_meta(self, _sid: str) -> None:
            self.touched = True

    manager = _TouchManager()
    handle_messages_tail(
        handler,
        session_id="s1",
        query="limit=20",
        manager=manager,
        deps=deps,
    )
    assert handler.unauthorized is True
    assert responses == []
    assert manager.touched is False
