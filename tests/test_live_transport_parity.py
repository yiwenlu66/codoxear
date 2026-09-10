from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

import codoxear.message_routes as message_routes
from codoxear.message_cursor import decode_message_cursor
from codoxear.message_cursor import encode_message_cursor
from codoxear.message_routes import MessageRouteDeps
from codoxear.message_routes import handle_messages_live
from codoxear.message_routes import handle_messages_live_stream
from codoxear.rollout_idle import _analyze_log_chunk
from codoxear.rollout_log import _compute_idle_from_log
from codoxear.rollout_log import _find_latest_token_update
from codoxear.session_log_metadata import turn_context_run_settings
from codoxear.session_log_runtime import SessionLogRuntimeCoordinator
from codoxear.session_model import Session
from codoxear.util import read_jsonl_from_offset


_SECRET = b"live-transport-parity"
_NORMALIZED_LIVE_FIELDS = (
    "transcript_state",
    "thread_id",
    "log_path",
    "live_cursor",
    "events",
    "meta_delta",
    "turn_start",
    "turn_end",
    "turn_aborted",
    "turn_boundaries",
    "turn_activity",
    "busy",
    "queue_len",
    "token",
)


def _row_bytes(row: dict[str, Any]) -> bytes:
    return (json.dumps(row, separators=(",", ":")) + "\n").encode()


def _session(log_path: Path, *, backend: str) -> Session:
    return Session(
        session_id="sid",
        thread_id="thread",
        broker_pid=99999999,
        codex_pid=99999999,
        agent_backend=backend,
        owned=False,
        start_ts=1.0,
        cwd=str(log_path.parent),
        log_path=log_path,
        sock_path=log_path.with_suffix(".sock"),
    )


def _runtime(session: Session) -> SessionLogRuntimeCoordinator:
    sessions = {session.session_id: session}
    import threading

    return SessionLogRuntimeCoordinator(
        lock=threading.Lock(),
        sessions=lambda: sessions,
        analyze_log_chunk=_analyze_log_chunk,
        turn_context_run_settings=lambda payload: turn_context_run_settings(
            payload,
            clean_optional_text=lambda value: value.strip() if isinstance(value, str) and value.strip() else None,
            display_reasoning_effort=lambda value: value.strip() if isinstance(value, str) and value.strip() else None,
        ),
        compute_idle_from_log=_compute_idle_from_log,
        read_jsonl_from_offset=read_jsonl_from_offset,
        find_latest_token_update=_find_latest_token_update,
    )


class _Manager:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.runtime = _runtime(session)

    def refresh_session_meta(self, _session_id: str) -> None:
        return None

    def get_session(self, _session_id: str) -> Session:
        return self.session

    def mark_log_delta(self, *args, **kwargs) -> bool:
        return self.runtime.mark_log_delta(*args, **kwargs)

    def _attach_notification_texts(self, events):
        return [{**event, "notification_text": f"notify:{event.get('text', '')}"} for event in events]


class _PollHandler:
    def _unauthorized(self) -> None:
        raise AssertionError("unexpected unauthorized")


class _SseHandler(_PollHandler):
    def __init__(self) -> None:
        self.wfile = BytesIO()
        self.status: int | None = None
        original_flush = self.wfile.flush

        def disconnect_after_message() -> None:
            original_flush()
            raise BrokenPipeError()

        self.wfile.flush = disconnect_after_message

    def send_response(self, status: int) -> None:
        self.status = status

    def send_header(self, _name: str, _value: str) -> None:
        return None

    def end_headers(self) -> None:
        return None


def _deps(responses: list[tuple[int, dict[str, Any]]] | None = None) -> MessageRouteDeps:
    sink = [] if responses is None else responses

    def encode_cursor(*, kind: str, session, pos: int) -> str:
        return encode_message_cursor(kind=kind, session=session, pos=pos, secret=_SECRET)

    def decode_cursor(token: str, *, kind: str, session) -> int:
        return decode_message_cursor(token, kind=kind, session=session, secret=_SECRET)

    return MessageRouteDeps(
        require_auth=lambda _handler: True,
        set_auth_cookie=lambda _handler: None,
        json_response=lambda _handler, status, payload: sink.append((status, payload)),
        launch_attempt_transcript_for_session_id=lambda _sid: None,
        transcript_export_max_bytes=1024,
        transcript_search_max_line_bytes=1024,
        decode_message_cursor=decode_cursor,
        encode_message_cursor=encode_cursor,
        record_metric=lambda _name, _value: None,
        message_runtime_snapshot=lambda _sid, session, **_kwargs: ({}, False, 0, session.token),
    )


def _sse_payload(handler: _SseHandler) -> dict[str, Any]:
    blocks = handler.wfile.getvalue().decode().strip().split("\n\n")
    messages = [json.loads(block.split("data: ", 1)[1]) for block in blocks if block.startswith("event: message")]
    assert len(messages) == 1
    return messages[0]


def _normalized(payload: dict[str, Any]) -> dict[str, Any]:
    return {field: payload.get(field) for field in _NORMALIZED_LIVE_FIELDS}


def _run_both(log_path: Path, *, backend: str, after_byte: int) -> tuple[dict[str, Any], dict[str, Any]]:
    poll_session = _session(log_path, backend=backend)
    poll_responses: list[tuple[int, dict[str, Any]]] = []
    cursor = encode_message_cursor(kind="live", session=poll_session, pos=after_byte, secret=_SECRET)
    handle_messages_live(
        _PollHandler(),
        session_id="sid",
        query=f"cursor={cursor}",
        manager=_Manager(poll_session),
        deps=_deps(poll_responses),
    )
    assert len(poll_responses) == 1 and poll_responses[0][0] == 200

    sse_session = _session(log_path, backend=backend)
    sse_handler = _SseHandler()
    handle_messages_live_stream(
        sse_handler,
        session_id="sid",
        query=f"cursor={cursor}",
        manager=_Manager(sse_session),
        deps=_deps(),
    )
    assert sse_handler.status == 200
    return poll_responses[0][1], _sse_payload(sse_handler)


def _cc_assistant(model: str, text: str, *, usage: int) -> dict[str, Any]:
    return {
        "type": "assistant",
        "sessionId": "thread",
        "timestamp": "2026-08-18T00:00:01.000Z",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "model": model,
            "usage": {"input_tokens": usage},
        },
    }


def test_poll_and_sse_project_token_clear_identically(tmp_path: Path) -> None:
    path = tmp_path / "cc-token-clear.jsonl"
    known = _cc_assistant("claude-sonnet-4-5", "older", usage=512)
    clear = _cc_assistant("claude-unmapped-9", "newer", usage=12000)
    path.write_bytes(_row_bytes(known))
    after_byte = path.stat().st_size
    with path.open("ab") as stream:
        stream.write(_row_bytes(clear))

    poll, sse = _run_both(path, backend="cc", after_byte=after_byte)

    assert _normalized(poll) == _normalized(sse)
    assert poll["token"] is None
    assert poll["events"][0]["notification_text"] == "notify:newer"


def test_poll_and_sse_project_cross_window_turn_close_identically(tmp_path: Path) -> None:
    from codoxear.rollout_chat_events import _NO_RESPONSE_TEXT

    path = tmp_path / "cc-split-close.jsonl"
    user = {
        "type": "user",
        "sessionId": "thread",
        "timestamp": "2026-08-18T00:00:00.000Z",
        "cwd": str(tmp_path),
        "message": {"role": "user", "content": [{"type": "text", "text": "silent"}]},
    }
    close = {
        "type": "system",
        "subtype": "turn_duration",
        "sessionId": "thread",
        "timestamp": "2026-08-18T00:00:03.000Z",
        "durationMs": 1234,
    }
    path.write_bytes(_row_bytes(user))
    after_byte = path.stat().st_size
    with path.open("ab") as stream:
        stream.write(_row_bytes(close))

    poll, sse = _run_both(path, backend="cc", after_byte=after_byte)

    assert _normalized(poll) == _normalized(sse)
    assert poll["turn_end"] is True
    assert [(event["role"], event["text"]) for event in poll["events"]] == [("assistant", _NO_RESPONSE_TEXT)]


def test_poll_and_sse_share_bound_cursor_read_window_over_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "large-live.jsonl"
    first = {"type": "event_msg", "payload": {"type": "user_message", "message": "first"}, "ts": 1.0}
    filler = {"type": "event_msg", "payload": {"type": "agent_reasoning_delta", "delta": "x" * 4096}, "ts": 2.0}
    payload = bytearray(_row_bytes(first))
    while len(payload) <= message_routes.LIVE_POLL_READ_MAX_BYTES + 64 * 1024:
        payload.extend(_row_bytes(filler))
    path.write_bytes(payload)

    real_reader = message_routes._rollout_log._read_jsonl_records_from_offset
    real_projection = message_routes._project_live_record_window
    seen_bounds: list[int] = []
    projected_windows: list[tuple[int, int, tuple[tuple[int, int], ...]]] = []

    def require_shared_bound(log_path: Path, offset: int, *, max_bytes: int):
        seen_bounds.append(max_bytes)
        return real_reader(log_path, offset, max_bytes=max_bytes)

    def observe_shared_projection(**kwargs):
        records = kwargs["records"]
        projected_windows.append(
            (
                kwargs["after_byte"],
                kwargs["next_after"],
                tuple((record.start, record.end) for record in records),
            )
        )
        return real_projection(**kwargs)

    monkeypatch.setattr(message_routes._rollout_log, "_read_jsonl_records_from_offset", require_shared_bound)
    monkeypatch.setattr(message_routes, "_project_live_record_window", observe_shared_projection)
    poll, sse = _run_both(path, backend="codex", after_byte=0)

    assert _normalized(poll) == _normalized(sse)
    assert seen_bounds == [message_routes.LIVE_POLL_READ_MAX_BYTES] * 2
    # This execution-level seam makes the parity suite sensitive to either
    # transport restoring a private projection: both must invoke the shared
    # operation with the identical bounded record window.
    assert len(projected_windows) == 2
    assert projected_windows[0] == projected_windows[1]
    returned = decode_message_cursor(poll["live_cursor"], kind="live", session=_session(path, backend="codex"), secret=_SECRET)
    assert 0 < returned < path.stat().st_size
