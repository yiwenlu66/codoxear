"""Performance floors for the bounded transcript-tail cache.

The fixture has a 1 GiB *logical* JSONL size but writes only its final 12 MiB,
so CI exercises the production bounded reverse-reader without allocating a
GiB.  Sparse leading bytes model old history that no steady-state tail poll
may replay.
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from unittest import mock

import codoxear.message_routes as message_routes

from codoxear.message_cursor import decode_message_cursor
from codoxear.message_cursor import encode_message_cursor
from codoxear.message_routes import MessageRouteDeps
from codoxear.message_routes import handle_messages_live
from codoxear.message_routes import handle_messages_tail
from codoxear.session_model import Session


_ONE_GIB = 1024 * 1024 * 1024
_TRAILING_BYTES = 12 * 1024 * 1024
_FIRST_TAIL_MAX_SECONDS = 0.100
_CACHED_TAIL_MAX_SECONDS = 0.005
_CURSOR_SECRET = b"perf-floor-regression-secret"


class _Handler:
    def _unauthorized(self) -> None:
        raise AssertionError("the test route must authenticate")


class _Manager:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.marked_log_deltas: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def refresh_session_meta(self, _session_id: str) -> None:
        return None

    def get_session(self, _session_id: str) -> Session:
        return self.session

    def mark_log_delta(self, *args: object, **kwargs: object) -> None:
        self.marked_log_deltas.append((args, kwargs))

    def _attach_notification_texts(self, events: list[dict[str, object]]) -> list[dict[str, object]]:
        return events


def _session(log_path: Path) -> Session:
    return Session(
        session_id="perf-session",
        thread_id="perf-thread",
        broker_pid=1,
        codex_pid=1,
        agent_backend="codex",
        owned=False,
        start_ts=0.0,
        cwd=str(log_path.parent),
        log_path=log_path,
        sock_path=log_path.parent / "perf.sock",
    )


def _route_deps(responses: list[tuple[int, dict[str, object]]]) -> MessageRouteDeps:
    def encode_cursor(*, kind: str, session: Session, pos: int) -> str:
        return encode_message_cursor(kind=kind, session=session, pos=pos, secret=_CURSOR_SECRET)

    def decode_cursor(token: str, *, kind: str, session: Session) -> int:
        return decode_message_cursor(token, kind=kind, session=session, secret=_CURSOR_SECRET)

    return MessageRouteDeps(
        require_auth=lambda _handler: True,
        set_auth_cookie=lambda _handler: None,
        json_response=lambda _handler, status, payload: responses.append((status, payload)),
        launch_attempt_transcript_for_session_id=lambda _session_id: None,
        transcript_export_max_bytes=50 * 1024 * 1024,
        transcript_search_max_line_bytes=64 * 1024,
        decode_message_cursor=decode_cursor,
        encode_message_cursor=encode_cursor,
        record_metric=lambda _name, _value: None,
        message_runtime_snapshot=lambda _session_id, _session, **_kwargs: ({}, False, 0, None),
    )


def _write_one_gib_sparse_log(path: Path) -> None:
    assistant = {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "latest"}],
            "phase": "final_answer",
        },
        "ts": 1.0,
    }
    assistant_line = json.dumps(assistant).encode("utf-8") + b"\n"
    filler_line = b'{"type":"debug","payload":"' + (b"x" * (64 * 1024)) + b'"}\n'
    trailing_without_assistant = _TRAILING_BYTES - len(assistant_line)

    with path.open("wb") as stream:
        stream.seek(_ONE_GIB - _TRAILING_BYTES)
        remaining = trailing_without_assistant
        while remaining >= len(filler_line):
            stream.write(filler_line)
            remaining -= len(filler_line)
        if remaining:
            stream.write((b"x" * (remaining - 1)) + b"\n")
        stream.write(assistant_line)
        assert stream.tell() == _ONE_GIB


def test_bounded_tail_cache_and_eof_poll_keep_one_gib_log_within_floor() -> None:
    with TemporaryDirectory() as td:
        log_path = Path(td) / "one-gib-rollout.jsonl"
        _write_one_gib_sparse_log(log_path)
        assert log_path.stat().st_size == _ONE_GIB

        session = _session(log_path)
        manager = _Manager(session)
        responses: list[tuple[int, dict[str, object]]] = []
        deps = _route_deps(responses)

        started = time.perf_counter()
        handle_messages_tail(_Handler(), session_id=session.session_id, query="limit=60", manager=manager, deps=deps)
        first_elapsed = time.perf_counter() - started
        first_status, first_body = responses.pop()

        started = time.perf_counter()
        handle_messages_tail(_Handler(), session_id=session.session_id, query="limit=60", manager=manager, deps=deps)
        cached_elapsed = time.perf_counter() - started
        second_status, second_body = responses.pop()

        assert first_status == second_status == 200
        assert first_body == second_body
        assert [event["text"] for event in first_body["events"]] == ["latest"]
        assert first_elapsed < _FIRST_TAIL_MAX_SECONDS
        assert cached_elapsed < _CACHED_TAIL_MAX_SECONDS

        live_cursor = first_body["live_cursor"]
        assert isinstance(live_cursor, str)
        with mock.patch.object(message_routes._rollout_log, "_read_jsonl_records_from_offset") as reader:
            handle_messages_live(
                _Handler(),
                session_id=session.session_id,
                query=f"cursor={live_cursor}",
                manager=manager,
                deps=deps,
            )
        eof_status, eof_body = responses.pop()

    assert eof_status == 200
    assert eof_body["events"] == []
    assert manager.marked_log_deltas == []
    reader.assert_not_called()
