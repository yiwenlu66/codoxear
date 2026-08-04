from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import urlencode

from codoxear.message_cursor import decode_message_cursor
from codoxear.message_cursor import encode_message_cursor
from codoxear.message_routes import MessageRouteDeps
from codoxear.message_routes import handle_messages_window
from codoxear.session_model import Session


_SECRET = b"messages-window-test-secret"


class _Handler:
    def __init__(self) -> None:
        self.unauthorized = False

    def _unauthorized(self) -> None:
        self.unauthorized = True


class _Manager:
    def __init__(self, session: Session) -> None:
        self.session = session

    def refresh_session_meta(self, _session_id: str) -> None:
        return None

    def get_session(self, _session_id: str) -> Session:
        return self.session

    def _attach_notification_texts(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return events


def _session(tmpdir: str, log_path: Path) -> Session:
    return Session(
        session_id="s1",
        thread_id="thread-1",
        broker_pid=1,
        codex_pid=1,
        agent_backend="codex",
        owned=False,
        start_ts=0.0,
        cwd=tmpdir,
        log_path=log_path,
        sock_path=Path(tmpdir) / "s1.sock",
    )


def _deps(responses: list[tuple[int, dict[str, Any]]]) -> MessageRouteDeps:
    def encode_cursor(*, kind: str, session: Session, pos: int) -> str:
        return encode_message_cursor(kind=kind, session=session, pos=pos, secret=_SECRET)

    def decode_cursor(token: str, *, kind: str, session: Session) -> int:
        return decode_message_cursor(token, kind=kind, session=session, secret=_SECRET)

    return MessageRouteDeps(
        require_auth=lambda _handler: True,
        set_auth_cookie=lambda _handler: None,
        json_response=lambda _handler, status, payload: responses.append((status, payload)),
        launch_attempt_transcript_for_session_id=lambda _session_id: None,
        transcript_export_max_bytes=1024 * 1024,
        transcript_search_max_line_bytes=1024 * 1024,
        decode_message_cursor=decode_cursor,
        encode_message_cursor=encode_cursor,
        record_metric=lambda _name, _value: None,
        message_runtime_snapshot=lambda _session_id, _session: ({}, False, 0, None),
    )


def _write_user_events(log_path: Path, messages: list[str]) -> list[int]:
    rows = [
        json.dumps({"type": "event_msg", "payload": {"type": "user_message", "message": message}}) + "\n"
        for message in messages
    ]
    offsets: list[int] = []
    offset = 0
    for row in rows:
        offsets.append(offset)
        offset += len(row.encode("utf-8"))
    log_path.write_text("".join(rows), encoding="utf-8")
    return offsets


def _window(session: Session, query: str) -> tuple[int, dict[str, Any]]:
    responses: list[tuple[int, dict[str, Any]]] = []
    handle_messages_window(
        _Handler(),
        session_id=session.session_id,
        query=query,
        manager=_Manager(session),
        deps=_deps(responses),
    )
    assert len(responses) == 1
    return responses[0]


def _history_cursor(session: Session, position: int) -> str:
    return encode_message_cursor(kind="history", session=session, pos=position, secret=_SECRET)


def test_messages_window_byte_boundary_starts_at_the_next_record_without_duplication() -> None:
    with TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "session.jsonl"
        offsets = _write_user_events(log_path, ["first", "second", "third"])
        session = _session(tmpdir, log_path)

        status, body = _window(
            session,
            urlencode({"cursor": _history_cursor(session, offsets[1]), "before": 0, "after": 0}),
        )

    assert status == 200
    assert [event["text"] for event in body["events"]] == ["second"]


def test_messages_window_cursor_after_final_event_returns_no_events() -> None:
    with TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "session.jsonl"
        _write_user_events(log_path, ["only event"])
        session = _session(tmpdir, log_path)

        status, body = _window(
            session,
            urlencode({"cursor": _history_cursor(session, log_path.stat().st_size), "before": 0, "after": 30}),
        )

    assert status == 200
    assert body["events"] == []
    assert body["has_newer"] is False


def test_messages_window_last_event_boundary_returns_empty_forward_window() -> None:
    with TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "session.jsonl"
        offsets = _write_user_events(log_path, ["first", "last"])
        session = _session(tmpdir, log_path)
        last_record_end = log_path.stat().st_size
        assert last_record_end > offsets[-1]

        status, body = _window(
            session,
            urlencode({"cursor": _history_cursor(session, last_record_end), "before": 0, "after": 0}),
        )

    assert status == 200
    assert body["events"] == []


def test_messages_window_rejects_invalid_cursor_with_bad_request() -> None:
    with TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "session.jsonl"
        _write_user_events(log_path, ["event"])
        session = _session(tmpdir, log_path)

        status, body = _window(session, "cursor=not-a-signed-cursor")

    assert status == 400
    assert body == {"error": "cursor_invalid"}


def test_messages_window_empty_log_returns_no_events() -> None:
    with TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "session.jsonl"
        log_path.write_text("", encoding="utf-8")
        session = _session(tmpdir, log_path)

        status, body = _window(
            session,
            urlencode({"cursor": _history_cursor(session, 0), "before": 0, "after": 30}),
        )

    assert status == 200
    assert body["events"] == []
    assert body["has_older"] is False
    assert body["has_newer"] is False
