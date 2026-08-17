from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

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
from codoxear.session_log_projection import log_revision
from codoxear.session_log_runtime import SessionLogRuntimeCoordinator
from codoxear.session_model import Session
from codoxear.util import read_jsonl_from_offset


_SECRET = b"ordered-projection-test"


def _row_bytes(row: dict) -> bytes:
    return (json.dumps(row, separators=(",", ":")) + "\n").encode()


def _pi_model(provider: str, model: str) -> dict:
    return {"type": "model_change", "provider": provider, "modelId": model}


def _cc_usage(model: str, tokens: int) -> dict:
    return {
        "type": "assistant",
        "sessionId": "thread",
        "timestamp": "2026-08-18T00:00:00.000Z",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": model}],
            "stop_reason": "end_turn",
            "model": model,
            "usage": {"input_tokens": tokens},
        },
    }


def _session(log_path: Path, *, backend: str = "pi") -> Session:
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
    def __init__(self, session: Session, runtime: SessionLogRuntimeCoordinator) -> None:
        self.session = session
        self.runtime = runtime

    def refresh_session_meta(self, _session_id: str) -> None:
        return None

    def get_session(self, _session_id: str) -> Session:
        return self.session

    def mark_log_delta(self, *args, **kwargs) -> bool:
        return self.runtime.mark_log_delta(*args, **kwargs)

    def _attach_notification_texts(self, events):
        return events


class _BlockingOldManager(_Manager):
    def __init__(self, session: Session, runtime: SessionLogRuntimeCoordinator) -> None:
        super().__init__(session, runtime)
        self.old_waiting = threading.Event()
        self.release_old = threading.Event()

    def mark_log_delta(self, *args, **kwargs) -> bool:
        if kwargs.get("start_off") == 0:
            self.old_waiting.set()
            assert self.release_old.wait(5)
        return super().mark_log_delta(*args, **kwargs)


class _Handler:
    def _unauthorized(self) -> None:
        raise AssertionError("unexpected unauthorized")


class _SseHandler(_Handler):
    def __init__(self) -> None:
        from io import BytesIO

        self.wfile = BytesIO()
        self.status = None
        original_flush = self.wfile.flush

        def disconnect() -> None:
            original_flush()
            raise BrokenPipeError()

        self.wfile.flush = disconnect

    def send_response(self, status: int) -> None:
        self.status = status

    def send_header(self, _name: str, _value: str) -> None:
        return None

    def end_headers(self) -> None:
        return None


def _deps(responses: list[tuple[int, dict]] | None = None) -> MessageRouteDeps:
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
        message_runtime_snapshot=lambda _sid, session, **kwargs: (
            {},
            False,
            0,
            kwargs.get("token_update").public_token
            if kwargs.get("token_update") is not None and kwargs["token_update"].observed
            else session.token,
        ),
    )


def test_newer_delta_then_older_delta_keeps_newer_settings(tmp_path: Path) -> None:
    path = tmp_path / "pi.jsonl"
    old = _pi_model("old-provider", "old-model")
    new = _pi_model("new-provider", "new-model")
    old_end = len(_row_bytes(old))
    path.write_bytes(_row_bytes(old) + _row_bytes(new))
    session = _session(path)
    runtime = _runtime(session)
    revision = log_revision(path)
    assert revision is not None

    assert runtime.mark_log_delta("sid", objs=[new], start_off=old_end, new_off=revision[2], revision=revision)
    assert not runtime.mark_log_delta("sid", objs=[old], start_off=0, new_off=old_end, revision=revision)
    assert (session.model_provider, session.model) == ("new-provider", "new-model")


def test_stale_cursor_over_large_log_cannot_regress_registry(tmp_path: Path) -> None:
    path = tmp_path / "pi-large.jsonl"
    old = _pi_model("old-provider", "old-model")
    filler = {"type": "message", "message": {"role": "assistant", "content": [{"type": "text", "text": "x" * 4096}]}}
    payload = bytearray(_row_bytes(old))
    while len(payload) <= message_routes.LIVE_POLL_READ_MAX_BYTES + 64 * 1024:
        payload.extend(_row_bytes(filler))
    new = _pi_model("new-provider", "new-model")
    new_start = len(payload)
    payload.extend(_row_bytes(new))
    path.write_bytes(payload)
    session = _session(path)
    runtime = _runtime(session)
    manager = _Manager(session, runtime)
    revision = log_revision(path)
    assert revision is not None
    assert runtime.mark_log_delta("sid", objs=[new], start_off=new_start, new_off=revision[2], revision=revision)

    responses: list[tuple[int, dict]] = []
    cursor = encode_message_cursor(kind="live", session=session, pos=0, secret=_SECRET)
    handle_messages_live(_Handler(), session_id="sid", query=f"cursor={cursor}", manager=manager, deps=_deps(responses))

    assert responses[0][0] == 200
    returned = decode_message_cursor(responses[0][1]["live_cursor"], kind="live", session=session, secret=_SECRET)
    assert returned < revision[2]
    assert (session.model_provider, session.model) == ("new-provider", "new-model")


def test_poll_and_sse_completing_out_of_order_keep_newer_projection(tmp_path: Path) -> None:
    path = tmp_path / "pi-race.jsonl"
    old = _pi_model("old-provider", "old-model")
    new = _pi_model("new-provider", "new-model")
    filler = {"type": "message", "message": {"role": "assistant", "content": [{"type": "text", "text": "x" * 4096}]}}
    payload = bytearray(_row_bytes(old))
    while len(payload) <= message_routes.LIVE_POLL_READ_MAX_BYTES + 64 * 1024:
        payload.extend(_row_bytes(filler))
    new_start = len(payload)
    payload.extend(_row_bytes(new))
    path.write_bytes(payload)
    session = _session(path)
    runtime = _runtime(session)
    manager = _BlockingOldManager(session, runtime)
    old_cursor = encode_message_cursor(kind="live", session=session, pos=0, secret=_SECRET)
    new_cursor = encode_message_cursor(kind="live", session=session, pos=new_start, secret=_SECRET)
    poll_responses: list[tuple[int, dict]] = []

    poll = threading.Thread(
        target=handle_messages_live,
        kwargs={"handler": _Handler(), "session_id": "sid", "query": f"cursor={old_cursor}", "manager": manager, "deps": _deps(poll_responses)},
    )
    poll.start()
    assert manager.old_waiting.wait(5)
    handle_messages_live_stream(
        _SseHandler(),
        session_id="sid",
        query=f"cursor={new_cursor}",
        manager=manager,
        deps=_deps(),
    )
    manager.release_old.set()
    poll.join(5)

    assert not poll.is_alive()
    assert poll_responses[0][0] == 200
    assert (session.model_provider, session.model) == ("new-provider", "new-model")


def test_old_token_update_and_clear_cannot_replace_newer_token(tmp_path: Path) -> None:
    path = tmp_path / "cc.jsonl"
    old_update = _cc_usage("claude-sonnet-4-5", 100)
    old_clear = _cc_usage("claude-unmapped-9", 200)
    new_update = _cc_usage("claude-sonnet-4-5", 300)
    old_update_end = len(_row_bytes(old_update))
    old_clear_end = old_update_end + len(_row_bytes(old_clear))
    path.write_bytes(_row_bytes(old_update) + _row_bytes(old_clear) + _row_bytes(new_update))
    session = _session(path, backend="cc")
    runtime = _runtime(session)
    revision = log_revision(path)
    assert revision is not None

    assert runtime.mark_log_delta("sid", objs=[new_update], start_off=old_clear_end, new_off=revision[2], revision=revision)
    newer_token = dict(session.token or {})
    assert not runtime.mark_log_delta("sid", objs=[old_update], start_off=0, new_off=old_update_end, revision=revision)
    assert session.token == newer_token
    assert not runtime.mark_log_delta("sid", objs=[old_clear], start_off=old_update_end, new_off=old_clear_end, revision=revision)
    assert session.token == newer_token


def test_append_between_stat_and_read_commits_consumed_boundary(tmp_path: Path) -> None:
    path = tmp_path / "pi-append-race.jsonl"
    old = _pi_model("old-provider", "old-model")
    path.write_bytes(_row_bytes(old))
    session = _session(path)
    runtime = _runtime(session)
    captured_revision = log_revision(path)
    assert captured_revision is not None

    new = _pi_model("new-provider", "new-model")
    with path.open("ab") as stream:
        stream.write(_row_bytes(new))
    objs, consumed_end = read_jsonl_from_offset(path, 0, max_bytes=1024 * 1024)
    grown_revision = log_revision(path)
    assert grown_revision is not None
    assert consumed_end > captured_revision[2]

    observation = runtime.observation_from_rows(
        agent_backend="pi",
        log_path=path,
        revision=captured_revision,
        start_off=0,
        end_off=consumed_end,
        objs=objs,
    )
    assert runtime.commit_log_observation("sid", observation)
    assert session.log_projection_end == consumed_end
    assert session.log_projection_revision == grown_revision
    assert (session.model_provider, session.model) == ("new-provider", "new-model")


def test_same_path_truncation_accepts_offset_reset_and_rejects_old_revision(tmp_path: Path) -> None:
    path = tmp_path / "pi.jsonl"
    old = _pi_model("old-provider", "old-model")
    padding = {"type": "message", "message": {"role": "assistant", "content": [{"type": "text", "text": "padding" * 100}]}}
    path.write_bytes(_row_bytes(old) + _row_bytes(padding))
    session = _session(path)
    runtime = _runtime(session)
    old_revision = log_revision(path)
    assert old_revision is not None
    assert runtime.mark_log_delta("sid", objs=[old], start_off=0, new_off=old_revision[2], revision=old_revision)

    new = _pi_model("new-provider", "new-model")
    path.write_bytes(_row_bytes(new))
    new_revision = log_revision(path)
    assert new_revision is not None and new_revision[2] < old_revision[2]
    assert runtime.mark_log_delta("sid", objs=[new], start_off=0, new_off=new_revision[2], revision=new_revision)
    assert not runtime.mark_log_delta("sid", objs=[old], start_off=0, new_off=old_revision[2], revision=old_revision)
    assert (session.model_provider, session.model) == ("new-provider", "new-model")


def test_different_log_rebind_accepts_offset_reset_and_rejects_prior_binding(tmp_path: Path) -> None:
    old_path = tmp_path / "old.jsonl"
    new_path = tmp_path / "new.jsonl"
    old = _pi_model("old-provider", "old-model")
    new = _pi_model("new-provider", "new-model")
    old_path.write_bytes(_row_bytes(old) * 10)
    new_path.write_bytes(_row_bytes(new))
    session = _session(old_path)
    runtime = _runtime(session)
    old_revision = log_revision(old_path)
    new_revision = log_revision(new_path)
    assert old_revision is not None and new_revision is not None
    assert runtime.mark_log_delta("sid", objs=[old], start_off=0, new_off=old_revision[2], revision=old_revision)

    session.log_path = new_path
    assert runtime.mark_log_delta("sid", objs=[new], start_off=0, new_off=new_revision[2], revision=new_revision)
    assert not runtime.mark_log_delta(
        "sid",
        objs=[old],
        start_off=0,
        new_off=old_revision[2],
        expected_log_path=old_path,
        revision=old_revision,
    )
    assert (session.model_provider, session.model) == ("new-provider", "new-model")
