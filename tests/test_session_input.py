from pathlib import Path
from typing import Any

import pytest

from codoxear.session_input import apply_confirmed_send_success
from codoxear.session_input import parse_confirmed_send_response
from codoxear.session_input import require_send_preconditions
from codoxear.session_input import SendResponseResult
from codoxear.session_model import Session


class NotReady(Exception):
    pass


class InjectionError(Exception):
    pass


class CommitUnknown(Exception):
    pass


def _session() -> Session:
    session = Session(
        session_id="s1",
        thread_id="t1",
        broker_pid=2,
        codex_pid=1,
        agent_backend="codex",
        owned=True,
        start_ts=0.0,
        cwd="/repo",
        log_path=Path("/tmp/log.jsonl"),
        sock_path=Path("/tmp/s1.sock"),
    )
    session.sync_send_supported = True
    return session


def test_require_send_preconditions_allows_valid_direct_send() -> None:
    session = _session()

    assert require_send_preconditions(
        session,
        local_queue_len=0,
        queue_item_id=None,
        allow_pending_attachment=False,
        not_ready_error=NotReady,
    ) == Path("/tmp/s1.sock")


def test_require_send_preconditions_preserves_not_ready_messages() -> None:
    cases: list[tuple[dict[str, Any], dict[str, Any], str]] = [
        ({"commit_unknown_send": {"text": "x"}}, {}, "resolve the unknown send before submitting more text"),
        ({"pending_attachment": True}, {}, "send the pending attachment explicitly before submitting other text"),
        ({}, {"local_queue_len": 1}, "send queued prompts before submitting new text"),
        ({"queue_sending_item_id": "other"}, {"queue_item_id": "item"}, "queued prompt is no longer active"),
        ({"sync_send_supported": False}, {}, "broker must be restarted before confirmed sends are available"),
    ]
    for attrs, kwargs, message in cases:
        session = _session()
        for key, value in attrs.items():
            setattr(session, key, value)
        with pytest.raises(NotReady, match=message):
            require_send_preconditions(
                session,
                local_queue_len=kwargs.get("local_queue_len", 0),
                queue_item_id=kwargs.get("queue_item_id"),
                allow_pending_attachment=False,
                not_ready_error=NotReady,
            )


def test_parse_confirmed_send_response_classifies_success_and_injection_error() -> None:
    unknown_messages: list[str] = []

    def raise_unknown(message: str) -> None:
        unknown_messages.append(message)
        raise CommitUnknown(message)

    result = parse_confirmed_send_response(
        {"queue_len": 0, "busy": False},
        raise_commit_unknown=raise_unknown,
        injection_error=InjectionError,
    )

    assert result.response == {"queue_len": 0, "busy": False}
    assert result.busy is False
    assert result.queue_len == 0
    assert unknown_messages == []

    with pytest.raises(InjectionError, match="bad"):
        parse_confirmed_send_response(
            {"queue_len": 0, "error": "bad"},
            raise_commit_unknown=raise_unknown,
            injection_error=InjectionError,
        )


def test_parse_confirmed_send_response_preserves_commit_unknown_messages() -> None:
    def parse(value: Any) -> str:
        messages: list[str] = []

        def raise_unknown(message: str) -> None:
            messages.append(message)
            raise CommitUnknown(message)

        with pytest.raises(CommitUnknown):
            parse_confirmed_send_response(value, raise_commit_unknown=raise_unknown, injection_error=InjectionError)
        return messages[-1]

    assert parse(None) == "send commit status unknown; broker response was malformed"
    assert parse({"commit_unknown": True, "queue_len": 0}) == "send commit status unknown; broker marked commit unknown"
    assert parse({"error": "empty response", "queue_len": 0}) == "send commit status unknown; broker response was empty"
    assert parse({"busy": False}) == "send commit status unknown; broker response was incomplete"
    assert parse({"queue_len": 0, "busy": "no"}) == "send commit status unknown; broker response was invalid"
    assert parse({"queue_len": True}) == "send commit status unknown; broker response was invalid"


def test_apply_confirmed_send_success_updates_runtime_boundary_state() -> None:
    session = _session()
    session.interrupted_idle = True
    result = SendResponseResult(response={"queue_len": 2}, busy=None, queue_len=2)

    apply_confirmed_send_success(
        session,
        result=result,
        pre_send_log_path=Path("/tmp/log.jsonl"),
        pre_send_log_size=123,
    )

    assert session.busy is True
    assert session.interrupted_idle is False
    assert session.queue_len == 2
    assert session.last_send_boundary_active is True
    assert session.last_send_log_path == Path("/tmp/log.jsonl")
    assert session.last_send_log_size == 123
