from __future__ import annotations

from contextlib import nullcontext
import io
from pathlib import Path
import threading
from typing import Any

import pytest

from codoxear.control_socket import ControlSocketCallError
from codoxear.send_journal import SendJournal
from codoxear.session_control import SessionControlCoordinator
from codoxear.session_model import Session
from codoxear.session_runtime import broker_send_verdict
from codoxear.session_send import SessionSendCoordinator


class CommitUnknown(Exception):
    pass


class NotReady(Exception):
    pass


class InjectionError(Exception):
    pass


def _session(tmp_path: Path) -> Session:
    session = Session(
        session_id="s1",
        thread_id="t1",
        broker_pid=2,
        codex_pid=1,
        agent_backend="pi",
        owned=True,
        start_ts=0.0,
        cwd="/repo",
        log_path=tmp_path / "session.jsonl",
        sock_path=tmp_path / "s1.sock",
    )
    session.sync_send_supported = True
    return session


def _control(
    sessions: dict[str, Session],
    *,
    sock_call: Any,
) -> SessionControlCoordinator:
    return SessionControlCoordinator(
        lock=threading.RLock(),
        sessions=lambda: sessions,
        sock_call=sock_call,
        pid_alive=lambda _pid: True,
        unlink_quiet=lambda _path: None,
        clear_deleted_session_state=lambda _sid: None,
        broker_busy_queue=lambda response: (bool(response["busy"]), int(response["queue_len"])),
        broker_interrupted_idle=lambda response: bool(response.get("interrupted_idle", False)),
        control_socket_call_error=ControlSocketCallError,
        commit_unknown_error=CommitUnknown,
    )


def test_broker_send_verdict_parser_accepts_only_the_protocol_shape() -> None:
    valid = {"send_verdict": {"send_id": "send-1", "verdict": "undelivered", "ts": 12}}

    assert broker_send_verdict(valid) == {"send_id": "send-1", "verdict": "undelivered", "ts": 12.0}
    assert broker_send_verdict({"busy": False, "queue_len": 0}) is None
    assert broker_send_verdict({"send_verdict": None}) is None

    with pytest.raises(ValueError, match="invalid broker send verdict"):
        broker_send_verdict([])  # type: ignore[arg-type]

    malformed = [
        "send-1",
        {},
        {"send_id": "", "verdict": "undelivered", "ts": 1.0},
        {"send_id": "x" * 121, "verdict": "undelivered", "ts": 1.0},
        {"send_id": "send-1", "verdict": "delivered", "ts": 1.0},
        {"send_id": "send-1", "verdict": "undelivered", "ts": True},
        {"send_id": "send-1", "verdict": "undelivered", "ts": float("inf")},
        {"send_id": "send-1", "verdict": "undelivered", "ts": -1.0},
    ]
    for value in malformed:
        with pytest.raises(ValueError, match="invalid broker send verdict"):
            broker_send_verdict({"send_verdict": value})


def test_get_state_copies_validated_broker_verdict_onto_session(tmp_path: Path) -> None:
    session = _session(tmp_path)
    response = {
        "busy": False,
        "queue_len": 0,
        "send_verdict": {"send_id": "send-2", "verdict": "undelivered", "ts": 44.5},
    }
    coordinator = _control({"s1": session}, sock_call=lambda _sock, request, **_kwargs: response if request == {"cmd": "state"} else None)

    assert coordinator.get_state("s1") is response
    assert session.send_verdict == {"send_id": "send-2", "verdict": "undelivered", "ts": 44.5}

    response["send_verdict"]["send_id"] = "changed-after-state"
    assert session.send_verdict == {"send_id": "send-2", "verdict": "undelivered", "ts": 44.5}


def test_send_coordinator_forwards_send_id_to_broker_request(tmp_path: Path) -> None:
    session = _session(tmp_path)
    sessions = {"s1": session}
    requests: list[dict[str, Any]] = []

    def sock_call(_sock: Path, request: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        requests.append(dict(request))
        return {"queued": False, "queue_len": 0}

    control = _control(sessions, sock_call=sock_call)
    coordinator = SessionSendCoordinator(
        lock=threading.RLock(),
        sessions=lambda: sessions,
        input_lock_for_session=lambda _sid: nullcontext(),
        queue_len=lambda _sid: 0,
        send_remote_ready=lambda *_args, **_kwargs: True,
        refresh_session_meta_if_sidecar_exists=lambda _sid: None,
        log_size_or_none=lambda _path: 0,
        call_confirmed_send=lambda session_id, **kwargs: control.call_confirmed_send(session_id, **kwargs),
        staged_attachments_for_session=lambda _sid: [],
        clear_staged_attachments=lambda *_args, **_kwargs: {"ok": True},
        enqueue=lambda _sid, _text: {"queued": True},
        set_pending_attachment=lambda _sid, _value: None,
        set_commit_unknown_send=lambda _sid, _value: None,
        record_prelog_user_message=lambda _session, _text: None,
        now=lambda: 50.0,
        send_commit_timeout_seconds=2.0,
        not_ready_error=NotReady,
        commit_unknown_error=CommitUnknown,
        injection_error=InjectionError,
        timeout_errors=(TimeoutError,),
        journal=SendJournal(tmp_path / "journal", now=lambda: 50.0),
        stderr=io.StringIO(),
    )

    result = coordinator.send("s1", "recoverable text", send_id="send-3")

    assert result["queue_len"] == 0
    assert requests == [{"cmd": "send", "text": "recoverable text", "sync": True, "send_id": "send-3"}]
