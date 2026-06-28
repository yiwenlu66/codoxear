from pathlib import Path
import threading
from typing import Any

import pytest

from codoxear.control_socket import ControlSocketCallError
from codoxear.session_control import SessionControlCoordinator
from codoxear.session_model import Session


class CommitUnknown(Exception):
    pass


def _session() -> Session:
    return Session(
        session_id="s1",
        thread_id="t1",
        broker_pid=2,
        codex_pid=1,
        agent_backend="codex",
        owned=True,
        start_ts=0.0,
        cwd="/repo",
        log_path=None,
        sock_path=Path("/tmp/s1.sock"),
    )


def _coordinator(
    *,
    sessions: dict[str, Session] | None = None,
    sock_call=lambda sock, req, **kwargs: {"busy": False, "queue_len": 0},
    alive=lambda pid: True,
) -> tuple[SessionControlCoordinator, list[Path], list[str]]:
    session_map = {"s1": _session()} if sessions is None else sessions
    unlinked: list[Path] = []
    cleared: list[str] = []
    coordinator = SessionControlCoordinator(
        lock=threading.Lock(),
        sessions=lambda: session_map,
        sock_call=sock_call,
        pid_alive=alive,
        unlink_quiet=unlinked.append,
        clear_deleted_session_state=cleared.append,
        broker_busy_queue=lambda response: (bool(response["busy"]), int(response["queue_len"])),
        broker_interrupted_idle=lambda response: bool(response.get("interrupted_idle", False)),
        control_socket_call_error=ControlSocketCallError,
        commit_unknown_error=CommitUnknown,
    )
    return coordinator, unlinked, cleared


def test_session_control_get_state_updates_runtime_cache() -> None:
    session = _session()
    coordinator, _unlinked, _cleared = _coordinator(
        sessions={"s1": session},
        sock_call=lambda sock, req, **kwargs: {"busy": True, "queue_len": 2, "interrupted_idle": True, "token": {"n": 1}},
    )

    resp = coordinator.get_state("s1")

    assert resp == {"busy": True, "queue_len": 2, "interrupted_idle": True, "token": {"n": 1}}
    assert session.busy is True
    assert session.queue_len == 2
    assert session.interrupted_idle is True
    assert session.token == {"n": 1}


def test_session_control_get_tail_validates_tail_shape() -> None:
    coordinator, _unlinked, _cleared = _coordinator(sock_call=lambda sock, req, **kwargs: {"tail": "hello"})

    assert coordinator.get_tail("s1") == "hello"

    bad, _unlinked, _cleared = _coordinator(sock_call=lambda sock, req, **kwargs: {"tail": 3})
    with pytest.raises(ValueError, match="invalid broker tail response"):
        bad.get_tail("s1")


def test_session_control_inject_keys_tracks_commit_unknown_after_request_sent() -> None:
    def sock_call(sock: Path, req: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        raise ControlSocketCallError("reset", request_sent=True)

    coordinator, _unlinked, _cleared = _coordinator(sock_call=sock_call)

    with pytest.raises(CommitUnknown, match="attachment commit status unknown; broker response failed"):
        coordinator.inject_keys("s1", "abc", track_request_sent=True)


def test_session_control_drops_dead_session_and_clears_state_for_get_state_only() -> None:
    def sock_call(sock: Path, req: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("dead")

    sessions = {"s1": _session()}
    coordinator, unlinked, cleared = _coordinator(sessions=sessions, sock_call=sock_call, alive=lambda pid: False)

    with pytest.raises(KeyError, match="unknown session"):
        coordinator.get_state("s1")

    assert sessions == {}
    assert unlinked == [Path("/tmp/s1.sock"), Path("/tmp/s1.json")]
    assert cleared == ["s1"]


def test_session_control_drops_dead_session_without_deleted_state_for_tail_and_keys() -> None:
    def sock_call(sock: Path, req: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("dead")

    sessions = {"s1": _session()}
    coordinator, unlinked, cleared = _coordinator(sessions=sessions, sock_call=sock_call, alive=lambda pid: False)

    with pytest.raises(KeyError, match="unknown session"):
        coordinator.get_tail("s1")

    assert sessions == {}
    assert unlinked == [Path("/tmp/s1.sock"), Path("/tmp/s1.json")]
    assert cleared == []

    sessions = {"s1": _session()}
    coordinator, unlinked, cleared = _coordinator(sessions=sessions, sock_call=sock_call, alive=lambda pid: False)
    with pytest.raises(KeyError, match="unknown session"):
        coordinator.inject_keys("s1", "abc")
    assert sessions == {}
    assert unlinked == [Path("/tmp/s1.sock"), Path("/tmp/s1.json")]
    assert cleared == []
