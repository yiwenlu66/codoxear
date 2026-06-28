from pathlib import Path
import threading
from typing import Any

import pytest

from codoxear.queue_store import QueueStore
from codoxear.session_model import Session
from codoxear.session_queue import SessionQueueCoordinator


class NotReady(Exception):
    pass


class InjectionError(Exception):
    pass


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
    tmp_path: Path,
    *,
    sessions: dict[str, Session] | None = None,
    queues: dict[str, list[dict[str, Any]]] | None = None,
    direct_unknowns: dict[str, dict[str, Any]] | None = None,
    remote_ready=lambda session_id, log_path: True,
    send=lambda session_id, text, **kwargs: {"queued": False, "queue_len": 0},
    now_values: list[float] | None = None,
) -> tuple[SessionQueueCoordinator, list[dict[str, Any]]]:
    saves: list[dict[str, Any]] = []
    session_map = {"s1": _session()} if sessions is None else sessions
    queue_map: dict[str, list[dict[str, Any]]] = {} if queues is None else queues
    unknown_map: dict[str, dict[str, Any]] = {} if direct_unknowns is None else direct_unknowns
    store = QueueStore(tmp_path / "queues.json")
    values = list(now_values or [10.0, 11.0, 12.0, 13.0])

    def now() -> float:
        return values.pop(0) if values else 99.0

    coordinator = SessionQueueCoordinator(
        lock=threading.Lock(),
        sessions=lambda: session_map,
        queues=lambda: queue_map,
        queue_store=lambda: store,
        commit_unknown_sends=lambda: unknown_map,
        save_queues=lambda: saves.append({sid: [dict(item) for item in items] for sid, items in queue_map.items()}),
        remote_ready=remote_ready,
        send=send,
        not_ready_error=NotReady,
        retryable_send_errors=(NotReady, InjectionError),
        commit_unknown_error=CommitUnknown,
        queue_idle_grace_seconds=5.0,
        now=now,
    )
    return coordinator, saves


def test_session_queue_coordinator_promotes_and_pops_sent_head(tmp_path: Path) -> None:
    sent: list[tuple[str, str, dict[str, Any]]] = []

    def send(session_id: str, text: str, **kwargs: Any) -> dict[str, Any]:
        sent.append((session_id, text, kwargs))
        return {"queued": False, "queue_len": 0}

    coordinator, saves = _coordinator(tmp_path, send=send)
    item, ql = coordinator.append_item_local("s1", "hello")

    resp = coordinator.promote_head_if_sendable("s1", require_idle_grace=False, expected_item_id=item["id"])

    assert ql == 1
    assert resp == {"queued": False, "queue_len": 0}
    assert sent == [("s1", "hello", {"queue_item_id": item["id"]})]
    assert coordinator.queues() == {}
    assert coordinator.sessions()["s1"].queue_sending_item_id is None
    assert saves[-1] == {}


def test_session_queue_coordinator_preserves_commit_unknown_marker(tmp_path: Path) -> None:
    def send(session_id: str, text: str, **kwargs: Any) -> dict[str, Any]:
        raise CommitUnknown("unknown")

    coordinator, _saves = _coordinator(tmp_path, send=send, now_values=[10.0, 20.0, 21.0])
    item, _ql = coordinator.append_item_local("s1", "hello")

    resp = coordinator.promote_head_if_sendable("s1", require_idle_grace=False, expected_item_id=item["id"])

    assert resp is not None
    assert resp["commit_unknown"] is True
    assert resp["queue_len"] == 1
    queued = coordinator.queues()["s1"][0]
    assert queued["id"] == item["id"]
    assert queued["commit_unknown"] is True
    assert queued["commit_unknown_ts"] == 21.0
    assert coordinator.sessions()["s1"].queue_sending_item_id is None


def test_session_queue_coordinator_blocks_recovery_barrier_with_retryable_error(tmp_path: Path) -> None:
    direct_unknowns = {"s1": {"text": "maybe", "created_ts": 1.0}}
    coordinator, _saves = _coordinator(tmp_path, direct_unknowns=direct_unknowns)

    with pytest.raises(NotReady, match="resolve the recovery queue"):
        coordinator.append_item_local("s1", "next", reject_recovery_barrier=True)


def test_session_queue_coordinator_remote_not_ready_resets_idle_without_sending(tmp_path: Path) -> None:
    sent: list[str] = []
    session = _session()
    session.queue_idle_since = 3.0
    coordinator, _saves = _coordinator(
        tmp_path,
        sessions={"s1": session},
        remote_ready=lambda session_id, log_path: False,
        send=lambda session_id, text, **kwargs: sent.append(text) or {"queued": False, "queue_len": 0},
    )
    item, _ql = coordinator.append_item_local("s1", "hello")

    assert coordinator.promote_head_if_sendable("s1", require_idle_grace=False, expected_item_id=item["id"]) is None
    assert sent == []
    assert session.queue_idle_since is None
    assert coordinator.queues()["s1"][0].get("commit_unknown") is not True
