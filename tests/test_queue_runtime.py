from pathlib import Path

from codoxear.queue_runtime import clear_queue_promotion
from codoxear.queue_runtime import queue_idle_grace_ready
from codoxear.queue_runtime import reset_queue_idle
from codoxear.queue_runtime import start_queue_promotion
from codoxear.session_model import Session


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


def test_queue_idle_grace_sets_first_idle_timestamp_then_allows_after_grace() -> None:
    session = _session()

    assert queue_idle_grace_ready(session, now_ts=10.0, grace_seconds=5.0, require_idle_grace=True) is False
    assert session.queue_idle_since == 10.0
    assert queue_idle_grace_ready(session, now_ts=14.0, grace_seconds=5.0, require_idle_grace=True) is False
    assert session.queue_idle_since == 10.0
    assert queue_idle_grace_ready(session, now_ts=15.0, grace_seconds=5.0, require_idle_grace=True) is True


def test_queue_idle_grace_can_be_disabled() -> None:
    session = _session()

    assert queue_idle_grace_ready(session, now_ts=10.0, grace_seconds=5.0, require_idle_grace=False) is True
    assert session.queue_idle_since is None


def test_queue_promotion_start_clear_and_idle_reset() -> None:
    session = _session()
    session.queue_idle_since = 3.0

    start_queue_promotion(session, "item-a")

    assert session.queue_idle_since is None
    assert session.queue_sending_item_id == "item-a"
    assert clear_queue_promotion(session, "other") is False
    assert session.queue_sending_item_id == "item-a"
    assert clear_queue_promotion(session, "item-a") is True
    assert session.queue_sending_item_id is None
    assert session.queue_idle_since is None

    session.queue_idle_since = 7.0
    reset_queue_idle(session)
    assert session.queue_idle_since is None
