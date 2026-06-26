from pathlib import Path

import pytest

from codoxear.session_model import Session
from codoxear.session_runtime import apply_history_backfill
from codoxear.session_runtime import apply_run_settings_backfill
from codoxear.session_runtime import broker_allows_interrupted_idle_override
from codoxear.session_runtime import broker_busy_queue
from codoxear.session_runtime import broker_interrupted_idle
from codoxear.session_runtime import broker_runtime_state
from codoxear.session_runtime import resolve_runtime_status
from codoxear.session_runtime import select_runtime_token


def _session(log_path: Path | None = Path("/tmp/log.jsonl")) -> Session:
    return Session(
        session_id="s1",
        thread_id="t1",
        broker_pid=2,
        codex_pid=1,
        agent_backend="codex",
        owned=True,
        start_ts=10.0,
        cwd="/repo",
        log_path=log_path,
        sock_path=Path("/tmp/sock"),
    )


def _status(state: dict, *, log_exists: bool, log_idle=None, boundary: bool = False):
    return resolve_runtime_status(
        broker=broker_runtime_state(state),
        log_exists=log_exists,
        log_idle=log_idle,
        send_boundary_unresolved=boundary,
    )


def test_apply_history_backfill_marks_scanned_and_updates_timestamp() -> None:
    session = _session()

    update = apply_history_backfill(session, expected_log_path=Path("/tmp/log.jsonl"), conversation_ts=20.0)

    assert update is not None
    assert update.updated_ts == 20.0
    assert session.last_chat_history_scanned is True
    assert session.last_chat_ts == 20.0


def test_apply_history_backfill_ignores_stale_binding_or_already_scanned_session() -> None:
    session = _session()
    assert apply_history_backfill(session, expected_log_path=Path("/tmp/other.jsonl"), conversation_ts=20.0) is None
    assert session.last_chat_history_scanned is False
    session.last_chat_history_scanned = True
    assert apply_history_backfill(session, expected_log_path=Path("/tmp/log.jsonl"), conversation_ts=20.0) is None


def test_apply_run_settings_backfill_fills_only_missing_fields() -> None:
    session = _session()
    session.model = "existing-model"
    session.preferred_auth_method = "api-key"

    update = apply_run_settings_backfill(
        session,
        expected_log_path=Path("/tmp/log.jsonl"),
        log_provider="provider",
        log_model="log-model",
        log_effort="high",
    )

    assert update is not None
    assert update.model_provider == "provider"
    assert update.preferred_auth_method == "api-key"
    assert update.model == "existing-model"
    assert update.reasoning_effort == "high"
    assert session.model_provider == "provider"
    assert session.model == "existing-model"
    assert session.reasoning_effort == "high"


def test_apply_run_settings_backfill_ignores_stale_log_binding() -> None:
    session = _session()

    assert apply_run_settings_backfill(
        session,
        expected_log_path=Path("/tmp/other.jsonl"),
        log_provider="provider",
        log_model="model",
        log_effort="high",
    ) is None
    assert session.model_provider is None
    assert session.model is None
    assert session.reasoning_effort is None


def test_runtime_status_uses_log_idle_over_stale_broker_busy() -> None:
    status = _status({"busy": True, "queue_len": 0}, log_exists=True, log_idle=True)
    assert status.busy is False
    assert status.remote_ready is True


def test_runtime_status_uses_busy_log_over_idle_broker() -> None:
    status = _status({"busy": False, "queue_len": 0}, log_exists=True, log_idle=False)
    assert status.busy is True
    assert status.remote_ready is False


def test_runtime_status_interrupted_idle_requires_idle_broker_and_empty_broker_queue() -> None:
    ready = _status({"busy": False, "queue_len": 0, "interrupted_idle": True}, log_exists=True, log_idle=False)
    assert ready.busy is False
    assert ready.remote_ready is True

    broker_busy = _status({"busy": True, "queue_len": 0, "interrupted_idle": True}, log_exists=True, log_idle=False)
    assert broker_busy.busy is True
    assert broker_busy.remote_ready is False

    queued = _status({"busy": False, "queue_len": 1, "interrupted_idle": True}, log_exists=True, log_idle=False)
    assert queued.busy is True
    assert queued.remote_ready is False


def test_runtime_status_send_boundary_dominates_without_log_idle() -> None:
    status = _status({"busy": False, "queue_len": 0, "interrupted_idle": True}, log_exists=True, log_idle=None, boundary=True)
    assert status.busy is True
    assert status.remote_ready is False


def test_runtime_status_no_log_uses_broker_for_remote_readiness_only() -> None:
    idle_broker = _status({"busy": False, "queue_len": 0}, log_exists=False)
    assert idle_broker.busy is False
    assert idle_broker.remote_ready is True

    busy_broker = _status({"busy": True, "queue_len": 0}, log_exists=False)
    assert busy_broker.busy is False
    assert busy_broker.remote_ready is False


def test_runtime_status_requires_log_idle_when_bound_and_not_boundary_protected() -> None:
    with pytest.raises(ValueError, match="log_idle"):
        _status({"busy": False, "queue_len": 0}, log_exists=True, log_idle=None)


def test_broker_state_validation_and_accessors() -> None:
    assert broker_busy_queue({"busy": False, "queue_len": 2, "interrupted_idle": True}) == (False, 2)
    assert broker_interrupted_idle({"busy": False, "queue_len": 0, "interrupted_idle": True}) is True
    assert broker_allows_interrupted_idle_override({"busy": False, "queue_len": 0, "interrupted_idle": True}) is True
    assert broker_allows_interrupted_idle_override({"busy": True, "queue_len": 0, "interrupted_idle": True}) is False
    for state in (
        {"busy": "false", "queue_len": 0},
        {"busy": False, "queue_len": True},
        {"busy": False, "queue_len": -1},
        {"busy": False, "queue_len": 0, "interrupted_idle": "true"},
    ):
        with pytest.raises(ValueError, match="invalid broker state response"):
            broker_runtime_state(state)


def test_runtime_token_prefers_live_log_state_over_stale_broker_token() -> None:
    assert select_runtime_token(
        broker_state={"busy": False, "queue_len": 0, "token": {"tokens_in_context": 0}},
        session_token={"tokens_in_context": 10},
        token_update=None,
        log_available=True,
    ) == {"tokens_in_context": 10}
    assert select_runtime_token(
        broker_state={"busy": False, "queue_len": 0, "token": {"tokens_in_context": 0}},
        session_token=None,
        token_update={"tokens_in_context": 20},
        log_available=True,
    ) == {"tokens_in_context": 20}
    assert select_runtime_token(
        broker_state={"busy": False, "queue_len": 0, "token": {"tokens_in_context": 0}},
        session_token=None,
        token_update=None,
        log_available=False,
    ) == {"tokens_in_context": 0}
    with pytest.raises(ValueError, match="invalid token"):
        select_runtime_token(
            broker_state={"busy": False, "queue_len": 0, "token": 0},
            session_token=None,
            token_update=None,
            log_available=False,
        )
