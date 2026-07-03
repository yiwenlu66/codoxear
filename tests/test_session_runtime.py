from pathlib import Path
import threading
from typing import Any

import pytest

from codoxear.session_model import Session
from codoxear.session_runtime import apply_history_backfill
from codoxear.session_runtime import apply_run_settings_backfill
from codoxear.session_runtime import consume_session_confirmed_send_boundary
from codoxear.session_runtime import broker_allows_interrupted_idle_override
from codoxear.session_runtime import broker_busy_queue
from codoxear.session_runtime import broker_interrupted_idle
from codoxear.session_runtime import ListingRuntimeProbes
from codoxear.session_runtime import broker_runtime_state
from codoxear.session_runtime import build_runtime_enriched_session_rows
from codoxear.session_runtime import log_path_size_or_none
from codoxear.session_runtime import session_allows_direct_send
from codoxear.session_runtime import session_allows_queue_promotion
from codoxear.session_runtime import session_runtime_readiness
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


def test_session_runtime_readiness_projects_send_queue_and_unattended_decisions() -> None:
    ready = session_runtime_readiness(_status({"busy": False, "queue_len": 0}, log_exists=True, log_idle=True), local_queue_len=0)
    assert ready.direct_send is True
    assert ready.queue_promotion is True
    assert ready.unattended_injection is True

    local_queued = session_runtime_readiness(_status({"busy": False, "queue_len": 0}, log_exists=True, log_idle=True), local_queue_len=1)
    assert local_queued.direct_send is True
    assert local_queued.queue_promotion is True
    assert local_queued.unattended_injection is False

    precondition_blocked = session_runtime_readiness(
        _status({"busy": False, "queue_len": 0}, log_exists=True, log_idle=True),
        direct_send_precondition=False,
        queue_promotion_precondition=False,
    )
    assert precondition_blocked.direct_send is False
    assert precondition_blocked.queue_promotion is False
    assert precondition_blocked.unattended_injection is True

    runtime_blocked = session_runtime_readiness(_status({"busy": True, "queue_len": 0}, log_exists=True, log_idle=False), local_queue_len=0)
    assert runtime_blocked.direct_send is False
    assert runtime_blocked.queue_promotion is False
    assert runtime_blocked.unattended_injection is False


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


class _RecentCwdStore:
    def __init__(self, *, dirty: bool = True) -> None:
        self.dirty = dirty
        self.calls: list[tuple[str, float]] = []

    def note_recent_cwd(self, cwd: str, updated_ts: float) -> bool:
        self.calls.append((cwd, updated_ts))
        return self.dirty


def _staged_listing_row(log_path: Path) -> dict[str, Any]:
    return {
        "session_id": "s1",
        "agent_backend": "codex",
        "updated_ts": 10.0,
        "start_ts": 10.0,
        "priority_offset": 0.0,
        "blocked": False,
        "snoozed": False,
        "log_exists": True,
        "needs_history_scan": True,
        "needs_run_settings": True,
        "state_busy": True,
        "broker_queue_len": 0,
        "interrupted_idle": False,
        "_log_path_obj": log_path,
        "_cwd_path_obj": Path("/repo"),
        "model_provider": None,
        "preferred_auth_method": None,
        "model": None,
        "reasoning_effort": None,
        "provider_choice": "default",
    }


def test_build_runtime_enriched_session_rows_applies_backfills_and_public_projection() -> None:
    log_path = Path("/tmp/log.jsonl")
    session = _session(log_path)
    store = _RecentCwdStore()

    result = build_runtime_enriched_session_rows(
        staged_rows=[_staged_listing_row(log_path)],
        sessions={"s1": session},
        lock=threading.Lock(),
        store=store,  # type: ignore[arg-type]
        probes=ListingRuntimeProbes(
            last_conversation_ts_from_tail=lambda path: 25.0,
            read_run_settings_from_log=lambda path, agent_backend: ("provider", "log-model", "high"),
            log_size_or_none=lambda path: 100,
            send_boundary_unresolved=lambda sid, path, size: False,
            idle_from_log_path=lambda sid, path: True,
            current_git_branch=lambda path: "main",
        ),
        now_ts=30.0,
        provider_choice_for_settings=lambda model_provider, preferred_auth_method: f"choice:{model_provider}:{preferred_auth_method}",
        priority_half_life_seconds=60.0,
        priority_bucket_seconds=1.0,
    )

    assert result.recent_cwd_dirty is True
    assert store.calls == [("/repo", 25.0)]
    assert session.last_chat_history_scanned is True
    assert session.last_chat_ts == 25.0
    assert session.model_provider == "provider"
    assert session.model == "log-model"
    assert session.reasoning_effort == "high"
    assert len(result.rows) == 1
    row = result.rows[0]
    assert row["updated_ts"] == 25.0
    assert row["busy"] is False
    assert row["git_branch"] == "main"
    assert row["model_provider"] == "provider"
    assert row["provider_choice"] == "choice:provider:None"
    assert "_log_path_obj" not in row
    assert "_cwd_path_obj" not in row


def test_build_runtime_enriched_session_rows_keeps_busy_when_send_boundary_unresolved() -> None:
    log_path = Path("/tmp/log.jsonl")
    session = _session(log_path)
    idle_calls = 0

    def idle_from_log_path(sid: str, path: Path) -> bool:
        nonlocal idle_calls
        idle_calls += 1
        return True

    result = build_runtime_enriched_session_rows(
        staged_rows=[{**_staged_listing_row(log_path), "needs_history_scan": False, "needs_run_settings": False, "state_busy": False}],
        sessions={"s1": session},
        lock=threading.Lock(),
        store=_RecentCwdStore(),  # type: ignore[arg-type]
        probes=ListingRuntimeProbes(
            last_conversation_ts_from_tail=lambda path: None,
            read_run_settings_from_log=lambda path, agent_backend: (None, None, None),
            log_size_or_none=lambda path: 100,
            send_boundary_unresolved=lambda sid, path, size: True,
            idle_from_log_path=idle_from_log_path,
            current_git_branch=lambda path: None,
        ),
        now_ts=30.0,
        provider_choice_for_settings=lambda model_provider, preferred_auth_method: "default",
        priority_half_life_seconds=60.0,
        priority_bucket_seconds=1.0,
    )

    assert result.rows[0]["busy"] is True
    assert idle_calls == 0


def test_log_path_size_or_none_uses_last_parseable_json_object(tmp_path: Path) -> None:
    log_path = tmp_path / "session.jsonl"
    first = b'{"ok": 1}\n'
    second = b'{"ok": 2}\n'
    log_path.write_bytes(first + second + b'{"partial"')

    assert log_path_size_or_none(log_path) == len(first + second)
    assert log_path_size_or_none(None) is None
    assert log_path_size_or_none(tmp_path / "missing.jsonl") is None


def test_consume_session_confirmed_send_boundary_clears_after_log_advances() -> None:
    log_path = Path("/tmp/log.jsonl")
    session = _session(log_path)
    session.last_send_boundary_active = True
    session.last_send_log_path = log_path
    session.last_send_log_size = 10

    assert consume_session_confirmed_send_boundary(session, log_path, 10) is True
    assert session.last_send_boundary_active is True

    assert consume_session_confirmed_send_boundary(session, log_path, 11) is False
    assert session.last_send_boundary_active is False
    assert session.last_send_log_path is None
    assert session.last_send_log_size is None


def test_session_runtime_readiness_preconditions_block_unknown_and_pending_attachment() -> None:
    session = _session()
    assert session_allows_direct_send(session, allow_pending_attachment=False) is True
    assert session_allows_queue_promotion(session) is True

    session.pending_attachment = True
    assert session_allows_direct_send(session, allow_pending_attachment=False) is False
    assert session_allows_direct_send(session, allow_pending_attachment=True) is True
    assert session_allows_queue_promotion(session) is False

    session.commit_unknown_send = {"text": "unknown"}
    assert session_allows_direct_send(session, allow_pending_attachment=True) is False
    assert session_allows_queue_promotion(session) is False
