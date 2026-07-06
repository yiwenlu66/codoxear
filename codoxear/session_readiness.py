from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, MutableMapping

from .session_model import Session
from .session_runtime import RuntimeStatus
from .session_runtime import broker_runtime_state_with_session_idle_authority
from .session_runtime import resolve_runtime_status
from .session_runtime import session_allows_direct_send
from .session_runtime import session_allows_queue_promotion
from .session_runtime import session_runtime_readiness


@dataclass(frozen=True)
class SessionReadinessCoordinator:
    lock: Any
    sessions: Callable[[], MutableMapping[str, Session]]
    refresh_session_meta_if_sidecar_exists: Callable[..., None]
    get_state: Callable[[str], dict[str, Any]]
    log_size_or_none: Callable[[Path | None], int | None]
    confirmed_send_boundary_unresolved_for_session: Callable[[str, Path | None, int | None], bool]
    idle_from_log: Callable[[str], bool]
    queue_len: Callable[[str], int]
    not_ready_error: type[BaseException]

    def runtime_status_from_state_and_log(self, session_id: str, state: dict[str, Any], log_path: Path | None) -> RuntimeStatus:
        # The interrupted-idle override is session-authoritative, not
        # broker-authoritative: the listing / log watcher may have suppressed a
        # stale broker ``interrupted_idle:true`` (stored Session.interrupted_idle
        # is then False). Raw broker busy / queue_len are still validated and
        # used as-sent. Building the BrokerRuntimeState from raw state here would
        # let the stale value reactivate send / queue / attachment / unattended
        # readiness while the sidebar projects busy (DEFECT a48ca8e).
        with self.lock:
            session = self.sessions().get(session_id)
            session_interrupted_idle = bool(session.interrupted_idle) if session is not None else False
        broker = broker_runtime_state_with_session_idle_authority(
            state, session_interrupted_idle=session_interrupted_idle
        )
        log_exists = isinstance(log_path, Path) and log_path.exists()
        if broker.queue_len > 0:
            return resolve_runtime_status(
                broker=broker,
                log_exists=log_exists,
                log_idle=None,
                send_boundary_unresolved=False,
            )
        log_size = self.log_size_or_none(log_path)
        boundary_unresolved = self.confirmed_send_boundary_unresolved_for_session(session_id, log_path, log_size)
        log_idle = bool(self.idle_from_log(session_id)) if log_exists and not boundary_unresolved else None
        return resolve_runtime_status(
            broker=broker,
            log_exists=log_exists,
            log_idle=log_idle,
            send_boundary_unresolved=boundary_unresolved,
        )

    def remote_ready_from_state_and_log(self, session_id: str, state: dict[str, Any], log_path: Path | None) -> bool:
        runtime = self.runtime_status_from_state_and_log(session_id, state, log_path)
        return session_runtime_readiness(runtime).direct_send

    def remote_state_after_metadata_probe(self, session_id: str, *, log_path_before_state: Path | None) -> tuple[dict[str, Any], Path | None]:
        state = self.get_state(session_id)
        self.refresh_session_meta_if_sidecar_exists(session_id, drain_queue=False)
        with self.lock:
            session = self.sessions().get(session_id)
            if not session:
                raise KeyError("unknown session")
            log_path = session.log_path
        if log_path != log_path_before_state:
            state = self.get_state(session_id)
        return state, log_path

    def send_remote_ready(self, session_id: str, *, allow_pending_attachment: bool = False) -> bool:
        self.refresh_session_meta_if_sidecar_exists(session_id, drain_queue=False)
        with self.lock:
            session = self.sessions().get(session_id)
            if not session:
                raise KeyError("unknown session")
            if not session_allows_direct_send(session, allow_pending_attachment=allow_pending_attachment):
                return False
            log_path_before_state = session.log_path
        state, log_path = self.remote_state_after_metadata_probe(session_id, log_path_before_state=log_path_before_state)
        runtime = self.runtime_status_from_state_and_log(session_id, state, log_path)
        return session_runtime_readiness(
            runtime,
            direct_send_precondition=session_allows_direct_send(session, allow_pending_attachment=allow_pending_attachment),
        ).direct_send

    def queue_remote_ready(self, session_id: str, *, log_path: Path | None) -> bool:
        self.refresh_session_meta_if_sidecar_exists(session_id, drain_queue=False)
        with self.lock:
            session = self.sessions().get(session_id)
            if not session:
                raise KeyError("unknown session")
            if not session_allows_queue_promotion(session):
                return False
            log_path_before_state = session.log_path
        state, refreshed_log_path = self.remote_state_after_metadata_probe(session_id, log_path_before_state=log_path_before_state)
        with self.lock:
            current = self.sessions().get(session_id)
            if not current:
                raise KeyError("unknown session")
            if not session_allows_queue_promotion(current):
                return False
        runtime = self.runtime_status_from_state_and_log(session_id, state, refreshed_log_path)
        return session_runtime_readiness(
            runtime,
            queue_promotion_precondition=session_allows_queue_promotion(current),
        ).queue_promotion

    def _attachment_ready(self, session_id: str, *, allow_existing_pending: bool, require_key_write_errors: bool) -> bool:
        self.refresh_session_meta_if_sidecar_exists(session_id, drain_queue=False)
        with self.lock:
            session = self.sessions().get(session_id)
            if not session:
                raise KeyError("unknown session")
            if session.commit_unknown_send:
                raise self.not_ready_error("resolve the unknown send before attaching a file")
            if (not session.sync_send_supported) or (require_key_write_errors and not session.key_write_errors_supported):
                raise self.not_ready_error("broker must be restarted before file attachments are available")
            if session.pending_attachment and not allow_existing_pending:
                return False
            if session.queue_sending_item_id is not None:
                return False
            if self.queue_len(session_id) > 0:
                return False
            log_path_before_state = session.log_path
        state, log_path = self.remote_state_after_metadata_probe(session_id, log_path_before_state=log_path_before_state)
        with self.lock:
            current = self.sessions().get(session_id)
            if not current:
                raise KeyError("unknown session")
            if current.commit_unknown_send:
                raise self.not_ready_error("resolve the unknown send before attaching a file")
            if current.pending_attachment and not allow_existing_pending:
                return False
            if current.queue_sending_item_id is not None:
                return False
            local_queue_len = self.queue_len(session_id)
            if local_queue_len > 0:
                return False
        runtime = self.runtime_status_from_state_and_log(session_id, state, log_path)
        return session_runtime_readiness(runtime, local_queue_len=local_queue_len).direct_send

    def attachment_injection_ready(self, session_id: str) -> bool:
        return self._attachment_ready(session_id, allow_existing_pending=False, require_key_write_errors=True)

    def attachment_staging_ready(self, session_id: str) -> bool:
        return self._attachment_ready(session_id, allow_existing_pending=True, require_key_write_errors=False)
