from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, MutableMapping

from .session_model import Session
from .session_runtime import RuntimeStatus
from .session_runtime import broker_runtime_state
from .session_runtime import resolve_runtime_status
from .session_runtime import session_allows_direct_send
from .session_runtime import session_allows_queue_promotion


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

    def remote_ready_from_state_and_log(self, session_id: str, state: dict[str, Any], log_path: Path | None) -> bool:
        broker = broker_runtime_state(state)
        if broker.queue_len > 0:
            return False
        log_exists = isinstance(log_path, Path) and log_path.exists()
        log_size = self.log_size_or_none(log_path)
        boundary_unresolved = self.confirmed_send_boundary_unresolved_for_session(session_id, log_path, log_size)
        log_idle = bool(self.idle_from_log(session_id)) if log_exists and not boundary_unresolved else None
        runtime = resolve_runtime_status(
            broker=broker,
            log_exists=log_exists,
            log_idle=log_idle,
            send_boundary_unresolved=boundary_unresolved,
        )
        return runtime.remote_ready

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
        return self.remote_ready_from_state_and_log(session_id, state, log_path)

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
        return self.remote_ready_from_state_and_log(session_id, state, refreshed_log_path)

    def attachment_injection_ready(self, session_id: str) -> bool:
        self.refresh_session_meta_if_sidecar_exists(session_id, drain_queue=False)
        with self.lock:
            session = self.sessions().get(session_id)
            if not session:
                raise KeyError("unknown session")
            if session.commit_unknown_send:
                raise self.not_ready_error("resolve the unknown send before attaching a file")
            if not (session.sync_send_supported and session.key_write_errors_supported):
                raise self.not_ready_error("broker must be restarted before file attachments are available")
            if session.pending_attachment:
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
            if current.pending_attachment:
                return False
            if current.queue_sending_item_id is not None:
                return False
            if self.queue_len(session_id) > 0:
                return False
        return self.remote_ready_from_state_and_log(session_id, state, log_path)
