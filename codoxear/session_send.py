from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, MutableMapping

from .file_upload import attachment_inject_text

from .session_input import apply_confirmed_send_success
from .session_input import parse_confirmed_send_response
from .session_input import require_send_preconditions
from .session_model import Session


_POST_CONFIRMATION_TAIL_ERRORS = (ValueError, OSError, KeyError)


def _tail_error_message(exc: BaseException) -> str:
    if len(getattr(exc, "args", ())) == 1 and isinstance(exc.args[0], str):
        return exc.args[0]
    message = str(exc)
    return message or exc.__class__.__name__


def _add_send_warning(response: dict[str, Any], field: str, message: str) -> dict[str, Any]:
    updated = dict(response)
    existing = updated.get(field)
    if existing:
        updated[field] = f"{existing}; {message}"
    else:
        updated[field] = message
    return updated


@dataclass(frozen=True)
class SessionSendCoordinator:
    lock: Any
    sessions: Callable[[], MutableMapping[str, Session]]
    input_lock_for_session: Callable[[str], Any]
    queue_len: Callable[[str], int]
    send_remote_ready: Callable[..., bool]
    log_size_or_none: Callable[[Path | None], int | None]
    call_confirmed_send: Callable[..., dict[str, Any]]
    staged_attachments_for_session: Callable[[str], list[dict[str, Any]]]
    clear_staged_attachments: Callable[..., dict[str, Any]]
    set_pending_attachment: Callable[[str, bool], None]
    set_commit_unknown_send: Callable[[str, dict[str, Any] | None], None]
    record_prelog_user_message: Callable[[Session, str], None]
    now: Callable[[], float]
    send_commit_timeout_seconds: float
    not_ready_error: type[BaseException]
    commit_unknown_error: type[BaseException]
    injection_error: type[BaseException]
    timeout_errors: tuple[type[BaseException], ...]

    def send(self, session_id: str, text: str, *, allow_pending_attachment: bool = False, queue_item_id: str | None = None) -> dict[str, Any]:
        input_lock = self.input_lock_for_session(session_id)
        with input_lock:
            with self.lock:
                session = self.sessions().get(session_id)
                if not session:
                    raise KeyError("unknown session")
                local_queue_len = self.queue_len(session_id)
                sock = require_send_preconditions(
                    session,
                    local_queue_len=local_queue_len,
                    queue_item_id=queue_item_id,
                    allow_pending_attachment=allow_pending_attachment,
                    not_ready_error=self.not_ready_error,
                )
            if not self.send_remote_ready(session_id, allow_pending_attachment=allow_pending_attachment):
                raise self.not_ready_error("session is busy; wait before sending")
            with self.lock:
                current = self.sessions().get(session_id)
                pre_send_log_path = current.log_path if current is not None else None
                pre_send_log_size = self.log_size_or_none(pre_send_log_path)
                staged_entries = self.staged_attachments_for_session(session_id) if allow_pending_attachment else []
            attachment_prefix = "".join(
                attachment_inject_text(idx, Path(str(entry.get("path") or "")))
                for idx, entry in enumerate(staged_entries, start=1)
            )
            committed_text = f"{attachment_prefix}{text}" if attachment_prefix else text

            def raise_commit_unknown(message: str, cause: BaseException | None = None) -> None:
                if queue_item_id is None:
                    self.set_commit_unknown_send(
                        session_id,
                        {"text": committed_text, "display_text": text, "created_ts": self.now(), "error": message},
                    )
                if cause is None:
                    raise self.commit_unknown_error(message)
                raise self.commit_unknown_error(message) from cause

            timeout_s = self.send_commit_timeout_seconds if self.send_commit_timeout_seconds > 0 else None
            response = self.call_confirmed_send(
                session_id,
                session=session,
                sock=sock,
                text=committed_text,
                timeout_s=timeout_s,
                raise_commit_unknown=raise_commit_unknown,
                not_ready_error=self.not_ready_error,
                timeout_errors=self.timeout_errors,
            )
            parsed_send = parse_confirmed_send_response(
                response,
                raise_commit_unknown=raise_commit_unknown,
                injection_error=self.injection_error,
            )
            with self.lock:
                try:
                    self.record_prelog_user_message(session, committed_text)
                except _POST_CONFIRMATION_TAIL_ERRORS as exc:
                    response = _add_send_warning(
                        response,
                        "send_state_cleanup_error",
                        f"prelog_user_message: {_tail_error_message(exc)}",
                    )
                current = self.sessions().get(session_id)
                if current:
                    apply_confirmed_send_success(
                        current,
                        result=parsed_send,
                        pre_send_log_path=pre_send_log_path,
                        pre_send_log_size=pre_send_log_size,
                    )
            if staged_entries:
                try:
                    # The confirmed message now owns these paths; session deletion reclaims their files.
                    self.clear_staged_attachments(session_id, delete_files=False)
                except _POST_CONFIRMATION_TAIL_ERRORS as exc:
                    response = _add_send_warning(
                        response,
                        "attachment_cleanup_error",
                        _tail_error_message(exc),
                    )
            else:
                try:
                    self.set_pending_attachment(session_id, False)
                except _POST_CONFIRMATION_TAIL_ERRORS as exc:
                    response = _add_send_warning(
                        response,
                        "send_state_cleanup_error",
                        f"pending_attachment: {_tail_error_message(exc)}",
                    )
            if queue_item_id is None:
                try:
                    self.set_commit_unknown_send(session_id, None)
                except _POST_CONFIRMATION_TAIL_ERRORS as exc:
                    response = _add_send_warning(
                        response,
                        "send_state_cleanup_error",
                        f"commit_unknown_send: {_tail_error_message(exc)}",
                    )
        return response


@dataclass(frozen=True)
class PrelogUserMessageRecorder:
    latest_launch_attempt: Callable[[str], dict[str, Any] | None]
    submitted_user_messages: Callable[[dict[str, Any] | None], list[dict[str, Any]]]
    clean_optional_text: Callable[[Any], str | None]
    record_launch_attempt: Callable[[dict[str, Any]], None]
    now: Callable[[], float]

    def record(self, session: Session, text: str, *, source: str) -> None:
        if not session.owned or session.log_path is not None or not session.launch_id:
            return
        previous = self.latest_launch_attempt(session.launch_id)
        messages = self.submitted_user_messages(previous)
        messages.append({"text": text, "ts": self.now(), "source": source})
        if len(messages) > 20:
            messages = messages[-20:]
        base: dict[str, Any] = dict(previous) if isinstance(previous, dict) else {}
        base.update(
            {
                "launch_id": session.launch_id,
                "state": self.clean_optional_text(base.get("state")) or "broker_meta_bound",
                "agent_backend": session.agent_backend,
                "cwd": session.cwd,
                "created_ts": base.get("created_ts", session.start_ts),
                "updated_ts": self.now(),
                "broker_pid": session.broker_pid,
                "agent_pid": session.codex_pid,
                "transport": session.transport,
                "tmux_session": session.tmux_session,
                "tmux_window": session.tmux_window,
                "spawn_nonce": session.spawn_nonce,
                "model_provider": session.model_provider,
                "preferred_auth_method": session.preferred_auth_method,
                "model": session.model,
                "reasoning_effort": session.reasoning_effort,
                "service_tier": session.service_tier,
                "submitted_user_messages": messages,
            }
        )
        self.record_launch_attempt(base)
