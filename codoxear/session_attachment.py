from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class SessionAttachmentCoordinator:
    input_lock_for_session: Callable[[str], Any]
    attachment_injection_ready: Callable[[str], bool]
    inject_keys: Callable[..., dict[str, Any]]
    set_pending_attachment: Callable[[str, bool], None]
    not_ready_error: type[BaseException]
    injection_error: type[BaseException]
    commit_unknown_error: type[BaseException]

    def inject_attachment_keys(self, session_id: str, seq: str) -> dict[str, Any]:
        input_lock = self.input_lock_for_session(session_id)
        with input_lock:
            if not self.attachment_injection_ready(session_id):
                raise self.not_ready_error("session is busy; wait before attaching a file")
            try:
                response = self.inject_keys(session_id, seq, track_request_sent=True)
            except self.commit_unknown_error:
                self.set_pending_attachment(session_id, True)
                raise
            if not isinstance(response, dict):
                self.set_pending_attachment(session_id, True)
                raise self.commit_unknown_error("attachment commit status unknown; broker response was malformed")
            if bool(response.get("commit_unknown")):
                self.set_pending_attachment(session_id, True)
                raise self.commit_unknown_error("attachment commit status unknown; broker marked commit unknown")
            if response.get("error"):
                err = str(response.get("error"))
                if bool(response.get("commit_unknown")) or err == "empty response":
                    self.set_pending_attachment(session_id, True)
                    raise self.commit_unknown_error(f"attachment commit status unknown; {err}")
                raise self.injection_error(err)
            if response.get("ok") is not True:
                self.set_pending_attachment(session_id, True)
                raise self.commit_unknown_error("attachment commit status unknown; broker response was incomplete")
            self.set_pending_attachment(session_id, True)
            return response
