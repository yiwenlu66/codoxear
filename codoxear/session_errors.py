from __future__ import annotations

from typing import Any

from .util import redacted_launch_attempt_response_record


class SessionLaunchError(RuntimeError):
    def __init__(self, record: dict[str, Any]):
        safe = redacted_launch_attempt_response_record(record)
        msg = str(safe.get("error") or safe.get("message") or "session launch failed")
        super().__init__(msg)
        self.record = safe


class SessionNotReadyError(RuntimeError):
    pass


class SessionInjectionError(RuntimeError):
    pass


class SessionCommitUnknownError(RuntimeError):
    pass
