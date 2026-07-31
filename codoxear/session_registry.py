from __future__ import annotations

import threading
from contextlib import AbstractContextManager
from typing import Any


class SessionRegistry:
    def __init__(self, *, lock: Any | None = None, stop_event: threading.Event | None = None) -> None:
        self.lock = threading.Lock() if lock is None else lock
        self.sessions: dict[str, Any] = {}
        self.stop_event = threading.Event() if stop_event is None else stop_event
        self.last_discover_ts = 0.0
        self.input_locks: dict[str, threading.RLock] = {}
        self.store: Any | None = None

    def acquire(self) -> AbstractContextManager[Any]:
        return self.lock

    def get(self, session_id: str) -> Any | None:
        return self.sessions.get(session_id)

    def set(self, session_id: str, session: Any) -> None:
        self.sessions[session_id] = session


def session_registry_for_manager(manager: Any) -> SessionRegistry:
    registry = getattr(manager, "_registry", None)
    if isinstance(registry, SessionRegistry):
        return registry
    registry = SessionRegistry()
    object.__setattr__(manager, "_registry", registry)
    return registry


def registry_backed_attr(registry_attr: str) -> property:
    def getter(manager: Any) -> Any:
        return getattr(session_registry_for_manager(manager), registry_attr)

    def setter(manager: Any, value: Any) -> None:
        setattr(session_registry_for_manager(manager), registry_attr, value)

    return property(getter, setter)
