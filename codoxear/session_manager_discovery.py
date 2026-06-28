from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .session_registry import session_registry_for_manager


def discover_existing_if_stale_for_manager(
    manager: Any,
    *,
    force: bool,
    discover_min_interval_seconds: float,
    now: Callable[[], float],
) -> None:
    registry = session_registry_for_manager(manager)
    now_ts = now()
    with registry.lock:
        last = float(registry.last_discover_ts)
    if (not force) and ((now_ts - last) < discover_min_interval_seconds):
        return
    try:
        manager._discover_existing(force=force)
    except TypeError:
        manager._discover_existing()



def discover_existing_for_manager(
    manager: Any,
    *,
    force: bool,
    discover_min_interval_seconds: float,
    sock_dir: Path,
    proc_root: Path,
    discover_sessions: Callable[..., Any],
    now: Callable[[], float],
) -> None:
    registry = session_registry_for_manager(manager)
    if not force:
        now_ts = now()
        with registry.lock:
            last = float(registry.last_discover_ts)
        if (now_ts - last) < discover_min_interval_seconds:
            return
    with registry.lock:
        hidden_sessions = set(getattr(manager, "_hidden_sessions", set()))
    result = discover_sessions(
        sock_dir,
        proc_root=proc_root,
        hidden_sessions=hidden_sessions,
        deps=manager._discovery_deps(),
    )
    manager._apply_discovery_result(result)
    with registry.lock:
        registry.last_discover_ts = now()
