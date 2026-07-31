from __future__ import annotations

from pathlib import Path
import threading
from typing import Any, Callable

from .session_registry import session_registry_for_manager


def seed_manager_in_memory_state(manager: Any) -> None:
    manager._session_store_for_manager().reset_in_memory_state()
    manager._queue_sweep_cursor = 0
    session_registry_for_manager(manager).input_locks = {}
    manager._include_launch_attempts = True
    manager._unattended_last_injected = {}
    manager._unattended_last_injected_scope = {}


def load_manager_persistent_state(manager: Any) -> None:
    manager._session_store_for_manager().load_persistent_state()
    manager._backfill_recent_cwds_from_logs()


def create_voice_push_coordinator(
    *,
    voice_push_factory: Callable[..., Any],
    app_dir: Path,
    stop_event: threading.Event,
    settings_path: Path,
    subscriptions_path: Path,
    delivery_ledger_path: Path,
    vapid_private_key_path: Path,
) -> Any:
    return voice_push_factory(
        app_dir=app_dir,
        stop_event=stop_event,
        settings_path=settings_path,
        subscriptions_path=subscriptions_path,
        delivery_ledger_path=delivery_ledger_path,
        vapid_private_key_path=vapid_private_key_path,
    )


def start_worker_thread(
    *,
    thread_factory: Callable[..., threading.Thread],
    target: Callable[[], None],
    name: str,
) -> threading.Thread:
    thread = thread_factory(target=target, name=name, daemon=True)
    thread.start()
    return thread


def input_lock_for_session(manager: Any, session_id: str, *, lock_factory: Callable[[], threading.RLock] = threading.RLock) -> threading.RLock:
    registry = session_registry_for_manager(manager)
    with registry.lock:
        locks = registry.input_locks
        if not isinstance(locks, dict):
            registry.input_locks = {}
            locks = registry.input_locks
        lock = locks.get(session_id)
        if lock is None:
            lock = lock_factory()
            locks[session_id] = lock
        return lock


def voice_push_scan_loop(manager: Any, *, wait_seconds: float, stderr: Any, print_exc: Callable[..., None]) -> None:
    stop_event = session_registry_for_manager(manager).stop_event
    while not stop_event.is_set():
        try:
            manager._voice_push_scan_sweep()
        except Exception as exc:
            stderr.write(f"error: voice-push scan failed: {type(exc).__name__}: {exc}\n")
            print_exc(file=stderr)
            stderr.flush()
        stop_event.wait(wait_seconds)


def unattended_loop(manager: Any, *, wait_seconds: float, stderr: Any, print_exc: Callable[..., None]) -> None:
    stop_event = session_registry_for_manager(manager).stop_event
    while not stop_event.is_set():
        try:
            manager._unattended_sweep()
        except Exception as exc:
            stderr.write(f"error: unattended sweep failed: {type(exc).__name__}: {exc}\n")
            print_exc(file=stderr)
            stderr.flush()
        stop_event.wait(wait_seconds)


def queue_loop(manager: Any, *, wait_seconds: float, stderr: Any) -> None:
    stop_event = session_registry_for_manager(manager).stop_event
    while not stop_event.is_set():
        try:
            manager._queue_sweep()
        except Exception:
            stderr.write("error: queue sweep crashed; continuing\n")
            stderr.flush()
        stop_event.wait(wait_seconds)


def start_manager_worker_threads(*, manager: Any, thread_factory: Callable[..., threading.Thread]) -> None:
    manager._unattended_thr = start_worker_thread(
        thread_factory=thread_factory,
        target=manager._unattended_loop,
        name="unattended",
    )
    manager._queue_thr = start_worker_thread(
        thread_factory=thread_factory,
        target=manager._queue_loop,
        name="queue",
    )
    manager._voice_push_scan_thr = start_worker_thread(
        thread_factory=thread_factory,
        target=manager._voice_push_scan_loop,
        name="voice-push-scan",
    )
