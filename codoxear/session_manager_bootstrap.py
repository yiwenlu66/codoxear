from __future__ import annotations

from pathlib import Path
import threading
from typing import Any, Callable


def seed_manager_in_memory_state(manager: Any) -> None:
    manager._unattended = {}
    manager._aliases = {}
    manager._sidebar_meta = {}
    manager._hidden_sessions = set()
    manager._files = {}
    manager._queues = {}
    manager._pending_attachment_ids = set()
    manager._commit_unknown_sends = {}
    manager._input_locks = {}
    manager._recent_cwds = {}
    manager._include_launch_attempts = True
    manager._unattended_last_injected = {}
    manager._unattended_last_injected_scope = {}


def load_manager_persistent_state(manager: Any) -> None:
    manager._load_unattended()
    manager._load_aliases()
    manager._load_sidebar_meta()
    manager._load_hidden_sessions()
    manager._load_files()
    manager._load_queues()
    manager._load_pending_attachments()
    manager._load_commit_unknown_sends()
    manager._load_recent_cwds()
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
