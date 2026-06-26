from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Session:
    session_id: str
    thread_id: str
    broker_pid: int
    codex_pid: int
    agent_backend: str
    owned: bool
    start_ts: float
    cwd: str
    log_path: Path | None
    sock_path: Path
    busy: bool = False
    queue_len: int = 0
    token: dict[str, Any] | None = None
    last_turn_id: str | None = None
    last_chat_ts: float | None = None
    last_chat_history_scanned: bool = False
    meta_thinking: int = 0
    meta_tools: int = 0
    meta_system: int = 0
    meta_log_off: int = 0
    delivery_log_off: int = 0
    idle_cache_log_off: int = -1
    idle_cache_value: bool | None = None
    queue_idle_since: float | None = None
    queue_sending_item_id: str | None = None
    model_provider: str | None = None
    preferred_auth_method: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    service_tier: str | None = None
    transport: str | None = None
    tmux_session: str | None = None
    tmux_window: str | None = None
    launch_id: str | None = None
    spawn_nonce: str | None = None
    resume_session_id: str | None = None
    pending_attachment: bool = False
    commit_unknown_send: dict[str, Any] | None = None
    sync_send_supported: bool = False
    key_write_errors_supported: bool = False
    interrupted_idle: bool = False
    last_send_boundary_active: bool = False
    last_send_log_path: Path | None = None
    last_send_log_size: int | None = None
