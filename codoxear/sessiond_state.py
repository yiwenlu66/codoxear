from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cc_log import cc_assistant_is_final_turn_end as _cc_assistant_is_final_turn_end
from .cc_log import cc_assistant_text as _cc_assistant_text
from .cc_log import cc_user_text as _cc_user_text
from .pi_log import pi_assistant_error_text as _pi_assistant_error_text
from .pi_log import pi_assistant_is_aborted_turn as _pi_assistant_is_aborted_turn
from .pi_log import pi_assistant_is_final_turn_end as _pi_assistant_is_final_turn_end
from .pi_log import pi_assistant_text as _pi_assistant_text
from .pi_log import pi_user_text as _pi_user_text
from .util import read_jsonl_from_offset as _read_jsonl_from_offset_impl


@dataclass
class State:
    session_id: str | None
    codex_pid: int
    log_path: Path
    sock_path: Path
    pty_master_fd: int
    start_ts: float
    busy: bool = False
    output_tail: str = ""
    output_tail_max: int = 64 * 1024
    log_off: int = 0


def _read_jsonl_from_offset(path: Path, offset: int, max_bytes: int = 256 * 1024) -> tuple[list[dict[str, Any]], int]:
    if not path.exists():
        return [], offset
    return _read_jsonl_from_offset_impl(path, offset, max_bytes=max_bytes)


def _log_busy_signals(obj: dict[str, Any]) -> tuple[bool, bool]:
    if obj.get("type") == "event_msg":
        p = obj.get("payload")
        if not isinstance(p, dict):
            raise ValueError("invalid rollout event_msg payload")
        pt = p.get("type")
        if pt == "user_message":
            return True, False
        if pt in {"turn_aborted", "thread_rolled_back", "task_complete", "turn_complete"}:
            return False, True
        if pt == "token_count":
            info = p.get("info")
            if isinstance(info, dict) and isinstance(info.get("total_token_usage"), dict):
                return False, True
        return False, False
    if obj.get("type") == "message":
        if _pi_user_text(obj):
            return True, False
        if _pi_assistant_is_aborted_turn(obj):
            return False, True
        if _pi_assistant_error_text(obj):
            return False, True
        if _pi_assistant_text(obj) and _pi_assistant_is_final_turn_end(obj):
            return False, True
    if obj.get("type") == "user" and _cc_user_text(obj):
        return True, False
    if obj.get("type") == "assistant" and _cc_assistant_text(obj) and _cc_assistant_is_final_turn_end(obj):
        return False, True
    return False, False


def _busy_value_after_log_batch(objs: list[dict[str, Any]]) -> bool | None:
    next_busy: bool | None = None
    for obj in objs:
        user_signal, turn_end_signal = _log_busy_signals(obj)
        if user_signal:
            next_busy = True
        if turn_end_signal:
            next_busy = False
    return next_busy
