from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .session_model import Session


@dataclass(frozen=True)
class SendResponseResult:
    response: dict[str, Any]
    busy: bool | None
    queue_len: int


def require_send_preconditions(
    session: Session,
    *,
    local_queue_len: int,
    queue_item_id: str | None,
    allow_pending_attachment: bool,
    not_ready_error: type[BaseException],
) -> Path:
    if session.commit_unknown_send:
        raise not_ready_error("resolve the unknown send before submitting more text")
    if session.pending_attachment and not allow_pending_attachment:
        raise not_ready_error("send the pending attachment explicitly before submitting other text")
    if queue_item_id is None and (local_queue_len > 0 or session.queue_sending_item_id is not None):
        raise not_ready_error("send queued prompts before submitting new text")
    if queue_item_id is not None and session.queue_sending_item_id != queue_item_id:
        raise not_ready_error("queued prompt is no longer active")
    if not session.sync_send_supported:
        raise not_ready_error("broker must be restarted before confirmed sends are available")
    return session.sock_path


def parse_confirmed_send_response(
    response: Any,
    *,
    raise_commit_unknown: Callable[[str], None],
    injection_error: type[BaseException],
) -> SendResponseResult:
    if not isinstance(response, dict):
        raise_commit_unknown("send commit status unknown; broker response was malformed")
    if bool(response.get("commit_unknown")):
        raise_commit_unknown("send commit status unknown; broker marked commit unknown")
    if response.get("error"):
        err = str(response.get("error"))
        if err == "empty response":
            raise_commit_unknown("send commit status unknown; broker response was empty")
        if bool(response.get("commit_unknown")):
            raise_commit_unknown(f"send commit status unknown; {err}")
        raise injection_error(err)
    if "queue_len" not in response:
        raise_commit_unknown("send commit status unknown; broker response was incomplete")
    busy_resp: bool | None = None
    if "busy" in response:
        busy_raw = response.get("busy")
        if not isinstance(busy_raw, bool):
            raise_commit_unknown("send commit status unknown; broker response was invalid")
        busy_resp = busy_raw
    queue_len_raw = response.get("queue_len")
    if isinstance(queue_len_raw, bool) or not isinstance(queue_len_raw, int) or queue_len_raw < 0:
        raise_commit_unknown("send commit status unknown; broker response was invalid")
    return SendResponseResult(response=response, busy=busy_resp, queue_len=int(queue_len_raw))


def apply_confirmed_send_success(
    session: Session,
    *,
    result: SendResponseResult,
    pre_send_log_path: Path | None,
    pre_send_log_size: int | None,
) -> None:
    session.busy = result.busy if result.busy is not None else True
    session.interrupted_idle = False
    session.queue_len = result.queue_len
    session.last_send_boundary_active = True
    session.last_send_log_path = pre_send_log_path
    session.last_send_log_size = pre_send_log_size
