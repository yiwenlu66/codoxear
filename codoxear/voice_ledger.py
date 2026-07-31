from __future__ import annotations

from typing import Any

from .voice_push_state import AnnouncementTask
from .voice_push_state import DELIVERY_LEDGER_MAX
from .voice_push_state import _clip_text


def mark_task_replaced(
    ledger: dict[str, dict[str, Any]],
    task: AnnouncementTask,
    *,
    now_ts: float,
) -> bool:
    dirty = False
    for message_id in dict.fromkeys(task.source_message_ids):
        row = ledger.get(message_id)
        if not isinstance(row, dict):
            continue
        row["last_error"] = "replaced by newer message"
        row["updated_ts"] = now_ts
        row["narrated_status"] = "skipped"
        if row.get("summary_status") == "pending":
            row["summary_status"] = "skipped"
        if row.get("push_status") == "pending":
            row["push_status"] = "skipped"
        ledger[message_id] = row
        dirty = True
    return dirty


def set_task_error(
    ledger: dict[str, dict[str, Any]],
    task: AnnouncementTask,
    error: str,
    *,
    now_ts: float,
) -> bool:
    clipped_error = _clip_text(error, limit=400)
    dirty = False
    for message_id in dict.fromkeys(task.source_message_ids):
        row = ledger.get(message_id)
        if not isinstance(row, dict):
            continue
        row["last_error"] = clipped_error
        row["updated_ts"] = now_ts
        row["narrated_status"] = "error"
        if row.get("summary_status") == "pending":
            row["summary_status"] = "error"
        if row.get("push_status") == "pending":
            row["push_status"] = "error"
        ledger[message_id] = row
        dirty = True
    return dirty


def mark_tasks_skipped_no_listener(
    ledger: dict[str, dict[str, Any]],
    tasks: list[AnnouncementTask],
    *,
    now_ts: float,
) -> bool:
    dirty = False
    for task in tasks:
        for message_id in dict.fromkeys(task.source_message_ids):
            row = ledger.get(message_id)
            if not isinstance(row, dict):
                continue
            row["narrated_status"] = "skipped"
            row["last_error"] = "no active listener"
            row["updated_ts"] = now_ts
            if row.get("summary_status") == "pending":
                row["summary_status"] = "skipped"
            ledger[message_id] = row
            dirty = True
    return dirty


def set_ledger_fields_many(
    ledger: dict[str, dict[str, Any]],
    message_ids: tuple[str, ...],
    patch: dict[str, Any],
    *,
    now_ts: float,
) -> bool:
    dirty = False
    for message_id in dict.fromkeys(message_ids):
        row = ledger.get(message_id)
        if not isinstance(row, dict):
            continue
        row.update(patch)
        row["updated_ts"] = now_ts
        ledger[message_id] = row
        dirty = True
    return dirty


def trim_delivery_ledger(
    ledger: dict[str, dict[str, Any]],
    *,
    limit: int = DELIVERY_LEDGER_MAX,
) -> None:
    if len(ledger) <= limit:
        return
    doomed = sorted(
        ledger.values(),
        key=lambda row: float(row.get("updated_ts") or row.get("created_ts") or 0.0),
    )[: len(ledger) - limit]
    for row in doomed:
        message_id = row.get("message_id")
        if isinstance(message_id, str):
            ledger.pop(message_id, None)
