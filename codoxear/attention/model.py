from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionItem:
    message_id: str
    session_id: str
    session_display_name: str
    notification_text: str
    updated_ts: float
    message_class: str
    summary_status: str
