from __future__ import annotations

from .session_model import Session


def reset_queue_idle(session: Session) -> None:
    session.queue_idle_since = None


def queue_idle_grace_ready(session: Session, *, now_ts: float, grace_seconds: float, require_idle_grace: bool) -> bool:
    if not require_idle_grace:
        return True
    idle_since = session.queue_idle_since
    if idle_since is None:
        session.queue_idle_since = float(now_ts)
        return False
    return (float(now_ts) - idle_since) >= grace_seconds


def start_queue_promotion(session: Session, item_id: str) -> None:
    session.queue_idle_since = None
    session.queue_sending_item_id = str(item_id)


def clear_queue_promotion(session: Session, item_id: str) -> bool:
    if session.queue_sending_item_id != str(item_id):
        return False
    session.queue_sending_item_id = None
    session.queue_idle_since = None
    return True
