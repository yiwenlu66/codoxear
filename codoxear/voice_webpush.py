from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from py_vapid import Vapid
from pywebpush import WebPushException
from pywebpush import webpush

from .voice_push_state import _b64u
from .voice_push_state import _chmod_private_file


@dataclass(frozen=True)
class WebPushDeliveryOutcome:
    record_id: str
    success: bool
    timestamp: float
    error: str = ""
    drop_subscription: bool = False


def ensure_vapid_public_key(private_key_path: Path) -> str:
    path = Path(private_key_path)
    if path.exists():
        _chmod_private_file(path)
        vapid = Vapid.from_file(str(path))
    else:
        vapid = Vapid()
        vapid.generate_keys()
        os.makedirs(path.parent, exist_ok=True)
        path.write_bytes(vapid.private_pem())
        _chmod_private_file(path)
    public_bytes = vapid.public_key.public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.UncompressedPoint,
    )
    return _b64u(public_bytes)


def push_payload_json(
    *,
    session_id: str,
    session_display_name: str,
    message_id: str,
    notification_text: str,
    timestamp: float | None,
) -> str:
    return json.dumps(
        {
            "session_id": session_id,
            "session_display_name": session_display_name,
            "message_id": message_id,
            "notification_text": notification_text,
            "timestamp": timestamp or time.time(),
        }
    )


def send_web_push_notifications(
    *,
    subscriptions: list[dict[str, Any]],
    private_key_path: Path,
    vapid_subject: str,
    payload_json: str,
) -> list[WebPushDeliveryOutcome]:
    vapid = Vapid.from_file(str(private_key_path))
    outcomes: list[WebPushDeliveryOutcome] = []
    for record in subscriptions:
        record_id = str(record.get("id") or "")
        try:
            response = webpush(
                subscription_info=record["subscription"],
                data=payload_json,
                vapid_private_key=vapid,
                vapid_claims={"sub": vapid_subject},
                ttl=300,
                timeout=10.0,
            )
            _ = response
            outcomes.append(WebPushDeliveryOutcome(record_id=record_id, success=True, timestamp=float(time.time())))
        except WebPushException as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            outcomes.append(
                WebPushDeliveryOutcome(
                    record_id=record_id,
                    success=False,
                    timestamp=float(time.time()),
                    error=str(e),
                    drop_subscription=status in {404, 410},
                )
            )
    return outcomes
