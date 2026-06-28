from __future__ import annotations

from pathlib import Path
from typing import Any

from .util import atomic_write_json
from .util import load_json_file
from .voice_push_state import _chmod_private_file
from .voice_push_state import _clean_ledger
from .voice_push_state import _clean_subscription_record
from .voice_push_state import _clean_voice_settings


def load_voice_settings(path: Path) -> dict[str, Any]:
    _chmod_private_file(path)
    raw = load_json_file(path, default={})
    return _clean_voice_settings(raw)


def save_voice_settings(path: Path, settings: dict[str, Any]) -> None:
    atomic_write_json(path, dict(settings))
    _chmod_private_file(path)


def load_subscription_records(path: Path) -> dict[str, dict[str, Any]]:
    raw = load_json_file(path, default=[])
    cleaned: dict[str, dict[str, Any]] = {}
    if isinstance(raw, list):
        for item in raw:
            record = _clean_subscription_record(item)
            if record is not None:
                cleaned[record["id"]] = record
    return cleaned


def save_subscription_records(path: Path, subscriptions: dict[str, dict[str, Any]]) -> None:
    atomic_write_json(path, list(subscriptions.values()))


def load_voice_delivery_ledger(path: Path) -> dict[str, dict[str, Any]]:
    raw = load_json_file(path, default={})
    return _clean_ledger(raw)


def save_voice_delivery_ledger(path: Path, ledger: dict[str, dict[str, Any]]) -> None:
    atomic_write_json(path, dict(ledger))
