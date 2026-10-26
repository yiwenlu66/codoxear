"""Attachment-resource persistence and upload identity validation.

The envelope is the sole durable authority for list, pending flag, revision and
upload outcomes. Callers serialize mutations and persistence under the registry
lock; never persist a detached list snapshot.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
from uuid import uuid4
from typing import Any


def upload_id_value(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9._:-]{1,120}", value) is None:
        raise ValueError("invalid upload_id")
    return value


class AttachmentPersistenceUncertain(OSError):
    """Replacement is visible but directory durability could not be confirmed."""


def durable_write_resource(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            os.chmod(temporary, 0o600)
            json.dump(payload, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError as exc:
            # Never report this as a pre-publication failure: rollback/unlink
            # would leave the published envelope referencing missing bytes.
            raise AttachmentPersistenceUncertain("attachment durability unconfirmed") from exc
    finally:
        temporary.unlink(missing_ok=True)
