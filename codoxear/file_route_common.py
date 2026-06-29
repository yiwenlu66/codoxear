from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .path_runtime import expanduser_path as _expanduser_path


@dataclass(frozen=True)
class FileRouteResponse:
    status: int
    payload: dict[str, Any]


class FileRouteError(Exception):
    def __init__(self, status: int, payload: dict[str, Any]) -> None:
        super().__init__(str(payload.get("error", "file route error")))
        self.status = int(status)
        self.payload = payload


@dataclass(frozen=True)
class SessionFileWriteRequest:
    path: str
    text: str
    create: bool
    git_path: bool
    version: str | None


def body_flag(obj: Mapping[str, Any], name: str) -> bool:
    raw = obj.get(name)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return False


def resolve_session_write_update_path(base: Path, raw_path: str) -> Path:
    if not isinstance(raw_path, str) or raw_path == "":
        raise ValueError("path required")
    if "\x00" in raw_path:
        raise ValueError("invalid path")
    p = Path(raw_path)
    if p.is_absolute():
        return _expanduser_path(p).resolve()
    resolved_base = _expanduser_path(base)
    if not resolved_base.is_absolute():
        resolved_base = resolved_base.resolve()
    resolved_base = resolved_base.resolve()
    resolved = (resolved_base / p).resolve()
    try:
        resolved.relative_to(resolved_base)
    except ValueError as e:
        raise ValueError("path escapes session cwd") from e
    return resolved
