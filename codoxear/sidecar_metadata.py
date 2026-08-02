from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

from .util import pid_alive as _pid_alive


def ignored_rollout_paths(meta: dict[str, Any], *, sock: Path) -> set[Path]:
    raw = meta.get("ignored_rollout_paths")
    if raw is None:
        return set()
    if not isinstance(raw, list):
        raise ValueError(f"invalid ignored_rollout_paths in metadata for socket {sock}")
    out: set[Path] = set()
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"invalid ignored_rollout_paths entry in metadata for socket {sock}")
        out.add(Path(item))
    return out


def read_metadata(meta_path: Path, *, sock: Path) -> dict[str, Any]:
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"invalid metadata json for socket {sock}: {type(e).__name__}: {e}") from e
    if not isinstance(meta, dict):
        raise ValueError(f"invalid metadata json for socket {sock}")
    return meta


def required_int(meta: dict[str, Any], key: str, *, sock: Path) -> int:
    value = meta.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"invalid {key} in metadata for socket {sock}")
    return int(value)


def required_live_pid(meta: dict[str, Any], key: str, *, sock: Path) -> int:
    pid = required_int(meta, key, sock=sock)
    if pid <= 0 or not _pid_alive(pid):
        raise ValueError(f"invalid {key} in metadata for socket {sock}")
    return pid


def required_text(meta: dict[str, Any], key: str, *, sock: Path) -> str:
    value = meta.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"invalid {key} in metadata for socket {sock}")
    return value


def log_path(meta: dict[str, Any], *, sock: Path) -> Path | None:
    if "log_path" not in meta:
        raise ValueError(f"missing log_path in metadata for socket {sock}")
    if meta.get("log_path") is None:
        return None
    raw = meta.get("log_path")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"invalid log_path in metadata for socket {sock}")
    path = Path(raw)
    if path.exists() and not path.is_file():
        raise ValueError(f"invalid log_path in metadata for socket {sock}")
    return path


def start_ts(meta: dict[str, Any], *, sock: Path) -> float:
    value = meta.get("start_ts")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"invalid start_ts in metadata for socket {sock}")
    try:
        timestamp = float(value)
    except (OverflowError, ValueError) as e:
        raise ValueError(f"invalid start_ts in metadata for socket {sock}") from e
    if not math.isfinite(timestamp):
        raise ValueError(f"invalid start_ts in metadata for socket {sock}")
    return timestamp


def log_invalid(context: str, sock: Path, error: Exception) -> None:
    sys.stderr.write(f"error: {context}: invalid sidecar metadata for {sock}: {error}\n")
    sys.stderr.flush()


def sync_send_supported(meta: dict[str, Any]) -> bool:
    caps = meta.get("control_capabilities")
    return meta.get("control_protocol_version") == 2 and isinstance(caps, dict) and caps.get("sync_send") is True


def key_write_errors_supported(meta: dict[str, Any]) -> bool:
    caps = meta.get("control_capabilities")
    return meta.get("control_protocol_version") == 2 and isinstance(caps, dict) and caps.get("key_write_errors") is True


def pi_thinking_command(meta: dict[str, Any]) -> bool:
    """Only an explicit true sidecar capability enables Pi /thinking."""


def _clean_optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    out = value.strip()
    return out or None


def detaches_current_log(meta: dict[str, Any], current_log_path: Path | None) -> bool:
    if current_log_path is None:
        return False
    return _clean_optional_text(meta.get("session_id")) is None and meta.get("log_path") is None
