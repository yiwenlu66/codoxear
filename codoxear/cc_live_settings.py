from __future__ import annotations

"""Live Claude Code effort observation from its active configuration file."""

import json
from pathlib import Path
from typing import Any

from .cc_log import CC_SUPPORTED_REASONING_EFFORTS


def cc_settings_mtime_ns(settings_path: Path) -> int | None:
    """Return the current settings-file revision, or ``None`` when unavailable."""
    try:
        return int(settings_path.stat().st_mtime_ns)
    except OSError:
        return None


def read_cc_settings_reasoning_effort(settings_path: Path) -> str | None:
    """Read Claude Code's persisted effective effort setting without guessing.

    Claude Code writes ``effortLevel`` when its interactive ``/effort`` control
    changes. Older/current config shapes also use ``effort`` or
    ``thinkingLevel``; retain those aliases because launch-default discovery
    already accepts them. A malformed or unsupported value is not evidence.
    """
    try:
        data: Any = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    for key in ("effortLevel", "effort", "thinkingLevel"):
        value = data.get(key)
        if not isinstance(value, str):
            continue
        effort = value.strip().lower()
        if effort in CC_SUPPORTED_REASONING_EFFORTS:
            return effort
    return None


def changed_cc_settings_reasoning_effort(
    settings_path: Path,
    *,
    previous_mtime_ns: int | None,
) -> tuple[int | None, str | None]:
    """Return a new valid effort only after the observed file revision changes."""
    current_mtime_ns = cc_settings_mtime_ns(settings_path)
    if current_mtime_ns == previous_mtime_ns:
        return current_mtime_ns, None
    return current_mtime_ns, read_cc_settings_reasoning_effort(settings_path)
