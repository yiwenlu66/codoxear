from __future__ import annotations

"""Backend-side session display names, read from the transcripts we already parse.

Each backend records a user-visible session title in its own log:

* Pi appends ``{"type": "session_info", "name": ...}``; the latest one wins.
* Claude Code appends ``{"type": "custom-title", "customTitle": ...}``.
* Codex keeps its thread title in the app-server state, never in the rollout.

This module only *observes*. The codoxear alias store stays the authority; an
observed name is used to dedupe a projection, and (when the session has no alias
at all) to adopt the name the user set in the terminal.
"""

import json
from pathlib import Path
from typing import Any

# A session log is read at bind time once. The cap keeps a pathological file
# from stalling the broker; a name beyond it is simply not adopted.
_MAX_SCAN_BYTES = 256 * 1024 * 1024


def _clean_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    name = " ".join(value.split()).strip()
    return name[:80].rstrip() if name else None


def backend_session_name_from_row(agent_backend: str, obj: Any) -> str | None:
    """The session name a single transcript row declares, or ``None``.

    Returns ``""`` for an explicit clear (Pi's empty ``session_info``), which is
    distinct from ``None`` = "this row says nothing about the name".
    """
    if not isinstance(obj, dict):
        return None
    row_type = obj.get("type")
    if agent_backend == "pi":
        if row_type != "session_info":
            return None
        cleaned = _clean_name(obj.get("name"))
        return cleaned if cleaned is not None else ""
    if agent_backend == "cc":
        if row_type != "custom-title":
            return None
        for key in ("customTitle", "custom_title", "title"):
            cleaned = _clean_name(obj.get(key))
            if cleaned is not None:
                return cleaned
        return ""
    return None


def read_last_backend_session_name(log_path: Path, agent_backend: str) -> str | None:
    """Scan a transcript for its latest recorded session name.

    Called once when the broker binds a log, because the incremental watcher
    starts at the current end of file and would otherwise never see a name set
    before codoxear attached.
    """
    if agent_backend not in ("pi", "cc"):
        return None
    found: str | None = None
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            consumed = 0
            for line in handle:
                consumed += len(line)
                if consumed > _MAX_SCAN_BYTES:
                    break
                if not line.startswith("{"):
                    continue
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    obj = json.loads(stripped)
                except ValueError:
                    continue
                observed = backend_session_name_from_row(agent_backend, obj)
                if observed is not None:
                    found = observed
    except OSError:
        return None
    return found
