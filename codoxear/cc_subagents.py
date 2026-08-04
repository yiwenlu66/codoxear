from __future__ import annotations

"""Claude Code SubagentStart/SubagentStop hook bridge for Codoxear-owned sessions."""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping

from .json_state import atomic_write_json
from .subagent_events import emit_subagent_event


_ENV_ROOT = "CODEX_WEB_CC_SUBAGENT_RUNS_ROOT"
_ENV_BROKER_PID = "CODEX_WEB_CC_SUBAGENT_BROKER_PID"
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,200}$")
_SESSION_ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


def _clean_id(value: Any, *, session: bool = False) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate:
        return None
    if session:
        return candidate if _SESSION_ID_RE.fullmatch(candidate) else None
    return candidate if _SAFE_ID_RE.fullmatch(candidate) else None


def _configured_root(environ: Mapping[str, str]) -> Path | None:
    raw = environ.get(_ENV_ROOT, "").strip()
    return Path(raw).expanduser() if raw else None


def _broker_pid(environ: Mapping[str, str]) -> int | None:
    try:
        value = int(environ.get(_ENV_BROKER_PID, ""))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _status_path(root: Path, *, session_id: str, agent_id: str) -> Path:
    return root / session_id / f"{agent_id}.json"


def handle_hook_event(event: Mapping[str, Any], *, environ: Mapping[str, str] | None = None) -> bool:
    """Apply one Claude Code hook event, returning whether it was accepted.

    The bridge is inert unless its web-owned launch environment supplies an
    explicit app-owned root and broker PID. This prevents a terminal session
    from accidentally becoming an indicator source.
    """
    env = os.environ if environ is None else environ
    if env.get("CODEX_WEB_OWNER") != "web" or env.get("CODEX_WEB_AGENT_BACKEND") != "cc":
        return False
    root = _configured_root(env)
    broker_pid = _broker_pid(env)
    session_id = _clean_id(event.get("session_id"), session=True)
    agent_id = _clean_id(event.get("agent_id"))
    event_name = event.get("hook_event_name")
    if root is None or broker_pid is None or session_id is None or agent_id is None:
        return False
    path = _status_path(root, session_id=session_id, agent_id=agent_id)
    if event_name == "SubagentStart":
        agent_type = event.get("agent_type")
        label = f"Subagent started{f' — {agent_type}' if isinstance(agent_type, str) and agent_type.strip() else ''}"
        atomic_write_json(
            path,
            {
                "version": 1,
                "state": "running",
                "parent_session_id": session_id,
                "agent_id": agent_id,
                "broker_pid": broker_pid,
                "event": emit_subagent_event("cc", event_id=agent_id, text=label),
            },
        )
        return True
    if event_name == "SubagentStop":
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        try:
            path.parent.rmdir()
        except OSError:
            pass
        return True
    return False


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except (OSError, ValueError, TypeError):
        return 0
    if isinstance(event, dict):
        handle_hook_event(event)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
