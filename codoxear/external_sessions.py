"""Pi-owned sessions: deliberately outside the broker registry and its workers.

Discovery is request-driven. The process incarnation owns identity, the bridge
owns current runtime state, and the existing positioned-log pipeline owns chat.
Nothing here can signal/kill a Pi process or enter broker lifecycle machinery.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
import time
from typing import Any

from .control_socket import call_control_socket
from .session_model import Session
from .session_log_projection import log_revision
from .rollout_log import _find_latest_token_update
from .slash_commands import PI_BUILTIN_SLASH_COMMANDS, PI_INTERACTIVE_SLASH_COMMANDS, _clean_commands
from .util import pid_alive


def external_dir() -> Path:
    base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share")
    return Path(os.environ.get("CODOXEAR_EXTERNAL_DIR") or base / "codoxear/external").expanduser()


def process_matches(pid: int, start: str | None) -> bool:
    if not pid_alive(pid):
        return False
    if start is not None and Path("/proc").is_dir():
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
            return fields[0] != "Z" and fields[19] == start
        except (OSError, IndexError):
            return False
    return True


class ExternalSessionRegistry:
    def __init__(self, directory: Path | None = None):
        self.directory = directory or external_dir()
        self.lock = threading.RLock()
        self.sessions: dict[str, Session] = {}
        self.hidden: set[str] = set()
        self.last_scan = float("-inf")
        self.drafts: dict[str, dict[str, Any]] = {}
        self.files: dict[str, list[str]] = {}

    def scan(self, *, force: bool = False) -> None:
        with self.lock:
            now = time.monotonic()
            if not force and now - self.last_scan < 0.5:
                return
            self.last_scan = now
            found = {}
            for marker in sorted(self.directory.glob("pi-*.json")):
                try:
                    data = json.loads(marker.read_text())
                    pid = int(data["pid"])
                    if marker.name != f"pi-{pid}.json" or pid <= 0 or not process_matches(pid, data.get("processStart")):
                        continue
                    if data.get("bridgeVersion", 0) < 4 or not {"send", "state", "interrupt"}.issubset(data.get("features", [])):
                        continue
                    incarnation = f"{pid}:{data.get('processStart') or data['startTs']}"
                    sid = f"external-pi-{pid}-" + hashlib.sha256(incarnation.encode()).hexdigest()[:12]
                    if sid in self.hidden:
                        continue
                    session = self.sessions.get(sid)
                    if session is None:
                        session = Session(session_id=sid, thread_id="", broker_pid=0, codex_pid=pid, agent_backend="pi", owned=False,
                                          start_ts=float(data["startTs"]), cwd=str(data["cwd"]), log_path=None,
                                          sock_path=marker.with_suffix(".bridge.sock"), external=True)
                    session.thread_id = str(data["sessionId"])
                    session.cwd = str(data["cwd"])
                    session.log_path = Path(data["sessionFile"]) if data.get("sessionFile") else None
                    self._apply_state(session, data)
                    revision = log_revision(session.log_path) if session.log_path is not None else None
                    if session.log_path is not None and revision is not None and revision != session.run_settings_log_revision:
                        session.token = _find_latest_token_update(session.log_path)
                        session.run_settings_log_revision = revision
                    # Marker writes happen at lifecycle boundaries, not at every
                    # idle transition. A live query is the busy authority.
                    try:
                        self._apply_state(session, self._call(session, "state"))
                    except (OSError, ValueError):
                        continue  # Unreachable is not a controllable session.
                    found[sid] = session
                except (OSError, ValueError, KeyError, TypeError):
                    continue
            self.sessions = found

    @staticmethod
    def _apply_state(session: Session, data: dict) -> None:
        session.busy = bool(data.get("busy"))
        session.backend_session_name = str(data.get("name") or "")
        session.model = data.get("model")
        session.model_provider = data.get("model_provider")
        session.reasoning_effort = data.get("reasoning_effort")
        # No TUI builtins: sendUserMessage dispatches only extension commands,
        # skills and templates. Known interactive extension names stay omitted.
        excluded = {item["name"] for item in PI_BUILTIN_SLASH_COMMANDS} | set(PI_INTERACTIVE_SLASH_COMMANDS) | {"effort", "thinking"}
        session.slash_commands = [item for item in _clean_commands(data.get("commands", [])) if item["name"] not in excluded]

    @staticmethod
    def _call(session: Session, op: str, **fields: Any) -> dict:
        result = call_control_socket(session.sock_path, {"op": op, "sessionId": session.thread_id, **fields}, timeout_s=0.5)
        if result.get("ok") is not True or result.get("pid") != session.codex_pid:
            raise ValueError(str(result.get("error") or "invalid Pi bridge response"))
        return result

    def get(self, sid: str) -> Session | None:
        if not sid.startswith("external-pi-"):
            return None
        self.scan()
        with self.lock:
            return self.sessions.get(sid)

    def require(self, sid: str) -> Session:
        session = self.get(sid)
        if session is None:
            raise KeyError("unknown external session")
        return session

    def call(self, sid: str, op: str, **fields: Any) -> dict:
        with self.lock:
            result = self._call(self.require(sid), op, **fields)
            self.last_scan = float("-inf")
            return result

    def dismiss(self, sid: str) -> bool:
        with self.lock:
            if self.get(sid) is None:
                return False
            self.hidden.add(sid)
            self.sessions.pop(sid, None)
            self.drafts.pop(sid, None)
            self.files.pop(sid, None)
            return True

    def rows(self) -> list[dict]:
        self.scan()
        with self.lock:
            rows = []
            for session in self.sessions.values():
                updated = session.start_ts
                if session.log_path is not None:
                    try:
                        updated = session.log_path.stat().st_mtime
                    except OSError:
                        pass
                rows.append({"session_id": session.session_id, "thread_id": session.thread_id,
                             "agent_backend": "pi", "external": True, "owned": False, "start_ts": session.start_ts,
                             "cwd": session.cwd, "log_path": str(session.log_path) if session.log_path else None,
                             "display_name": session.backend_session_name or Path(session.cwd).name or "Pi",
                             "busy": session.busy, "queue_len": 0, "updated_ts": updated,
                             "model": session.model, "model_provider": session.model_provider,
                             "reasoning_effort": session.reasoning_effort, "token": session.token,
                             "slash_commands": session.slash_commands, "files": list(self.files.get(session.session_id, [])),
                             "staged_attachments": []})
            return sorted(rows, key=lambda row: (-row["updated_ts"], row["session_id"]))


def registry_for_manager(manager: Any) -> ExternalSessionRegistry:
    # Constructed eagerly by real managers; lazy compatibility for __new__ fixtures.
    registry = getattr(manager, "_external_sessions", None)
    if registry is None:
        registry = ExternalSessionRegistry()
        manager._external_sessions = registry
    return registry
