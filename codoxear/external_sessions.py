"""Pi-owned sessions: deliberately outside the broker registry and its workers.

Discovery is request-driven. The extension instance owns identity, the bridge
owns current runtime state, and the existing positioned-log pipeline owns chat.
Nothing here can signal/kill a Pi process or enter broker lifecycle machinery.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
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


def raft_agent_name(cwd: str, *, home: Path | None = None) -> str:
    """The Raft agent name for a session whose cwd is a Raft agent folder.

    Raft runs each agent in ``~/.raft/agents/<uuid>`` and never sets a Pi
    session name; the only record of the agent's name is the ``# <name>``
    heading that opens that folder's ``MEMORY.md``. Anything else is not a Raft
    agent and yields ``""``.
    """
    folder = Path(cwd)
    if not cwd or folder.parent != (home or Path.home()) / ".raft" / "agents":
        return ""
    try:
        with (folder / "MEMORY.md").open(encoding="utf-8", errors="replace") as handle:
            head = handle.read(4096)
    except OSError:
        return ""
    for line in head.splitlines():
        if line.strip():
            match = re.match(r"#\s+(.+)", line.strip())
            return match.group(1).strip()[:200] if match else ""
    return ""


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
                    instance_id = data.get("instanceId")
                    version = int(data.get("bridgeVersion", 0))
                    # Version 4 processes can remain alive across a server
                    # upgrade; their single-process endpoint remains readable.
                    if instance_id is None and version == 4:
                        expected_name = f"pi-{pid}.json"
                    elif isinstance(instance_id, str) and re.fullmatch(r"[0-9a-f]{16}", instance_id):
                        expected_name = f"pi-{pid}-{instance_id}.json"
                    else:
                        continue
                    if marker.name != expected_name or pid <= 0:
                        continue
                    if not process_matches(pid, data.get("processStart")):
                        self._clean_entry(marker, data)
                        continue
                    if version < 4 or not {"send", "state", "interrupt"}.issubset(data.get("features", [])):
                        continue
                    incarnation = f"{pid}:{data.get('processStart') or data['startTs']}" + (f":{instance_id}" if instance_id else "")
                    sid = f"external-pi-{pid}-" + hashlib.sha256(incarnation.encode()).hexdigest()[:12]
                    if sid in self.hidden:
                        continue
                    session = self.sessions.get(sid)
                    if session is None:
                        session = Session(session_id=sid, thread_id="", broker_pid=0, codex_pid=pid, agent_backend="pi", owned=False,
                                          start_ts=float(data["startTs"]), cwd=str(data["cwd"]), log_path=None,
                                          sock_path=marker.with_suffix(".bridge.sock"), external=True, external_instance_id=instance_id)
                    self._apply_state(session, data)
                    # A PID may host many SDK sessions and outlive a disposed
                    # one. Only this instance's current ctx/socket proves life.
                    try:
                        self._apply_state(session, self._call(session, "state"))
                    except (OSError, ValueError, TypeError):
                        self._clean_entry(marker, data)
                        continue
                    revision = log_revision(session.log_path) if session.log_path is not None else None
                    if session.log_path is not None and revision is not None and revision != session.run_settings_log_revision:
                        session.token = _find_latest_token_update(session.log_path)
                        session.run_settings_log_revision = revision
                    found[sid] = session
                except (OSError, ValueError, KeyError, TypeError):
                    continue
            for sid in self.sessions.keys() - found.keys():
                self.drafts.pop(sid, None)
                self.files.pop(sid, None)
            self.sessions = found

    @staticmethod
    def _clean_entry(marker: Path, observed: dict) -> None:
        try:
            # Do not erase a lifecycle rewrite that raced the failed query.
            if json.loads(marker.read_text()) != observed:
                return
            for file in (marker, marker.with_suffix(".bridge.sock"), Path(f"{marker}.caps"), Path(f"{marker}.receipt")):
                file.unlink(missing_ok=True)
        except (OSError, ValueError):
            pass

    @staticmethod
    def _apply_state(session: Session, data: dict) -> None:
        if "sessionId" in data:
            session.thread_id = str(data["sessionId"])
        if "cwd" in data:
            session.cwd = str(data["cwd"])
        if "sessionFile" in data:
            session.log_path = Path(data["sessionFile"]) if data["sessionFile"] else None
        session.busy = bool(data.get("busy"))
        session.backend_session_name = str(data.get("name") or "")
        session.model = data.get("model")
        session.model_provider = data.get("model_provider")
        session.reasoning_effort = data.get("reasoning_effort")
        # No TUI builtins: sendUserMessage dispatches only extension commands,
        # skills and templates. Known interactive extension names stay omitted.
        excluded = {item["name"] for item in PI_BUILTIN_SLASH_COMMANDS} | set(PI_INTERACTIVE_SLASH_COMMANDS) | {"effort", "thinking"}
        # Pi queues busy sendUserMessage calls as literal steering text; it
        # does not dispatch their extension commands. Do not offer that false
        # affordance during a run.
        session.slash_commands = [] if session.busy else [item for item in _clean_commands(data.get("commands", [])) if item["name"] not in excluded]

    @staticmethod
    def _call(session: Session, op: str, **fields: Any) -> dict:
        request = {"op": op, **fields}
        if op != "state":
            request["sessionId"] = session.thread_id
        if session.external_instance_id is not None:
            request["instanceId"] = session.external_instance_id
        result = call_control_socket(session.sock_path, request, timeout_s=0.5)
        if not isinstance(result, dict):
            raise ValueError("invalid Pi bridge response")
        if result.get("ok") is not True or result.get("pid") != session.codex_pid:
            raise ValueError(str(result.get("error") or "invalid Pi bridge response"))
        if session.external_instance_id is not None and result.get("instanceId") != session.external_instance_id:
            raise ValueError("stale Pi bridge instance")
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
                             "pid": session.codex_pid, "external_instance_id": session.external_instance_id,
                             "cwd": session.cwd, "log_path": str(session.log_path) if session.log_path else None,
                             "display_name": session.backend_session_name or raft_agent_name(session.cwd)
                             or Path(session.cwd).name or "Pi",
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
