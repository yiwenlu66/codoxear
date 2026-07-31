from __future__ import annotations

import json
import os
import pwd
import re
import shlex
import sys
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from codoxear.agent_backend import normalize_agent_backend
from codoxear.util import default_app_dir as _default_app_dir
from codoxear.util import read_session_meta_payload as _read_session_meta_payload


SHELL_PRE_EXEC_MARKER = "\x1b]777;codoxear=agent-exec-start\x07"
SHELL_PRE_EXEC_MARKER_BYTES = SHELL_PRE_EXEC_MARKER.encode("utf-8")


def _resume_session_id_from_args(args: list[str], *, agent_backend: str) -> str | None:
    backend = normalize_agent_backend(agent_backend)
    if backend == "pi":
        for idx, token in enumerate(args):
            if token != "--session":
                continue
            if (idx + 1) >= len(args):
                return None
            resume_id = str(args[idx + 1] or "").strip()
            if not resume_id:
                return None
            if resume_id.endswith(".jsonl"):
                try:
                    payload = _read_session_meta_payload(Path(resume_id), agent_backend="pi", timeout_s=0.0)
                except Exception:
                    return None
                if isinstance(payload, dict):
                    sid = payload.get("id")
                    if isinstance(sid, str) and sid:
                        return sid
                return None
            return resume_id
        return None
    for idx, token in enumerate(args):
        expected = "--resume" if backend == "cc" else "resume"
        if token != expected:
            continue
        if (idx + 1) >= len(args):
            return None
        resume_id = str(args[idx + 1] or "").strip()
        return resume_id or None
    return None


def _session_log_path_from_args(*, args: list[str], agent_backend: str, sessions_dir: Path) -> Path | None:
    if normalize_agent_backend(agent_backend) != "pi":
        return None
    for idx, token in enumerate(args):
        if token != "--session":
            continue
        if (idx + 1) >= len(args):
            return None
        raw = str(args[idx + 1] or "").strip()
        if (not raw) or (not raw.endswith(".jsonl")):
            return None
        path = Path(raw).expanduser()
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        try:
            resolved.relative_to(sessions_dir.resolve())
        except Exception:
            return None
        return resolved
    return None


def _pi_session_dir_name(cwd: str) -> str:
    normalized = cwd.lstrip("/\\").replace("/", "-").replace("\\", "-").replace(":", "-")
    return f"--{normalized}--"


def _pi_session_dir_from_args(*, args: list[str], cwd: str, sessions_dir: Path) -> Path | None:
    for idx, token in enumerate(args):
        if token == "--no-session":
            return None
        if token != "--session-dir":
            continue
        if (idx + 1) >= len(args):
            return None
        raw = str(args[idx + 1] or "").strip()
        if not raw:
            return None
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (Path(cwd) / path).resolve()
        return path
    return sessions_dir / _pi_session_dir_name(cwd)


def _pi_new_session_log_path(*, cwd: str, sessions_dir: Path) -> Path:
    session_dir = sessions_dir / _pi_session_dir_name(cwd)
    session_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    filename = f"{timestamp.replace(':', '-').replace('.', '-')}_{uuid.uuid4()}.jsonl"
    return session_dir / filename


def _pi_active_session_marker_path(*, broker_pid: int | None = None) -> Path:
    pid = int(os.getpid() if broker_pid is None else broker_pid)
    return _default_app_dir() / "pi-active-sessions" / f"broker-{pid}.json"


def _pi_bridge_extension_path() -> Path:
    return Path(__file__).resolve().parent / "pi_active_session_bridge.ts"


def _ensure_pi_bridge_args(*, args: list[str], marker_path: Path, agent_backend: str) -> list[str]:
    if normalize_agent_backend(agent_backend) != "pi":
        return list(args)
    bridge = str(_pi_bridge_extension_path())
    out = list(args)
    for idx, token in enumerate(out[:-1]):
        if token in ("--extension", "-e") and out[idx + 1] == bridge:
            os.environ["CODEX_WEB_PI_ACTIVE_SESSION_FILE"] = str(marker_path)
            return out
    os.environ["CODEX_WEB_PI_ACTIVE_SESSION_FILE"] = str(marker_path)
    out.extend(["--extension", bridge])
    return out


def _reset_pi_active_session_marker(marker_path: Path) -> None:
    try:
        marker_path.unlink()
    except FileNotFoundError:
        return
    except Exception:
        return


def _read_pi_active_session_marker(marker_path: Path, *, sessions_dir: Path) -> Path | None:
    try:
        data = json.loads(marker_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception:
        return None
    if not isinstance(data, dict) or data.get("version") != 1:
        return None
    raw = data.get("sessionFile")
    if not isinstance(raw, str) or not raw.strip() or not raw.endswith(".jsonl"):
        return None
    path = Path(raw).expanduser()
    try:
        resolved = path.resolve()
    except Exception:
        resolved = path
    try:
        resolved.relative_to(sessions_dir.resolve())
    except Exception:
        return None
    return resolved


def _ensure_pi_session_arg(*, args: list[str], cwd: str, sessions_dir: Path, agent_backend: str) -> list[str]:
    if normalize_agent_backend(agent_backend) != "pi":
        return list(args)
    out = list(args)
    session_dir = _pi_session_dir_from_args(args=out, cwd=cwd, sessions_dir=sessions_dir)
    if session_dir is None:
        return out
    for token in out:
        if token == "--session":
            return out
    if session_dir == sessions_dir / _pi_session_dir_name(cwd):
        log_path = _pi_new_session_log_path(cwd=cwd, sessions_dir=sessions_dir)
    else:
        session_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        filename = f"{timestamp.replace(':', '-').replace('.', '-')}_{uuid.uuid4()}.jsonl"
        log_path = session_dir / filename
    out.extend(["--session", str(log_path)])
    return out


def _expand_cwd(cwd: str) -> str:
    if not isinstance(cwd, str) or not cwd.strip():
        raise ValueError("cwd must be a non-empty string")
    home = str(Path.home())
    s = cwd.strip().replace("${HOME}", home)
    s = re.sub(r"\$HOME(?![A-Za-z0-9_])", home, s)
    return os.path.abspath(os.path.expanduser(os.path.expandvars(s)))


def _user_shell() -> str:
    sh = os.environ.get("SHELL")
    if isinstance(sh, str) and sh.strip():
        return sh.strip()
    try:
        return pwd.getpwuid(os.getuid()).pw_shell
    except Exception:
        return "/bin/zsh"


def _shell_argv_for_command(cmd: str, *, user_shell: Callable[[], str] = _user_shell) -> list[str]:
    shell = user_shell()
    # -l: login (read profile); -i: interactive (read rc); -c: run command; command begins with exec to avoid wrapper processes.
    return [shell, "-l", "-i", "-c", cmd]


def _agent_shell_command(argv: list[str], *, pty_slave_path: str) -> str:
    q = shlex.quote
    script = (
        "import os, sys\n"
        "fd = os.open(sys.argv[1], os.O_RDWR)\n"
        "try:\n"
        "    for target in (0, 1, 2):\n"
        "        os.dup2(fd, target)\n"
        "finally:\n"
        "    if fd > 2:\n"
        "        os.close(fd)\n"
        "os.write(1, sys.argv[2].encode('utf-8'))\n"
        "argv = sys.argv[3:]\n"
        "os.execvpe(argv[0], argv, os.environ)\n"
    )
    trampoline = [sys.executable, "-c", script, pty_slave_path, SHELL_PRE_EXEC_MARKER, *argv]
    return "exec " + " ".join(q(x) for x in trampoline)
