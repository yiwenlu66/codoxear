from __future__ import annotations

import os
import sys
from pathlib import Path

from .agent_backend import get_agent_backend
from .agent_backend import normalize_agent_backend
from .session_log_paths import _is_cc_session_log_path
from .session_log_paths import _is_codex_rollout_log_path
from .session_log_paths import _is_pi_session_log_path


def _macos_children(pid: int) -> list[int]:
    try:
        import subprocess
        result = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True)
        out: list[int] = []
        for line in result.stdout.splitlines():
            try:
                out.append(int(line.strip()))
            except ValueError:
                continue
        return out
    except Exception:
        return []


def _macos_descendants(root_pid: int) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    stack: list[int] = [root_pid]
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
        stack.extend(_macos_children(pid))
    return out


def _macos_open_rollout_logs(root_pid: int, *, agent_backend: str = "codex") -> set[Path]:
    import subprocess
    backend_name = normalize_agent_backend(agent_backend)
    sessions_dir = get_agent_backend(backend_name).sessions_dir()
    pids = _macos_descendants(root_pid)
    if not pids:
        return set()
    pid_arg = ",".join(str(p) for p in pids)
    try:
        result = subprocess.run(
            ["lsof", "-p", pid_arg, "-F", "n"],
            capture_output=True, text=True, timeout=5,
        )
    except Exception:
        return set()
    out: set[Path] = set()
    for line in result.stdout.splitlines():
        if not line.startswith("n"):
            continue
        tgt = line[1:]
        if not tgt.startswith("/") or not tgt.endswith(".jsonl"):
            continue
        path = Path(tgt)
        if backend_name == "codex":
            if _is_codex_rollout_log_path(path):
                out.add(path)
            continue
        if backend_name == "pi" and _is_pi_session_log_path(path, sessions_dir=sessions_dir):
            out.add(path)
            continue
        if backend_name == "cc" and _is_cc_session_log_path(path, sessions_dir=sessions_dir):
            out.add(path)
    return out


def _proc_pid_uid(proc_root: Path, pid: int) -> int | None:
    try:
        return int((proc_root / str(pid)).stat().st_uid)
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _proc_children(proc_root: Path, pid: int) -> list[int]:
    p = proc_root / str(pid) / "task" / str(pid) / "children"
    try:
        raw = p.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return []
    except Exception:
        return []
    if not raw:
        return []
    out: list[int] = []
    for s in raw.split():
        try:
            out.append(int(s))
        except ValueError:
            continue
    return out


def _proc_descendants(proc_root: Path, root_pid: int) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    stack: list[int] = [int(root_pid)]
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
        stack.extend(_proc_children(proc_root, pid))
    return out


def _proc_fd_flags(proc_root: Path, pid: int, fd_name: str) -> int | None:
    info_path = proc_root / str(pid) / "fdinfo" / fd_name
    try:
        raw = info_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except Exception:
        return None
    for line in raw.splitlines():
        if not line.startswith("flags:"):
            continue
        flags_raw = line.split(":", 1)[1].strip().split()[0]
        try:
            return int(flags_raw, 8)
        except ValueError:
            return None
    return None


def _fd_has_write_intent(flags: int) -> bool:
    access_mode = int(flags) & int(os.O_ACCMODE)
    return access_mode in (int(os.O_WRONLY), int(os.O_RDWR))


def proc_open_rollout_logs(proc_root: Path, root_pid: int, *, agent_backend: str = "codex") -> set[Path]:
    return proc_open_rollout_logs_for_backend(proc_root, root_pid, agent_backend=agent_backend)


def proc_open_rollout_logs_for_backend(proc_root: Path, root_pid: int, *, agent_backend: str) -> set[Path]:
    backend_name = normalize_agent_backend(agent_backend)
    if sys.platform == "darwin":
        return _macos_open_rollout_logs(root_pid, agent_backend=backend_name)
    uid = int(os.getuid())
    sessions_dir = get_agent_backend(backend_name).sessions_dir()
    out: set[Path] = set()
    for pid in _proc_descendants(proc_root, root_pid):
        puid = _proc_pid_uid(proc_root, pid)
        if (puid is not None) and (puid != uid):
            continue
        fd_dir = proc_root / str(pid) / "fd"
        try:
            entries = list(fd_dir.iterdir())
        except FileNotFoundError:
            continue
        except Exception:
            continue
        for ent in entries:
            try:
                tgt = os.readlink(ent)
            except OSError:
                continue
            if tgt.endswith(" (deleted)"):
                continue
            if (not tgt.startswith("/")) or (not tgt.endswith(".jsonl")):
                continue
            path = Path(tgt)
            if backend_name == "codex":
                if not _is_codex_rollout_log_path(path):
                    continue
                out.add(path)
                continue
            if backend_name == "pi" and _is_pi_session_log_path(path, sessions_dir=sessions_dir):
                out.add(path)
                continue
            if backend_name == "cc" and _is_cc_session_log_path(path, sessions_dir=sessions_dir):
                out.add(path)
    return out


def proc_open_writable_rollout_logs(proc_root: Path, root_pid: int, *, agent_backend: str = "codex") -> set[Path]:
    return proc_open_writable_rollout_logs_for_backend(proc_root, root_pid, agent_backend=agent_backend)


def proc_open_writable_rollout_logs_for_backend(proc_root: Path, root_pid: int, *, agent_backend: str) -> set[Path]:
    backend_name = normalize_agent_backend(agent_backend)
    if sys.platform == "darwin":
        return _macos_open_rollout_logs(root_pid, agent_backend=backend_name)
    uid = int(os.getuid())
    sessions_dir = get_agent_backend(backend_name).sessions_dir()
    out: set[Path] = set()
    for pid in _proc_descendants(proc_root, root_pid):
        puid = _proc_pid_uid(proc_root, pid)
        if (puid is not None) and (puid != uid):
            continue
        fd_dir = proc_root / str(pid) / "fd"
        try:
            entries = list(fd_dir.iterdir())
        except FileNotFoundError:
            continue
        except Exception:
            continue
        for ent in entries:
            flags = _proc_fd_flags(proc_root, pid, ent.name)
            if flags is None or (not _fd_has_write_intent(flags)):
                continue
            try:
                tgt = os.readlink(ent)
            except OSError:
                continue
            if tgt.endswith(" (deleted)"):
                continue
            if (not tgt.startswith("/")) or (not tgt.endswith(".jsonl")):
                continue
            path = Path(tgt)
            if backend_name == "codex":
                if not _is_codex_rollout_log_path(path):
                    continue
                out.add(path)
                continue
            if backend_name == "pi" and _is_pi_session_log_path(path, sessions_dir=sessions_dir):
                out.add(path)
                continue
            if backend_name == "cc" and _is_cc_session_log_path(path, sessions_dir=sessions_dir):
                out.add(path)
    return out
