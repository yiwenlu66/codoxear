from __future__ import annotations

import datetime
import errno
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from .agent_backend import get_agent_backend
from .agent_backend import infer_agent_backend_from_log_path
from .agent_backend import normalize_agent_backend
from .pi_log import read_pi_log_cwd
from .pi_log import read_pi_session_header
from .pi_log import read_pi_session_id


_LEGACY_WARNED = False


def _log_error(msg: str) -> None:
    sys.stderr.write(msg.rstrip("\n") + "\n")
    sys.stderr.flush()


def _log_exception(context: str, exc: BaseException) -> None:
    ts = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    _log_error(f"error: {context}: {type(exc).__name__}: {exc}")
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip("\n")
    if tb:
        _log_error(f"traceback ({ts}):\n{tb}")


def _socket_peer_disconnected(exc: BaseException) -> bool:
    if isinstance(exc, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
        return True
    return isinstance(exc, OSError) and exc.errno in (
        errno.EPIPE,
        errno.ECONNRESET,
        errno.ECONNABORTED,
        errno.ENOTCONN,
        errno.ESHUTDOWN,
    )


def _send_socket_json_line(conn: socket.socket, payload: dict[str, Any]) -> None:
    conn.sendall((json.dumps(payload) + "\n").encode("utf-8"))


def default_app_dir() -> Path:
    base = Path.home() / ".local" / "share"
    new = base / "codoxear"
    old = base / "codex-web"
    if old.exists():
        global _LEGACY_WARNED
        if not _LEGACY_WARNED:
            _LEGACY_WARNED = True
            _log_error(
                f"error: legacy runtime dir detected at {old}; it is no longer used. "
                f"migrate runtime state to {new}."
            )
    return new


def now() -> float:
    return time.time()


def _is_codex_rollout_log_path(path: Path) -> bool:
    return path.name.startswith("rollout-") and path.suffix == ".jsonl"


def _is_pi_session_log_path(path: Path, *, sessions_dir: Path | None = None) -> bool:
    if path.suffix != ".jsonl":
        return False
    if sessions_dir is None:
        return "/.pi/agent/sessions/" in str(path).replace("\\", "/")
    try:
        path.resolve().relative_to(sessions_dir.resolve())
    except Exception:
        return False
    return True


def _paths_match(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except Exception:
        try:
            return a.absolute() == b.absolute()
        except Exception:
            return str(a) == str(b)


def _path_in_set(path: Path, paths: set[Path]) -> bool:
    for candidate in paths:
        if _paths_match(path, candidate):
            return True
    return False


def _read_session_meta_payload_once(log_path: Path, *, max_bytes: int) -> dict[str, Any] | None:
    try:
        with log_path.open("rb") as f:
            data = f.read(int(max_bytes))
    except FileNotFoundError:
        return None
    except Exception as e:
        _log_exception(f"read session log {log_path}", e)
        raise

    for raw in data.splitlines():
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        except Exception as e:
            _log_exception(f"decode session log line from {log_path}", e)
            raise
        if obj.get("type") != "session_meta":
            continue
        payload = obj.get("payload")
        if not isinstance(payload, dict):
            raise ValueError(f"invalid session_meta payload in {log_path}")
        return payload
    return None


def read_session_meta_payload(
    log_path: Path,
    *,
    agent_backend: str | None = None,
    timeout_s: float = 0.0,
    poll_s: float = 0.05,
    max_bytes: int = 64 * 1024,
) -> dict[str, Any] | None:
    backend_name = normalize_agent_backend(
        agent_backend if agent_backend is not None else infer_agent_backend_from_log_path(log_path) or "codex"
    )
    if backend_name == "pi":
        return read_pi_session_header(log_path)
    deadline = now() + float(timeout_s)
    while True:
        payload = _read_session_meta_payload_once(log_path, max_bytes=max_bytes)
        if payload is not None:
            return payload
        if timeout_s <= 0:
            return None
        if now() >= deadline:
            return None
        time.sleep(float(poll_s))


def is_subagent_session_meta(payload: dict[str, Any]) -> bool:
    src = payload.get("source")
    return isinstance(src, dict) and ("subagent" in src)


def subagent_parent_thread_id(payload: dict[str, Any]) -> str | None:
    src = payload.get("source")
    if not isinstance(src, dict):
        return None
    sub = src.get("subagent")
    if not isinstance(sub, dict):
        return None
    spawn = sub.get("thread_spawn")
    if not isinstance(spawn, dict):
        return None
    parent = spawn.get("parent_thread_id")
    return parent if isinstance(parent, str) and parent else None


def classify_session_log(log_path: Path, *, agent_backend: str | None = None, timeout_s: float = 0.0) -> str | None:
    payload = read_session_meta_payload(log_path, agent_backend=agent_backend, timeout_s=timeout_s)
    if payload is None:
        return None
    return "subagent" if is_subagent_session_meta(payload) else "main"


def iter_session_logs(sessions_dir: Path, *, agent_backend: str = "codex") -> list[Path]:
    backend_name = normalize_agent_backend(agent_backend)
    if not sessions_dir.exists():
        return []
    out: list[tuple[float, Path]] = []
    pattern = "rollout-*.jsonl" if backend_name == "codex" else "*.jsonl"
    for p in sessions_dir.rglob(pattern):
        if backend_name == "codex" and not _is_codex_rollout_log_path(p):
            continue
        if backend_name == "pi" and not _is_pi_session_log_path(p, sessions_dir=sessions_dir):
            continue
        try:
            mt = float(p.stat().st_mtime)
        except FileNotFoundError:
            continue
        except Exception as e:
            _log_exception(f"stat {p}", e)
            raise
        out.append((mt, p))
    out.sort(key=lambda t: t[0], reverse=True)
    return [p for _mt, p in out]


def find_session_log_for_session_id(sessions_dir: Path, session_id: str, *, agent_backend: str = "codex") -> Path | None:
    backend_name = normalize_agent_backend(agent_backend)
    if not session_id:
        return None
    for p in iter_session_logs(sessions_dir, agent_backend=backend_name):
        if backend_name == "codex":
            if session_id in p.name:
                return p
            continue
        if read_pi_session_id(p) == session_id:
            return p
    return None


def find_new_session_log(
    *,
    sessions_dir: Path,
    agent_backend: str = "codex",
    cwd: str | None = None,
    after_ts: float,
    preexisting: set[Path],
    exclude_paths: set[Path] | None = None,
    timeout_s: float,
) -> tuple[str, Path] | None:
    backend_name = normalize_agent_backend(agent_backend)
    if cwd is not None:
        if not isinstance(cwd, str) or (not cwd.strip()):
            raise ValueError("cwd must be a non-empty string when provided")
    deadline = now() + float(timeout_s)
    while True:
        matches: list[tuple[str, Path]] = []
        for p in iter_session_logs(sessions_dir, agent_backend=backend_name):
            if _path_in_set(p, preexisting):
                continue
            if exclude_paths and _path_in_set(p, exclude_paths):
                continue
            try:
                if p.stat().st_mtime < after_ts - 2:
                    continue
            except FileNotFoundError:
                continue
            payload = read_session_meta_payload(p, agent_backend=backend_name, timeout_s=0.0)
            if not payload:
                continue
            if backend_name == "codex" and is_subagent_session_meta(payload):
                continue
            if cwd is not None:
                pcwd = payload.get("cwd")
                if not (isinstance(pcwd, str) and pcwd == cwd):
                    continue
            if backend_name == "pi":
                sid = read_pi_session_id(p)
            else:
                sid = payload.get("id")
            if isinstance(sid, str) and sid:
                matches.append((sid, p))
        if len(matches) == 1:
            return matches[0]
        if now() >= deadline:
            return None
        time.sleep(0.2)


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


def _macos_open_rollout_logs(root_pid: int) -> set[Path]:
    import subprocess
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
        if "/rollout-" not in tgt:
            continue
        out.add(Path(tgt))
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
        return _macos_open_rollout_logs(root_pid)
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
            if _is_pi_session_log_path(path, sessions_dir=sessions_dir):
                out.add(path)
    return out


def proc_open_writable_rollout_logs(proc_root: Path, root_pid: int, *, agent_backend: str = "codex") -> set[Path]:
    return proc_open_writable_rollout_logs_for_backend(proc_root, root_pid, agent_backend=agent_backend)


def proc_open_writable_rollout_logs_for_backend(proc_root: Path, root_pid: int, *, agent_backend: str) -> set[Path]:
    backend_name = normalize_agent_backend(agent_backend)
    if sys.platform == "darwin":
        return _macos_open_rollout_logs(root_pid)
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
            if _is_pi_session_log_path(path, sessions_dir=sessions_dir):
                out.add(path)
    return out


def proc_find_open_rollout_log(
    *,
    proc_root: Path,
    root_pid: int,
    agent_backend: str = "codex",
    cwd: str | None = None,
    ignored_paths: set[Path] | None = None,
) -> Path | None:
    backend_name = normalize_agent_backend(agent_backend)
    cands = list(proc_open_writable_rollout_logs_for_backend(proc_root, root_pid, agent_backend=backend_name))
    if not cands:
        return None
    ignored_resolved: set[Path] = set()
    for p in ignored_paths or set():
        try:
            ignored_resolved.add(p.resolve())
        except Exception:
            ignored_resolved.add(p)
    try:
        cands.sort(key=lambda p: float(p.stat().st_mtime), reverse=True)
    except Exception:
        pass
    matches: list[Path] = []
    for p in cands:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        if rp in ignored_resolved:
            continue
        payload = read_session_meta_payload(p, agent_backend=backend_name, timeout_s=0.0)
        if not payload:
            continue
        if backend_name == "codex" and is_subagent_session_meta(payload):
            continue
        if cwd is not None:
            pcwd = payload.get("cwd")
            if not (isinstance(pcwd, str) and pcwd == cwd):
                continue
        matches.append(p)
    if len(matches) != 1:
        return None
    return matches[0]


def read_jsonl_from_offset(path: Path, offset: int, *, max_bytes: int) -> tuple[list[dict[str, Any]], int]:
    try:
        with path.open("rb") as f:
            prev_byte = b"\n"
            if int(offset) > 0:
                f.seek(int(offset) - 1)
                prev_byte = f.read(1)
            f.seek(offset)
            target = max(1, int(max_bytes))
            chunk_size = max(64 * 1024, min(target, 1024 * 1024))
            data = f.read(target)
            if b"\n" not in data:
                extra: list[bytes] = []
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    extra.append(chunk)
                    if b"\n" in chunk:
                        break
                if extra:
                    data += b"".join(extra)
    except Exception as e:
        _log_exception(f"read jsonl {path} from offset {offset}", e)
        raise

    if not data:
        return [], int(offset)

    start_off = int(offset)

    # If the caller seeks into the middle of a record, skip the leading
    # fragment so JSON decoding starts on a real line boundary.
    if int(offset) > 0 and prev_byte != b"\n":
        first_nl = data.find(b"\n")
        if first_nl < 0:
            return [], int(offset)
        data = data[first_nl + 1 :]
        start_off = int(offset) + int(first_nl) + 1
        if not data:
            return [], start_off

    # When tailing a live JSONL file, we can read a chunk that ends in the middle
    # of the last record, including the middle of a multibyte UTF-8 sequence.
    # Only parse newline-terminated records, and do not advance the offset past
    # the last newline we observed.
    last_nl = data.rfind(b"\n")
    if last_nl < 0:
        return [], start_off
    data = data[: last_nl + 1]
    new_off = start_off + int(last_nl) + 1

    lines = data.splitlines()
    out: list[dict[str, Any]] = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        except Exception as e:
            _log_exception(f"decode jsonl line from {path}", e)
            raise
    return out, new_off
