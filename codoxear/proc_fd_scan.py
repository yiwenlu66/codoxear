from __future__ import annotations

import os
from pathlib import Path


def _proc_children_pids(proc_root: Path, pid: int) -> list[int]:
    try:
        data = (proc_root / str(pid) / "task" / str(pid) / "children").read_text("utf-8", errors="replace")
    except FileNotFoundError:
        return []
    toks = [t for t in data.strip().split() if t.isdigit()]
    out: list[int] = []
    for t in toks:
        try:
            out.append(int(t))
        except Exception:
            continue
    return out


def _proc_descendants(proc_root: Path, root_pid: int) -> set[int]:
    seen: set[int] = set()
    q: list[int] = [int(root_pid)]
    while q:
        pid = q.pop()
        if pid <= 0 or pid in seen:
            continue
        seen.add(pid)
        for c in _proc_children_pids(proc_root, pid):
            if c not in seen:
                q.append(c)
    return seen


def _fd_is_writable(proc_root: Path, pid: int, fd: int) -> bool:
    try:
        data = (proc_root / str(pid) / "fdinfo" / str(fd)).read_text("utf-8", errors="replace")
    except FileNotFoundError:
        return False
    flags = None
    for line in data.splitlines():
        if line.startswith("flags:"):
            flags = line.split(":", 1)[1].strip()
            break
    if not flags:
        return False
    try:
        v = int(flags, 8)
    except Exception:
        return False
    acc = v & os.O_ACCMODE
    return acc != os.O_RDONLY


def _rollout_path_from_fd_link(link: str) -> Path | None:
    s = str(link).strip()
    if not s:
        return None
    if s.endswith(" (deleted)"):
        s = s[: -len(" (deleted)")].rstrip()
    if not s.startswith("/"):
        return None
    p = Path(s)
    if (not p.name.startswith("rollout-")) or (not p.name.endswith(".jsonl")):
        return None
    return p


def _iter_writable_rollout_paths(
    *,
    proc_root: Path,
    pid: int,
    sessions_dir: Path,
) -> list[Path]:
    try:
        fd_dir = proc_root / str(pid) / "fd"
        entries = list(fd_dir.iterdir())
    except FileNotFoundError:
        return []
    out: list[Path] = []
    sessions_root = sessions_dir.resolve()
    for e in entries:
        name = e.name
        if not name.isdigit():
            continue
        try:
            fd = int(name)
        except Exception:
            continue
        try:
            link = os.readlink(e)
        except FileNotFoundError:
            continue
        except PermissionError:
            raise
        except Exception:
            continue
        p = _rollout_path_from_fd_link(link)
        if p is None:
            continue
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        try:
            rp.relative_to(sessions_root)
        except Exception:
            continue
        if not _fd_is_writable(proc_root, pid, fd):
            continue
        out.append(rp)
    return out

