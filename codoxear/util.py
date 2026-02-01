from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def default_app_dir() -> Path:
    base = Path.home() / ".local" / "share"
    new = base / "codoxear"
    old = base / "codex-web"
    if old.exists() and not new.exists():
        return old
    return new


def now() -> float:
    return time.time()


def iter_session_logs(sessions_dir: Path) -> list[Path]:
    if not sessions_dir.exists():
        return []
    return sorted(sessions_dir.rglob("rollout-*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)


def find_new_session_log(
    *,
    sessions_dir: Path,
    after_ts: float,
    preexisting: set[Path],
    timeout_s: float,
) -> tuple[str, Path] | None:
    deadline = now() + float(timeout_s)
    while now() < deadline:
        for p in iter_session_logs(sessions_dir):
            if p in preexisting:
                continue
            try:
                if p.stat().st_mtime < after_ts - 2:
                    continue
            except FileNotFoundError:
                continue
            try:
                with p.open("r", encoding="utf-8") as f:
                    first = f.readline().strip()
                obj = json.loads(first)
                if obj.get("type") == "session_meta":
                    sid = obj.get("payload", {}).get("id")
                    if isinstance(sid, str) and sid:
                        return sid, p
            except Exception:
                continue
        time.sleep(0.2)
    return None


def read_jsonl_from_offset(path: Path, offset: int, *, max_bytes: int) -> tuple[list[dict[str, Any]], int]:
    try:
        with path.open("rb") as f:
            f.seek(offset)
            data = f.read(int(max_bytes))
            new_off = f.tell()
    except FileNotFoundError:
        return [], offset

    lines = data.splitlines()
    out: list[dict[str, Any]] = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out, new_off

