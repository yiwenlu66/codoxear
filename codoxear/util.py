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


def _read_session_meta_payload_once(log_path: Path, *, max_bytes: int) -> dict[str, Any] | None:
    try:
        with log_path.open("rb") as f:
            data = f.read(int(max_bytes))
    except FileNotFoundError:
        return None
    except Exception:
        return None

    for raw in data.splitlines():
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        if obj.get("type") != "session_meta":
            continue
        payload = obj.get("payload") or {}
        return payload if isinstance(payload, dict) else {}
    return None


def read_session_meta_payload(
    log_path: Path,
    *,
    timeout_s: float = 0.0,
    poll_s: float = 0.05,
    max_bytes: int = 64 * 1024,
) -> dict[str, Any] | None:
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


def classify_session_log(log_path: Path, *, timeout_s: float = 0.0) -> str | None:
    payload = read_session_meta_payload(log_path, timeout_s=timeout_s)
    if payload is None:
        return None
    return "subagent" if is_subagent_session_meta(payload) else "main"


def iter_session_logs(sessions_dir: Path) -> list[Path]:
    if not sessions_dir.exists():
        return []
    out: list[tuple[float, Path]] = []
    for p in sessions_dir.rglob("rollout-*.jsonl"):
        try:
            mt = float(p.stat().st_mtime)
        except FileNotFoundError:
            continue
        except Exception:
            mt = 0.0
        out.append((mt, p))
    out.sort(key=lambda t: t[0], reverse=True)
    return [p for _mt, p in out]


def find_session_log_for_session_id(sessions_dir: Path, session_id: str) -> Path | None:
    if not session_id:
        return None
    for p in iter_session_logs(sessions_dir):
        if session_id in p.name:
            return p
    return None


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
                payload = read_session_meta_payload(p, timeout_s=0.0)
                if not payload:
                    continue
                if is_subagent_session_meta(payload):
                    continue
                sid = payload.get("id")
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
