from __future__ import annotations

import hashlib
import os
import secrets
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = int(str(raw).strip())
    except ValueError:
        return default
    return max(0, value)


VIDEO_PREVIEW_CACHE_MAX_FILES = _positive_int_env("CODEX_WEB_VIDEO_PREVIEW_MAX_FILES", 256)
VIDEO_PREVIEW_CACHE_MAX_BYTES = _positive_int_env("CODEX_WEB_VIDEO_PREVIEW_MAX_BYTES", 10 * 1024 * 1024 * 1024)
VIDEO_PREVIEW_FAILURE_TTL_SECONDS = _positive_int_env("CODEX_WEB_VIDEO_PREVIEW_FAILURE_TTL_SECONDS", 15)
VIDEO_PREVIEW_FAILURE_MAX_ENTRIES = _positive_int_env("CODEX_WEB_VIDEO_PREVIEW_FAILURE_MAX_ENTRIES", 512)


class _PreviewLock:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.refs = 0


_PREVIEW_LOCKS_GUARD = threading.Lock()
_PREVIEW_LOCKS: dict[Path, _PreviewLock] = {}
_PREVIEW_FAILURES_GUARD = threading.Lock()
_PREVIEW_FAILURES: dict[Path, tuple[float, str]] = {}


def video_response_payload(
    *,
    path_obj: Path,
    size: int,
    content_type: str | None,
    video_url: str,
    preview_url: str,
    rel: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": True,
        "kind": "video",
        "content_type": content_type or "application/octet-stream",
        "preview_content_type": "video/mp4",
        "path": str(path_obj),
        "size": int(size),
        "video_url": video_url,
        "video_preview_url": preview_url,
    }
    if rel is not None:
        payload["rel"] = str(rel)
    return payload


def video_preview_path(path: Path, *, preview_dir: Path) -> Path:
    st = path.stat()
    payload = f"{path.resolve()}\0{int(st.st_size)}\0{int(st.st_mtime_ns)}".encode("utf-8", errors="surrogateescape")
    return preview_dir / f"{hashlib.sha256(payload).hexdigest()}.mp4"


def _preview_ready(path: Path) -> bool:
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


def _acquire_preview_lock(path: Path) -> _PreviewLock:
    with _PREVIEW_LOCKS_GUARD:
        entry = _PREVIEW_LOCKS.get(path)
        if entry is None:
            entry = _PreviewLock()
            _PREVIEW_LOCKS[path] = entry
        entry.refs += 1
    entry.lock.acquire()
    return entry


def _release_preview_lock(path: Path, entry: _PreviewLock) -> None:
    entry.lock.release()
    with _PREVIEW_LOCKS_GUARD:
        entry.refs -= 1
        if entry.refs <= 0 and _PREVIEW_LOCKS.get(path) is entry:
            _PREVIEW_LOCKS.pop(path, None)


def _prune_preview_failures_locked(now: float) -> None:
    expired = [path for path, (expires_at, _message) in _PREVIEW_FAILURES.items() if expires_at <= now]
    for path in expired:
        _PREVIEW_FAILURES.pop(path, None)
    cap = max(0, int(VIDEO_PREVIEW_FAILURE_MAX_ENTRIES))
    if cap <= 0:
        _PREVIEW_FAILURES.clear()
        return
    overflow = len(_PREVIEW_FAILURES) - cap
    if overflow <= 0:
        return
    oldest = sorted(_PREVIEW_FAILURES.items(), key=lambda item: (item[1][0], str(item[0])))[:overflow]
    for path, _cached in oldest:
        _PREVIEW_FAILURES.pop(path, None)


def _cached_preview_failure(path: Path) -> str | None:
    ttl = max(0, int(VIDEO_PREVIEW_FAILURE_TTL_SECONDS))
    if ttl <= 0:
        return None
    now = time.monotonic()
    with _PREVIEW_FAILURES_GUARD:
        _prune_preview_failures_locked(now)
        cached = _PREVIEW_FAILURES.get(path)
        if not cached:
            return None
        return cached[1]


def _remember_preview_failure(path: Path, exc: BaseException) -> None:
    ttl = max(0, int(VIDEO_PREVIEW_FAILURE_TTL_SECONDS))
    cap = max(0, int(VIDEO_PREVIEW_FAILURE_MAX_ENTRIES))
    if ttl <= 0 or cap <= 0:
        return
    message = str(exc).strip() or type(exc).__name__
    now = time.monotonic()
    with _PREVIEW_FAILURES_GUARD:
        _prune_preview_failures_locked(now)
        _PREVIEW_FAILURES[path] = (now + ttl, message)
        _prune_preview_failures_locked(now)


def _clear_preview_failure(path: Path) -> None:
    with _PREVIEW_FAILURES_GUARD:
        _PREVIEW_FAILURES.pop(path, None)


def _generate_video_preview(path: Path, out: Path, *, preview_dir: Path) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for compatible video previews")
    preview_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.stem}.{os.getpid()}.{secrets.token_hex(8)}.tmp.mp4")
    unlink_quiet(tmp)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-vf",
        "scale=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(tmp),
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(err or f"ffmpeg exited with code {proc.returncode}")
        if not tmp.exists() or tmp.stat().st_size <= 0:
            raise RuntimeError("ffmpeg produced an empty preview")
        os.replace(tmp, out)
        prune_video_preview_cache(preview_dir, keep=out)
        return out
    finally:
        unlink_quiet(tmp)


def ensure_video_preview(path: Path, *, preview_dir: Path) -> Path:
    out = video_preview_path(path, preview_dir=preview_dir)
    if _preview_ready(out):
        prune_video_preview_cache(preview_dir, keep=out)
        return out
    entry = _acquire_preview_lock(out)
    try:
        if _preview_ready(out):
            prune_video_preview_cache(preview_dir, keep=out)
            return out
        cached_failure = _cached_preview_failure(out)
        if cached_failure:
            raise RuntimeError(f"recent video preview generation failed: {cached_failure}")
        try:
            generated = _generate_video_preview(path, out, preview_dir=preview_dir)
        except RuntimeError as exc:
            _remember_preview_failure(out, exc)
            raise
        _clear_preview_failure(out)
        return generated
    finally:
        _release_preview_lock(out, entry)


def prune_video_preview_cache(
    preview_dir: Path,
    *,
    keep: Path | None = None,
    max_files: int | None = None,
    max_bytes: int | None = None,
) -> None:
    file_cap = VIDEO_PREVIEW_CACHE_MAX_FILES if max_files is None else max(0, int(max_files))
    byte_cap = VIDEO_PREVIEW_CACHE_MAX_BYTES if max_bytes is None else max(0, int(max_bytes))
    if not preview_dir.exists():
        return
    try:
        keep_resolved = keep.resolve() if keep is not None else None
    except OSError:
        keep_resolved = keep
    entries: list[tuple[float, int, Path, bool]] = []
    total = 0
    for path in preview_dir.glob("*.mp4"):
        if path.name.startswith("."):
            continue
        try:
            st = path.stat()
        except OSError:
            continue
        if not path.is_file():
            continue
        try:
            is_keep = keep_resolved is not None and path.resolve() == keep_resolved
        except OSError:
            is_keep = keep is not None and path == keep
        size = max(0, int(st.st_size))
        total += size
        entries.append((float(st.st_mtime), size, path, is_keep))
    entries.sort(key=lambda item: (item[0], item[2].name))
    count = len(entries)
    for _mtime, size, path, is_keep in entries:
        over_files = file_cap > 0 and count > file_cap
        over_bytes = byte_cap > 0 and total > byte_cap
        if not over_files and not over_bytes:
            break
        if is_keep:
            continue
        unlink_quiet(path)
        count -= 1
        total = max(0, total - size)


def unlink_quiet(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass
