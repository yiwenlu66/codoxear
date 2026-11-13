from __future__ import annotations

import hashlib
import os
import secrets
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from .git_ops import path_json_text
from .system_dependencies import resolve_executable


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        raw = os.environ.get(name.replace("MEDIA_PREVIEW", "VIDEO_PREVIEW"))
    if raw is None or not str(raw).strip():
        return default
    try:
        value = int(str(raw).strip())
    except ValueError:
        return default
    return max(0, value)


MEDIA_PREVIEW_CACHE_MAX_FILES = _positive_int_env("CODEX_WEB_MEDIA_PREVIEW_MAX_FILES", 256)
MEDIA_PREVIEW_CACHE_MAX_BYTES = _positive_int_env("CODEX_WEB_MEDIA_PREVIEW_MAX_BYTES", 10 * 1024 * 1024 * 1024)
MEDIA_PREVIEW_FAILURE_TTL_SECONDS = _positive_int_env("CODEX_WEB_MEDIA_PREVIEW_FAILURE_TTL_SECONDS", 15)
MEDIA_PREVIEW_FAILURE_MAX_ENTRIES = _positive_int_env("CODEX_WEB_MEDIA_PREVIEW_FAILURE_MAX_ENTRIES", 512)


MEDIA_PREVIEW_KINDS = {"video": {"suffix": ".mp4", "content_type": "video/mp4"}, "audio": {"suffix": ".mp3", "content_type": "audio/mpeg"}}


class _PreviewLock:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.refs = 0


_PREVIEW_LOCKS_GUARD = threading.Lock()
_PREVIEW_LOCKS: dict[Path, _PreviewLock] = {}
_PREVIEW_FAILURES_GUARD = threading.Lock()
_PREVIEW_FAILURES: dict[Path, tuple[float, str]] = {}


def media_response_payload(
    *,
    path_obj: Path,
    size: int,
    content_type: str | None,
    kind: str,
    media_url: str,
    preview_url: str,
    rel: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": True,
        "kind": kind,
        "content_type": content_type or "application/octet-stream",
        "preview_content_type": MEDIA_PREVIEW_KINDS[kind]["content_type"],
        "path": path_json_text(path_obj),
        "size": int(size),
        "media_url": media_url,
        "media_preview_url": preview_url,
    }
    if rel is not None:
        payload["rel"] = str(rel)
    return payload


def media_preview_path(path: Path, *, kind: str, preview_dir: Path) -> Path:
    st = path.stat()
    payload = f"{path.resolve()}\0{int(st.st_size)}\0{int(st.st_mtime_ns)}".encode("utf-8", errors="surrogateescape")
    return preview_dir / f"{hashlib.sha256(payload).hexdigest()}{MEDIA_PREVIEW_KINDS[kind]['suffix']}"


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
    cap = max(0, int(MEDIA_PREVIEW_FAILURE_MAX_ENTRIES))
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
    ttl = max(0, int(MEDIA_PREVIEW_FAILURE_TTL_SECONDS))
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
    ttl = max(0, int(MEDIA_PREVIEW_FAILURE_TTL_SECONDS))
    cap = max(0, int(MEDIA_PREVIEW_FAILURE_MAX_ENTRIES))
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


def _generate_media_preview(path: Path, out: Path, *, kind: str, preview_dir: Path) -> Path:
    ffmpeg = resolve_executable("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for compatible media previews")
    preview_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.stem}.{os.getpid()}.{secrets.token_hex(8)}.tmp{out.suffix}")
    unlink_quiet(tmp)
    if kind == "audio":
        # MP3 is the one lossy format every target client decodes, including the
        # iOS WebKit clients that cannot read Ogg/Vorbis, Ogg/Opus or WMA.
        stream_args = ["-vn", "-map", "0:a:0", "-c:a", "libmp3lame", "-q:a", "2", "-f", "mp3"]
    else:
        stream_args = [
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            # yuv420p needs even dimensions; odd source sizes gain one padding row/column.
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
        ]
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(path), *stream_args, str(tmp)]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(err or f"ffmpeg exited with code {proc.returncode}")
        if not tmp.exists() or tmp.stat().st_size <= 0:
            raise RuntimeError("ffmpeg produced an empty preview")
        os.replace(tmp, out)
        prune_media_preview_cache(preview_dir, keep=out)
        return out
    finally:
        unlink_quiet(tmp)


def ensure_media_preview(path: Path, *, kind: str, preview_dir: Path) -> Path:
    out = media_preview_path(path, kind=kind, preview_dir=preview_dir)
    if _preview_ready(out):
        prune_media_preview_cache(preview_dir, keep=out)
        return out
    entry = _acquire_preview_lock(out)
    try:
        if _preview_ready(out):
            prune_media_preview_cache(preview_dir, keep=out)
            return out
        cached_failure = _cached_preview_failure(out)
        if cached_failure:
            raise RuntimeError(f"recent media preview generation failed: {cached_failure}")
        try:
            generated = _generate_media_preview(path, out, kind=kind, preview_dir=preview_dir)
        except RuntimeError as exc:
            _remember_preview_failure(out, exc)
            raise
        _clear_preview_failure(out)
        return generated
    finally:
        _release_preview_lock(out, entry)


def prune_media_preview_cache(
    preview_dir: Path,
    *,
    keep: Path | None = None,
    max_files: int | None = None,
    max_bytes: int | None = None,
) -> None:
    file_cap = MEDIA_PREVIEW_CACHE_MAX_FILES if max_files is None else max(0, int(max_files))
    byte_cap = MEDIA_PREVIEW_CACHE_MAX_BYTES if max_bytes is None else max(0, int(max_bytes))
    if not preview_dir.exists():
        return
    try:
        keep_resolved = keep.resolve() if keep is not None else None
    except OSError:
        keep_resolved = keep
    entries: list[tuple[float, int, Path, bool]] = []
    total = 0
    for path in preview_dir.iterdir():
        if path.suffix not in {item["suffix"] for item in MEDIA_PREVIEW_KINDS.values()}:
            continue
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
