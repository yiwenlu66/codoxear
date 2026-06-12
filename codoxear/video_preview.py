from __future__ import annotations

import hashlib
import os
import secrets
import shutil
import subprocess
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


def ensure_video_preview(path: Path, *, preview_dir: Path) -> Path:
    out = video_preview_path(path, preview_dir=preview_dir)
    if out.exists() and out.stat().st_size > 0:
        prune_video_preview_cache(preview_dir, keep=out)
        return out
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
