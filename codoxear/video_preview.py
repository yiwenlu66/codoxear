from __future__ import annotations

import hashlib
import os
import secrets
import shutil
import subprocess
from pathlib import Path
from typing import Any


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
        return out
    finally:
        unlink_quiet(tmp)


def unlink_quiet(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass
