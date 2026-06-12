from __future__ import annotations

import os
from pathlib import Path
from typing import Callable


def safe_filename(name: str, *, default: str = "file") -> str:
    base = Path(str(name or "")).name
    out = []
    for ch in base:
        if ch.isalnum() or ch in ("-", "_", ".", " "):
            out.append(ch)
    s = "".join(out).strip().replace(" ", "_")
    if not s:
        return default
    return s[:96]


def stage_uploaded_file(
    session_id: str,
    filename: str,
    raw: bytes,
    *,
    upload_dir: Path,
    now_fn: Callable[[], float],
    max_bytes: int,
) -> Path:
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id required")
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("filename required")
    if not isinstance(raw, (bytes, bytearray)):
        raise ValueError("file bytes required")
    data = bytes(raw)
    if len(data) > int(max_bytes):
        raise ValueError(f"file too large (max {int(max_bytes)} bytes)")
    safe_name = safe_filename(filename, default="file")
    subdir = (upload_dir / session_id).resolve()
    subdir.mkdir(parents=True, exist_ok=True)
    out_path = (subdir / f"{int(now_fn() * 1000)}_{safe_name}").resolve()
    if not str(out_path).startswith(str(subdir) + os.sep):
        raise ValueError("bad path")
    out_path.write_bytes(data)
    os.chmod(out_path, 0o600)
    return out_path


def attachment_inject_text(attachment_index: int, path: Path) -> str:
    idx = int(attachment_index)
    if idx <= 0:
        raise ValueError("attachment_index must be >= 1")
    return f"Attachment {idx}: {path}\n"
