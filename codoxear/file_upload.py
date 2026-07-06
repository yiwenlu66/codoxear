from __future__ import annotations

import os
import shutil
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
    stamp = int(now_fn() * 1000)
    out_path = (subdir / f"{stamp}_{safe_name}").resolve()
    if not str(out_path).startswith(str(subdir) + os.sep):
        raise ValueError("bad path")
    counter = 2
    while out_path.exists():
        out_path = (subdir / f"{stamp}_{counter}_{safe_name}").resolve()
        if not str(out_path).startswith(str(subdir) + os.sep):
            raise ValueError("bad path")
        counter += 1
    out_path.write_bytes(data)
    os.chmod(out_path, 0o600)
    return out_path


def remove_session_uploads(upload_root: Path, session_id: str) -> bool:
    """Remove the per-session staged-attachment entry ``<upload_root>/<session_id>``.

    This is the cleanup counterpart to :func:`stage_uploaded_file`: when a session
    is deleted, the staged attachment bytes written under its session-scoped
    subdirectory must not outlive it. ``session_id`` is validated strictly so a
    blank or traversal-bearing id can never widen the removal to the upload root
    or escape it.

    Only the single ``<sid>`` directory entry is removed: a symlink entry is
    unlinked itself (never followed, so a tampered link resolving outside the
    upload root cannot delete or damage its target); a real directory is removed
    recursively; any other file entry is unlinked. Returns ``True`` when an entry
    was removed and ``False`` when the session had no staged-upload entry.
    """
    if not isinstance(upload_root, (str, Path)):
        raise ValueError("upload_root required")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id required")
    sid = session_id.strip()
    if sid in (".", "..") or "/" in sid or "\\" in sid or os.sep in sid:
        raise ValueError("invalid session_id")
    # Operate on the literal path; do NOT resolve() the target, because
    # resolving would follow a symlink and either widen the removal onto the
    # link's destination or (via the parent check below) refuse to clean up a
    # tampered entry, leaving stale bytes behind. ``sid`` carries no path
    # separators and is not ``.``/``..``, so ``root / sid`` is by construction a
    # direct child entry of ``root``; the parent check is a redundant guard.
    root = Path(upload_root)
    target = root / sid
    if target.parent != root:
        raise ValueError("invalid session_id")
    if target.is_symlink():
        # Unlink the link itself; never follow it, even if it resolves outside
        # the upload root.
        target.unlink()
        return True
    if not target.exists():
        return False
    if target.is_dir():
        shutil.rmtree(target)
    else:
        # A stray non-directory entry where the session dir should be is not a
        # valid staged-upload tree; remove it loudly rather than leaving bytes
        # behind or silently recursing into something unexpected.
        target.unlink()
    return True


def remove_staged_attachment_file(upload_root: Path, session_id: str, staged_path: str | Path) -> bool:
    """Remove one staged attachment file without following symlinks.

    The stored staged path must name a direct child of ``<upload_root>/<session_id>``.
    A symlink at that child path is unlinked as an entry; its target is never
    followed. Returns ``True`` when a file/link was removed and ``False`` when
    the stored entry was already absent.
    """
    if not isinstance(upload_root, (str, Path)):
        raise ValueError("upload_root required")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id required")
    sid = session_id.strip()
    if sid in (".", "..") or "/" in sid or "\\" in sid or os.sep in sid:
        raise ValueError("invalid session_id")
    target = Path(staged_path)
    if not target.is_absolute():
        raise ValueError("staged_path must be absolute")
    root = Path(upload_root)
    subdir = root / sid
    if subdir.is_symlink():
        raise ValueError("session upload directory is a symlink")
    subdir_resolved = subdir.resolve()
    parent_resolved = target.parent.resolve()
    if parent_resolved != subdir_resolved:
        raise ValueError("staged_path outside session uploads")
    if target.name in ("", ".", ".."):
        raise ValueError("invalid staged_path")
    if target.is_dir() and not target.is_symlink():
        raise ValueError("staged_path is not a file")
    try:
        target.unlink()
        return True
    except FileNotFoundError:
        return False


def attachment_inject_text(attachment_index: int, path: Path) -> str:
    idx = int(attachment_index)
    if idx <= 0:
        raise ValueError("attachment_index must be >= 1")
    return f"Attachment {idx}: {path}\n"
