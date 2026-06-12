from __future__ import annotations

import hashlib
import os
import secrets
from pathlib import Path

from .file_types import MARKDOWN_EXTENSIONS
from .file_types import TEXTUAL_EXTENSIONS
from .file_types import TEXTUAL_FILENAMES

FILE_READ_MAX_BYTES = int(os.environ.get("CODEX_WEB_FILE_READ_MAX_BYTES", str(2 * 1024 * 1024)))


def read_text_file_strict(path: Path, *, max_bytes: int) -> tuple[str, int]:
    st = path.stat()
    size = int(st.st_size)
    if size > max_bytes:
        raise ValueError(f"file too large (max {max_bytes} bytes)")
    data = path.read_bytes()
    if b"\x00" in data:
        raise ValueError("binary file not supported")
    text = data.decode("utf-8", errors="replace")
    return text, size


def file_content_version(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_extension(path: Path) -> str:
    suffix = str(path.suffix or "").lower()
    if not suffix.startswith("."):
        return ""
    return suffix[1:]


def markdown_kind(path: Path) -> str:
    return "markdown" if file_extension(path) in MARKDOWN_EXTENSIONS else "text"


def path_looks_textual(path: Path) -> bool:
    ext = file_extension(path)
    if ext in TEXTUAL_EXTENSIONS:
        return True
    return str(path.name or "").strip().lower() in TEXTUAL_FILENAMES


def looks_like_text_bytes(raw: bytes) -> bool:
    if b"\x00" in raw:
        return False
    for b in raw:
        if b < 32 and b not in (9, 10, 12, 13, 27):
            return False
    return True


def decode_text_for_client(raw: bytes) -> tuple[str, bool]:
    try:
        return raw.decode("utf-8"), True
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace"), False


def decode_text_view_for_client(path: Path, raw: bytes) -> tuple[str, bool, str] | None:
    if b"\x00" in raw:
        return None
    try:
        text = raw.decode("utf-8")
        editable = True
    except UnicodeDecodeError:
        if not path_looks_textual(path) and not looks_like_text_bytes(raw):
            return None
        text = raw.decode("utf-8", errors="replace")
        editable = False
    return text, editable, file_content_version(raw)


def read_text_file_for_client(path: Path, *, max_bytes: int) -> tuple[str, int, bool, str]:
    st = path.stat()
    size = int(st.st_size)
    if size > max_bytes:
        raise ValueError(f"file too large (max {max_bytes} bytes)")
    data = path.read_bytes()
    if b"\x00" in data:
        raise ValueError("binary file not supported")
    text, editable = decode_text_for_client(data)
    return text, size, editable, file_content_version(data)


def read_text_file_for_write(path: Path, *, max_bytes: int) -> tuple[str, int, str]:
    st = path.stat()
    size = int(st.st_size)
    if size > max_bytes:
        raise ValueError(f"file too large (max {max_bytes} bytes)")
    data = path.read_bytes()
    if b"\x00" in data:
        raise ValueError("binary file not supported")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError("file is not editable as utf-8 text") from e
    return text, size, file_content_version(data)


def write_text_file_atomic(path: Path, *, text: str, max_bytes: int = FILE_READ_MAX_BYTES) -> tuple[int, str]:
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    if path.is_symlink():
        raise ValueError("symlink file not supported")
    data = text.encode("utf-8")
    size = len(data)
    if size > max_bytes:
        raise ValueError(f"file too large (max {max_bytes} bytes)")
    st = path.stat()
    tmp = path.with_name(f".{path.name}.codoxear-tmp-{secrets.token_hex(6)}")
    try:
        tmp.write_bytes(data)
        os.chmod(tmp, st.st_mode & 0o777)
        os.replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
    return size, file_content_version(data)


def write_new_text_file_atomic(path: Path, *, text: str, max_bytes: int = FILE_READ_MAX_BYTES) -> tuple[int, str]:
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    if path.is_symlink():
        raise ValueError("symlink file not supported")
    parent = path.parent
    if not parent.exists():
        raise FileNotFoundError("parent directory not found")
    if not parent.is_dir():
        raise ValueError("parent path is not a directory")
    if parent.is_symlink():
        raise ValueError("symlink parent directory not supported")
    if path.exists():
        raise FileExistsError("file already exists")
    data = text.encode("utf-8")
    size = len(data)
    if size > max_bytes:
        raise ValueError(f"file too large (max {max_bytes} bytes)")
    tmp = path.with_name(f".{path.name}.codoxear-tmp-{secrets.token_hex(6)}")
    try:
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.link(str(tmp), str(path))
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
    return size, file_content_version(data)
