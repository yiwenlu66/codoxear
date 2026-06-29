from __future__ import annotations

from contextlib import contextmanager
import errno
import hashlib
import os
import secrets
import stat
from pathlib import Path
from typing import BinaryIO, Iterator

from .file_types import MARKDOWN_EXTENSIONS
from .file_types import TEXTUAL_EXTENSIONS
from .file_types import TEXTUAL_FILENAMES

FILE_READ_MAX_BYTES = int(os.environ.get("CODEX_WEB_FILE_READ_MAX_BYTES", str(2 * 1024 * 1024)))


def _dir_open_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


def _read_open_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)


def _write_open_flags() -> int:
    return os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


def _path_leaf_name(path: Path) -> str:
    name = path.name
    if not name:
        raise ValueError("path is not a file")
    return name


def _map_parent_open_error(exc: OSError, *, parent_fd: int, component: str) -> Exception:
    if exc.errno == errno.ENOENT:
        return FileNotFoundError("parent directory not found")
    if exc.errno == errno.ELOOP:
        return ValueError("symlink parent directory not supported")
    if exc.errno == errno.ENOTDIR:
        try:
            st = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
        except OSError:
            st = None
        if st is not None and stat.S_ISLNK(st.st_mode):
            return ValueError("symlink parent directory not supported")
        return ValueError("parent path is not a directory")
    return exc


@contextmanager
def _open_parent_dir_fd_no_symlink(path: Path) -> Iterator[int]:
    parent = Path(path).parent
    if parent.is_absolute():
        fd = os.open(os.sep, _dir_open_flags())
        components = parent.parts[1:]
    else:
        fd = os.open(".", _dir_open_flags())
        components = parent.parts
    try:
        for component in components:
            if component in {"", "."}:
                continue
            try:
                next_fd = os.open(component, _dir_open_flags(), dir_fd=fd)
            except OSError as exc:
                mapped = _map_parent_open_error(exc, parent_fd=fd, component=component)
                os.close(fd)
                fd = -1
                raise mapped from exc
            os.close(fd)
            fd = next_fd
        yield fd
    finally:
        if fd >= 0:
            os.close(fd)


def stat_path_no_symlink(path: Path) -> os.stat_result:
    path = Path(path)
    if path.name == "":
        try:
            return path.stat(follow_symlinks=False)
        except FileNotFoundError:
            raise FileNotFoundError("file not found")
    name = _path_leaf_name(path)
    with _open_parent_dir_fd_no_symlink(path) as parent_fd:
        try:
            return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            raise FileNotFoundError("file not found")
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise ValueError("symlink file not supported") from exc
            raise


@contextmanager
def open_regular_file_no_symlink(path: Path) -> Iterator[tuple[BinaryIO, os.stat_result]]:
    name = _path_leaf_name(Path(path))
    with _open_parent_dir_fd_no_symlink(path) as parent_fd:
        try:
            pre_open_st = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            raise FileNotFoundError("file not found")
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise ValueError("symlink file not supported") from exc
            raise
        if stat.S_ISLNK(pre_open_st.st_mode):
            raise ValueError("symlink file not supported")
        if not stat.S_ISREG(pre_open_st.st_mode):
            raise ValueError("path is not a file")
        fd: int | None = None
        try:
            fd = os.open(name, _read_open_flags(), dir_fd=parent_fd)
            opened_st = os.fstat(fd)
            if not stat.S_ISREG(opened_st.st_mode):
                raise ValueError("path is not a file")
            with os.fdopen(fd, "rb") as fh:
                fd = None
                yield fh, opened_st
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise ValueError("symlink file not supported") from exc
            if exc.errno == errno.ENOENT:
                raise FileNotFoundError("file not found") from exc
            if exc.errno == errno.EACCES:
                raise PermissionError("permission denied") from exc
            raise
        finally:
            if fd is not None:
                os.close(fd)


def read_regular_file_prefix_no_symlink(path: Path, byte_count: int) -> tuple[bytes, int]:
    with open_regular_file_no_symlink(path) as (fh, st):
        return fh.read(byte_count), int(st.st_size)


def read_regular_file_bytes_no_symlink(path: Path, *, max_bytes: int | None = None) -> tuple[bytes, int]:
    with open_regular_file_no_symlink(path) as (fh, st):
        if max_bytes is not None and int(st.st_size) > max_bytes:
            raise ValueError(f"file too large (max {max_bytes} bytes)")
        data = fh.read()
    if max_bytes is not None and len(data) > max_bytes:
        raise ValueError(f"file too large (max {max_bytes} bytes)")
    return data, len(data)


def _write_tmp_file(parent_fd: int, tmp_name: str, data: bytes, mode: int, *, chmod_mode: int | None = None) -> None:
    fd: int | None = None
    try:
        fd = os.open(tmp_name, _write_open_flags(), mode, dir_fd=parent_fd)
        with os.fdopen(fd, "wb") as fh:
            fd = None
            fh.write(data)
            fh.flush()
            if chmod_mode is not None:
                os.fchmod(fh.fileno(), chmod_mode)
    finally:
        if fd is not None:
            os.close(fd)


def _unlink_child_quiet(parent_fd: int, name: str) -> None:
    try:
        os.unlink(name, dir_fd=parent_fd)
    except FileNotFoundError:
        return
    except OSError:
        return


def _existing_regular_child_stat(parent_fd: int, name: str) -> os.stat_result:
    try:
        st = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        raise FileNotFoundError("file not found")
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError("symlink file not supported") from exc
        raise
    if stat.S_ISLNK(st.st_mode):
        raise ValueError("symlink file not supported")
    if not stat.S_ISREG(st.st_mode):
        raise ValueError("path is not a file")
    return st


def _raise_if_child_exists(parent_fd: int, name: str) -> None:
    try:
        st = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError("symlink file not supported") from exc
        raise
    if stat.S_ISLNK(st.st_mode):
        raise ValueError("symlink file not supported")
    raise FileExistsError("file already exists")


def read_text_file_strict(path: Path, *, max_bytes: int) -> tuple[str, int]:
    data, size = read_regular_file_bytes_no_symlink(path, max_bytes=max_bytes)
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
    data, size = read_regular_file_bytes_no_symlink(path, max_bytes=max_bytes)
    if b"\x00" in data:
        raise ValueError("binary file not supported")
    text, editable = decode_text_for_client(data)
    return text, size, editable, file_content_version(data)


def read_text_file_for_write(path: Path, *, max_bytes: int) -> tuple[str, int, str]:
    data, size = read_regular_file_bytes_no_symlink(path, max_bytes=max_bytes)
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
    name = _path_leaf_name(Path(path))
    data = text.encode("utf-8")
    size = len(data)
    if size > max_bytes:
        raise ValueError(f"file too large (max {max_bytes} bytes)")
    tmp_name = f".{name}.codoxear-tmp-{secrets.token_hex(6)}"
    with _open_parent_dir_fd_no_symlink(path) as parent_fd:
        st = _existing_regular_child_stat(parent_fd, name)
        try:
            mode = st.st_mode & 0o777
            _write_tmp_file(parent_fd, tmp_name, data, mode, chmod_mode=mode)
            os.replace(tmp_name, name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        finally:
            _unlink_child_quiet(parent_fd, tmp_name)
    return size, file_content_version(data)


def write_new_text_file_atomic(path: Path, *, text: str, max_bytes: int = FILE_READ_MAX_BYTES) -> tuple[int, str]:
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    name = _path_leaf_name(Path(path))
    data = text.encode("utf-8")
    size = len(data)
    if size > max_bytes:
        raise ValueError(f"file too large (max {max_bytes} bytes)")
    tmp_name = f".{name}.codoxear-tmp-{secrets.token_hex(6)}"
    with _open_parent_dir_fd_no_symlink(path) as parent_fd:
        _raise_if_child_exists(parent_fd, name)
        try:
            _write_tmp_file(parent_fd, tmp_name, data, 0o666)
            os.link(tmp_name, name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        finally:
            _unlink_child_quiet(parent_fd, tmp_name)
    return size, file_content_version(data)
