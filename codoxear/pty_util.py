from __future__ import annotations

import codecs
import fcntl
import os
import re
import struct
import termios
import time


_ESCAPE_ONLY_RE = re.compile(
    r"(?:\\[\\'\"abfnrtv]|\\x[0-9A-Fa-f]{2}|\\u[0-9A-Fa-f]{4}|\\U[0-9A-Fa-f]{8}|\\N\{[^}]+\}|\\[0-7]{1,3})+\Z"
)
_BRACKETED_PASTE_START = b"\x1b[200~"
_BRACKETED_PASTE_END = b"\x1b[201~"


def write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        n = os.write(fd, view)
        if n <= 0:
            raise OSError("short write to PTY")
        view = view[n:]


def inject_bracketed_paste(fd: int, *, text: str, suffix: bytes, delay_s: float = 0.05) -> None:
    write_all(fd, _BRACKETED_PASTE_START + text.encode("utf-8") + _BRACKETED_PASTE_END)
    if not suffix:
        return
    if delay_s > 0:
        time.sleep(delay_s)
    write_all(fd, suffix)


def set_winsize(fd: int, rows: int, cols: int) -> None:
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    ws = struct.pack("HHHH", rows, cols, 0, 0)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, ws)


def seq_bytes(raw: str) -> bytes:
    t = raw.strip().upper()
    if t in ("NONE", "EMPTY", "NOENTER", "NO_ENTER"):
        return b""
    if t in ("ESC", "ESCAPE"):
        return b"\x1b"
    if t in ("ENTER", "CR"):
        return b"\r"
    if t in ("LF",):
        return b"\n"
    if t in ("CRLF",):
        return b"\r\n"
    if not _ESCAPE_ONLY_RE.fullmatch(raw):
        return raw.encode("utf-8")
    try:
        decoded = codecs.decode(raw, "unicode_escape")
    except Exception:
        return raw.encode("utf-8")
    return decoded.encode("utf-8")
