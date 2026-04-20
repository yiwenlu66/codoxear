from __future__ import annotations

import codecs
import fcntl
import re
import struct
import termios

_ESCAPE_ONLY_RE = re.compile(
    r"(?:\\[\\'\"abfnrtv]|\\x[0-9A-Fa-f]{2}|\\u[0-9A-Fa-f]{4}|\\U[0-9A-Fa-f]{8}|\\N\{[^}]+\}|\\[0-7]{1,3})+\Z"
)


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
