from __future__ import annotations

import codecs
import fcntl
import struct
import termios


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
    try:
        decoded = codecs.decode(raw.encode("utf-8"), "unicode_escape")
        return decoded.encode("utf-8")
    except Exception:
        return raw.encode("utf-8", errors="replace")

