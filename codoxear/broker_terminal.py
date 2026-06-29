from __future__ import annotations

import traceback
from collections.abc import Callable


_TERMINAL_QUERY_RESPONSES: tuple[tuple[bytes, bytes], ...] = (
    (b"\x1b[5n", b"\x1b[0n"),
    (b"\x1b[6n", b"\x1b[1;1R"),
    (b"\x1b[c", b"\x1b[?1;2c"),
    (b"\x1b[>c", b"\x1b[>0;0;0c"),
    (b"\x1b[?u", b"\x1b[?1u"),
    (b"\x1b]10;?\x1b\\", b"\x1b]10;rgb:c0c0/c0c0/c0c0\x1b\\"),
    (b"\x1b]11;?\x1b\\", b"\x1b]11;rgb:0000/0000/0000\x1b\\"),
)


def _reply_to_terminal_queries(
    *,
    term_query_buf: bytes,
    fd: int,
    chunk: bytes,
    write_all: Callable[[int, bytes], None],
    max_buffer: int = 256,
) -> bytes:
    buf = (term_query_buf + chunk)[-max_buffer:]
    for query, response in _TERMINAL_QUERY_RESPONSES:
        if query not in buf:
            continue
        try:
            write_all(fd, response)
        except Exception:
            traceback.print_exc()
        buf = buf.replace(query, b"")
    return buf
