from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import threading
from typing import Iterator


@contextmanager
def file_write_lock(
    path: Path,
    *,
    locks_lock: threading.Lock,
    locks: dict[str, tuple[threading.Lock, int]],
) -> Iterator[None]:
    key = str(path)
    with locks_lock:
        entry = locks.get(key)
        if entry is None:
            lock = threading.Lock()
            refcount = 0
        else:
            lock, refcount = entry
        locks[key] = (lock, refcount + 1)
    try:
        with lock:
            yield
    finally:
        with locks_lock:
            entry = locks.get(key)
            if entry is not None and entry[0] is lock:
                refcount = entry[1] - 1
                if refcount <= 0:
                    locks.pop(key, None)
                else:
                    locks[key] = (lock, refcount)
