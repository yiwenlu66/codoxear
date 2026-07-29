from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
import os
from pathlib import Path
from typing import Any


CWD_SUGGEST_LIMIT = 50
CWD_SUGGEST_TIMEOUT_SECONDS = 2.0
_CWD_SUGGEST_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cwd-suggest")


def _list_child_directories(path: str, *, prefix: str) -> list[dict[str, str]]:
    base = Path(path or "/").expanduser()
    include_hidden = prefix.startswith(".")
    directories: list[dict[str, str]] = []
    try:
        with os.scandir(base) as entries:
            for entry in entries:
                if len(directories) >= CWD_SUGGEST_LIMIT:
                    break
                if entry.name.startswith(".") and not include_hidden:
                    continue
                if any(0xDC80 <= ord(char) <= 0xDCFF for char in entry.name):
                    continue
                try:
                    if not entry.is_dir():
                        continue
                except OSError:
                    continue
                directories.append({"name": entry.name, "path": str(base / entry.name)})
    except (OSError, ValueError):
        return []
    return directories


def cwd_suggestions(path: Any, *, prefix: Any = "") -> list[dict[str, str]]:
    """Return up to 50 immediate directory children without stalling a request.

    Filesystem metadata can block on an unavailable mount. The bounded worker lets
    the HTTP request return an empty list after two seconds while the New Session
    dialog remains usable.
    """
    path_text = str(path or "")
    prefix_text = str(prefix or "")
    future = _CWD_SUGGEST_EXECUTOR.submit(_list_child_directories, path_text, prefix=prefix_text)
    try:
        return future.result(timeout=CWD_SUGGEST_TIMEOUT_SECONDS)
    except FutureTimeoutError:
        future.cancel()
        return []
    except (OSError, ValueError):
        return []
