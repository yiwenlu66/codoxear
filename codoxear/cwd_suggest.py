from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any


CWD_SUGGEST_LIMIT = 50
CWD_SUGGEST_TIMEOUT_SECONDS = 2.0
_CWD_SUGGEST_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cwd-suggest")


@dataclass(frozen=True)
class CwdSuggestions:
    """Immediate child directories of one path, plus whether the cap hid some.

    ``truncated`` is part of the contract, not diagnostics: the New Session cwd
    field completes the typed fragment from exactly this list, and a longest
    common prefix over a capped list can name a directory that does not exist.
    A caller that sees ``truncated`` may only complete to a name it received.
    """

    directories: list[dict[str, str]]
    truncated: bool


def _list_child_directories(path: str, *, prefix: str) -> CwdSuggestions:
    base = Path(path or "/").expanduser()
    include_hidden = prefix.startswith(".")
    folded_prefix = prefix.casefold()
    names: list[str] = []
    try:
        with os.scandir(base) as entries:
            for entry in entries:
                name = entry.name
                if name.startswith(".") and not include_hidden:
                    continue
                # Case-insensitive prefix match, the same rule the dialog applies
                # to the rendered rows. Filtering here (before the cap) is what
                # makes the cap count *matching* names, so a large directory
                # cannot hide the fragment the user is completing.
                if folded_prefix and not name.casefold().startswith(folded_prefix):
                    continue
                if any(0xDC80 <= ord(char) <= 0xDCFF for char in name):
                    continue
                try:
                    if not entry.is_dir():
                        continue
                except OSError:
                    continue
                names.append(name)
    except (OSError, ValueError):
        return CwdSuggestions(directories=[], truncated=False)
    # Deterministic order: the cap must always hide the same tail, otherwise a
    # repeat request could return a different subset for the same fragment.
    names.sort(key=str.casefold)
    truncated = len(names) > CWD_SUGGEST_LIMIT
    directories = [{"name": name, "path": str(base / name)} for name in names[:CWD_SUGGEST_LIMIT]]
    return CwdSuggestions(directories=directories, truncated=truncated)


def cwd_suggestions(path: Any, *, prefix: Any = "") -> CwdSuggestions:
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
    except (OSError, ValueError):
        pass
    return CwdSuggestions(directories=[], truncated=False)
