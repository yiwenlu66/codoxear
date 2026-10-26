"""Quick-session workspaces: one numbered directory per quick launch.

A quick session is created on a single browser gesture with no directory chosen
by the user. It gets its own workspace under ``~/codoxear-quick`` so two quick
launches cannot share a cwd, and it gets a slot number that names both the
directory (``quick-07``) and the sidebar label (``Quick Session 07``).

The slot is the identity. It is allocated monotonically from a counter in the
app dir and never reused, so a name always refers to exactly one workspace and
two live sessions can never collide. The counter file is advisory: the
directories are the floor (``next_slot = max(counter, highest existing slot+1)``),
so losing the counter can skip numbers but can never allocate a slot whose
directory already holds work.

Overflow past 99 is not an error: ``f"{slot:02d}"`` is a minimum width, so slot
100 renders ``Quick Session 100`` / ``quick-100``. Only lexicographic ``ls``
order breaks there (``quick-100`` sorts before ``quick-99``); nothing sorts
sessions by name, and ``ls -v`` restores natural order.

This module never creates the workspace directory. ``prepare_launch_plan``
already creates a missing cwd for any launch, so an allocation that never
reaches a launch costs a number and no filesystem state.
"""

from __future__ import annotations

import os
import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

from .json_state import atomic_write_json
from .json_state import load_json_file


QUICK_DIR_NAME = "codoxear-quick"
QUICK_STATE_FILENAME = "quick_sessions.json"

_SLOT_DIR_RE = re.compile(r"^quick-(\d+)$")
# One server process allocates; the lock covers read-modify-write of the counter.
_ALLOCATION_LOCK = threading.Lock()


def quick_root() -> Path:
    """The single quick-workspace root. Hardcoded by design: no env knob."""
    return Path.home() / QUICK_DIR_NAME


def quick_state_path(app_dir: Path) -> Path:
    return Path(app_dir) / QUICK_STATE_FILENAME


def quick_dir_name(slot: int) -> str:
    return f"quick-{int(slot):02d}"


def quick_display_name(slot: int) -> str:
    return f"Quick Session {int(slot):02d}"


def quick_cwd(slot: int, *, root: Path | None = None) -> Path:
    return (root or quick_root()) / quick_dir_name(slot)


@lru_cache(maxsize=8)
def _root_prefixes_cached(root_text: str) -> tuple[str, ...]:
    candidates = [root_text]
    try:
        candidates.append(str(Path(root_text).resolve()))
    except OSError:  # pragma: no cover - resolve is best-effort
        pass
    prefixes: list[str] = []
    for candidate in candidates:
        normalised = os.path.normpath(candidate)
        if not normalised or normalised == os.sep:
            continue
        prefixes.append(normalised.rstrip(os.sep) + os.sep)
    return tuple(dict.fromkeys(prefixes))


def _root_prefixes(root: Path | None) -> tuple[str, ...]:
    return _root_prefixes_cached(str(root or quick_root()))


def _normalised(cwd: Any) -> str:
    if not isinstance(cwd, str):
        return ""
    text = cwd.strip()
    if not text:
        return ""
    return os.path.normpath(text)


def is_quick_cwd(cwd: Any, *, root: Path | None = None) -> bool:
    """True when ``cwd`` lives anywhere under the quick-workspace root.

    Compares normalised paths, and also accepts the root's resolved form, so a
    symlinked ``~/codoxear-quick`` still matches the resolved cwd a session
    actually records.
    """
    text = _normalised(cwd)
    if not text or text in {".", os.sep}:
        return False
    return any(text.startswith(prefix) for prefix in _root_prefixes(root))


def slot_from_cwd(cwd: Any, *, root: Path | None = None) -> int | None:
    """The slot whose own directory is exactly ``cwd``, else ``None``."""
    text = _normalised(cwd)
    if not text:
        return None
    for prefix in _root_prefixes(root):
        if not text.startswith(prefix):
            continue
        leaf = text[len(prefix):]
        if not leaf or os.sep in leaf or (os.altsep and os.altsep in leaf):
            return None
        match = _SLOT_DIR_RE.match(leaf)
        if match:
            return int(match.group(1))
    return None


def quick_display_name_for_cwd(cwd: Any, *, root: Path | None = None) -> str:
    slot = slot_from_cwd(cwd, root=root)
    return quick_display_name(slot) if slot is not None else ""


def _load_next_slot(state_path: Path) -> int:
    # The counter is advisory: a missing, unreadable, or malformed file falls
    # back to 1, and `_highest_existing_slot` keeps that from colliding with a
    # directory that already holds work.
    try:
        raw = load_json_file(state_path, default=None)
    except (OSError, ValueError):
        return 1
    if isinstance(raw, dict):
        value = raw.get("next_slot")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return 1


def _save_next_slot(state_path: Path, value: int) -> None:
    atomic_write_json(state_path, {"next_slot": int(value)})


def _highest_existing_slot(root: Path | None) -> int:
    base = root or quick_root()
    try:
        entries = list(base.iterdir())
    except OSError:
        return 0
    highest = 0
    for entry in entries:
        match = _SLOT_DIR_RE.match(entry.name)
        if match is None:
            continue
        try:
            if not entry.is_dir():
                continue
        except OSError:  # pragma: no cover - race with a concurrent removal
            continue
        highest = max(highest, int(match.group(1)))
    return highest


def allocate(*, state_path: Path, root: Path | None = None) -> dict[str, Any]:
    """Reserve the next slot. Creates no directory; spawns nothing."""
    with _ALLOCATION_LOCK:
        slot = max(_load_next_slot(state_path), _highest_existing_slot(root) + 1, 1)
        _save_next_slot(state_path, slot + 1)
    return {
        "slot": slot,
        "name": quick_display_name(slot),
        "cwd": str(quick_cwd(slot, root=root)),
    }


def release_workspace(cwd: Any, *, root: Path | None = None) -> str:
    """Tidy a quick workspace whose session was just deleted.

    Removes the directory when it is empty (``rmdir`` semantics, never
    recursive) and returns ``""``; returns the path when it still holds files so
    the caller can report it. Slots are never reused, so trimming an empty
    directory frees no number — it only keeps the root free of empty reminders.
    """
    if slot_from_cwd(cwd, root=root) is None:
        return ""
    path = Path(str(cwd))
    try:
        if not path.is_dir():
            return ""
    except OSError:  # pragma: no cover - defensive
        return ""
    try:
        path.rmdir()
    except OSError:
        return str(path)
    return ""
