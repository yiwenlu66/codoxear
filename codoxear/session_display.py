"""The one definition of a session's displayed name.

A session is shown by its alias when it has one, otherwise by the basename of
its working directory. The listing row carries that value as `display_name`;
`app_display.js` renders it, and the tmux window mirror titles windows with it,
so a window is identifiable by exactly the name the sidebar shows.

Leaf module on purpose: the listing layer and the launch-attempt ledger both
need it, and neither should import the other.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Mapping


def sidebar_display_name(row: Mapping[str, Any]) -> str:
    """The sidebar's name for one row: trimmed alias, else the cwd basename.

    Empty when the row names neither; the client then falls back to a timestamp
    label and the tmux mirror leaves the window's launch name alone.
    """
    alias = row.get("alias")
    if isinstance(alias, str) and alias.strip():
        return alias.strip()
    cwd = row.get("cwd")
    if isinstance(cwd, str) and cwd.strip():
        name = PurePosixPath(cwd.rstrip("/")).name
        if name:
            return name
    return ""
