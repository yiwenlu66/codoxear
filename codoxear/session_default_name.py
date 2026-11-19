"""The name a session gets when nobody has chosen one.

A codoxear session always has a name. The user's alias is that name when they
set one; otherwise codoxear assigns a default (``session_ui_state``) and projects
it onto the backend's own title (``session_name_projection``), so the backend
never keeps an empty session name while the sidebar shows something else. That
gap is what made a Pi session present itself as a generated
``subagent-chat-<session id>`` label to intercom and to delegated subagents, and
what left Codex threads and Claude Code resume rows untitled.

The default is the working directory's basename — the same string the sidebar
already fell back to — except for a codoxear-managed quick workspace, whose label
(``Quick Session 07``) the launch already declares. Only a *live* name that would
duplicate it earns a ``-N`` suffix: two sessions in the same directory become
``repo`` and ``repo-2`` instead of two indistinguishable ``repo`` rows.
Suffixes are assigned once and never renumbered, so a name does not drift while
sessions come and go.

Leaf module: the alias store and the name projector both need it, and neither
should import the other.
"""

from __future__ import annotations

from typing import Any, Iterable

from .quick_workspace import quick_display_name_for_cwd
from .session_display import cwd_display_basename


def default_session_name(cwd: Any, taken: Iterable[Any]) -> str:
    """The default name for a session in ``cwd``, avoiding every name in ``taken``.

    ``taken`` holds the names already in use by live sessions. Matching is
    case-insensitive, because two names that differ only in case read as one
    row in any sidebar. An empty cwd yields no name: there is nothing to derive.
    """
    base = quick_display_name_for_cwd(cwd) or cwd_display_basename(cwd)
    if not base:
        return ""
    claimed = {
        str(name).strip().casefold()
        for name in taken
        if isinstance(name, str) and name.strip()
    }
    if base.casefold() not in claimed:
        return base
    index = 2
    while f"{base}-{index}".casefold() in claimed:
        index += 1
    return f"{base}-{index}"
