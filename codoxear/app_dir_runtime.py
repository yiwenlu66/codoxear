from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppDirResolution:
    app_dir: Path
    legacy_warned: bool
    warning: str | None = None


def resolve_default_app_dir(*, legacy_warned: bool, home: Path | None = None) -> AppDirResolution:
    base = (home or Path.home()) / ".local" / "share"
    new = base / "codoxear"
    old = base / "codex-web"
    warning = None
    if old.exists() and not legacy_warned:
        legacy_warned = True
        warning = (
            f"error: legacy runtime dir detected at {old}; it is no longer used. "
            f"migrate runtime state to {new}."
        )
    return AppDirResolution(app_dir=new, legacy_warned=legacy_warned, warning=warning)
