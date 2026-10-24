"""Shared executable discovery for runtime features and the explicit doctor CLI.

PATH wins. Known user/native installation locations follow, so a narrow service
PATH can still find Homebrew and user-installed tools. No shell is executed.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
from typing import Callable


def resolve_executable(
    name: str,
    *,
    path: str | None = None,
    platform: str | None = None,
    home: str | Path | None = None,
    which: Callable[..., str | None] = shutil.which,
) -> str | None:
    """Resolve using current PATH then known native locations; return None if absent.

    Keyword overrides are injection seams for isolated tests, not configuration.
    LibreOffice's macOS app bundle is supported without requiring a shell alias.
    """
    platform = sys.platform if platform is None else platform
    home = Path.home() if home is None else Path(home)
    search = os.environ.get("PATH", "") if path is None else path
    fallbacks = [str(home / ".local/bin"), str(home / "bin")]
    if platform in {"linux", "darwin"}:
        fallbacks += ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin"]
    if platform == "linux":
        fallbacks += [str(home / ".linuxbrew/bin"), "/home/linuxbrew/.linuxbrew/bin"]
    # An entirely empty/unset PATH must not introduce a new implicit cwd entry.
    # Nonempty caller PATHs retain their normal component semantics.
    resolved = which(name, path=os.pathsep.join([*([search] if search else []), *fallbacks]))
    if resolved is None and platform == "darwin" and name == "soffice":
        resolved = which("/Applications/LibreOffice.app/Contents/MacOS/soffice")
    return resolved
