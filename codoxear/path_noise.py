"""Directory names that are never a place a session or a file belongs.

Suggestions are a *shortlist*, not a directory dump: a cwd picker that offers
`__pycache__` and `.git` beside `src` wastes the reader's attention on names no
one will ever choose. These are tool and build artifacts; the field still accepts
them typed by hand.

Exact names only — never a substring or suffix rule, so a project that happens to
be called `environment` or `targets` keeps its place in the list.
"""

from __future__ import annotations

NOISE_DIRECTORY_NAMES = frozenset(
    {
        "__pycache__",
        "__pypackages__",
        "node_modules",
        "site-packages",
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        ".gradle",
        ".next",
        ".nuxt",
        ".turbo",
        ".parcel-cache",
        ".cache",
        ".idea",
    }
)


def is_noise_directory(name: str) -> bool:
    """True when this exact name is a build/tooling artifact directory."""
    return str(name or "") in NOISE_DIRECTORY_NAMES


def path_enters_noise_directory(path: str) -> bool:
    """True when any component of this path is an artifact directory.

    Suggested listings stop at the artifact boundary: nothing *inside*
    `__pycache__`, `node_modules` or `.git` is a candidate either, so navigating
    into one yields no suggestions instead of a wall of generated files.
    """
    parts = [part for part in str(path or "").replace("\\", "/").split("/") if part and part not in (".", "..")]
    return any(is_noise_directory(part) for part in parts)
