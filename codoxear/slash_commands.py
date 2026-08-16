from __future__ import annotations

from typing import Any, Mapping

# Pi's built-in commands are intentionally split: commands below are safe to
# send as text from a browser; interactive UI commands are omitted.
PI_BUILTIN_SLASH_COMMANDS = (
    {"name": "model", "description": "Select model (opens selector UI)"},
    {"name": "new", "description": "Start a new session"},
    {"name": "share", "description": "Share session as a secret GitHub gist"},
    {"name": "copy", "description": "Copy last agent message to clipboard"},
    {"name": "name", "description": "Set session display name"},
    {"name": "session", "description": "Show session info and stats"},
    {"name": "changelog", "description": "Show changelog entries"},
    {"name": "hotkeys", "description": "Show all keyboard shortcuts"},
    {"name": "logout", "description": "Remove provider authentication"},
    {"name": "compact", "description": "Manually compact the session context"},
    {"name": "reload", "description": "Reload extensions, skills, prompts, themes, and context files"},
)
# Interactive TUI commands are omitted from the browser menu. Session-
# lifecycle commands other than /new (resume, fork, clone) stay excluded: they
# rebind the transcript like /new does, but their TUI flows pick targets
# interactively, which the browser cannot drive.
PI_INTERACTIVE_SLASH_COMMANDS = (
    "settings", "model", "scoped-models", "import", "tree", "login",
    "resume", "quit", "fork", "clone", "trust",
)
CC_SLASH_COMMANDS = (
    {"name": "model", "description": "Select model"},
    {"name": "effort", "description": "Set reasoning effort"},
    {"name": "compact", "description": "Compact conversation context"},
    {"name": "cost", "description": "Show session cost and token usage"},
    {"name": "status", "description": "Show session status"},
    {"name": "doctor", "description": "Diagnose Claude Code installation"},
)


def _clean_commands(value: Any) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    if not isinstance(value, (list, tuple)):
        return out
    for item in value:
        if isinstance(item, Mapping):
            raw_name, raw_description = item.get("name"), item.get("description", "")
        else:
            continue
        name = str(raw_name or "").strip().lstrip("/").lower()
        description = str(raw_description or "").strip()
        if not name or name in seen or any(ch.isspace() for ch in name):
            continue
        seen.add(name)
        out.append({"name": name, "description": description})
    return out


def default_slash_commands(backend: str, *, pi_bridge_capable: bool = False) -> list[dict[str, str]]:
    backend = str(backend or "").strip().lower()
    if backend == "pi":
        commands = list(PI_BUILTIN_SLASH_COMMANDS)
        if pi_bridge_capable:
            commands += [
                {"name": "effort", "description": "Set the reasoning effort (thinking level) for the current model"},
                {"name": "thinking", "description": "Alias for /effort"},
            ]
        return _clean_commands(commands)
    if backend == "cc":
        return _clean_commands(CC_SLASH_COMMANDS)
    return []


def slash_commands_for_backend(backend: str, caps: Any = None, *, pi_bridge_capable: bool = False) -> list[dict[str, str]]:
    """Project commands the browser can invoke from a live backend session.

    Codex capabilities are emitted only by a broker that successfully attached
    its typed app-server control transport; preserving those rows makes the
    browser picker available. Claude Code has a stable browser-safe command
    set, while Pi filters its live extension registry for commands whose TUI
    flow is interactive.
    """
    backend = str(backend or "").strip().lower()
    if backend == "codex":
        return _clean_commands(caps)
    if backend == "cc":
        return default_slash_commands("cc")
    if backend != "pi":
        return []
    if pi_bridge_capable and isinstance(caps, (list, tuple)):
        return [command for command in _clean_commands(caps) if command["name"] not in PI_INTERACTIVE_SLASH_COMMANDS or command["name"] == "model"]
    return default_slash_commands("pi", pi_bridge_capable=False)
