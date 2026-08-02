from __future__ import annotations

from pathlib import Path

from .base import AgentBackend
from .cc import ClaudeCodeBackend
from .codex import CodexBackend
from .pi import PiBackend


CODEX_BACKEND = CodexBackend(
    name="codex",
    bin_env_var="CODEX_BIN",
    home_env_var="CODEX_HOME",
    default_bin="codex",
    default_home_dirname=".codex",
    sessions_relpath=("sessions",),
)

PI_BACKEND = PiBackend(
    name="pi",
    bin_env_var="PI_BIN",
    home_env_var="PI_HOME",
    default_bin="pi",
    default_home_dirname=".pi",
    sessions_relpath=("agent", "sessions"),
)

CC_BACKEND = ClaudeCodeBackend(
    name="cc",
    bin_env_var="CLAUDE_BIN",
    home_env_var="CLAUDE_CONFIG_DIR",
    default_bin="claude",
    default_home_dirname=".claude",
    sessions_relpath=("projects",),
)

_BACKENDS: dict[str, AgentBackend] = {
    CODEX_BACKEND.name: CODEX_BACKEND,
    PI_BACKEND.name: PI_BACKEND,
    CC_BACKEND.name: CC_BACKEND,
}


def normalize_agent_backend(value: object, *, default: str = "codex") -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        raw = default
    if raw not in _BACKENDS:
        allowed = ", ".join(sorted(_BACKENDS))
        raise ValueError(f"agent_backend must be one of {allowed}")
    return raw


def get_agent_backend(value: object, *, default: str = "codex") -> AgentBackend:
    return _BACKENDS[normalize_agent_backend(value, default=default)]


def infer_agent_backend_from_log_path(path: Path) -> str | None:
    if CODEX_BACKEND.is_session_log_path(path):
        return "codex"
    if PI_BACKEND.is_session_log_path(path):
        return "pi"
    if CC_BACKEND.is_session_log_path(path):
        return "cc"
    if CC_BACKEND.is_session_log_path(path, sessions_dir=CC_BACKEND.sessions_dir()):
        return "cc"
    return None


__all__ = [
    "AgentBackend",
    "CodexBackend",
    "PiBackend",
    "ClaudeCodeBackend",
    "CODEX_BACKEND",
    "PI_BACKEND",
    "CC_BACKEND",
    "get_agent_backend",
    "normalize_agent_backend",
    "infer_agent_backend_from_log_path",
]
