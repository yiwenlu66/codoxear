from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AgentBackend:
    name: str
    bin_env_var: str
    home_env_var: str
    default_bin: str
    default_home_dirname: str
    sessions_relpath: tuple[str, ...]

    def cli_bin(self, env: dict[str, str] | None = None) -> str:
        env_map = os.environ if env is None else env
        value = str(env_map.get(self.bin_env_var) or "").strip()
        return value or self.default_bin

    def home(self, env: dict[str, str] | None = None) -> Path:
        env_map = os.environ if env is None else env
        raw = str(env_map.get(self.home_env_var) or "").strip()
        if raw:
            return Path(raw).expanduser()
        return Path.home() / self.default_home_dirname

    def sessions_dir(self, env: dict[str, str] | None = None) -> Path:
        return self.home(env).joinpath(*self.sessions_relpath)


CODEX_BACKEND = AgentBackend(
    name="codex",
    bin_env_var="CODEX_BIN",
    home_env_var="CODEX_HOME",
    default_bin="codex",
    default_home_dirname=".codex",
    sessions_relpath=("sessions",),
)

PI_BACKEND = AgentBackend(
    name="pi",
    bin_env_var="PI_BIN",
    home_env_var="PI_HOME",
    default_bin="pi",
    default_home_dirname=".pi",
    sessions_relpath=("agent", "sessions"),
)

CLAUDE_BACKEND = AgentBackend(
    name="claude",
    bin_env_var="CLAUDE_BIN",
    home_env_var="CLAUDE_HOME",
    default_bin="claude",
    default_home_dirname=".claude",
    sessions_relpath=("projects",),
)

_BACKENDS: dict[str, AgentBackend] = {
    CODEX_BACKEND.name: CODEX_BACKEND,
    PI_BACKEND.name: PI_BACKEND,
    CLAUDE_BACKEND.name: CLAUDE_BACKEND,
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
    name = path.name
    if name.startswith("rollout-") and name.endswith(".jsonl"):
        return "codex"
    path_text = str(path).replace("\\", "/")
    if "/.pi/agent/sessions/" in path_text and name.endswith(".jsonl"):
        return "pi"
    if "/.claude/projects/" in path_text and name.endswith(".jsonl"):
        return "claude"
    return None

