from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


CODEX_CLI = "codex"
CLAUDE_CLI = "claude"
SUPPORTED_CLIS = (CODEX_CLI, CLAUDE_CLI)

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


def normalize_cli_name(raw: str | None, *, default: str = CODEX_CLI) -> str:
    s = (raw or "").strip().lower()
    if not s:
        return default
    if s in ("claude", "claude-code", "claude_code"):
        return CLAUDE_CLI
    if s in ("codex", "openai-codex", "codex-cli"):
        return CODEX_CLI
    return default


def parse_cli_name(raw: str | None, *, default: str = CODEX_CLI) -> str:
    s = (raw or "").strip()
    if not s:
        return default
    out = normalize_cli_name(s, default="")
    if out not in SUPPORTED_CLIS:
        raise ValueError(f"unsupported cli: {raw}")
    return out


def default_cli_name() -> str:
    return normalize_cli_name(os.environ.get("CODEX_WEB_DEFAULT_CLI"), default=CODEX_CLI)


def cli_home(cli: str) -> Path:
    c = normalize_cli_name(cli, default=CODEX_CLI)
    if c == CLAUDE_CLI:
        raw = os.environ.get("CLAUDE_HOME")
        if raw is None or (not raw.strip()):
            return Path.home() / ".claude"
        return Path(raw)
    raw = os.environ.get("CODEX_HOME")
    if raw is None or (not raw.strip()):
        return Path.home() / ".codex"
    return Path(raw)


def cli_logs_dir(cli: str) -> Path:
    c = normalize_cli_name(cli, default=CODEX_CLI)
    home = cli_home(c)
    if c == CLAUDE_CLI:
        return home / "projects"
    return home / "sessions"


def cli_bin(cli: str) -> str:
    c = normalize_cli_name(cli, default=CODEX_CLI)
    if c == CLAUDE_CLI:
        raw = os.environ.get("CLAUDE_BIN")
        if raw is not None and raw.strip():
            return raw.strip()
        return "claude"
    raw = os.environ.get("CODEX_BIN")
    if raw is not None and raw.strip():
        return raw.strip()
    return "codex"


def is_codex_rollout_log_path(path: Path) -> bool:
    name = path.name
    return name.startswith("rollout-") and name.endswith(".jsonl")


def is_claude_subagent_log_path(path: Path) -> bool:
    return "subagents" in path.parts


def is_claude_project_log_path(path: Path, *, claude_projects_dir: Path | None = None) -> bool:
    if path.suffix != ".jsonl":
        return False
    if is_claude_subagent_log_path(path):
        return False
    if not _UUID_RE.fullmatch(path.stem or ""):
        return False
    if claude_projects_dir is None:
        return "/.claude/projects/" in str(path)
    try:
        path.resolve().relative_to(claude_projects_dir.resolve())
    except Exception:
        return False
    return True


def infer_cli_from_log_path(path: Path) -> str | None:
    if is_codex_rollout_log_path(path):
        return CODEX_CLI
    if is_claude_project_log_path(path, claude_projects_dir=cli_logs_dir(CLAUDE_CLI)):
        return CLAUDE_CLI
    if _UUID_RE.fullmatch(path.stem or "") and path.suffix == ".jsonl" and not is_claude_subagent_log_path(path):
        return CLAUDE_CLI
    return None


def session_id_from_log_path(path: Path, *, cli: str | None = None) -> str | None:
    c = normalize_cli_name(cli, default="")
    if not c:
        guessed = infer_cli_from_log_path(path)
        c = guessed if guessed is not None else CODEX_CLI
    if c == CLAUDE_CLI:
        sid = path.stem
        return sid if _UUID_RE.fullmatch(sid or "") else None
    # Codex rollout filename contains session id as trailing UUID.
    m = re.findall(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", path.name, re.I)
    return m[-1] if m else None


def read_claude_log_cwd(path: Path, *, max_bytes: int = 256 * 1024) -> str | None:
    try:
        with path.open("rb") as f:
            data = f.read(int(max_bytes))
    except Exception:
        return None
    for raw in data.splitlines():
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        cwd = obj.get("cwd")
        if isinstance(cwd, str) and cwd:
            return cwd
    return None


def claude_user_text(obj: dict[str, Any]) -> str | None:
    if obj.get("type") != "user":
        return None
    msg = obj.get("message")
    if isinstance(msg, str):
        t = msg.strip()
        return t if t else None
    if not isinstance(msg, dict):
        return None
    content = msg.get("content")
    if isinstance(content, str):
        t = content.strip()
        return t if t else None
    if isinstance(content, list):
        out: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") != "text":
                continue
            text = part.get("text")
            if not isinstance(text, str) or (not text.strip()):
                continue
            out.append(text)
        if out:
            return "".join(out)
    return None


def claude_assistant_content_parts(obj: dict[str, Any]) -> list[dict[str, Any]]:
    if obj.get("type") != "assistant":
        return []
    msg = obj.get("message")
    if not isinstance(msg, dict):
        return []
    content = msg.get("content")
    if not isinstance(content, list):
        return []
    return [part for part in content if isinstance(part, dict)]


def claude_assistant_text(obj: dict[str, Any]) -> str | None:
    out: list[str] = []
    for part in claude_assistant_content_parts(obj):
        if part.get("type") != "text":
            continue
        text = part.get("text")
        if isinstance(text, str) and text:
            out.append(text)
    if out:
        return "".join(out)
    return None


def claude_assistant_tool_use_count(obj: dict[str, Any]) -> int:
    n = 0
    for part in claude_assistant_content_parts(obj):
        if part.get("type") == "tool_use":
            n += 1
    return n


def claude_assistant_thinking_count(obj: dict[str, Any]) -> int:
    n = 0
    for part in claude_assistant_content_parts(obj):
        if part.get("type") == "thinking":
            n += 1
    return n

