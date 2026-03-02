from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


CODEX_CLI = "codex"
CLAUDE_CLI = "claude"
GEMINI_CLI = "gemini"
SUPPORTED_CLIS = (CODEX_CLI, CLAUDE_CLI, GEMINI_CLI)

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


def normalize_cli_name(raw: str | None, *, default: str = CODEX_CLI) -> str:
    s = (raw or "").strip().lower()
    if not s:
        return default
    if s in ("gemini", "google-gemini", "gemini-cli", "gemini_cli"):
        return GEMINI_CLI
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
    if c == GEMINI_CLI:
        raw = os.environ.get("GEMINI_HOME")
        if raw is None or (not raw.strip()):
            return Path.home() / ".gemini"
        return Path(raw)
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
    if c == GEMINI_CLI:
        return home / "tmp"
    if c == CLAUDE_CLI:
        return home / "projects"
    return home / "sessions"


def cli_bin(cli: str) -> str:
    c = normalize_cli_name(cli, default=CODEX_CLI)
    if c == GEMINI_CLI:
        raw = os.environ.get("GEMINI_BIN")
        if raw is not None and raw.strip():
            return raw.strip()
        return "gemini"
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


def is_gemini_chat_log_path(path: Path, *, gemini_tmp_dir: Path | None = None) -> bool:
    if path.suffix != ".json":
        return False
    if not path.name.startswith("session-"):
        return False
    if path.parent.name != "chats":
        return False
    if gemini_tmp_dir is None:
        return "/.gemini/tmp/" in str(path).replace("\\", "/")
    try:
        path.resolve().relative_to(gemini_tmp_dir.resolve())
    except Exception:
        return False
    return True


def infer_cli_from_log_path(path: Path) -> str | None:
    if is_codex_rollout_log_path(path):
        return CODEX_CLI
    if is_gemini_chat_log_path(path, gemini_tmp_dir=cli_logs_dir(GEMINI_CLI)):
        return GEMINI_CLI
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
    if c == GEMINI_CLI:
        return read_gemini_session_id(path)
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


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _text_parts(value: Any) -> list[str]:
    out: list[str] = []
    if isinstance(value, str):
        t = value.strip()
        return [t] if t else []
    if isinstance(value, dict):
        t = value.get("text")
        if isinstance(t, str) and t.strip():
            out.append(t)
        return out
    if not isinstance(value, list):
        return out
    for part in value:
        if isinstance(part, str):
            t = part.strip()
            if t:
                out.append(t)
            continue
        if not isinstance(part, dict):
            continue
        t = part.get("text")
        if isinstance(t, str) and t.strip():
            out.append(t)
    return out


def _gemini_user_text(msg: dict[str, Any]) -> str | None:
    joined = "".join(_text_parts(msg.get("content")))
    if joined:
        return joined
    t = msg.get("text")
    if isinstance(t, str) and t.strip():
        return t
    return None


def _gemini_assistant_text(msg: dict[str, Any]) -> str | None:
    joined = "".join(_text_parts(msg.get("content")))
    if joined:
        return joined
    t = msg.get("text")
    if isinstance(t, str) and t.strip():
        return t
    return None


def _gemini_assistant_thinking_count(msg: dict[str, Any]) -> int:
    thoughts = msg.get("thoughts")
    if isinstance(thoughts, list) and thoughts:
        return 1
    tok = msg.get("tokens")
    if isinstance(tok, dict):
        th = tok.get("thoughts")
        if isinstance(th, (int, float)) and float(th) > 0:
            return 1
    return 0


def _gemini_assistant_tool_use_count(msg: dict[str, Any]) -> int:
    tool_uses = msg.get("toolUses")
    if isinstance(tool_uses, list) and tool_uses:
        return 1
    tok = msg.get("tokens")
    if isinstance(tok, dict):
        tv = tok.get("tool")
        if isinstance(tv, (int, float)) and float(tv) > 0:
            return 1
    return 0


def _gemini_assistant_is_turn_end(msg: dict[str, Any], *, text: str | None) -> bool:
    if isinstance(text, str) and text.strip():
        return True
    status = msg.get("status")
    if isinstance(status, str):
        sv = status.strip().lower()
        if sv in ("done", "complete", "completed", "final", "finished", "end"):
            return True
    done = msg.get("done")
    if isinstance(done, bool):
        return done
    final = msg.get("final")
    if isinstance(final, bool):
        return final
    is_final = msg.get("isFinal")
    if isinstance(is_final, bool):
        return is_final
    return False


def read_gemini_session_id(path: Path) -> str | None:
    obj = _read_json_object(path)
    if not isinstance(obj, dict):
        return None
    sid = obj.get("sessionId")
    if isinstance(sid, str) and _UUID_RE.fullmatch(sid or ""):
        return sid
    return None


def read_gemini_log_cwd(path: Path) -> str | None:
    cands: list[Path] = []
    cands.append(path.parent / ".project_root")
    cands.append(path.parent.parent / ".project_root")
    seen: set[str] = set()
    for cand in cands:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        try:
            raw = cand.read_text("utf-8").strip()
        except Exception:
            continue
        if raw:
            return raw
    obj = _read_json_object(path)
    if isinstance(obj, dict):
        cwd = obj.get("cwd")
        if isinstance(cwd, str) and cwd:
            return cwd
    return None


def read_gemini_rollout_objs(path: Path) -> list[dict[str, Any]]:
    obj = _read_json_object(path)
    if not isinstance(obj, dict):
        return []
    msgs = obj.get("messages")
    if not isinstance(msgs, list):
        return []
    out: list[dict[str, Any]] = []
    for msg in msgs:
        if not isinstance(msg, dict):
            continue
        typ = str(msg.get("type") or "").strip().lower()
        ts = msg.get("timestamp")
        ts_val = ts if isinstance(ts, str) and ts else None
        if typ == "user":
            text = _gemini_user_text(msg)
            if not text:
                continue
            row: dict[str, Any] = {
                "type": "user",
                "message": {"content": [{"type": "text", "text": text}]},
            }
            if ts_val is not None:
                row["timestamp"] = ts_val
            out.append(row)
            continue
        if typ not in ("gemini", "assistant", "model"):
            continue
        text = _gemini_assistant_text(msg)
        turn_end = _gemini_assistant_is_turn_end(msg, text=text)
        content: list[dict[str, Any]] = []
        if text:
            content.append({"type": "text", "text": text})
        if _gemini_assistant_thinking_count(msg) > 0:
            content.append({"type": "thinking"})
        if _gemini_assistant_tool_use_count(msg) > 0:
            content.append({"type": "tool_use", "name": "gemini_tool"})
        if not content:
            continue
        row2: dict[str, Any] = {
            "type": "assistant",
            "message": {"content": content},
        }
        if turn_end:
            row2["_gemini_turn_end"] = True
        if ts_val is not None:
            row2["timestamp"] = ts_val
        out.append(row2)
    return out


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
