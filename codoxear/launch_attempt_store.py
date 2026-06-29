from __future__ import annotations

import json
import re
import uuid
from collections import deque
from pathlib import Path
from typing import Any


_SENSITIVE_LAUNCH_FIELD_RE = re.compile(r"(?:TOKEN|SECRET|KEY|PASSWORD|CREDENTIAL|AUTH)", re.IGNORECASE)
_LAUNCH_ERROR_RESPONSE_FIELDS = {
    "type",
    "launch_id",
    "spawn_nonce",
    "state",
    "stage",
    "error",
    "message",
    "agent_backend",
    "cwd",
    "transport",
    "tmux_session",
    "tmux_window",
    "model_provider",
    "provider_choice",
    "preferred_auth_method",
    "model",
    "reasoning_effort",
    "service_tier",
    "created_ts",
    "updated_ts",
    "agent_exit_status",
    "broker_exit_status",
}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return str(value)


def redact_launch_failure_text(value: Any, *, strip: bool = True) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip() if strip else value
    if not text:
        return ""
    sensitive_key = r"[A-Z0-9_.-]*(?:TOKEN|SECRET|KEY|PASSWORD|CREDENTIAL|AUTH)[A-Z0-9_.-]*"
    secret_value = r"(?:(?:Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+|\"[^\"]*(?:\"|$)|'[^']*(?:'|$)|[^\s\"',;}\[\]]+)"
    text = re.sub(
        rf"\b({sensitive_key})\s*=\s*{secret_value}",
        r"\1=[redacted]",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        rf"(^|[^A-Z0-9_.-])([\"']?{sensitive_key}[\"']?\s*:\s*){secret_value}",
        r"\1\2[redacted]",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        rf"(^|[^A-Z0-9_.-])([\"']?{sensitive_key}[\"']?\s*[:=]\s*)\[redacted\]\s+[A-Za-z0-9._~+/=-]{{12,}}(?=$|[\s,;}}\]])",
        r"\1\2[redacted]",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+", r"\1 [redacted]", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(sk-[A-Za-z0-9_-]{12,}|xox[baprs]-[A-Za-z0-9-]{12,})\b", "[redacted-token]", text)
    return text


def redact_launch_failure_value(value: Any, *, key: str = "", strip: bool = True) -> Any:
    if isinstance(value, str):
        redacted = redact_launch_failure_text(value, strip=strip)
        if _SENSITIVE_LAUNCH_FIELD_RE.search(key) and redacted == value and value:
            return "[redacted]"
        return redacted
    if isinstance(value, dict):
        return {str(k): redact_launch_failure_value(v, key=str(k), strip=strip) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_launch_failure_value(v, key=key, strip=strip) for v in value]
    if isinstance(value, tuple):
        return [redact_launch_failure_value(v, key=key, strip=strip) for v in value]
    return value


def redacted_launch_attempt_persist_record(record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    safe = redact_launch_failure_value(record, strip=False)
    return safe if isinstance(safe, dict) else {}


def redacted_launch_attempt_response_record(record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    return {
        key: redact_launch_failure_value(record[key], key=key)
        for key in _LAUNCH_ERROR_RESPONSE_FIELDS
        if key in record
    }


def append_launch_attempt(record: dict[str, Any], *, path: Path, now_ts: float) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise TypeError("launch attempt record must be a dict")
    ts = float(now_ts)
    out = _jsonable(record)
    if not isinstance(out, dict):
        raise TypeError("launch attempt record must remain a dict after normalization")
    out.setdefault("type", "launch_attempt")
    out.setdefault("launch_id", f"launch-{int(ts * 1000)}-{uuid.uuid4().hex[:8]}")
    out.setdefault("state", "starting")
    out.setdefault("updated_ts", ts)
    out.setdefault("created_ts", out.get("updated_ts", ts))
    target = path
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(out, sort_keys=True) + "\n")
    return out


def read_launch_attempts(
    *,
    path: Path,
    max_records: int = 200,
    max_age_s: float = 24 * 3600,
    now_ts: float,
) -> list[dict[str, Any]]:
    target = path
    try:
        with target.open("r", encoding="utf-8") as f:
            lines = deque(f, maxlen=max(max_records * 4, max_records))
    except FileNotFoundError:
        return []
    cutoff = float(now_ts) - max(float(max_age_s), 0.0)
    latest: dict[str, dict[str, Any]] = {}
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or obj.get("type") != "launch_attempt":
            continue
        launch_id = obj.get("launch_id")
        if not isinstance(launch_id, str) or not launch_id:
            continue
        ts = obj.get("updated_ts", obj.get("created_ts"))
        if isinstance(ts, (int, float)) and float(ts) < cutoff:
            continue
        latest[launch_id] = obj
    out = list(latest.values())
    out.sort(key=lambda item: float(item.get("updated_ts", item.get("created_ts", 0.0)) or 0.0))
    return out[-max_records:]
