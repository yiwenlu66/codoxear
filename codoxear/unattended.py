from __future__ import annotations

from pathlib import Path
from typing import Any

from .util import atomic_write_json
from .util import load_json_file


class UnattendedStore:
    def __init__(self, *, path: Path, default_idle_minutes: int, default_max_injections: int) -> None:
        self.path = path
        self.default_idle_minutes = int(default_idle_minutes)
        self.default_max_injections = int(default_max_injections)

    def clean_cooldown_minutes(self, raw: Any) -> int:
        return clean_unattended_cooldown_minutes(raw, default_idle_minutes=self.default_idle_minutes)

    def clean_remaining_injections(self, raw: Any, *, allow_zero: bool) -> int:
        return clean_unattended_remaining_injections(raw, default_max_injections=self.default_max_injections, allow_zero=allow_zero)

    def load(self) -> dict[str, dict[str, Any]]:
        obj = load_json_file(self.path, default={})
        if not isinstance(obj, dict):
            raise ValueError("invalid unattended.json (expected object)")
        cleaned: dict[str, dict[str, Any]] = {}
        for sid, v in obj.items():
            if not isinstance(sid, str) or not sid:
                continue
            if not isinstance(v, dict):
                continue
            enabled = bool(v.get("enabled")) if "enabled" in v else False
            if "text" in v:
                raise ValueError(f"invalid unattended config for session {sid!r} (use 'request', not 'text')")
            request = v.get("request")
            if request is None:
                request = ""
            if not isinstance(request, str):
                raise ValueError(f"invalid unattended request for session {sid!r}")
            cooldown_minutes = self.clean_cooldown_minutes(v.get("cooldown_minutes"))
            remaining_injections = self.clean_remaining_injections(v.get("remaining_injections"), allow_zero=True)
            cleaned[sid] = {
                "enabled": enabled,
                "request": request,
                "cooldown_minutes": cooldown_minutes,
                "remaining_injections": remaining_injections,
            }
        return cleaned

    def save(self, obj: dict[str, dict[str, Any]]) -> None:
        atomic_write_json(self.path, obj)


def render_unattended_prompt(request: str | None, *, prompt_prefix: str) -> str:
    base = prompt_prefix.rstrip()
    r = (request or "").strip()
    if not r:
        return base + "\n"
    return base + "\n\n---\n\nAdditional request from user: " + r + "\n"


def clean_unattended_cooldown_minutes(raw: Any, *, default_idle_minutes: int) -> int:
    if raw is None:
        return int(default_idle_minutes)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError("unattended cooldown_minutes must be an integer")
    if raw < 1:
        raise ValueError("unattended cooldown_minutes must be at least 1")
    return raw


def clean_unattended_remaining_injections(raw: Any, *, default_max_injections: int, allow_zero: bool) -> int:
    if raw is None:
        return int(default_max_injections)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError("unattended remaining_injections must be an integer")
    minimum = 0 if allow_zero else 1
    if raw < minimum:
        lower = "0" if allow_zero else "1"
        raise ValueError(f"unattended remaining_injections must be at least {lower}")
    return raw
