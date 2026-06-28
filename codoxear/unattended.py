from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .util import atomic_write_json
from .util import load_json_file


UNATTENDED_PROMPT_PREFIX = """Unattended-mode instructions (optimize for 8+ hours, minimal turns, minimal repetition, maximal progress)

- Maintain four internal sections:
  1. Deliverables
     - The concrete outputs the agent owes the user by the end of the task.
     - Stable unless the user changes the request.
  2. Completed
     - Verified facts already established while producing the Deliverables.
  3. Next actions
     - Ordered concrete steps from the current state toward the Deliverables.
  4. Parked user decisions
     - Decisions or inputs that only the user can provide.

- Working rules:
  - Keep these sections internal. Surface them only when yielding is necessary.
  - Default to continuing in the same turn.
  - Before each action, reason until the approach, failure modes, and verification path are clear.
  - Exploration should happen through reading, tracing, inspection, and reasoning.
  - Avoid trial and error.
  - Resolve crashes, bugs, and design mistakes yourself unless a true user decision is required.
  - Use the strongest available verification.
  - Do not repeat the same command, edit, or analysis without a concrete new reason.

- Yield only when:
  - all Deliverables are finished and supported by Completed;
  - the only remaining gap is a Parked user decision;
  - or the next step is irreversible or high-risk and needs explicit user confirmation.

- End-of-turn gate (only when yielding is necessary):
  - Run a clean-room adversarial review via a dedicated subagent.
  - Give it: user intent, Deliverables, Completed, remaining Next actions, Parked user decisions, constraints, and changed artifacts.
  - Apply findings before yielding, or surface the exact remaining user decision or risk.
"""



@dataclass(frozen=True)
class UnattendedConfigState:
    enabled: bool
    request: str
    cooldown_minutes: int
    cooldown_seconds: float
    remaining_injections: int


@dataclass(frozen=True)
class UnattendedPromptDecision:
    prompt: str
    cooldown_seconds: float
    config: dict[str, Any]
    disabled_exhausted: bool


@dataclass(frozen=True)
class UnattendedSuccessUpdate:
    config: dict[str, Any]
    remaining_injections: int
    enabled: bool


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


def unattended_config_state(
    config: dict[str, Any],
    *,
    default_idle_minutes: int,
    default_max_injections: int,
) -> UnattendedConfigState:
    cooldown_minutes = clean_unattended_cooldown_minutes(config.get("cooldown_minutes"), default_idle_minutes=default_idle_minutes)
    remaining_injections = clean_unattended_remaining_injections(
        config.get("remaining_injections"),
        default_max_injections=default_max_injections,
        allow_zero=True,
    )
    request = config.get("request")
    if not isinstance(request, str):
        request = ""
    return UnattendedConfigState(
        enabled=bool(config.get("enabled")),
        request=request,
        cooldown_minutes=cooldown_minutes,
        cooldown_seconds=float(cooldown_minutes * 60),
        remaining_injections=remaining_injections,
    )


def unattended_scope_key(*, thread_id: str | None, log_path: Path) -> str:
    return f"thread:{thread_id}" if thread_id else f"log:{str(log_path)}"


def unattended_cooldown_blocked(*, now_ts: float, cooldown_seconds: float, session_last_ts: float, scope_last_ts: float) -> bool:
    return bool(
        (session_last_ts and (now_ts - session_last_ts) < cooldown_seconds)
        or (scope_last_ts and (now_ts - scope_last_ts) < cooldown_seconds)
    )


def unattended_tail_allows_injection(last: tuple[str, float] | None, *, now_ts: float, cooldown_seconds: float) -> bool:
    if not last:
        return False
    role, ts = last
    if role != "assistant":
        return False
    return (now_ts - float(ts)) >= cooldown_seconds


def disable_unattended_if_exhausted(
    config: dict[str, Any],
    *,
    default_max_injections: int,
) -> tuple[dict[str, Any], bool]:
    cur = dict(config)
    remaining_injections = clean_unattended_remaining_injections(
        cur.get("remaining_injections"),
        default_max_injections=default_max_injections,
        allow_zero=True,
    )
    if remaining_injections > 0:
        return cur, False
    cur["enabled"] = False
    cur["remaining_injections"] = 0
    return cur, True


def unattended_prompt_decision(
    config: dict[str, Any],
    *,
    now_ts: float,
    session_last_ts: float,
    scope_last_ts: float,
    prompt_prefix: str,
    default_idle_minutes: int,
    default_max_injections: int,
) -> UnattendedPromptDecision:
    cur = dict(config)
    state = unattended_config_state(
        cur,
        default_idle_minutes=default_idle_minutes,
        default_max_injections=default_max_injections,
    )
    if state.remaining_injections <= 0:
        cur["enabled"] = False
        cur["remaining_injections"] = 0
        return UnattendedPromptDecision(prompt="", cooldown_seconds=state.cooldown_seconds, config=cur, disabled_exhausted=True)
    if not state.enabled or unattended_cooldown_blocked(
        now_ts=now_ts,
        cooldown_seconds=state.cooldown_seconds,
        session_last_ts=session_last_ts,
        scope_last_ts=scope_last_ts,
    ):
        return UnattendedPromptDecision(prompt="", cooldown_seconds=state.cooldown_seconds, config=cur, disabled_exhausted=False)
    return UnattendedPromptDecision(
        prompt=render_unattended_prompt(state.request, prompt_prefix=prompt_prefix),
        cooldown_seconds=state.cooldown_seconds,
        config=cur,
        disabled_exhausted=False,
    )


def record_unattended_success(
    config: dict[str, Any],
    *,
    default_max_injections: int,
) -> UnattendedSuccessUpdate:
    cur = dict(config)
    current_remaining = clean_unattended_remaining_injections(
        cur.get("remaining_injections"),
        default_max_injections=default_max_injections,
        allow_zero=True,
    )
    next_remaining = max(0, current_remaining - 1)
    cur["remaining_injections"] = next_remaining
    if next_remaining <= 0:
        cur["enabled"] = False
    return UnattendedSuccessUpdate(config=cur, remaining_injections=next_remaining, enabled=bool(cur.get("enabled")))


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
