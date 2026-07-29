from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .util import atomic_write_json
from .util import load_json_file


UNATTENDED_PROMPT_PREFIX = """Unattended-mode operating constitution

1. Recall the objective.
What is the user's goal? What does done look like? Ground every action in the original intent, not in process artifacts. When in doubt, return to the objective.

2. Understand current status.
What has been accomplished? What evidence exists? Compare the actual state of the world against the desired state. Be honest about gaps — wishful thinking wastes turns.

3. Replan toward the objective.
Given the current status, what is the shortest path to the objective? Adjust the plan based on new evidence. Eliminate work that does not serve the goal. Prioritize the highest-leverage next action over the most comfortable one.

4. Continue execution with delegation.
Execute the plan. Delegate bounded work to subagents when parallelizable: implementation, verification, exploration. Maintain ownership of integration and judgment. Verify delegated results against the objective, not against the subagent's self-assessment.

Operating principles:
- Maximize useful progress per turn. This is not about minimizing turns — it is about maximizing signal per turn.
- Verification is mandatory. Claims must be grounded in evidence: test results, browser observations, file contents, or direct measurement. No assertion without proof.
- Delegation is a first-class tool. Dispatch subagents for bounded execution while the main agent owns decisions, integration, and the causal model.
- Learn from failure. When an approach fails, understand why before trying the next thing. A failed result is evidence — use it.
- Yield control to the user only when: the objective is met, a genuine user decision is required, or the next action is irreversible and high-risk. Otherwise, continue.
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
