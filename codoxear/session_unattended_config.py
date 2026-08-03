from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, MutableMapping

from .session_model import Session
from .unattended import unattended_config_key


def _config_for_session(unattended: MutableMapping[str, dict[str, Any]], session: Session) -> tuple[str, dict[str, Any]]:
    config_key = unattended_config_key(session)
    scoped = unattended.get(config_key)
    legacy = unattended.get(session.session_id)
    raw = scoped if isinstance(scoped, dict) else legacy
    return config_key, dict(raw) if isinstance(raw, dict) else {}


@dataclass(frozen=True)
class SessionUnattendedConfigCoordinator:
    lock: Any
    sessions: Callable[[], MutableMapping[str, Session]]
    unattended: Callable[[], MutableMapping[str, dict[str, Any]]]
    unattended_last_injected: Callable[[], MutableMapping[str, float]]
    input_lock_for_session: Callable[[str], Any]
    save_unattended: Callable[[], None]
    clean_unattended_cooldown_minutes: Callable[[Any], int]
    clean_unattended_remaining_injections: Callable[..., int]

    def get(self, session_id: str) -> dict[str, Any]:
        with self.lock:
            session = self.sessions().get(session_id)
            if not session:
                raise KeyError("unknown session")
            _config_key, cfg = _config_for_session(self.unattended(), session)
        request = cfg.get("request")
        if not isinstance(request, str):
            request = ""
        cooldown_minutes = self.clean_unattended_cooldown_minutes(cfg.get("cooldown_minutes"))
        remaining_injections = self.clean_unattended_remaining_injections(cfg.get("remaining_injections"), allow_zero=True)
        enabled = bool(cfg.get("enabled")) and remaining_injections > 0
        return {
            "enabled": enabled,
            "request": request,
            "cooldown_minutes": cooldown_minutes,
            "remaining_injections": remaining_injections,
        }

    def set(
        self,
        session_id: str,
        *,
        enabled: bool | None = None,
        request: str | None = None,
        cooldown_minutes: int | None = None,
        remaining_injections: int | None = None,
    ) -> dict[str, Any]:
        input_lock = self.input_lock_for_session(session_id)
        with input_lock:
            with self.lock:
                session = self.sessions().get(session_id)
                if not session:
                    raise KeyError("unknown session")
                config_key, cur = _config_for_session(self.unattended(), session)
                if enabled is not None:
                    cur["enabled"] = bool(enabled)
                if request is not None:
                    cur["request"] = str(request)
                if cooldown_minutes is not None:
                    cur["cooldown_minutes"] = self.clean_unattended_cooldown_minutes(cooldown_minutes)
                if remaining_injections is not None:
                    cur["remaining_injections"] = self.clean_unattended_remaining_injections(remaining_injections, allow_zero=True)
                cur["cooldown_minutes"] = self.clean_unattended_cooldown_minutes(cur.get("cooldown_minutes"))
                cur["remaining_injections"] = self.clean_unattended_remaining_injections(cur.get("remaining_injections"), allow_zero=True)
                if int(cur["remaining_injections"]) <= 0:
                    cur["enabled"] = False
                self.unattended()[config_key] = cur
                if not bool(cur.get("enabled")):
                    self.unattended_last_injected().pop(session_id, None)
            self.save_unattended()
        return self.get(session_id)
