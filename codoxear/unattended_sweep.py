from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import traceback
from typing import Any, Callable, MutableMapping

from .session_model import Session
from .session_runtime import RuntimeStatus
from .session_runtime import session_runtime_readiness
from .unattended import disable_unattended_if_exhausted
from .unattended import record_unattended_success
from .unattended import unattended_config_state
from .unattended import unattended_cooldown_blocked
from .unattended import unattended_prompt_decision
from .unattended import unattended_scope_key
from .unattended import unattended_tail_allows_injection


@dataclass(frozen=True)
class UnattendedSweepCoordinator:
    lock: Any
    sessions: Callable[[], MutableMapping[str, Session]]
    unattended: Callable[[], MutableMapping[str, Any]]
    unattended_last_injected: Callable[[], MutableMapping[str, float]]
    unattended_last_injected_scope: Callable[[], MutableMapping[str, float]]
    discover_existing_if_stale: Callable[[], None]
    prune_dead_sessions: Callable[[], None]
    input_lock_for_session: Callable[[str], Any]
    save_unattended: Callable[[], None]
    get_state: Callable[[str], dict[str, Any]]
    runtime_status_from_state: Callable[[str, dict[str, Any], Path | None], RuntimeStatus]
    queue_len: Callable[[str], int]
    last_chat_role_ts_from_tail: Callable[..., tuple[str, float] | None]
    send: Callable[[str, str], dict[str, Any]]
    now: Callable[[], float]
    prompt_prefix: str | Callable[[], str]
    default_idle_minutes: int
    default_max_injections: int
    max_scan_bytes: int

    def sweep(self) -> None:
        now_ts = self.now()
        self.discover_existing_if_stale()
        self.prune_dead_sessions()
        with self.lock:
            items: list[tuple[str, Session, dict[str, Any], float]] = []
            for sid, session in self.sessions().items():
                cfg0 = self.unattended().get(sid)
                cfg = dict(cfg0) if isinstance(cfg0, dict) else {}
                last_injected = float(self.unattended_last_injected().get(sid, 0.0))
                items.append((sid, session, cfg, last_injected))

        for sid, session, cfg, last_injected in items:
            if not bool(cfg.get("enabled")):
                continue
            try:
                state = unattended_config_state(
                    cfg,
                    default_idle_minutes=self.default_idle_minutes,
                    default_max_injections=self.default_max_injections,
                )
                if state.remaining_injections <= 0:
                    input_lock = self.input_lock_for_session(sid)
                    save_zero_cleanup = False
                    with input_lock:
                        with self.lock:
                            cur0 = self.unattended().get(sid)
                            cur = dict(cur0) if isinstance(cur0, dict) else {}
                            disabled, did_disable = disable_unattended_if_exhausted(
                                cur,
                                default_max_injections=self.default_max_injections,
                            )
                            if did_disable:
                                self.unattended()[sid] = disabled
                                self.unattended_last_injected().pop(sid, None)
                                save_zero_cleanup = True
                    if save_zero_cleanup:
                        self.save_unattended()
                    continue
                log_path = session.log_path
                if log_path is None or (not log_path.exists()):
                    continue
                scope_key = unattended_scope_key(thread_id=session.thread_id, log_path=log_path)
                with self.lock:
                    scope_last = float(self.unattended_last_injected_scope().get(scope_key, 0.0))
                if unattended_cooldown_blocked(
                    now_ts=now_ts,
                    cooldown_seconds=state.cooldown_seconds,
                    session_last_ts=last_injected,
                    scope_last_ts=scope_last,
                ):
                    continue
                broker_state = self.get_state(sid)
                if not isinstance(broker_state, dict):
                    raise ValueError("invalid broker state response")
                local_queue_len = self.queue_len(sid)
                runtime = self.runtime_status_from_state(sid, broker_state, log_path)
                if not session_runtime_readiness(runtime, local_queue_len=local_queue_len).unattended_injection:
                    continue
                last = self.last_chat_role_ts_from_tail(log_path, max_scan_bytes=self.max_scan_bytes, final_assistant_only=True)
                if not unattended_tail_allows_injection(last, now_ts=now_ts, cooldown_seconds=state.cooldown_seconds):
                    continue
                with self.lock:
                    scope_last = float(self.unattended_last_injected_scope().get(scope_key, 0.0))
                if unattended_cooldown_blocked(
                    now_ts=now_ts,
                    cooldown_seconds=state.cooldown_seconds,
                    session_last_ts=0.0,
                    scope_last_ts=scope_last,
                ):
                    continue
                input_lock = self.input_lock_for_session(sid)
                save_after_disable = False
                prompt = ""
                live_cooldown_seconds = state.cooldown_seconds
                with input_lock:
                    with self.lock:
                        cur0 = self.unattended().get(sid)
                        cur = dict(cur0) if isinstance(cur0, dict) else {}
                        live_last_injected = float(self.unattended_last_injected().get(sid, 0.0))
                        live_scope_last = float(self.unattended_last_injected_scope().get(scope_key, 0.0))
                        decision = unattended_prompt_decision(
                            cur,
                            now_ts=now_ts,
                            session_last_ts=live_last_injected,
                            scope_last_ts=live_scope_last,
                            prompt_prefix=self.prompt_prefix() if callable(self.prompt_prefix) else self.prompt_prefix,
                            default_idle_minutes=self.default_idle_minutes,
                            default_max_injections=self.default_max_injections,
                        )
                        live_cooldown_seconds = decision.cooldown_seconds
                        if decision.disabled_exhausted:
                            self.unattended()[sid] = decision.config
                            self.unattended_last_injected().pop(sid, None)
                            save_after_disable = True
                        prompt = decision.prompt
                    if save_after_disable:
                        self.save_unattended()
                    if not prompt:
                        continue
                    live_last = self.last_chat_role_ts_from_tail(log_path, max_scan_bytes=self.max_scan_bytes, final_assistant_only=True)
                    if not unattended_tail_allows_injection(live_last, now_ts=now_ts, cooldown_seconds=live_cooldown_seconds):
                        continue
                    self.send(sid, prompt)
                    with self.lock:
                        self.unattended_last_injected()[sid] = now_ts
                        self.unattended_last_injected_scope()[scope_key] = now_ts
                        cur0 = self.unattended().get(sid)
                        cur = dict(cur0) if isinstance(cur0, dict) else {}
                        update = record_unattended_success(
                            cur,
                            default_max_injections=self.default_max_injections,
                        )
                        if not update.enabled:
                            self.unattended_last_injected().pop(sid, None)
                        self.unattended()[sid] = update.config
                    self.save_unattended()
            except Exception as exc:
                sys.stderr.write(f"error: unattended session {sid} skipped: {type(exc).__name__}: {exc}\n")
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
