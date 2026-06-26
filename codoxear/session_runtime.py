from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class BrokerRuntimeState:
    busy: bool
    queue_len: int
    interrupted_idle: bool = False

    @property
    def allows_interrupted_idle_override(self) -> bool:
        return (not self.busy) and self.queue_len == 0 and self.interrupted_idle


@dataclass(frozen=True)
class RuntimeStatus:
    broker: BrokerRuntimeState
    log_exists: bool
    log_idle: bool | None
    send_boundary_unresolved: bool
    busy: bool
    remote_ready: bool


def broker_runtime_state(state: Mapping[str, Any]) -> BrokerRuntimeState:
    if not isinstance(state, Mapping) or "busy" not in state or "queue_len" not in state:
        raise ValueError("invalid broker state response")
    busy_raw = state.get("busy")
    queue_len_raw = state.get("queue_len")
    if not isinstance(busy_raw, bool):
        raise ValueError("invalid broker state response")
    if isinstance(queue_len_raw, bool) or not isinstance(queue_len_raw, int) or queue_len_raw < 0:
        raise ValueError("invalid broker state response")
    interrupted_idle_raw = state.get("interrupted_idle", False)
    if not isinstance(interrupted_idle_raw, bool):
        raise ValueError("invalid broker state response")
    return BrokerRuntimeState(busy=busy_raw, queue_len=int(queue_len_raw), interrupted_idle=interrupted_idle_raw)


def broker_busy_queue(state: Mapping[str, Any]) -> tuple[bool, int]:
    broker = broker_runtime_state(state)
    return broker.busy, broker.queue_len


def broker_interrupted_idle(state: Mapping[str, Any]) -> bool:
    return broker_runtime_state(state).interrupted_idle


def broker_allows_interrupted_idle_override(state: Mapping[str, Any]) -> bool:
    return broker_runtime_state(state).allows_interrupted_idle_override


def resolve_runtime_status(
    *,
    broker: BrokerRuntimeState,
    log_exists: bool,
    log_idle: bool | None,
    send_boundary_unresolved: bool,
) -> RuntimeStatus:
    log_bound = bool(log_exists)
    boundary = bool(send_boundary_unresolved)
    if log_bound and (not boundary) and not isinstance(log_idle, bool):
        raise ValueError("log_idle is required for a bound transcript log")
    if boundary:
        busy = True
    elif not log_bound:
        busy = False
    else:
        busy = not (bool(log_idle) or broker.allows_interrupted_idle_override)

    if broker.queue_len > 0:
        remote_ready = False
    elif boundary:
        remote_ready = False
    elif log_bound and (log_idle is not True) and not broker.allows_interrupted_idle_override:
        remote_ready = False
    elif broker.busy and log_idle is not True:
        remote_ready = False
    else:
        remote_ready = True

    return RuntimeStatus(
        broker=broker,
        log_exists=log_bound,
        log_idle=log_idle if log_bound else None,
        send_boundary_unresolved=boundary,
        busy=busy,
        remote_ready=remote_ready,
    )


def select_runtime_token(
    *,
    broker_state: Mapping[str, Any],
    session_token: dict[str, Any] | None,
    token_update: dict[str, Any] | None,
    log_available: bool,
) -> dict[str, Any] | None:
    if "token" in broker_state:
        broker_token = broker_state.get("token")
        if not (isinstance(broker_token, dict) or broker_token is None):
            raise ValueError("invalid token from broker state response")
    if isinstance(token_update, dict):
        return token_update
    if isinstance(session_token, dict):
        return session_token
    if (not log_available) and "token" in broker_state and isinstance(broker_state.get("token"), dict):
        return broker_state.get("token")
    return None
