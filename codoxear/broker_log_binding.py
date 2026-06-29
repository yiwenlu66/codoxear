from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from codoxear.agent_backend import normalize_agent_backend
from codoxear.broker_turn_state import State
from codoxear.broker_turn_state import _close_turn_state
from codoxear.cc_log import cc_current_turn_state_before as _cc_current_turn_state_before
from codoxear.pi_log import PiPendingToolCallId as _PiPendingToolCallId
from codoxear.pi_log import pi_complete_jsonl_offset_before as _pi_complete_jsonl_offset_before
from codoxear.pi_log import pi_current_turn_state_before as _pi_current_turn_state_before
from codoxear.util import _paths_match
from codoxear.util import find_session_log_for_session_id as _find_session_log_for_session_id
from codoxear.util import is_subagent_session_meta as _is_subagent_session_meta
from codoxear.util import read_session_meta_payload as _read_session_meta_payload
from codoxear.util import subagent_parent_thread_id as _subagent_parent_thread_id


@dataclass(frozen=True)
class BrokerLogBinding:
    log_path: Path
    session_id: str


@dataclass(frozen=True)
class BrokerLogSeed:
    log_offset: int
    pending_calls: set[str | _PiPendingToolCallId]
    idle: bool | None


@dataclass(frozen=True)
class BrokerLogStateApplyResult:
    have_sock: bool
    previous_log_path: Path | None


def _resolve_broker_log_binding(
    *,
    log_path: Path,
    sessions_dir: Path,
    agent_backend: str,
    session_id_from_rollout_path: Callable[[Path], str | None],
) -> BrokerLogBinding | None:
    backend = normalize_agent_backend(agent_backend)
    try:
        lp = log_path.resolve()
    except Exception:
        lp = log_path
    try:
        lp.resolve().relative_to(sessions_dir.resolve())
    except Exception:
        return None
    if backend == "codex":
        if not (lp.name.startswith("rollout-") and lp.name.endswith(".jsonl")):
            return None
    elif lp.suffix != ".jsonl":
        return None

    payload = _read_session_meta_payload(lp, agent_backend=backend, timeout_s=1.5)
    if not payload:
        return None
    if backend == "codex" and _is_subagent_session_meta(payload):
        parent = _subagent_parent_thread_id(payload)
        if not parent:
            return None
        parent_log = _find_session_log_for_session_id(sessions_dir, parent, agent_backend=backend)
        if not parent_log:
            return None
        parent_payload = _read_session_meta_payload(parent_log, agent_backend=backend, timeout_s=0.2)
        if not parent_payload:
            return None
        if _is_subagent_session_meta(parent_payload):
            return None
        lp = parent_log
        payload = parent_payload

    sid = payload.get("id")
    if not isinstance(sid, str) or not sid:
        sid = session_id_from_rollout_path(lp)
        if sid is None:
            raise RuntimeError(f"unable to determine session_id from rollout filename: {lp}")
    if not sid:
        return None
    return BrokerLogBinding(log_path=lp, session_id=sid)


def _seed_broker_log_state(*, log_path: Path, agent_backend: str) -> BrokerLogSeed:
    backend = normalize_agent_backend(agent_backend)
    try:
        log_size = int(log_path.stat().st_size)
    except Exception:
        log_size = 0
    seed_pending: set[str | _PiPendingToolCallId] = set()
    seed_idle: bool | None = None
    seed_log_off = log_size
    if backend == "cc" and log_size > 0:
        try:
            seed_pending, seed_idle = _cc_current_turn_state_before(log_path, log_size)
        except Exception:
            seed_pending = set()
            seed_idle = None
    elif backend == "pi" and log_size > 0:
        try:
            seed_log_off = _pi_complete_jsonl_offset_before(log_path, log_size)
            seed_pending, seed_idle = _pi_current_turn_state_before(log_path, seed_log_off)
        except Exception:
            seed_log_off = 0
            seed_pending = set()
            seed_idle = None
    return BrokerLogSeed(log_offset=seed_log_off, pending_calls=seed_pending, idle=seed_idle)


def _apply_broker_log_binding_to_state(
    st: State,
    *,
    binding: BrokerLogBinding,
    seed: BrokerLogSeed,
) -> BrokerLogStateApplyResult | None:
    lp = binding.log_path
    last = st.last_rollout_path
    if last is not None and _paths_match(last, lp):
        return None
    st.last_rollout_path = lp
    have_sock = st.sock_path is not None
    prev_lp = st.log_path
    if prev_lp is None or not _paths_match(prev_lp, lp):
        st.last_interrupt_request_ts = 0.0
        st.last_interrupted_idle_ts = 0.0
    st.session_id = binding.session_id
    st.log_path = lp
    st.known_rollout_paths.add(lp)
    st.log_off = seed.log_offset
    st.pending_calls = set(seed.pending_calls)
    if seed.pending_calls or seed.idle is False:
        st.busy = True
        st.turn_open = True
        st.turn_has_completion_candidate = False
    elif seed.idle is True:
        _close_turn_state(st)
    return BrokerLogStateApplyResult(have_sock=have_sock, previous_log_path=prev_lp)
