from __future__ import annotations

from collections.abc import Callable
from typing import Any

from codoxear.broker_turn_state import State
from codoxear.broker_turn_state import _apply_rollout_obj_to_state
from codoxear.pi_log import pi_context_token_update as _pi_context_token_update
from codoxear.pi_log import pi_token_update as _pi_token_update


def _pop_key_queue_if_idle(st: State) -> tuple[int | None, list[bytes]]:
    if st.busy or st.turn_open or st.pending_calls:
        return None, []
    if not st.key_queue:
        return None, []
    fd = st.pty_master_fd
    if fd is None:
        return None, []
    kq = st.key_queue[:]
    st.key_queue.clear()
    return fd, kq


def _clear_resume_delivery_mute_if_idle(st: State) -> bool:
    if st.resume_session_id and (not st.busy) and (not st.turn_open) and (not st.pending_calls):
        st.resume_session_id = None
        return True
    return False


def _apply_log_objects_to_state(st: State, objs: list[dict[str, Any]], *, now: Callable[[], float]) -> None:
    for obj in objs:
        now_ts = now()
        token_update = _pi_token_update(obj)
        if token_update is not None:
            st.token = token_update
        if obj.get("type") == "event_msg":
            p = obj.get("payload")
            if not isinstance(p, dict):
                raise ValueError("invalid rollout event_msg payload")
            pt = p.get("type")
            if pt == "token_count":
                info = p.get("info")
                if isinstance(info, dict) and isinstance(info.get("total_token_usage"), dict):
                    ctx = info.get("model_context_window")
                    last = info.get("last_token_usage")
                    if isinstance(ctx, int) and isinstance(last, dict):
                        tt = last.get("total_tokens")
                        if isinstance(tt, int):
                            token_update = _pi_context_token_update(
                                context_window=ctx,
                                tokens_in_context=tt,
                                as_of=obj.get("timestamp") if isinstance(obj.get("timestamp"), str) else None,
                            )
                            st.token = token_update
        _apply_rollout_obj_to_state(st, obj, now_ts=now_ts)
