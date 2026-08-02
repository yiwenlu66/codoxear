from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codoxear.cc_log import cc_apply_tool_result_to_pending as _cc_apply_tool_result_to_pending
from codoxear.cc_log import cc_assistant_is_final_turn_end as _cc_assistant_is_final_turn_end
from codoxear.cc_log import cc_assistant_pending_tool_use_ids as _cc_assistant_pending_tool_use_ids
from codoxear.cc_log import cc_assistant_text as _cc_assistant_text
from codoxear.cc_log import cc_assistant_thinking_count as _cc_assistant_thinking_count
from codoxear.cc_log import cc_assistant_tool_use_count as _cc_assistant_tool_use_count
from codoxear.cc_log import cc_is_turn_end as _cc_is_turn_end
from codoxear.cc_log import cc_message_role as _cc_message_role
from codoxear.cc_log import cc_user_text as _cc_user_text
from codoxear.pi_log import PiPendingToolCallId as _PiPendingToolCallId
from codoxear.pi_log import pi_apply_assistant_tool_calls_to_pending as _pi_apply_assistant_tool_calls_to_pending
from codoxear.pi_log import pi_apply_tool_result_to_pending as _pi_apply_tool_result_to_pending
from codoxear.pi_log import pi_assistant_text as _pi_assistant_text
from codoxear.pi_log import pi_assistant_error_text as _pi_assistant_error_text
from codoxear.pi_log import pi_assistant_is_aborted_turn as _pi_assistant_is_aborted_turn
from codoxear.pi_log import pi_assistant_is_final_turn_end as _pi_assistant_is_final_turn_end
from codoxear.pi_log import pi_assistant_is_terminal_no_visible_response as _pi_assistant_is_terminal_no_visible_response
from codoxear.pi_log import pi_assistant_thinking_count as _pi_assistant_thinking_count
from codoxear.pi_log import pi_assistant_tool_use_count as _pi_assistant_tool_use_count
from codoxear.pi_log import pi_message_role as _pi_message_role
from codoxear.pi_log import pi_user_text as _pi_user_text


INTERRUPT_HINT_TAIL_MAX = 4096
_PI_PROGRESS_ACTIVE_SEQUENCE = "\x1b]9;4;3\x07"

_ANSI_OSC_RE = re.compile("\x1B\\][^\x07]*(?:\x07|\x1B\\\\)")
_ANSI_CSI_RE = re.compile("\x1B(?:[@-Z\\\\-_]|\\[[0-?]*[ -/]*[@-~])")
_PI_RETRY_STATUS_RE = re.compile(r"Retrying \(\d+/\d+\) in \d+s\.\.\. \([^\r\n]{0,48} to cancel\)", re.IGNORECASE)


def _codex_error_affects_turn_status(payload: dict[str, Any]) -> bool:
    info = payload.get("codex_error_info")
    if info == "thread_rollback_failed":
        return False
    return not (isinstance(info, dict) and "thread_rollback_failed" in info)


def _strip_ansi(text: str) -> str:
    return _ANSI_CSI_RE.sub("", _ANSI_OSC_RE.sub("", text))


def _hint_seen_in_new_text(*, tail: str, cleaned: str, phrase: str) -> bool:
    low_cleaned = cleaned.lower()
    low_phrase = phrase.lower()
    if low_phrase in low_cleaned:
        return True
    overlap = max(len(low_phrase) - 1, 0)
    if overlap <= 0:
        return False
    stitched = tail[-overlap:].lower() + low_cleaned
    pos = stitched.find(low_phrase)
    if pos < 0:
        return False
    return (pos + len(low_phrase)) > overlap


def _interrupt_hint_seen_in_new_text(*, tail: str, cleaned: str) -> bool:
    return _hint_seen_in_new_text(tail=tail, cleaned=cleaned, phrase="esc to interrupt")


def _compacting_hint_seen_in_new_text(*, tail: str, cleaned: str) -> bool:
    return (
        _hint_seen_in_new_text(tail=tail, cleaned=cleaned, phrase="compacting context")
        or _hint_seen_in_new_text(tail=tail, cleaned=cleaned, phrase="compacting conversation")
    )


def _pi_retry_hint_seen_in_new_text(*, tail: str, cleaned: str) -> bool:
    stitched = tail[-160:] + cleaned
    return _PI_RETRY_STATUS_RE.search(stitched) is not None


def _pi_working_hint_seen_in_new_text(*, tail: str, cleaned: str, raw: str) -> bool:
    return _PI_PROGRESS_ACTIVE_SEQUENCE in raw or _hint_seen_in_new_text(tail=tail, cleaned=cleaned, phrase="Working...")


def _update_busy_from_pty_text(st: "State", text: str, now_ts: float) -> None:
    cleaned = _strip_ansi(text)
    tail = st.interrupt_hint_tail
    if _pi_working_hint_seen_in_new_text(tail=tail, cleaned=cleaned, raw=text):
        # A retry attempt has left backoff and started a new request. Retire the
        # prior error probe; a failing attempt will persist a fresh error row.
        _clear_pi_error_probe(st)
    if not cleaned:
        return
    st.interrupt_hint_tail = (st.interrupt_hint_tail + cleaned)[-st.interrupt_hint_tail_max :]
    if _pi_retry_hint_seen_in_new_text(tail=tail, cleaned=cleaned):
        # Pi's JSONL error row is identical for retryable and terminal failures.
        # The TUI supplies the missing positive discriminator during backoff.
        st.pi_retry_status_active = True
        st.last_pi_retry_hint_ts = now_ts
    if _interrupt_hint_seen_in_new_text(tail=tail, cleaned=cleaned):
        st.busy = True
        st.last_interrupt_hint_ts = now_ts
        if now_ts > st.last_turn_activity_ts:
            st.last_turn_activity_ts = now_ts
        return
    if _compacting_hint_seen_in_new_text(tail=tail, cleaned=cleaned):
        st.busy = True
        if now_ts > st.last_turn_activity_ts:
            st.last_turn_activity_ts = now_ts
        return


def _response_call_started(payload: dict[str, Any]) -> str | None:
    t = payload.get("type")
    if t not in ("function_call", "custom_tool_call"):
        return None
    call_id = payload.get("call_id")
    return call_id if isinstance(call_id, str) and call_id else None


def _response_call_finished(payload: dict[str, Any]) -> str | None:
    t = payload.get("type")
    if t not in ("function_call_output", "custom_tool_call_output"):
        return None
    call_id = payload.get("call_id")
    return call_id if isinstance(call_id, str) and call_id else None


def _should_clear_busy_state(
    st: "State",
    now_ts: float,
    *,
    busy_quiet_seconds: float,
    busy_interrupt_grace_seconds: float,
) -> bool:
    if not st.busy:
        return False
    if st.last_pi_error_probe_ts > 0.0:
        if st.pi_retry_status_active:
            return False
        # Pi renders the retry status immediately after persisting the failed
        # assistant row. Once the quiet window passes without that status, the
        # TUI has settled back at its editor after a terminal failure.
        return (now_ts - st.last_pi_error_probe_ts) >= busy_quiet_seconds
    if st.pending_calls:
        return False
    if st.turn_open and (not st.turn_has_completion_candidate):
        if st.last_interrupt_request_ts <= 0.0:
            return False
        if (now_ts - st.last_interrupt_request_ts) < busy_interrupt_grace_seconds:
            return False
    if st.last_interrupt_hint_ts > 0.0 and (now_ts - st.last_interrupt_hint_ts) < busy_interrupt_grace_seconds:
        return False
    if st.last_turn_activity_ts <= 0.0:
        return False
    return (now_ts - st.last_turn_activity_ts) >= busy_quiet_seconds


def _mark_explicit_interrupt_request(st: "State", now_ts: float) -> None:
    st.last_interrupt_request_ts = now_ts
    st.last_interrupted_idle_ts = 0.0
    # Esc cancels Pi's backoff. The old retry frame must not keep the error
    # probe busy after the user has explicitly ended that retry wait.
    st.pi_retry_status_active = False
    if now_ts > st.last_turn_activity_ts:
        st.last_turn_activity_ts = now_ts


def _clear_pi_error_probe(st: "State") -> None:
    st.last_pi_error_probe_ts = 0.0
    st.last_pi_retry_hint_ts = 0.0
    st.pi_retry_status_active = False


def _mark_busy_state_idle(st: "State", now_ts: float) -> None:
    interrupted_idle = st.turn_open and st.last_interrupt_request_ts > 0.0
    terminal_error_idle = st.turn_open and st.last_pi_error_probe_ts > 0.0 and not st.pi_retry_status_active
    st.busy = False
    st.turn_open = False
    st.turn_has_completion_candidate = False
    st.last_turn_activity_ts = 0.0
    st.last_interrupt_hint_ts = 0.0
    st.last_interrupt_request_ts = 0.0
    _clear_pi_error_probe(st)
    # ``interrupted_idle`` is the control protocol's established override for
    # a non-final log tail whose live PTY has nevertheless settled. A terminal
    # Pi error has the same projection need as an explicit interrupt.
    st.last_interrupted_idle_ts = now_ts if interrupted_idle or terminal_error_idle else 0.0


def _reopen_turn_on_activity(st: "State") -> None:
    if st.turn_open:
        return
    st.turn_open = True
    st.turn_has_completion_candidate = False


def _close_turn_state(st: "State") -> None:
    st.pending_calls.clear()
    st.busy = False
    st.turn_open = False
    st.turn_has_completion_candidate = False
    st.last_interrupt_hint_ts = 0.0
    st.last_interrupt_request_ts = 0.0
    _clear_pi_error_probe(st)
    st.last_interrupted_idle_ts = 0.0
    st.last_turn_activity_ts = 0.0


def _apply_rollout_obj_to_state(st: "State", obj: dict[str, Any], now_ts: float) -> None:
    typ = obj.get("type")

    if typ == "event_msg":
        payload = obj.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("invalid rollout event_msg payload")
        ev_type = payload.get("type")
        if ev_type == "user_message":
            msg = payload.get("message")
            if isinstance(msg, str) and msg.strip():
                st.pending_calls.clear()
                st.busy = True
                st.turn_open = True
                st.turn_has_completion_candidate = False
                st.last_interrupt_hint_ts = 0.0
                st.last_interrupt_request_ts = 0.0
                st.last_interrupted_idle_ts = 0.0
                st.last_turn_activity_ts = now_ts
            return
        if ev_type in ("turn_aborted", "thread_rolled_back"):
            _close_turn_state(st)
            return
        if ev_type in ("task_complete", "turn_complete"):
            _close_turn_state(st)
            return
        if ev_type == "error":
            if _codex_error_affects_turn_status(payload):
                _close_turn_state(st)
            return
        if ev_type == "agent_message":
            msg = payload.get("message")
            if isinstance(msg, str) and msg.strip() and st.turn_open:
                st.turn_has_completion_candidate = True
            st.busy = True
            st.last_turn_activity_ts = now_ts
            return
        if ev_type == "agent_reasoning":
            _reopen_turn_on_activity(st)
            if st.turn_open:
                st.turn_has_completion_candidate = False
            st.busy = True
            st.last_turn_activity_ts = now_ts
            return
        if ev_type == "token_count" and st.busy:
            st.last_turn_activity_ts = now_ts
            return
        return

    if typ == "message":
        user_text = _pi_user_text(obj)
        if isinstance(user_text, str) and user_text:
            _clear_pi_error_probe(st)
            st.pending_calls.clear()
            st.busy = True
            st.turn_open = True
            st.turn_has_completion_candidate = False
            st.last_interrupt_hint_ts = 0.0
            st.last_interrupt_request_ts = 0.0
            st.last_interrupted_idle_ts = 0.0
            st.last_turn_activity_ts = now_ts
            return

        role = _pi_message_role(obj)
        has_text = bool(_pi_assistant_text(obj))
        thinking_count = _pi_assistant_thinking_count(obj)
        tool_count = _pi_assistant_tool_use_count(obj)
        is_tool_result = role == "toolResult"

        if role == "assistant" and _pi_assistant_is_aborted_turn(obj):
            _close_turn_state(st)
            return

        if _pi_assistant_error_text(obj) and role == "assistant":
            # A Pi error row can precede its automatic retry and carries no
            # terminal/retry discriminator. It is turn activity, never a
            # close; clear stale calls from the failed request before retrying.
            st.pending_calls.clear()
            _reopen_turn_on_activity(st)
            if st.turn_open:
                st.turn_has_completion_candidate = False
            st.busy = True
            st.last_turn_activity_ts = now_ts
            st.last_pi_error_probe_ts = now_ts
            return

        if role == "assistant" and _pi_assistant_is_terminal_no_visible_response(obj):
            _close_turn_state(st)
            return

        if is_tool_result:
            _pi_apply_tool_result_to_pending(obj, st.pending_calls)

        if tool_count > 0:
            _pi_apply_assistant_tool_calls_to_pending(obj, st.pending_calls)

        if has_text and role == "assistant" and _pi_assistant_is_final_turn_end(obj):
            _close_turn_state(st)
            return

        if is_tool_result or tool_count > 0 or thinking_count > 0:
            _clear_pi_error_probe(st)
            _reopen_turn_on_activity(st)
            if st.turn_open:
                st.turn_has_completion_candidate = False
            st.busy = True
            st.last_turn_activity_ts = now_ts
            return

        return

    if typ == "user":
        user_text = _cc_user_text(obj)
        if isinstance(user_text, str) and user_text:
            st.pending_calls.clear()
            st.busy = True
            st.turn_open = True
            st.turn_has_completion_candidate = False
            st.last_interrupt_hint_ts = 0.0
            st.last_interrupt_request_ts = 0.0
            st.last_interrupted_idle_ts = 0.0
            st.last_turn_activity_ts = now_ts
            return
        if _cc_message_role(obj) == "toolResult":
            _cc_apply_tool_result_to_pending(obj, st.pending_calls)
            _reopen_turn_on_activity(st)
            if st.turn_open:
                st.turn_has_completion_candidate = False
            st.busy = True
            st.last_turn_activity_ts = now_ts
            return
        return

    if typ == "assistant":
        from .cc_log import cc_assistant_is_api_error
        has_text = bool(_cc_assistant_text(obj))
        thinking_count = _cc_assistant_thinking_count(obj)
        tool_count = _cc_assistant_tool_use_count(obj)
        if cc_assistant_is_api_error(obj):
            _close_turn_state(st)
            return
        if has_text and _cc_assistant_is_final_turn_end(obj):
            if st.pending_calls:
                _reopen_turn_on_activity(st)
                if st.turn_open:
                    st.turn_has_completion_candidate = False
                st.busy = True
                st.last_turn_activity_ts = now_ts
                return
            _close_turn_state(st)
            return
        if tool_count > 0 or thinking_count > 0:
            if tool_count > 0:
                st.pending_calls.update(_cc_assistant_pending_tool_use_ids(obj))
            _reopen_turn_on_activity(st)
            if st.turn_open:
                st.turn_has_completion_candidate = False
            st.busy = True
            st.last_turn_activity_ts = now_ts
            return
        if has_text:
            if st.turn_open:
                st.turn_has_completion_candidate = True
            st.busy = True
            st.last_turn_activity_ts = now_ts
            return
        return

    if typ == "system" and _cc_is_turn_end(obj):
        if st.pending_calls:
            _reopen_turn_on_activity(st)
            if st.turn_open:
                st.turn_has_completion_candidate = False
            st.busy = True
            st.last_turn_activity_ts = now_ts
            return
        _close_turn_state(st)
        return

    if typ != "response_item":
        return
    payload = obj.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("invalid rollout response_item payload")

    started = _response_call_started(payload)
    if started is not None:
        st.pending_calls.add(started)
        _reopen_turn_on_activity(st)
        if st.turn_open:
            st.turn_has_completion_candidate = False
        st.busy = True
        st.last_turn_activity_ts = now_ts
        return

    finished = _response_call_finished(payload)
    if finished is not None:
        st.pending_calls.discard(finished)
        _reopen_turn_on_activity(st)
        if st.turn_open:
            st.turn_has_completion_candidate = False
        st.busy = True
        st.last_turn_activity_ts = now_ts
        return

    item_type = payload.get("type")
    role = payload.get("role")
    if item_type in (
        "reasoning",
        "function_call",
        "function_call_output",
        "custom_tool_call",
        "custom_tool_call_output",
        "web_search_call",
        "local_shell_call",
    ):
        _reopen_turn_on_activity(st)
        if st.turn_open:
            st.turn_has_completion_candidate = False
        st.busy = True
        st.last_turn_activity_ts = now_ts
        return
    if item_type == "message" and role == "assistant":
        content = payload.get("content")
        if not isinstance(content, list):
            raise ValueError("invalid assistant message content")
        has_text = any(
            isinstance(part, dict)
            and part.get("type") == "output_text"
            and isinstance(part.get("text"), str)
            and part.get("text")
            for part in content
        )
        if has_text and st.turn_open:
            st.turn_has_completion_candidate = True
        st.busy = True
        st.last_turn_activity_ts = now_ts
        return


@dataclass
class State:
    codex_pid: int
    pty_master_fd: int
    cwd: str
    start_ts: float
    codex_home: Path
    sessions_dir: Path
    log_path: Path | None = None
    session_id: str | None = None
    sock_path: Path | None = None
    busy: bool = False
    stdin_eof: bool = False
    key_queue: list[bytes] = field(default_factory=list)
    output_tail: str = ""
    output_tail_max: int = 256 * 1024
    shell_pre_exec_marker_seen: bool = False
    shell_pre_exec_marker_ts: float = 0.0
    shell_pre_exec_marker_tail: bytes = b""
    prelog_failure_recorded: bool = False
    log_off: int = 0
    last_local_input_ts: float = 0.0
    last_turn_activity_ts: float = 0.0
    last_interrupt_hint_ts: float = 0.0
    last_interrupt_request_ts: float = 0.0
    last_interrupted_idle_ts: float = 0.0
    last_pi_error_probe_ts: float = 0.0
    last_pi_retry_hint_ts: float = 0.0
    pi_retry_status_active: bool = False
    pending_calls: set[str | _PiPendingToolCallId] = field(default_factory=set)
    turn_open: bool = False
    turn_has_completion_candidate: bool = False
    interrupt_hint_tail: str = ""
    interrupt_hint_tail_max: int = INTERRUPT_HINT_TAIL_MAX
    detach_trigger_tail: str = ""
    detach_trigger_tail_max: int = 8192
    token: dict[str, Any] | None = None
    declared_log_path: Path | None = None
    last_rollout_path: Path | None = None
    last_detected_rollout_path: Path | None = None
    ignored_rollout_paths: set[Path] = field(default_factory=set)
    known_rollout_paths: set[Path] = field(default_factory=set)
    resume_session_id: str | None = None
