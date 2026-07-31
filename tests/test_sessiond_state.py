from pathlib import Path

import pytest

from codoxear.sessiond_state import State
from codoxear.sessiond_state import _busy_value_after_log_batch
from codoxear.sessiond_state import _log_busy_signals
from codoxear.sessiond_state import _read_jsonl_from_offset


def test_sessiond_state_dataclass_and_missing_log_reader_contract(tmp_path: Path) -> None:
    state = State(
        session_id=None,
        codex_pid=123,
        log_path=tmp_path / "missing.jsonl",
        sock_path=tmp_path / "session.sock",
        pty_master_fd=4,
        start_ts=1.5,
    )

    assert state.busy is False
    assert state.output_tail == ""
    assert state.log_off == 0
    assert _read_jsonl_from_offset(state.log_path, 99) == ([], 99)


def test_sessiond_busy_reducer_handles_codex_pi_and_claude_rows_in_order() -> None:
    codex_user = {"type": "event_msg", "payload": {"type": "user_message", "message": "run"}}
    codex_done = {"type": "event_msg", "payload": {"type": "turn_complete"}}
    pi_user = {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "run"}]}}
    pi_done = {"type": "message", "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}], "stopReason": "end_turn"}}
    cc_user = {"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": "run"}]}}
    cc_done = {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn"}}

    assert _log_busy_signals(codex_user) == (True, False)
    assert _log_busy_signals(codex_done) == (False, True)
    assert _log_busy_signals(pi_user) == (True, False)
    assert _log_busy_signals(pi_done) == (False, True)
    assert _log_busy_signals(cc_user) == (True, False)
    assert _log_busy_signals(cc_done) == (False, True)
    assert _busy_value_after_log_batch([codex_user, codex_done, pi_user, pi_done, cc_user, cc_done]) is False
    assert _busy_value_after_log_batch([codex_done, cc_user]) is True


def test_sessiond_busy_reducer_rejects_malformed_codex_event_payload() -> None:
    with pytest.raises(ValueError, match="invalid rollout event_msg payload"):
        _log_busy_signals({"type": "event_msg", "payload": "bad"})
