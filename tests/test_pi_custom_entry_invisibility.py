import json
from pathlib import Path
from tempfile import TemporaryDirectory

from codoxear.broker_turn_state import State
from codoxear.broker_turn_state import _apply_rollout_obj_to_state
from codoxear.pi_log import pi_current_turn_state_before
from codoxear.rollout_jsonl import _read_jsonl_records_from_offset
from codoxear.rollout_log import _extract_positioned_chat_events


def _state() -> State:
    return State(
        codex_pid=1,
        pty_master_fd=1,
        cwd="/tmp",
        start_ts=0.0,
        codex_home=Path("/tmp"),
        sessions_dir=Path("/tmp"),
    )


def test_pi_custom_materialization_entry_is_invisible_and_turn_neutral() -> None:
    rows = [
        {"type": "session", "id": "session-id", "cwd": "/tmp", "timestamp": "2026-01-01T00:00:00.000Z"},
        {
            "type": "custom",
            "customType": "codoxear",
            "data": {"kind": "session-materialize", "reason": "new"},
            "id": "custom-id",
            "parentId": None,
            "timestamp": "2026-01-01T00:00:01.000Z",
        },
    ]
    with TemporaryDirectory() as td:
        log_path = Path(td) / "session.jsonl"
        log_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        records, _next_off = _read_jsonl_records_from_offset(log_path, 0, max_bytes=max(log_path.stat().st_size, 1))
        pending, idle = pi_current_turn_state_before(log_path, log_path.stat().st_size)

    assert _extract_positioned_chat_events(records) == []
    assert pending == set()
    assert idle is None

    st = _state()
    for row in rows:
        _apply_rollout_obj_to_state(st, row, now_ts=10.0)
    assert not st.busy
    assert not st.turn_open
    assert st.pending_calls == set()
