from __future__ import annotations

import json
import threading
from pathlib import Path

from codoxear.rollout_idle import _analyze_log_chunk
from codoxear.session_log_runtime import SessionLogRuntimeCoordinator
from codoxear.session_model import Session
from codoxear.util import read_jsonl_from_offset


def _session(log_path: Path, *, thinking: int, tools: int, turn_open: bool, queue_len: int = 0) -> Session:
    return Session(
        session_id="sid",
        thread_id="thread",
        broker_pid=1,
        codex_pid=2,
        agent_backend="pi",
        owned=False,
        start_ts=1.0,
        cwd="/tmp",
        log_path=log_path,
        sock_path=log_path.with_suffix(".sock"),
        busy=True,
        queue_len=queue_len,
        meta_thinking=thinking,
        meta_tools=tools,
        meta_turn_open=turn_open,
    )


def _runtime(session: Session) -> SessionLogRuntimeCoordinator:
    sessions = {session.session_id: session}
    return SessionLogRuntimeCoordinator(
        lock=threading.Lock(),
        sessions=lambda: sessions,
        analyze_log_chunk=_analyze_log_chunk,
        turn_context_run_settings=lambda _payload: (None, None),
        compute_idle_from_log=lambda _path: False,
        read_jsonl_from_offset=read_jsonl_from_offset,
        find_latest_token_update=lambda _path: None,
    )


def _append(log_path: Path, *rows: dict) -> None:
    with log_path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def _pi_user(text: str) -> dict:
    return {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": text}]}}


def _pi_tool(call_id: str) -> dict:
    return {
        "type": "message",
        "message": {
            "role": "assistant",
            "content": [{"type": "toolCall", "id": call_id, "name": "bash", "arguments": {"command": "pwd"}}],
            "stopReason": "toolUse",
        },
    }


def test_queued_turn_resets_counters_when_user_arrives_after_cross_chunk_close(tmp_path: Path) -> None:
    log_path = tmp_path / "pi.jsonl"
    log_path.touch()
    session = _session(log_path, thinking=2, tools=1, turn_open=True, queue_len=1)
    runtime = _runtime(session)

    # Scan A closes turn 1, but queue state keeps the public/runtime session busy.
    _append(
        log_path,
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "turn one done"}],
                "stopReason": "stop",
            },
        },
    )
    runtime.update_meta_counters()
    assert session.busy is True
    assert session.meta_turn_open is False
    assert (session.meta_thinking, session.meta_tools) == (2, 1)

    # Scan B starts the queued turn. Its first user row arrived while the
    # persisted counting turn was closed, so predecessor counts are replaced.
    _append(log_path, _pi_user("queued turn"), _pi_tool("queued-tool"))
    runtime.update_meta_counters()
    assert session.meta_turn_open is True
    assert (session.meta_thinking, session.meta_tools) == (0, 1)


def test_cross_chunk_steer_preserves_open_turn_counters(tmp_path: Path) -> None:
    log_path = tmp_path / "pi.jsonl"
    log_path.touch()
    session = _session(log_path, thinking=1, tools=1, turn_open=True)
    runtime = _runtime(session)

    # A tool in scan A keeps the persisted turn open.
    _append(log_path, _pi_tool("before-steer"))
    runtime.update_meta_counters()
    assert session.meta_turn_open is True
    assert (session.meta_thinking, session.meta_tools) == (1, 2)

    # The user row in scan B is steering because the turn was already open.
    # Both tool deltas remain in the same counting window.
    _append(log_path, _pi_user("steer"), _pi_tool("after-steer"))
    runtime.update_meta_counters()
    assert session.meta_turn_open is True
    assert (session.meta_thinking, session.meta_tools) == (1, 3)
