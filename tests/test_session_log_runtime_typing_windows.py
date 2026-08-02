from __future__ import annotations

import json
import threading
from pathlib import Path

from codoxear.rollout_idle import _analyze_log_chunk
from codoxear.session_log_runtime import SessionLogRuntimeCoordinator
from codoxear.session_model import Session
from codoxear.util import read_jsonl_from_offset


def _session(
    log_path: Path,
    *,
    thinking: int,
    thinking_tokens: int = 0,
    tools: int,
    turn_open: bool,
    queue_len: int = 0,
) -> Session:
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
        meta_thinking_tokens=thinking_tokens,
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


def _pi_thinking(text: str, reasoning_tokens: int) -> dict:
    return {
        "type": "message",
        "message": {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": text}],
            "usage": {"reasoning": reasoning_tokens},
        },
    }


def _pi_tool(call_id: str) -> dict:
    return {
        "type": "message",
        "message": {
            "role": "assistant",
            "content": [{"type": "toolCall", "id": call_id, "name": "bash", "arguments": {"command": "pwd"}}],
            "stopReason": "toolUse",
        },
    }


def _pi_final(text: str) -> dict:
    return {
        "type": "message",
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}], "stopReason": "stop"},
    }


def _pi_subagent_result() -> dict:
    return {
        "type": "message",
        "message": {
            "role": "user",
            "content": [
                {"type": "text", "text": "**📨 From subagent-result** (/workspace)\n\n"},
                {"type": "text", "text": "subagent results"},
            ],
        },
    }


def test_subagent_delivery_reopens_turn_without_resetting_episode_counters(tmp_path: Path) -> None:
    log_path = tmp_path / "pi.jsonl"
    log_path.touch()
    session = _session(log_path, thinking=0, thinking_tokens=0, tools=0, turn_open=False)
    runtime = _runtime(session)

    # Human input begins the episode. Two tools and their reasoning tokens are
    # retained after the assistant's first model turn completes.
    _append(log_path, _pi_user("human request"), _pi_thinking("first", 20), _pi_tool("one"), _pi_tool("two"), _pi_final("first response"))
    runtime.update_meta_counters()
    assert session.meta_turn_open is False
    assert (session.meta_thinking, session.meta_thinking_tokens, session.meta_tools) == (1, 20, 2)

    # The subagent result is a Pi user-role row but an episode-internal
    # delivery: it reopens busy/turn state and adds to the same counters.
    _append(log_path, _pi_subagent_result(), _pi_thinking("second", 16), _pi_tool("three"))
    runtime.update_meta_counters()
    assert session.meta_turn_open is True
    assert (session.meta_thinking, session.meta_thinking_tokens, session.meta_tools) == (2, 36, 3)


def test_human_input_after_final_resets_episode_counters(tmp_path: Path) -> None:
    log_path = tmp_path / "pi.jsonl"
    log_path.touch()
    session = _session(log_path, thinking=2, thinking_tokens=36, tools=3, turn_open=False)
    runtime = _runtime(session)

    _append(log_path, _pi_user("next human request"), _pi_thinking("fresh", 8), _pi_tool("fresh-tool"))
    runtime.update_meta_counters()
    assert session.meta_turn_open is True
    assert (session.meta_thinking, session.meta_thinking_tokens, session.meta_tools) == (1, 8, 1)


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


def test_reasoning_tokens_reset_after_closed_turn_and_accumulate_across_steer(tmp_path: Path) -> None:
    log_path = tmp_path / "pi.jsonl"
    log_path.touch()
    session = _session(log_path, thinking=2, thinking_tokens=20, tools=1, turn_open=True, queue_len=1)
    runtime = _runtime(session)

    # A closed turn leaves its counter visible until queued work starts.
    _append(
        log_path,
        {
            "type": "message",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "turn one done"}], "stopReason": "stop"},
        },
    )
    runtime.update_meta_counters()
    assert session.meta_turn_open is False
    assert session.meta_thinking_tokens == 20

    # A user row after that close replaces the window; the following assistant
    # row contributes Pi's recorded (not estimated) reasoning usage.
    _append(log_path, _pi_user("queued turn"), _pi_thinking("queued thought", 40))
    runtime.update_meta_counters()
    assert session.meta_turn_open is True
    assert session.meta_thinking_tokens == 40

    # The user row is a steer while the same turn remains open, so exact token
    # usage is preserved and extended rather than reset.
    _append(log_path, _pi_user("steer"), _pi_thinking("steered thought", 16))
    runtime.update_meta_counters()
    assert session.meta_turn_open is True
    assert session.meta_thinking_tokens == 56


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
