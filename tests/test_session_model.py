from __future__ import annotations

from pathlib import Path

from codoxear.session_model import Session
from codoxear import server


def test_server_reexports_canonical_session_model() -> None:
    assert server.Session is Session


def test_session_model_preserves_defaults_for_runtime_state() -> None:
    session = Session(
        session_id="sid",
        thread_id="thread",
        broker_pid=1,
        codex_pid=2,
        agent_backend="codex",
        owned=True,
        start_ts=123.0,
        cwd="/tmp/work",
        log_path=None,
        sock_path=Path("/tmp/sid.sock"),
    )

    assert session.busy is False
    assert session.queue_len == 0
    assert session.token is None
    assert session.last_chat_history_scanned is False
    assert session.meta_log_off == 0
    assert session.delivery_log_off == 0
    assert session.idle_cache_log_off == -1
    assert session.transport is None
    assert session.pending_attachment is False
    assert session.commit_unknown_send is None
    assert session.sync_send_supported is False
    assert session.key_write_errors_supported is False
    assert session.interrupted_idle is False
    assert session.last_send_boundary_active is False
    assert session.last_send_log_path is None
    assert session.last_send_log_size is None
