"""Pi bridge readiness: a send injected before the backend can read input.

A submission injected while Pi is still booting waits in the PTY's input buffer
(or is consumed by a pre-startup gate), so it has not been offered to the
backend at all. The broker must not judge such a send from the moment it was
injected; the bridge's live session marker (written from ``session_start``,
after the interactive UI started reading stdin) is the readiness evidence it
uses to start the send-latch quiet window.
"""

import json
import threading
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from codoxear import broker as broker_mod
from codoxear.broker import Broker
from codoxear.broker_launch import PiActiveSessionMarkerObserver
from codoxear.broker_launch import _pi_bridge_session_is_live
from codoxear.broker_turn_state import State


def _state(root: Path) -> State:
    return State(
        codex_pid=42,
        pty_master_fd=1,
        cwd=str(root),
        start_ts=0.0,
        codex_home=root,
        sessions_dir=root,
        sock_path=root / "broker.sock",
    )


def _write_marker(marker_path: Path, *, pid: int, session_file: Path) -> None:
    marker_path.write_text(
        json.dumps(
            {
                "version": 1,
                "bridgeVersion": 2,
                "reason": "startup",
                "sessionFile": str(session_file),
                "sessionId": "pi-session",
                "cwd": "/tmp",
                "pid": pid,
                "updatedAt": "2026-10-22T00:00:00.000Z",
            }
        ),
        encoding="utf-8",
    )


def _broker(root: Path, state: State, marker_path: Path) -> Broker:
    broker = Broker.__new__(Broker)
    broker.sessions_dir = root
    broker.state = state
    broker._lock = threading.Lock()
    broker.pi_active_session_marker_path = marker_path
    broker._pi_active_session_marker_observer = PiActiveSessionMarkerObserver()
    return broker


def test_live_bridge_marker_of_this_process_marks_backend_ready() -> None:
    with TemporaryDirectory() as td:
        root = Path(td)
        session_file = root / "2026-10-22_session.jsonl"
        session_file.write_text("", encoding="utf-8")
        marker_path = root / "active.json"
        _write_marker(marker_path, pid=42, session_file=session_file)
        state = _state(root)

        with patch.object(broker_mod, "AGENT_BACKEND", "pi"):
            _broker(root, state, marker_path)._refresh_pi_active_session_observability(process_pid=42)

        assert state.backend_ready_ts > 0.0


def test_marker_written_by_another_process_is_not_readiness() -> None:
    with TemporaryDirectory() as td:
        root = Path(td)
        session_file = root / "2026-10-22_session.jsonl"
        session_file.write_text("", encoding="utf-8")
        marker_path = root / "active.json"
        _write_marker(marker_path, pid=7, session_file=session_file)
        state = _state(root)

        with patch.object(broker_mod, "AGENT_BACKEND", "pi"):
            _broker(root, state, marker_path)._refresh_pi_active_session_observability(process_pid=42)

        assert state.backend_ready_ts == 0.0
        assert not _pi_bridge_session_is_live(marker_path, sessions_dir=root, process_pid=42)


def test_marker_naming_a_file_outside_the_sessions_dir_is_not_readiness() -> None:
    with TemporaryDirectory() as sessions_td, TemporaryDirectory() as other_td:
        root = Path(sessions_td)
        foreign = Path(other_td) / "foreign.jsonl"
        foreign.write_text("", encoding="utf-8")
        marker_path = root / "active.json"
        _write_marker(marker_path, pid=42, session_file=foreign)

        assert not _pi_bridge_session_is_live(marker_path, sessions_dir=root, process_pid=42)


def test_absent_or_malformed_marker_is_not_readiness() -> None:
    with TemporaryDirectory() as td:
        root = Path(td)
        marker_path = root / "active.json"

        assert not _pi_bridge_session_is_live(marker_path, sessions_dir=root, process_pid=42)

        marker_path.write_text("{not json", encoding="utf-8")
        assert not _pi_bridge_session_is_live(marker_path, sessions_dir=root, process_pid=42)

        state = _state(root)
        with patch.object(broker_mod, "AGENT_BACKEND", "pi"):
            _broker(root, state, marker_path)._refresh_pi_active_session_observability(process_pid=42)
        assert state.backend_ready_ts == 0.0
