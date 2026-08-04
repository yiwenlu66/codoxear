from __future__ import annotations

import json
from pathlib import Path

from codoxear.broker_launch import PiActiveSessionMarkerObserver
from codoxear.broker_launch import _read_pi_active_session_marker_state


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_pi_bridge_marker_state_and_observer_distinguish_handover_stale_and_caps_anomaly(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    marker = tmp_path / "broker-10.json"
    log = sessions_dir / "active.jsonl"
    log.write_text("{}\n", encoding="utf-8")
    process_pid = 101
    _write_json(
        marker,
        {
            "version": 1,
            "bridgeVersion": 2,
            "pid": process_pid,
            "sessionFile": str(log),
            "sessionId": "session-a",
            "reason": "session_start",
        },
    )
    _write_json(
        Path(f"{marker}.caps"),
        {
            "bridgeVersion": 2,
            "pid": process_pid,
            "features": ["effort", "thinking"],
            "commands": [{"name": "effort"}, {"name": "thinking"}],
        },
    )

    observer = PiActiveSessionMarkerObserver()
    active = _read_pi_active_session_marker_state(marker, sessions_dir=sessions_dir, process_pid=process_pid)
    assert active["marker"]["active"] is True
    assert active["marker"]["session_file"] == str(log.resolve())
    assert active["caps"]["thinking_capable"] is True
    assert active["caps"]["command_names"] == ["effort", "thinking"]
    assert active["commands_without_thinking_capability"] is False
    assert observer.observe(active) == []

    handover_pid = process_pid + 1
    _write_json(
        marker,
        {
            "version": 1,
            "bridgeVersion": 2,
            "pid": handover_pid,
            "sessionFile": str(log),
        },
    )
    handover = _read_pi_active_session_marker_state(marker, sessions_dir=sessions_dir, process_pid=process_pid)
    assert handover["marker"]["current_process"] is False
    assert observer.observe(handover) == [f"Pi bridge marker handover: writer pid changed {process_pid} -> {handover_pid}"]

    marker.unlink()
    stale = _read_pi_active_session_marker_state(marker, sessions_dir=sessions_dir, process_pid=process_pid)
    assert stale["marker"]["present"] is False
    assert observer.observe(stale) == ["Pi bridge marker went stale: marker file was deleted"]

    _write_json(
        Path(f"{marker}.caps"),
        {"bridgeVersion": 2, "pid": process_pid, "commands": [{"name": "effort"}]},
    )
    registry_without_caps = _read_pi_active_session_marker_state(marker, sessions_dir=sessions_dir, process_pid=process_pid)
    assert registry_without_caps["caps"]["commands_registered"] is True
    assert registry_without_caps["caps"]["thinking_capable"] is False
    assert registry_without_caps["commands_without_thinking_capability"] is True
    assert observer.observe(registry_without_caps) == [
        "Pi bridge caps anomaly: command registry (1 commands) has no current thinking capability"
    ]


def test_pi_bridge_marker_state_keeps_out_of_tree_session_file_untrusted(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    marker = tmp_path / "broker-10.json"
    outside_log = tmp_path / "outside.jsonl"
    _write_json(
        marker,
        {"version": 1, "bridgeVersion": 2, "pid": 101, "sessionFile": str(outside_log)},
    )

    state = _read_pi_active_session_marker_state(marker, sessions_dir=sessions_dir, process_pid=101)

    assert state["marker"]["parse_valid"] is True
    assert state["marker"]["session_file_valid"] is False
    assert state["marker"]["active"] is False
    assert state["marker"]["session_file"] is None
