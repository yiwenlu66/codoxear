from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from codoxear import broker
from codoxear.broker_turn_state import State


def test_cc_settings_rewrite_updates_live_run_settings_only_after_mtime_change(tmp_path: Path, monkeypatch) -> None:
    """A Claude /effort rewrite becomes live sidecar evidence, not launch intent."""
    settings_path = tmp_path / "settings.json"
    settings_path.write_text('{"effortLevel":"low"}', encoding="utf-8")
    initial_mtime_ns = settings_path.stat().st_mtime_ns
    socket_dir = tmp_path / "socks"
    socket_path = socket_dir / "cc-live.sock"
    state = State(
        codex_pid=os.getpid(),
        pty_master_fd=-1,
        cwd=str(tmp_path),
        start_ts=1.0,
        codex_home=tmp_path,
        sessions_dir=tmp_path / "projects",
        sock_path=socket_path,
        reasoning_effort="low",
    )
    live_broker = object.__new__(broker.Broker)
    live_broker._cc_settings_path = settings_path
    live_broker._cc_settings_mtime_ns = initial_mtime_ns
    live_broker._lock = threading.Lock()
    live_broker.state = state

    monkeypatch.setattr(broker, "AGENT_BACKEND", "cc")
    monkeypatch.setattr(broker, "SOCK_DIR", socket_dir)

    # An unchanged pre-launch preference cannot override this session's launch effort.
    live_broker._refresh_cc_live_effort()
    assert state.reasoning_effort == "low"
    assert state.live_run_settings is None
    assert not socket_path.with_suffix(".json").exists()

    settings_path.write_text('{"effortLevel":"high"}', encoding="utf-8")
    os.utime(settings_path, ns=(initial_mtime_ns + 1_000_000, initial_mtime_ns + 1_000_000))

    live_broker._refresh_cc_live_effort()

    sidecar = json.loads(socket_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert state.reasoning_effort == "high"
    assert state.live_run_settings == {"reasoning_effort": "high"}
    assert sidecar["live_run_settings"] == {"reasoning_effort": "high"}
    assert sidecar["reasoning_effort"] == "high"
