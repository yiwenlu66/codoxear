from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from codoxear import broker
from codoxear.broker_turn_state import State
from codoxear.cc_live_settings import changed_cc_settings_reasoning_effort
from codoxear.session_runtime import session_run_settings_from_meta


def _cc_settings_effort_from_meta(meta: dict[str, object]) -> str | None:
    _provider, _auth, _model, effort = session_run_settings_from_meta(
        meta=meta,
        log_path=None,
        agent_backend="cc",
        clean_optional_text=lambda value: value.strip() if isinstance(value, str) and value.strip() else None,
        normalize_requested_preferred_auth_method=lambda _value: None,
        display_reasoning_effort=lambda _value: None,
        display_pi_reasoning_effort=lambda _value: None,
        normalize_requested_cc_reasoning_effort=lambda value: (
            value.strip().lower()
            if isinstance(value, str) and value.strip().lower() in {"low", "medium", "high", "xhigh", "max", "auto"}
            else None
        ),
        read_run_settings_from_log=lambda _path, **_kwargs: (None, None, None),
    )
    return effort


def test_cc_broker_publishes_post_launch_settings_effort_and_listing_prefers_it(tmp_path: Path, monkeypatch) -> None:
    """A fake Claude settings rewrite travels through the real broker sidecar
    and the same metadata reconciliation used to build a sidebar session."""
    settings_path = tmp_path / "settings.json"
    settings_path.write_text('{"effortLevel":"low"}', encoding="utf-8")
    initial_mtime_ns = int(settings_path.stat().st_mtime_ns)
    socket_dir = tmp_path / "socks"
    socket_path = socket_dir / "broker-test.sock"
    st = State(
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
    live_broker.state = st

    monkeypatch.setattr(broker, "AGENT_BACKEND", "cc")
    monkeypatch.setattr(broker, "SOCK_DIR", socket_dir)

    settings_path.write_text('{"effortLevel":"high"}', encoding="utf-8")
    # Filesystems with coarse write timestamps still need a distinct observed
    # revision for this behavioral test.
    os.utime(settings_path, ns=(initial_mtime_ns + 1_000_000, initial_mtime_ns + 1_000_000))

    live_broker._refresh_cc_live_effort()

    meta = json.loads(socket_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["reasoning_effort"] == "high"
    assert meta["live_run_settings"] == {"reasoning_effort": "high"}

    stale_launch_meta = {**meta, "reasoning_effort": "low"}
    assert _cc_settings_effort_from_meta(stale_launch_meta) == "high"


def test_cc_settings_observation_requires_new_valid_revision(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text('{"effortLevel":"medium"}', encoding="utf-8")
    initial_mtime_ns = int(settings_path.stat().st_mtime_ns)

    same_mtime_ns, same_effort = changed_cc_settings_reasoning_effort(
        settings_path,
        previous_mtime_ns=initial_mtime_ns,
    )
    assert same_mtime_ns == initial_mtime_ns
    assert same_effort is None

    settings_path.write_text('{"effortLevel":"unsupported"}', encoding="utf-8")
    os.utime(settings_path, ns=(initial_mtime_ns + 2_000_000, initial_mtime_ns + 2_000_000))
    changed_mtime_ns, changed_effort = changed_cc_settings_reasoning_effort(
        settings_path,
        previous_mtime_ns=initial_mtime_ns,
    )
    assert changed_mtime_ns != initial_mtime_ns
    assert changed_effort is None
