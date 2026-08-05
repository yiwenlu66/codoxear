from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from codoxear.broker_launch import _pi_active_session_marker_path
from codoxear.broker_launch import _read_pi_active_session_marker
from codoxear.diagnostics_routes import handle_diagnostics_get_route
from codoxear.server import _match_session_route
from codoxear.server_route_deps import ServerRouteDepsFactory
from codoxear.session_model import Session
from codoxear.session_runtime import broker_runtime_state
from codoxear.session_runtime import resolve_runtime_status


class _Handler:
    def __init__(self) -> None:
        self.unauthorized = False

    def _unauthorized(self) -> None:
        self.unauthorized = True


class _Manager:
    def __init__(self, session: Session) -> None:
        self.session = session

    def refresh_session_meta(self, _session_id: str) -> None:
        return None

    def get_session(self, _session_id: str) -> Session:
        return self.session

    def get_state(self, _session_id: str) -> dict[str, object]:
        return {"busy": False, "queue_len": 0}

    def _runtime_status_from_state_and_log(self, _session_id: str, state: dict[str, object], log_path: Path | None):
        return resolve_runtime_status(
            broker=broker_runtime_state(state),
            log_exists=log_path is not None and log_path.exists(),
            log_idle=True if log_path is not None and log_path.exists() else None,
            send_boundary_unresolved=False,
        )

    def sidebar_meta_get(self, _session_id: str) -> dict[str, object]:
        return {"priority_offset": 0.0, "snooze_until": None, "dependency_session_id": None}

    def _queue_len(self, _session_id: str) -> int:
        return 0


def test_pi_remote_marker_binds_pid_scoped_log_and_diagnostics_exposes_bridge_state(tmp_path: Path, monkeypatch) -> None:
    app_dir = tmp_path / ".local" / "share" / "codoxear"
    pi_home = tmp_path / "pi-home"
    sessions_dir = pi_home / "agent" / "sessions"
    log_path = sessions_dir / "remote-session.jsonl"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("{}\n", encoding="utf-8")
    broker_pid = 4242
    process_pid = 4343

    monkeypatch.setattr("codoxear.broker_launch._default_app_dir", lambda: app_dir)
    monkeypatch.setenv("PI_HOME", str(pi_home))
    marker_path = _pi_active_session_marker_path(broker_pid=broker_pid)
    assert marker_path == app_dir / "pi-active-sessions" / f"broker-{broker_pid}.json"
    marker_path.parent.mkdir(parents=True)
    marker_path.write_text(
        json.dumps(
            {
                "version": 1,
                "bridgeVersion": 2,
                "pid": process_pid,
                "sessionFile": str(log_path),
                "sessionId": "remote-session",
            }
        ),
        encoding="utf-8",
    )

    assert _read_pi_active_session_marker(marker_path, sessions_dir=sessions_dir) == log_path.resolve()

    responses: list[tuple[int, dict[str, object]]] = []
    server = SimpleNamespace(
        _require_auth=lambda _handler: True,
        _json_response=lambda _handler, status, body: responses.append((status, body)),
        _provider_choice_for_settings=lambda **_kwargs: "default",
        _read_run_settings_from_log=lambda _path, **_kwargs: (None, None, None),
        _resolve_session_cwd=lambda cwd: Path(cwd),
        _current_git_branch=lambda _cwd: None,
        _sidebar_time_priority_from_elapsed_seconds=lambda _elapsed: 0.0,
        _clip01=lambda value: value,
        time=SimpleNamespace(time=lambda: 100.0),
    )
    deps = ServerRouteDepsFactory(server=server, config=SimpleNamespace()).diagnostics_route_deps()
    session = Session(
        session_id="s1",
        thread_id="remote-session",
        broker_pid=broker_pid,
        codex_pid=process_pid,
        agent_backend="pi",
        owned=False,
        start_ts=100.0,
        cwd=str(tmp_path),
        log_path=log_path,
        sock_path=tmp_path / "s1.sock",
    )

    assert handle_diagnostics_get_route(
        _Handler(),
        path="/api/sessions/s1/diagnostics",
        manager=_Manager(session),
        deps=deps,
        match_session_route=_match_session_route,
    ) is True

    assert responses[0][0] == 200
    pi_bridge_marker = responses[0][1]["pi_bridge_marker"]
    assert pi_bridge_marker["marker"]["active"] is True
    assert pi_bridge_marker["marker"]["session_file"] == str(log_path.resolve())
    assert pi_bridge_marker["marker"]["pid"] == process_pid
