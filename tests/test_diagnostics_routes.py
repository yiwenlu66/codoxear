from pathlib import Path
import tempfile

from codoxear.diagnostics_routes import DiagnosticsRouteDeps
from codoxear.diagnostics_routes import handle_diagnostics_get_route
from codoxear.server import Session
from codoxear.server import _match_session_route


class Handler:
    def __init__(self):
        self.unauthorized = False

    def _unauthorized(self):
        self.unauthorized = True


class Manager:
    def __init__(self, session):
        self.session = session
        self.refreshed = []
        self.state = {"busy": True, "queue_len": 0, "token": {"tokens_in_context": 1}}
        self.idle = True
        self.boundary = False
        self.queue_len = 3
        self.sidebar = {"priority_offset": 0.2, "snooze_until": None, "dependency_session_id": None}

    def refresh_session_meta(self, session_id):
        self.refreshed.append(session_id)

    def get_session(self, session_id):
        return self.session

    def get_state(self, session_id):
        return dict(self.state)

    def _log_size_or_none(self, log_path):
        return log_path.stat().st_size if log_path is not None and log_path.exists() else None

    def _confirmed_send_boundary_unresolved_for_session(self, session_id, log_path, log_size):
        return self.boundary

    def idle_from_log(self, session_id):
        return self.idle

    def sidebar_meta_get(self, session_id):
        return dict(self.sidebar)

    def _queue_len(self, session_id):
        return self.queue_len


def _deps(responses, *, auth=True, log_settings=("anthropic", "claude", "medium"), now=130.0):
    return DiagnosticsRouteDeps(
        require_auth=lambda _handler: auth,
        json_response=lambda _handler, status, obj: responses.append((status, obj)),
        provider_choice_for_settings=lambda **kwargs: kwargs.get("preferred_auth_method") or kwargs.get("model_provider") or "default",
        read_run_settings_from_log=lambda _path, **_kwargs: log_settings,
        resolve_session_cwd=lambda cwd: Path(cwd),
        current_git_branch=lambda _cwd: "main",
        sidebar_time_priority_from_elapsed_seconds=lambda elapsed: round(elapsed / 100.0, 3),
        clip01=lambda value: max(0.0, min(1.0, value)),
        time_fn=lambda: now,
    )


def _session(log_path: Path | None) -> Session:
    return Session(
        session_id="s1",
        thread_id="t1",
        broker_pid=11,
        codex_pid=12,
        agent_backend="codex",
        owned=False,
        start_ts=100.0,
        cwd="/tmp",
        log_path=log_path,
        sock_path=Path("/tmp/s1.sock"),
        busy=True,
        queue_len=0,
        last_chat_ts=120.0,
        token={"tokens_in_context": 9},
        tmux_session="tmux-s",
        tmux_window="win",
    )


def test_diagnostics_route_uses_runtime_log_idle_over_stale_broker_busy_and_log_token() -> None:
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "rollout.jsonl"
        log_path.write_text("{}\n", encoding="utf-8")
        session = _session(log_path)
        responses = []
        manager = Manager(session)
        handled = handle_diagnostics_get_route(
            Handler(),
            path="/api/sessions/s1/diagnostics",
            manager=manager,
            deps=_deps(responses),
            match_session_route=_match_session_route,
        )
    assert handled is True
    assert manager.refreshed == ["s1"]
    status, body = responses[0]
    assert status == 200
    assert body["busy"] is False
    assert body["broker_busy"] is True
    assert body["queue_len"] == 3
    assert body["token"] == {"tokens_in_context": 9}
    assert body["model_provider"] == "anthropic"
    assert body["model"] == "claude"
    assert body["reasoning_effort"] == "medium"
    assert body["provider_choice"] == "anthropic"
    assert body["git_branch"] == "main"
    assert body["time_priority"] == 0.1
    assert abs(body["base_priority"] - 0.3) < 1e-9
    assert abs(body["final_priority"] - 0.3) < 1e-9


def test_diagnostics_route_boundary_forces_busy_without_idle_parse() -> None:
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "rollout.jsonl"
        log_path.write_text("{}\n", encoding="utf-8")
        session = _session(log_path)
        manager = Manager(session)
        manager.boundary = True
        manager.idle_from_log = lambda _sid: (_ for _ in ()).throw(AssertionError("boundary should skip log idle"))
        responses = []
        handle_diagnostics_get_route(
            Handler(),
            path="/api/sessions/s1/diagnostics",
            manager=manager,
            deps=_deps(responses),
            match_session_route=_match_session_route,
        )
    assert responses[0][1]["busy"] is True


def test_diagnostics_route_uses_broker_token_when_no_log() -> None:
    session = _session(None)
    session.token = None
    manager = Manager(session)
    responses = []
    handle_diagnostics_get_route(
        Handler(),
        path="/api/sessions/s1/diagnostics",
        manager=manager,
        deps=_deps(responses),
        match_session_route=_match_session_route,
    )
    assert responses[0][1]["busy"] is False
    assert responses[0][1]["token"] == {"tokens_in_context": 1}


def test_diagnostics_route_auth_and_unknown_session() -> None:
    session = _session(None)
    manager = Manager(session)
    responses = []
    handler = Handler()
    assert handle_diagnostics_get_route(
        handler,
        path="/api/sessions/s1/diagnostics",
        manager=manager,
        deps=_deps(responses, auth=False),
        match_session_route=_match_session_route,
    ) is True
    assert handler.unauthorized is True
    assert responses == []

    manager.session = None
    assert handle_diagnostics_get_route(
        Handler(),
        path="/api/sessions/s1/diagnostics",
        manager=manager,
        deps=_deps(responses),
        match_session_route=_match_session_route,
    ) is True
    assert responses == [(404, {"error": "unknown session"})]
