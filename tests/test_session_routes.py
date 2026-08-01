from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import urllib.parse

from codoxear.session_routes import SessionRouteDeps
from codoxear.session_routes import handle_session_get_route
from codoxear.session_routes import handle_session_post_route


class _FakeHandler:
    def __init__(self) -> None:
        self.unauthorized = False

    def _unauthorized(self) -> None:
        self.unauthorized = True


class _ValidationError(Exception):
    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


class _LaunchError(RuntimeError):
    def __init__(self, message: str, record: dict[str, object]) -> None:
        super().__init__(message)
        self.record = record


class _FakeManager:
    def __init__(self) -> None:
        self.spawn_args: dict[str, object] | None = None

    def list_sessions(self):
        return [{"id": "s1"}]

    def recent_cwds(self):
        return ["/repo"]

    def alias_get(self, sid: str) -> str:
        return f"alias-{sid}"

    def get_tail(self, sid: str) -> str:
        if sid == "missing":
            raise KeyError(sid)
        return "tail text"

    def unattended_get(self, sid: str):
        if sid == "missing":
            raise KeyError(sid)
        return {"enabled": True, "request": "continue"}

    def spawn_web_session(self, **kwargs):
        self.spawn_args = kwargs
        return {"id": "new-session"}


def _match_session_route(path: str, *parts: str) -> str | None:
    expected = "/api/sessions/s1/" + "/".join(parts)
    if path == expected:
        return "s1"
    missing = "/api/sessions/missing/" + "/".join(parts)
    if path == missing:
        return "missing"
    return None


def _deps(**overrides):
    responses: list[tuple[int, dict[str, object]]] = []
    etag_payloads: list[dict[str, object]] = []
    metrics: list[tuple[str, float]] = []
    counter = iter([10.0, 10.123])

    def json_response(_handler, status: int, payload: dict[str, object]) -> None:
        responses.append((status, payload))

    def json_response_with_etag(_handler, payload: dict[str, object]) -> None:
        etag_payloads.append(payload)

    deps = SessionRouteDeps(
        require_auth=lambda _handler: True,
        json_response=json_response,
        json_response_with_etag=json_response_with_etag,
        read_json_body=lambda _handler, **_kwargs: {},
        read_new_session_defaults=lambda: {"backend": "codex"},
        tmux_available=lambda: True,
        tmux_session_name="codoxear",
        metrics_snapshot=lambda: {"api_sessions_ms": {"count": 1}},
        record_metric=lambda name, value: metrics.append((name, value)),
        perf_counter=lambda: next(counter),
        normalize_agent_backend=lambda raw, *, default="codex": raw or default,
        default_agent_backend="codex",
        resolve_dir_target=lambda raw, *, field_name: Path(raw),
        describe_session_cwd=lambda cwd: {"cwd": str(cwd), "exists": True},
        list_resume_candidates_for_cwd=lambda cwd, *, agent_backend: [{"session_id": "s1", "log_path": "/tmp/log.jsonl"}],
        first_user_message_preview_from_log=lambda path: f"first:{path.name}",
        parse_new_session_launch_request=lambda obj: SimpleNamespace(
            cwd="/repo",
            args=["--search"],
            agent_backend="codex",
            resume_session_id=None,
            worktree_branch=None,
            model_provider="openai",
            preferred_auth_method="api-key",
            model="gpt-5",
            reasoning_effort="medium",
            service_tier="default",
            create_in_tmux=False,
        ),
        launch_request_validation_error=_ValidationError,
        session_launch_error=_LaunchError,
    )
    for name, value in overrides.items():
        object.__setattr__(deps, name, value)
    return deps, responses, etag_payloads, metrics


def test_handle_session_get_route_lists_sessions_with_etag_and_metric() -> None:
    deps, responses, etag_payloads, metrics = _deps()
    handled = handle_session_get_route(
        _FakeHandler(),
        path="/api/sessions",
        query="",
        manager=_FakeManager(),
        deps=deps,
        match_session_route=_match_session_route,
    )
    assert handled is True
    assert responses == []
    assert etag_payloads == [
        {
            "sessions": [{"id": "s1"}],
            "recent_cwds": ["/repo"],
            "new_session_defaults": {"backend": "codex"},
            "tmux_available": True,
            "tmux_session_name": "codoxear",
        }
    ]
    assert metrics[0][0] == "api_sessions_ms"
    assert abs(metrics[0][1] - 123.0) < 1e-9


def test_handle_session_get_route_resume_candidates_adds_alias_and_preview() -> None:
    deps, responses, _etag_payloads, _metrics = _deps()
    handled = handle_session_get_route(
        _FakeHandler(),
        path="/api/session_resume_candidates",
        query="cwd=/repo&agent_backend=pi",
        manager=_FakeManager(),
        deps=deps,
        match_session_route=_match_session_route,
    )
    assert handled is True
    assert responses == [
        (
            200,
            {
                "ok": True,
                "cwd": "/repo",
                "exists": True,
                "sessions": [
                    {
                        "session_id": "s1",
                        "log_path": "/tmp/log.jsonl",
                        "alias": "alias-s1",
                        "first_user_message": "first:log.jsonl",
                    }
                ],
            },
        )
    ]


def test_handle_session_get_route_cwd_suggest_lists_immediate_directories() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        (root / "project").mkdir()
        (root / ".hidden").mkdir()
        (root / "notes.txt").write_text("not a directory", encoding="utf-8")
        deps, responses, _etag_payloads, _metrics = _deps()
        query = urllib.parse.urlencode({"path": str(root)})
        assert handle_session_get_route(
            _FakeHandler(),
            path="/api/cwd-suggest",
            query=query,
            manager=_FakeManager(),
            deps=deps,
            match_session_route=_match_session_route,
        ) is True
        assert responses == [(200, {"directories": [{"name": "project", "path": str(root / "project")}]})]

        responses.clear()
        assert handle_session_get_route(
            _FakeHandler(),
            path="/api/cwd-suggest",
            query=urllib.parse.urlencode({"path": str(root), "prefix": "."}),
            manager=_FakeManager(),
            deps=deps,
            match_session_route=_match_session_route,
        ) is True
        assert responses[0][0] == 200
        directories = responses[0][1]["directories"]
        assert isinstance(directories, list)
        assert {(item["name"], item["path"]) for item in directories if isinstance(item, dict)} == {
            ("project", str(root / "project")),
            (".hidden", str(root / ".hidden")),
        }


def test_handle_session_get_route_tail_and_unattended_map_unknown_session() -> None:
    deps, responses, _etag_payloads, _metrics = _deps()
    manager = _FakeManager()
    assert handle_session_get_route(
        _FakeHandler(),
        path="/api/sessions/s1/tail",
        query="",
        manager=manager,
        deps=deps,
        match_session_route=_match_session_route,
    ) is True
    assert handle_session_get_route(
        _FakeHandler(),
        path="/api/sessions/missing/unattended",
        query="",
        manager=manager,
        deps=deps,
        match_session_route=_match_session_route,
    ) is True
    assert responses == [(200, {"tail": "tail text"}), (404, {"error": "unknown session"})]


def test_handle_session_post_route_validation_error_preserves_field() -> None:
    deps, responses, _etag_payloads, _metrics = _deps(
        parse_new_session_launch_request=lambda _obj: (_ for _ in ()).throw(_ValidationError("bad cwd", field="cwd"))
    )
    handled = handle_session_post_route(
        _FakeHandler(),
        path="/api/sessions",
        manager=_FakeManager(),
        deps=deps,
    )
    assert handled is True
    assert responses == [(400, {"error": "bad cwd", "field": "cwd"})]


def test_handle_session_post_route_spawns_with_launch_request_fields() -> None:
    deps, responses, _etag_payloads, _metrics = _deps()
    manager = _FakeManager()
    handled = handle_session_post_route(
        _FakeHandler(),
        path="/api/sessions",
        manager=manager,
        deps=deps,
    )
    assert handled is True
    assert responses == [(200, {"ok": True, "id": "new-session"})]
    assert manager.spawn_args == {
        "cwd": "/repo",
        "args": ["--search"],
        "agent_backend": "codex",
        "resume_session_id": None,
        "worktree_branch": None,
        "model_provider": "openai",
        "preferred_auth_method": "api-key",
        "model": "gpt-5",
        "reasoning_effort": "medium",
        "service_tier": "default",
        "create_in_tmux": False,
    }
