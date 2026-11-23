from __future__ import annotations

import http.client
import http.server
import json
from pathlib import Path
import subprocess
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from codoxear import server
from codoxear import source_workspace
from codoxear.auth import sign_cookie
from codoxear.server_handler import make_server_handler
from codoxear.server_route_deps import ServerRouteDepsFactory
from codoxear.source_workspace_routes import SourceWorkspaceRouteDeps
from codoxear.source_workspace_routes import handle_source_workspace_get_route
from codoxear.source_workspace_routes import handle_source_workspace_post_route


class _FakeHandler:
    def __init__(self) -> None:
        self.unauthorized = False

    def _unauthorized(self) -> None:
        self.unauthorized = True


def _route_deps(
    *,
    auth: bool,
    discover: Any,
    setup: Any,
) -> tuple[SourceWorkspaceRouteDeps, list[tuple[int, dict[str, Any]]], list[bytes]]:
    responses: list[tuple[int, dict[str, Any]]] = []
    bodies: list[bytes] = []
    return SourceWorkspaceRouteDeps(
        require_auth=lambda _handler: auth,
        json_response=lambda _handler, status, payload: responses.append((status, payload)),
        read_body=lambda _handler: bodies.append(b"{}") or b"{}",
        discover=discover,
        setup=setup,
    ), responses, bodies


def test_routes_require_auth_and_ignore_unowned_paths() -> None:
    calls: list[str] = []
    deps, responses, bodies = _route_deps(
        auth=False,
        discover=lambda: calls.append("discover") or {},
        setup=lambda: calls.append("setup") or {},
    )
    handler = _FakeHandler()

    assert handle_source_workspace_get_route(handler, path="/api/other", deps=deps) is False
    assert handle_source_workspace_post_route(handler, path="/api/other", deps=deps) is False
    assert handle_source_workspace_get_route(handler, path="/api/source-workspace", deps=deps) is True
    assert handle_source_workspace_post_route(handler, path="/api/source-workspace", deps=deps) is True
    assert handler.unauthorized is True
    assert calls == []
    assert responses == []
    assert bodies == []


def test_get_returns_discovery_shape_and_post_consumes_body() -> None:
    unavailable = {"available": False, "path": None, "can_setup": True, "reason": "missing"}
    ready = {"available": True, "path": "/source", "can_setup": False, "reason": "source workspace is ready"}
    deps, responses, bodies = _route_deps(auth=True, discover=lambda: unavailable, setup=lambda: ready)

    assert handle_source_workspace_get_route(_FakeHandler(), path="/api/source-workspace", deps=deps) is True
    assert handle_source_workspace_post_route(_FakeHandler(), path="/api/source-workspace", deps=deps) is True

    assert responses == [(200, unavailable), (200, ready)]
    assert bodies == [b"{}"]


def test_post_projects_controlled_setup_error() -> None:
    def fail() -> dict[str, Any]:
        raise source_workspace.SourceWorkspaceSetupError("occupied; refusing to replace it", status=409)

    deps, responses, bodies = _route_deps(auth=True, discover=lambda: {}, setup=fail)
    assert handle_source_workspace_post_route(_FakeHandler(), path="/api/source-workspace", deps=deps) is True
    assert responses == [(409, {"error": "occupied; refusing to replace it"})]
    assert bodies == [b"{}"]


def _make_remote(path: Path) -> Path:
    path.mkdir()
    (path / "AGENTS.md").write_text("# Source fixture\n", encoding="utf-8")
    (path / "pyproject.toml").write_text(
        "[project]\nname = \"codoxear\"\n\n"
        "[project.scripts]\n"
        "codoxear-server = \"codoxear.server:main\"\n"
        "codoxear-broker = \"codoxear.broker:main\"\n"
        "codoxear-sessiond = \"codoxear.sessiond:main\"\n",
        encoding="utf-8",
    )
    tracked = ["AGENTS.md", "pyproject.toml"]
    for module in ("server", "broker", "sessiond"):
        module_path = path / "codoxear" / f"{module}.py"
        module_path.parent.mkdir(exist_ok=True)
        module_path.write_text("def main():\n    pass\n", encoding="utf-8")
        tracked.append(f"codoxear/{module}.py")
    for name in ("codoxear-report-issue", "codoxear-contribute-fix"):
        skill = path / ".agents" / "skills" / name / "SKILL.md"
        skill.parent.mkdir(parents=True, exist_ok=True)
        skill.write_text(f"---\nname: {name}\n---\nTest source setup.\n", encoding="utf-8")
        projection = path / ".claude" / "skills" / name
        projection.parent.mkdir(parents=True, exist_ok=True)
        projection.symlink_to(f"../../.agents/skills/{name}", target_is_directory=True)
    tracked.extend([".agents/skills", ".claude/skills"])
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "routes@example.invalid"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Route Test"], cwd=path, check=True)
    subprocess.run(["git", "add", "--", *tracked], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "source fixture"], cwd=path, check=True, capture_output=True)
    return path


def _read_json(response: http.client.HTTPResponse) -> tuple[int, dict[str, Any]]:
    body = response.read()
    return response.status, json.loads(body)


def test_real_http_route_absent_get_then_explicit_clone_and_connection_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = _make_remote(tmp_path / "canonical-remote")
    app_dir = tmp_path / "runtime"
    monkeypatch.setattr(source_workspace, "CANONICAL_CLONE_URL", str(remote))
    monkeypatch.setattr(server, "URL_PREFIX", "")
    monkeypatch.setattr(server, "HMAC_SECRET", b"source-workspace-route-test")
    monkeypatch.setattr(server, "MANAGER", SimpleNamespace(_voice_push=None))

    def route_deps(factory: ServerRouteDepsFactory) -> SourceWorkspaceRouteDeps:
        route_server = factory.server
        kwargs = {
            "app_dir": app_dir,
            "working_dir": tmp_path / "server-cwd",
            "module_root": tmp_path / "installed-package",
            "direct_url": {},
        }
        return SourceWorkspaceRouteDeps(
            require_auth=route_server._require_auth,
            json_response=route_server._json_response,
            read_body=route_server._read_body,
            discover=lambda: source_workspace.discover_source_workspace(**kwargs),
            setup=lambda: source_workspace.setup_source_workspace(**kwargs),
        )

    monkeypatch.setattr(ServerRouteDepsFactory, "source_workspace_route_deps", route_deps)
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), make_server_handler(server))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    cookie = f"{server.COOKIE_NAME}={sign_cookie({'v': 1}, secret=server.HMAC_SECRET)}"
    try:
        connection = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=10)
        try:
            connection.request("GET", "/api/source-workspace")
            status, payload = _read_json(connection.getresponse())
            assert (status, payload) == (401, {"error": "unauthorized"})

            connection.request("GET", "/api/source-workspace", headers={"Cookie": cookie})
            status, before = _read_json(connection.getresponse())
            assert status == 200
            assert before["available"] is False
            assert before["can_setup"] is True
            assert not app_dir.exists(), "GET must not create or clone the workspace"

            # This body is deliberately hostile-looking. The route takes no
            # fields: setup must use only the server-side canonical remote.
            body = json.dumps({"clone_url": "file:///definitely-not-the-source", "source": "/tmp/other"}).encode()
            connection.request(
                "POST",
                "/api/source-workspace",
                body=body,
                headers={"Cookie": cookie, "Content-Type": "application/json", "Content-Length": str(len(body))},
            )
            status, after = _read_json(connection.getresponse())
            assert status == 200
            assert after["available"] is True
            managed = app_dir / "source-workspace"
            assert after["path"] == str(managed.resolve())
            origin = subprocess.run(
                ["git", "-C", str(managed), "remote", "get-url", "origin"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            assert origin == str(remote)

            # A second request on this same HTTP/1.1 connection proves the POST
            # body was consumed rather than parsed as the next request line.
            connection.request("GET", "/api/source-workspace", headers={"Cookie": cookie})
            status, repeated = _read_json(connection.getresponse())
            assert status == 200
            assert repeated == after
        finally:
            connection.close()
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=3)
