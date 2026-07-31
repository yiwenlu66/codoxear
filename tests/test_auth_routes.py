from __future__ import annotations

from io import BytesIO

from codoxear.auth_routes import AuthRouteDeps
from codoxear.auth_routes import handle_auth_get_route
from codoxear.auth_routes import handle_auth_post_route


class _FakeHandler:
    def __init__(self) -> None:
        self.status: int | None = None
        self.headers: list[tuple[str, str]] = []
        self.wfile = BytesIO()
        self.unauthorized = False

    def _unauthorized(self) -> None:
        self.unauthorized = True

    def send_response(self, status: int) -> None:
        self.status = status

    def send_header(self, name: str, value: str) -> None:
        self.headers.append((name, value))

    def end_headers(self) -> None:
        return None


def _deps(*, body: dict[str, object] | None = None, auth: bool = True, password_ok: bool = True):
    responses: list[tuple[int, dict[str, object]]] = []
    cookie_set: list[bool] = []

    def json_response(_handler, status: int, payload: dict[str, object]) -> None:
        responses.append((status, payload))

    def set_auth_cookie(handler) -> None:
        cookie_set.append(True)
        handler.send_header("Set-Cookie", "codoxear_auth=signed; Path=/; HttpOnly; SameSite=Strict")

    deps = AuthRouteDeps(
        require_auth=lambda _handler: auth,
        json_response=json_response,
        read_json_body=lambda _handler, **_kwargs: dict(body or {}),
        is_same_password=lambda password: password_ok and password == "pw",
        set_auth_cookie=set_auth_cookie,
        cookie_name="codoxear_auth",
        cookie_path="/app",
    )
    return deps, responses, cookie_set


def test_auth_get_me_requires_auth_and_returns_ok() -> None:
    handler = _FakeHandler()
    deps, responses, _cookie_set = _deps(auth=True)
    assert handle_auth_get_route(handler, path="/api/me", deps=deps) is True
    assert responses == [(200, {"ok": True})]
    assert handler.unauthorized is False

    handler = _FakeHandler()
    deps, responses, _cookie_set = _deps(auth=False)
    assert handle_auth_get_route(handler, path="/api/me", deps=deps) is True
    assert responses == []
    assert handler.unauthorized is True


def test_auth_post_login_bad_password_uses_json_403() -> None:
    handler = _FakeHandler()
    deps, responses, cookie_set = _deps(body={"password": "wrong"}, password_ok=True)
    assert handle_auth_post_route(handler, path="/api/login", deps=deps) is True
    assert responses == [(403, {"error": "bad password"})]
    assert cookie_set == []
    assert handler.status is None


def test_auth_post_login_sets_cookie_and_json_body_without_content_length() -> None:
    handler = _FakeHandler()
    deps, responses, cookie_set = _deps(body={"password": "pw"})
    assert handle_auth_post_route(handler, path="/api/login", deps=deps) is True
    assert responses == []
    assert cookie_set == [True]
    assert handler.status == 200
    assert ("Content-Type", "application/json; charset=utf-8") in handler.headers
    assert handler.wfile.getvalue() == b'{"ok":true}'


def test_auth_post_logout_requires_auth_and_clears_cookie() -> None:
    handler = _FakeHandler()
    deps, responses, _cookie_set = _deps(auth=False)
    assert handle_auth_post_route(handler, path="/api/logout", deps=deps) is True
    assert responses == []
    assert handler.unauthorized is True

    handler = _FakeHandler()
    deps, responses, _cookie_set = _deps(auth=True)
    assert handle_auth_post_route(handler, path="/api/logout", deps=deps) is True
    assert responses == []
    assert handler.status == 200
    assert (
        "Set-Cookie",
        "codoxear_auth=deleted; Path=/app; Max-Age=0; HttpOnly; SameSite=Strict",
    ) in handler.headers
    assert ("Content-Type", "application/json; charset=utf-8") in handler.headers
    assert handler.wfile.getvalue() == b'{"ok":true}'


def test_auth_routes_ignore_unowned_paths() -> None:
    deps, responses, _cookie_set = _deps()
    assert handle_auth_get_route(_FakeHandler(), path="/api/sessions", deps=deps) is False
    assert handle_auth_post_route(_FakeHandler(), path="/api/sessions", deps=deps) is False
    assert responses == []
