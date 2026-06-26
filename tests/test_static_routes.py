from __future__ import annotations

from io import BytesIO
from pathlib import Path

from codoxear.static_routes import CONTENT_SECURITY_POLICY
from codoxear.static_routes import StaticRouteDeps
from codoxear.static_routes import handle_static_get_route
from codoxear.static_routes import read_static_bytes
from codoxear.static_routes import static_cache_control_headers
from codoxear.static_routes import static_content_type
from codoxear.static_routes import static_route_asset


class _FakeHandler:
    def __init__(self) -> None:
        self.status: int | None = None
        self.errors: list[int] = []
        self.headers: list[tuple[str, str]] = []
        self.ended = False
        self.wfile = BytesIO()

    def send_response(self, status: int) -> None:
        self.status = status

    def send_header(self, name: str, value: str) -> None:
        self.headers.append((name, value))

    def end_headers(self) -> None:
        self.ended = True

    def send_error(self, status: int) -> None:
        self.errors.append(status)


def _deps(root: Path) -> StaticRouteDeps:
    return StaticRouteDeps(
        static_dir=root,
        top_level_static_assets=(("/", "index.html"), ("/app.js", "app.js")),
        read_static_bytes=lambda path: read_static_bytes(path, attach_upload_max_bytes=123),
        static_cache_control_headers=lambda: static_cache_control_headers(enabled=False),
        content_security_policy=CONTENT_SECURITY_POLICY,
    )


def test_static_route_asset_maps_top_level_and_static_prefix() -> None:
    assert static_route_asset("/app.js", top_level_static_assets=(("/app.js", "app.js"),)) == "app.js"
    assert static_route_asset("/static/logos/cc.svg", top_level_static_assets=()) == "logos/cc.svg"
    assert static_route_asset("/api/me", top_level_static_assets=(("/", "index.html"),)) is None


def test_static_get_route_serves_html_headers_and_body(tmp_path: Path) -> None:
    index = tmp_path / "index.html"
    index.write_text("hello __CODOXEAR_ATTACH_MAX_BYTES__", encoding="utf-8")
    handler = _FakeHandler()

    assert handle_static_get_route(handler, path="/", deps=_deps(tmp_path)) is True

    assert handler.status == 200
    assert handler.errors == []
    assert ("Content-Type", "text/html; charset=utf-8") in handler.headers
    assert ("Content-Security-Policy", CONTENT_SECURITY_POLICY) in handler.headers
    assert ("X-Frame-Options", "DENY") in handler.headers
    assert ("Cache-Control", "no-store") in handler.headers
    assert ("Pragma", "no-cache") in handler.headers
    assert ("Expires", "0") in handler.headers
    assert handler.wfile.getvalue() == b"hello 123"
    assert ("Content-Length", str(len(handler.wfile.getvalue()))) in handler.headers
    assert handler.ended is True


def test_static_get_route_rejects_missing_and_escaped_paths(tmp_path: Path) -> None:
    handler = _FakeHandler()
    assert handle_static_get_route(handler, path="/missing.js", deps=_deps(tmp_path)) is False
    assert handler.errors == []

    handler = _FakeHandler()
    assert handle_static_get_route(handler, path="/static/../secret.txt", deps=_deps(tmp_path)) is True
    assert handler.errors == [404]
    assert handler.status is None


def test_static_content_type_policy() -> None:
    assert static_content_type(Path("index.html")) == "text/html; charset=utf-8"
    assert static_content_type(Path("app.js")) == "text/javascript; charset=utf-8"
    assert static_content_type(Path("app.css")) == "text/css; charset=utf-8"
    assert static_content_type(Path("manifest.webmanifest")) == "application/manifest+json; charset=utf-8"
    assert static_content_type(Path("icon.png")) == "image/png"
    assert static_content_type(Path("photo.jpeg")) == "image/jpeg"
    assert static_content_type(Path("image.webp")) == "image/webp"
    assert static_content_type(Path("logo.svg")) == "image/svg+xml; charset=utf-8"
    assert static_content_type(Path("favicon.ico")) == "image/x-icon"
    assert static_content_type(Path("download.bin")) == "application/octet-stream"
