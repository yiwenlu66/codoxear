"""HTML cache-validator behavior for the static routes.

The HTML response embeds server-side substitutions (content-hashed asset
version, attachment limit), so its validator must identify the *served*
bytes. A release that changes only frontend assets leaves index.html's stat
untouched while the served representation changes; a stat-derived ETag would
then answer 304 for a changed representation and pin clients to immutable
old asset URLs. These tests execute the real route code through a request
handler harness and assert the 200/304 outcomes of conditional requests.
"""

from __future__ import annotations

import gzip
import hashlib
import os
from io import BytesIO
from pathlib import Path

from codoxear.static_routes import CONTENT_SECURITY_POLICY
from codoxear.static_routes import StaticRouteDeps
from codoxear.static_routes import handle_static_get_route
from codoxear.static_routes import read_static_bytes
from codoxear.static_routes import static_asset_version
from codoxear.static_routes import static_cache_control_headers

INDEX_HTML = (
    "<!doctype html>\n"
    "<html><head>\n"
    "<script>window.CODOXEAR_ATTACH_MAX_BYTES = __CODOXEAR_ATTACH_MAX_BYTES__;</script>\n"
    '<link rel="stylesheet" href="app.css?v=__CODOXEAR_ASSET_VERSION__" />\n'
    + "<!-- padding so substituted HTML exceeds the gzip threshold -->\n" * 40
    + "</head><body>shell</body></html>\n"
)


class _Headers(list[tuple[str, str]]):
    def __init__(self) -> None:
        super().__init__()
        self.request: dict[str, str] = {}

    def get(self, name: str, default: str | None = None) -> str | None:
        return self.request.get(name, default)


class _FakeHandler:
    def __init__(self) -> None:
        self.status: int | None = None
        self.errors: list[int] = []
        self.headers = _Headers()
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

    def header(self, name: str) -> str | None:
        return next((value for key, value in self.headers if key == name), None)

    def body(self) -> bytes:
        raw = self.wfile.getvalue()
        if self.header("Content-Encoding") == "gzip":
            return gzip.decompress(raw)
        return raw


def _deps(root: Path, *, attach_max: int, read_calls: list[Path] | None = None) -> StaticRouteDeps:
    def read(path: Path) -> bytes:
        if read_calls is not None:
            read_calls.append(path)
        return read_static_bytes(path, attach_upload_max_bytes=attach_max)

    return StaticRouteDeps(
        static_dir=root,
        top_level_static_assets=(("/", "index.html"), ("/app.css", "app.css")),
        read_static_bytes=read,
        static_cache_control_headers=lambda *, versioned=False, is_html=False: static_cache_control_headers(
            versioned=versioned, is_html=is_html
        ),
        content_security_policy=CONTENT_SECURITY_POLICY,
    )


def _get(path: str, deps: StaticRouteDeps, *, query: str = "", **request_headers: str) -> _FakeHandler:
    handler = _FakeHandler()
    handler.headers.request.update(request_headers)
    handle_static_get_route(handler, path=path, query=query, deps=deps)
    return handler


def _fresh_asset_version_cache() -> None:
    # A restarted server process recomputes the memoized asset version.
    static_asset_version.cache_clear()


def _fixture(root: Path) -> None:
    (root / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    (root / "app.css").write_text("body { color: black; }\n", encoding="utf-8")


def test_html_conditional_get_returns_200_when_assets_change_and_index_stat_is_frozen(tmp_path: Path) -> None:
    _fixture(tmp_path)
    index = tmp_path / "index.html"
    frozen_stat = index.stat()
    try:
        _fresh_asset_version_cache()
        before = _get("/", _deps(tmp_path, attach_max=16 * 1024 * 1024))
        assert before.status == 200
        etag_before = before.header("ETag")
        version_before = before.body().split(b'app.css?v=')[1].split(b'"')[0].decode()

        # Asset-only release: sibling asset bytes change, index.html untouched.
        (tmp_path / "app.css").write_text("body { color: white; }\n", encoding="utf-8")
        # Restore index.html's stat exactly, as a git worktree checkout does
        # for a file whose content did not change between commits.
        os.utime(index, ns=(frozen_stat.st_atime_ns, frozen_stat.st_mtime_ns))
        _fresh_asset_version_cache()

        fresh = _get("/", _deps(tmp_path, attach_max=16 * 1024 * 1024))
        assert fresh.status == 200
        version_after = fresh.body().split(b'app.css?v=')[1].split(b'"')[0].decode()
        assert version_after != version_before

        # A client holding the pre-release HTML must not be told 304: the
        # representation it would keep no longer matches the served bytes.
        revalidated = _get("/", _deps(tmp_path, attach_max=16 * 1024 * 1024), **{"If-None-Match": etag_before})
        assert revalidated.status == 200
        assert b"shell" in revalidated.body()
        assert revalidated.header("ETag") != etag_before

        # The validator must be representation-derived, not stat-derived: an
        # untouched index.html cannot keep the old validator once the served
        # asset URLs change.
        assert etag_before == f'W/"{hashlib.sha256(before.body()).hexdigest()}"'
    finally:
        _fresh_asset_version_cache()


def test_html_conditional_get_returns_200_when_only_attachment_limit_changes(tmp_path: Path) -> None:
    _fixture(tmp_path)
    index = tmp_path / "index.html"
    frozen_stat = index.stat()
    try:
        _fresh_asset_version_cache()
        before = _get("/", _deps(tmp_path, attach_max=16 * 1024 * 1024))
        etag_before = before.header("ETag")
        assert b"CODOXEAR_ATTACH_MAX_BYTES = 16777216" in before.body()

        # Attachment-limit-only release: no file changed at all; only the
        # injected limit differs (environment change on a restarted server).
        os.utime(index, ns=(frozen_stat.st_atime_ns, frozen_stat.st_mtime_ns))
        _fresh_asset_version_cache()
        revalidated = _get("/", _deps(tmp_path, attach_max=8 * 1024 * 1024), **{"If-None-Match": etag_before})

        assert revalidated.status == 200
        assert b"CODOXEAR_ATTACH_MAX_BYTES = 8388608" in revalidated.body()
        assert revalidated.header("ETag") != etag_before
    finally:
        _fresh_asset_version_cache()


def test_unchanged_html_representation_still_revalidates_304_even_when_mtime_moves(tmp_path: Path) -> None:
    _fixture(tmp_path)
    try:
        _fresh_asset_version_cache()
        deps = _deps(tmp_path, attach_max=123)
        before = _get("/", deps)
        etag = before.header("ETag")

        immediate = _get("/", deps, **{"If-None-Match": etag})
        assert immediate.status == 304
        assert immediate.header("ETag") == etag

        # The validator tracks content, not stat: rewriting index.html's
        # timestamps without changing bytes must not invalidate it.
        st = (tmp_path / "index.html").stat()
        os.utime(tmp_path / "index.html", ns=(st.st_atime_ns + 10**9, st.st_mtime_ns + 10**9))
        after_touch = _get("/", deps, **{"If-None-Match": etag})
        assert after_touch.status == 304
        assert after_touch.header("ETag") == etag
    finally:
        _fresh_asset_version_cache()


def test_bare_index_edits_change_the_html_validator(tmp_path: Path) -> None:
    # An HTML file without placeholders still serves its raw bytes; editing it
    # must produce a new validator and a 200 for holders of the old one.
    (tmp_path / "index.html").write_text("<html><body>first</body></html>\n", encoding="utf-8")
    deps = _deps(tmp_path, attach_max=1)
    before = _get("/", deps)
    etag = before.header("ETag")
    assert before.body() == b"<html><body>first</body></html>\n"

    (tmp_path / "index.html").write_text("<html><body>second</body></html>\n", encoding="utf-8")
    changed = _get("/", deps, **{"If-None-Match": etag})
    assert changed.status == 200
    assert changed.body() == b"<html><body>second</body></html>\n"
    assert changed.header("ETag") != etag


def test_html_validator_is_shared_by_root_and_static_index_routes(tmp_path: Path) -> None:
    _fixture(tmp_path)
    try:
        _fresh_asset_version_cache()
        deps = _deps(tmp_path, attach_max=5)
        root = _get("/", deps)
        static_route = _get("/static/index.html", deps)
        assert root.status == static_route.status == 200
        assert root.header("ETag") == static_route.header("ETag")

        via_static = _get("/static/index.html", deps, **{"If-None-Match": root.header("ETag")})
        assert via_static.status == 304
    finally:
        _fresh_asset_version_cache()


def test_html_validator_is_weak_and_shared_across_gzip_and_identity(tmp_path: Path) -> None:
    _fixture(tmp_path)
    try:
        _fresh_asset_version_cache()
        deps = _deps(tmp_path, attach_max=7)
        identity = _get("/", deps)
        gzipped = _get("/", deps, **{"Accept-Encoding": "gzip"})

        assert identity.header("ETag").startswith('W/"')
        assert gzipped.header("ETag") == identity.header("ETag")
        assert gzipped.header("Content-Encoding") == "gzip"
        assert gzipped.header("Vary") == "Accept-Encoding"
        assert gzip.decompress(gzipped.wfile.getvalue()) == identity.body()

        # A response cached from one encoding must revalidate coherently via
        # the other: weak identity is shared across content codings.
        revalidated = _get("/", deps, **{"Accept-Encoding": "gzip", "If-None-Match": identity.header("ETag")})
        assert revalidated.status == 304
        assert revalidated.header("ETag") == identity.header("ETag")
    finally:
        _fresh_asset_version_cache()


def test_html_is_read_exactly_once_per_request(tmp_path: Path) -> None:
    _fixture(tmp_path)
    try:
        _fresh_asset_version_cache()
        read_calls: list[Path] = []
        deps = _deps(tmp_path, attach_max=9, read_calls=read_calls)

        assert _get("/", deps).status == 200
        assert _get("/", deps, **{"If-None-Match": _get("/", deps).header("ETag")}).status == 304
        # Three requests, one read each: substitution and hashing share the
        # same bytes; the 304 path must not add a second read.
        assert len(read_calls) == 3
        assert all(path.name == "index.html" for path in read_calls)
    finally:
        _fresh_asset_version_cache()


def test_non_html_validator_stays_stat_based_and_immutable_policy_is_preserved(tmp_path: Path) -> None:
    _fixture(tmp_path)
    css = tmp_path / "app.css"
    try:
        _fresh_asset_version_cache()
        deps = _deps(tmp_path, attach_max=11)

        versioned = _get("/app.css", deps, query="v=abc123")
        assert versioned.status == 200
        assert versioned.header("Cache-Control") == "public, max-age=31536000, immutable"

        stat = css.stat()
        stat_etag = f'W/"{stat.st_mtime_ns:x}-{stat.st_size:x}"'
        unversioned = _get("/app.css", deps)
        assert unversioned.header("ETag") == stat_etag
        assert unversioned.header("Cache-Control") == "no-cache"

        # Non-HTML 304s keep working from stat alone...
        assert _get("/app.css", deps, **{"If-None-Match": stat_etag}).status == 304

        # ...and stat semantics are preserved: moving only mtime refreshes the
        # non-HTML validator even though the bytes are identical.
        os.utime(css, ns=(stat.st_atime_ns + 10**9, stat.st_mtime_ns + 10**9))
        moved = _get("/app.css", deps)
        assert moved.header("ETag") != stat_etag
        assert _get("/app.css", deps, **{"If-None-Match": stat_etag}).status == 200
    finally:
        _fresh_asset_version_cache()
