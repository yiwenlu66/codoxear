from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import urllib.parse

import pytest

from codoxear.file_routes import FileGetRouteDeps
from codoxear.file_routes import FileRouteResponse
from codoxear.file_routes import FileWriteRouteDeps
from codoxear.file_routes import GlobalFileRequest
from codoxear.file_routes import GlobalFileRouteDeps
from codoxear.file_routes import SessionFileWriteRequest
from codoxear.file_routes import global_file_read_payload
from codoxear.file_routes import handle_absolute_file_preview_route
from codoxear.file_routes import handle_file_get_route
from codoxear.file_routes import handle_file_write_post_route
from codoxear.file_routes import handle_global_file_post_route
from codoxear.file_routes import resolve_session_write_update_path
from codoxear.file_routes import session_file_read_payload
from codoxear.file_routes import session_file_write_response
from codoxear.file_view import ClientFileView
from codoxear.file_view import read_client_file_view
from codoxear.file_response import send_inline_file_response
from codoxear.path_runtime import resolve_existing_session_file
from codoxear.file_get_routes import decode_session_raw_file_path
from codoxear.file_types import RAW_FILE_CONTENT_TYPES
from codoxear.file_get_routes import encode_session_raw_file_path
from codoxear.server_routing import match_session_file_raw_route


@contextmanager
def _null_write_lock(_path: Path):
    yield


def _resolve_create_path(base: Path, rel: str) -> Path:
    if "\x00" in rel:
        raise ValueError("invalid path")
    path = (base / rel).resolve()
    path.relative_to(base.resolve())
    return path


def _unexpected_git_resolver(_raw_path: str) -> tuple[Path, str]:
    raise AssertionError("git resolver should not be called")


def test_session_file_write_response_creates_file_and_records_path() -> None:
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        recorded: list[str] = []
        response = session_file_write_response(
            body={"path": "notes/new.md", "text": "hello\n", "create": True},
            session_base=base,
            resolve_create_path=_resolve_create_path,
            resolve_git_existing_regular_file=_unexpected_git_resolver,
            file_write_lock=_null_write_lock,
            record_file=recorded.append,
        )
        created = base / "notes" / "new.md"
        assert isinstance(response, FileRouteResponse)
        assert response.status == 404
        assert response.payload == {"error": "parent directory not found"}
        assert recorded == []

        (base / "notes").mkdir()
        response = session_file_write_response(
            body={"path": "notes/new.md", "text": "hello\n", "create": True},
            session_base=base,
            resolve_create_path=_resolve_create_path,
            resolve_git_existing_regular_file=_unexpected_git_resolver,
            file_write_lock=_null_write_lock,
            record_file=recorded.append,
        )
        assert response.status == 200
        assert response.payload["path"] == str(created.resolve())
        assert response.payload["rel"] == "notes/new.md"
        assert response.payload["editable"] is True
        assert created.read_text(encoding="utf-8") == "hello\n"
        assert recorded == [str(created.resolve())]


def test_session_file_write_response_checks_conflict_under_injected_lock() -> None:
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        path = base / "note.txt"
        path.write_text("old\n", encoding="utf-8")
        events: list[str] = []

        @contextmanager
        def tracking_lock(lock_path: Path):
            events.append(f"enter:{lock_path.name}")
            try:
                yield
            finally:
                events.append(f"exit:{lock_path.name}")

        response = session_file_write_response(
            body={"path": "note.txt", "text": "new\n", "version": "wrong"},
            session_base=base,
            resolve_create_path=_resolve_create_path,
            resolve_git_existing_regular_file=_unexpected_git_resolver,
            file_write_lock=tracking_lock,
        )
        assert response.status == 409
        assert response.payload["conflict"] is True
        assert response.payload["version"] == hashlib.sha256(b"old\n").hexdigest()
        assert path.read_text(encoding="utf-8") == "old\n"
        assert events == ["enter:note.txt", "exit:note.txt"]


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlink required")
def test_session_file_write_response_rejects_relative_symlink_parent_escape() -> None:
    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as outside_td:
        base = Path(td)
        outside = Path(outside_td)
        target = outside / "note.txt"
        target.write_text("outside\n", encoding="utf-8")
        os.symlink(outside, base / "link")
        version = hashlib.sha256(b"outside\n").hexdigest()
        response = session_file_write_response(
            body={"path": "link/note.txt", "text": "mutated\n", "version": version},
            session_base=base,
            resolve_create_path=_resolve_create_path,
            resolve_git_existing_regular_file=_unexpected_git_resolver,
            file_write_lock=_null_write_lock,
        )
        assert response.status == 400
        assert "escapes session cwd" in str(response.payload.get("error"))
        assert target.read_text(encoding="utf-8") == "outside\n"


def test_resolve_session_write_update_path_preserves_legacy_absolute_update_paths() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "note.txt"
        path.write_text("ok\n", encoding="utf-8")
        assert resolve_session_write_update_path(Path("/does/not/matter"), str(path)) == path.resolve()


def test_session_file_read_payload_builds_html_preview_url_only_for_html() -> None:
    html_payload = session_file_read_payload(
        session_id="sid",
        path_obj=Path("/tmp/docs/my report.html"),
        rel="docs/my report.html",
        view=ClientFileView(kind="html", size=12, text="<h1>preview</h1>", editable=True, version="v1"),
        git_path=False,
    )
    assert html_payload["preview_url"] == "/api/sessions/sid/file/raw/docs/my%20report.html"
    for kind, name in (("markdown", "notes.md"), ("text", "notes.txt")):
        payload = session_file_read_payload(
            session_id="sid",
            path_obj=Path("/tmp") / name,
            rel=name,
            view=ClientFileView(kind=kind, size=5, text="text", editable=True, version="v1"),
            git_path=False,
        )
        assert "preview_url" not in payload


def test_session_file_read_payload_builds_git_aware_video_urls() -> None:
    payload = session_file_read_payload(
        session_id="sid",
        path_obj=Path("/tmp/movie.mp4"),
        rel="dir/movie.mp4",
        view=ClientFileView(kind="video", size=12, content_type="video/mp4"),
        git_path=True,
    )
    assert payload["kind"] == "video"
    assert payload["video_url"] == "/api/sessions/sid/file/blob?path=dir/movie.mp4&git_path=1"
    assert payload["video_preview_url"] == "/api/sessions/sid/file/video_preview?path=dir/movie.mp4&git_path=1"


class _FakeHandler:
    def __init__(self) -> None:
        self.unauthorized = False

    def _unauthorized(self) -> None:
        self.unauthorized = True


class _StreamingHandler(_FakeHandler):
    def __init__(self) -> None:
        super().__init__()
        self.headers: dict[str, str] = {}
        self.status: int | None = None
        self.sent_headers: list[tuple[str, str]] = []
        self.wfile = io.BytesIO()
        self.error: tuple[int, str] | None = None

    def send_response(self, status: int) -> None:
        self.status = status

    def send_header(self, name: str, value: str) -> None:
        self.sent_headers.append((name, value))

    def end_headers(self) -> None:
        return None

    def send_error(self, status: int, message: str = "") -> None:
        self.status = status
        self.error = (status, message)


class _FakeManager:
    def __init__(self, cwd: str) -> None:
        self.cwd = cwd
        self.refreshed: list[str] = []
        self.recorded: list[tuple[str, str]] = []

    def refresh_session_meta(self, session_id: str) -> None:
        self.refreshed.append(session_id)

    def get_session(self, _session_id: str) -> object:
        return SimpleNamespace(cwd=self.cwd)

    def files_add(self, session_id: str, path: str) -> None:
        self.recorded.append((session_id, path))


def _match_session_route(path: str, *parts: str) -> str | None:
    expected = "/api/sessions/s/" + "/".join(parts)
    return "s" if path == expected else None


def _file_get_deps(**overrides):
    responses: list[tuple[int, dict[str, object]]] = []
    inline: list[tuple[Path, str]] = []
    attachments: list[tuple[Path, int, str]] = []

    def json_response(_handler, status: int, payload: dict[str, object]) -> None:
        responses.append((status, payload))

    def send_inline(_handler, path: Path, content_type: str) -> None:
        inline.append((path, content_type))

    def send_attachment(_handler, path: Path, *, size: int, content_disposition: str) -> None:
        attachments.append((path, size, content_disposition))

    deps = FileGetRouteDeps(
        require_auth=lambda _handler: True,
        json_response=json_response,
        resolve_session_cwd=lambda raw: Path(raw),
        resolve_existing_session_file=lambda base, rel: base / rel,
        resolve_session_path=lambda base, rel: base / rel,
        resolve_git_client_file_view=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected git view")),
        resolve_git_existing_regular_file=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected git file")),
        resolve_existing_absolute_file=lambda raw: Path(raw),
        read_client_file_view=lambda path: ClientFileView(
            kind="text",
            size=5,
            text="hello",
            editable=True,
            version="v1",
        ),
        read_regular_file_prefix=lambda path, byte_count: (path.read_bytes()[:byte_count], path.stat().st_size),
        search_session_relative_files=lambda base, *, query, limit: {
            "query": query,
            "mode": "literal",
            "matches": [],
            "scanned": 0,
            "truncated": False,
        },
        list_session_relative_files=lambda _base: ["note.txt"],
        list_session_relative_file_entries=lambda _base: [{"path": "note.txt"}],
        file_kind=lambda _path, _prefix: ("text", None),
        ensure_video_preview=lambda path: path.with_suffix(".preview.mp4"),
        ensure_office_preview=lambda path: path.with_suffix(".preview.pdf"),
        inspect_downloadable_file=lambda _path: 5,
        download_disposition=lambda path: f"attachment; filename={path.name}",
        send_inline_file_response=send_inline,
        send_attachment_file_response=send_attachment,
        file_search_limit=80,
    )
    for name, value in overrides.items():
        object.__setattr__(deps, name, value)
    return deps, responses, inline, attachments


def test_handle_file_get_route_session_read_records_file() -> None:
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        path = base / "note.txt"
        manager = _FakeManager(str(base))
        deps, responses, _inline, _attachments = _file_get_deps(
            resolve_existing_session_file=lambda base_path, rel: path if base_path == base and rel == "note.txt" else (_ for _ in ()).throw(AssertionError()),
        )
        handled = handle_file_get_route(
            _FakeHandler(),
            path="/api/sessions/s/file/read",
            query="path=note.txt",
            manager=manager,
            deps=deps,
            match_session_route=_match_session_route,
        )
        assert handled is True
        assert responses == [
            (
                200,
                {
                    "ok": True,
                    "kind": "text",
                    "path": str(path),
                    "rel": "note.txt",
                    "size": 5,
                    "text": "hello",
                    "editable": True,
                    "version": "v1",
                },
            )
        ]
        assert manager.refreshed == ["s"]
        assert manager.recorded == [("s", str(path))]


def test_session_file_read_reports_html_kind_and_preserves_existing_kinds() -> None:
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        files = {
            "page.html": b"<p>html</p>",
            "page.htm": b"<p>htm</p>",
            "page.HTML": b"<p>uppercase</p>",
            "notes.md": b"# notes\n",
            "notes.txt": b"text\n",
            "icon.png": b"\x89PNG\r\n\x1a\nbytes",
        }
        expected_kinds = {
            "page.html": "html",
            "page.htm": "html",
            "page.HTML": "html",
            "notes.md": "markdown",
            "notes.txt": "text",
            "icon.png": "image",
        }
        for name, raw in files.items():
            (base / name).write_bytes(raw)
        manager = _FakeManager(str(base))
        deps, responses, _inline, _attachments = _file_get_deps(
            resolve_existing_session_file=resolve_existing_session_file,
            read_client_file_view=read_client_file_view,
        )
        for name, expected_kind in expected_kinds.items():
            assert handle_file_get_route(
                _FakeHandler(),
                path="/api/sessions/s/file/read",
                query=urllib.parse.urlencode({"path": name}),
                manager=manager,
                deps=deps,
                match_session_route=_match_session_route,
            ) is True
            status, payload = responses.pop()
            assert status == 200
            assert payload["kind"] == expected_kind
            if expected_kind == "html":
                assert payload["preview_url"] == f"/api/sessions/s/file/raw/{name}"
            else:
                assert "preview_url" not in payload


def test_session_file_raw_route_decodes_paths_streams_sandboxed_documents_and_preserves_relative_urls() -> None:
    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as outside_td:
        base = Path(td)
        document = base / "docs" / "my report.html"
        image = base / "docs" / "img" / "x.png"
        document.parent.mkdir()
        image.parent.mkdir()
        document.write_text("<h1>preview</h1>", encoding="utf-8")
        image.write_bytes(b"png")
        external = Path(outside_td) / "outside.xhtml"
        external.write_text("<html></html>", encoding="utf-8")
        manager = _FakeManager(str(base))
        deps, responses, _inline, _attachments = _file_get_deps(
            resolve_existing_session_file=resolve_existing_session_file,
            send_inline_file_response=send_inline_file_response,
        )

        handler = _StreamingHandler()
        document_url = "/api/sessions/s/file/raw/docs/my%20report.html"
        assert handle_file_get_route(
            handler,
            path=document_url,
            query="",
            manager=manager,
            deps=deps,
            match_session_route=_match_session_route,
        ) is True
        assert responses == []
        assert handler.status == 200
        assert dict(handler.sent_headers)["Content-Type"] == "text/html; charset=utf-8"
        assert dict(handler.sent_headers)["X-Content-Type-Options"] == "nosniff"
        assert dict(handler.sent_headers)["Content-Security-Policy"] == "sandbox allow-same-origin"
        assert handler.wfile.getvalue() == b"<h1>preview</h1>"

        sibling_url = urllib.parse.urljoin(document_url, "img/x.png")
        assert sibling_url == "/api/sessions/s/file/raw/docs/img/x.png"
        sibling_match = match_session_file_raw_route(sibling_url)
        assert sibling_match == ("s", ("docs", "img", "x.png"))
        assert decode_session_raw_file_path(sibling_match[1]) == "docs/img/x.png"
        image_handler = _StreamingHandler()
        assert handle_file_get_route(
            image_handler,
            path=sibling_url,
            query="",
            manager=manager,
            deps=deps,
            match_session_route=_match_session_route,
        ) is True
        assert image_handler.status == 200
        assert dict(image_handler.sent_headers)["Content-Type"] == "image/png"
        assert dict(image_handler.sent_headers)["X-Content-Type-Options"] == "nosniff"
        assert "Content-Security-Policy" not in dict(image_handler.sent_headers)

        absolute_handler = _StreamingHandler()
        absolute_url = f"/api/sessions/s/file/raw/{encode_session_raw_file_path(str(external))}"
        assert absolute_url.startswith("/api/sessions/s/file/raw//")
        assert handle_file_get_route(
            absolute_handler,
            path=absolute_url,
            query="",
            manager=manager,
            deps=deps,
            match_session_route=_match_session_route,
        ) is True
        assert absolute_handler.status == 200
        assert dict(absolute_handler.sent_headers)["Content-Type"] == "application/xhtml+xml; charset=utf-8"
        assert dict(absolute_handler.sent_headers)["Content-Security-Policy"] == "sandbox allow-same-origin"


def test_session_file_raw_route_maps_auth_session_and_resolution_errors() -> None:
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        (base / "directory").mkdir()
        manager = _FakeManager(str(base))
        deps, responses, _inline, _attachments = _file_get_deps(resolve_existing_session_file=resolve_existing_session_file)

        unauth_handler = _FakeHandler()
        unauth_deps, _unauth_responses, _inline, _attachments = _file_get_deps(
            require_auth=lambda _handler: False,
            resolve_existing_session_file=resolve_existing_session_file,
        )
        assert handle_file_get_route(
            unauth_handler,
            path="/api/sessions/s/file/raw/missing.html",
            query="",
            manager=manager,
            deps=unauth_deps,
            match_session_route=_match_session_route,
        ) is True
        assert unauth_handler.unauthorized is True

        class MissingSessionManager(_FakeManager):
            def get_session(self, _session_id: str) -> object | None:
                return None

        assert handle_file_get_route(
            _FakeHandler(),
            path="/api/sessions/missing/file/raw/page.html",
            query="",
            manager=MissingSessionManager(str(base)),
            deps=deps,
            match_session_route=_match_session_route,
        ) is True
        assert responses == [(404, {"error": "unknown session"})]
        responses.clear()

        for raw_path, expected_status, expected_error in (
            ("missing.html", 404, "file not found"),
            ("directory", 400, "path is not a file"),
        ):
            assert handle_file_get_route(
                _FakeHandler(),
                path=f"/api/sessions/s/file/raw/{encode_session_raw_file_path(raw_path)}",
                query="",
                manager=manager,
                deps=deps,
                match_session_route=_match_session_route,
            ) is True
            assert responses == [(expected_status, {"error": expected_error})]
            responses.clear()


def test_session_raw_file_path_codec_round_trips_utf8_and_surrogate_bytes() -> None:
    raw_byte_name = b"bad-\xff.html".decode("utf-8", "surrogateescape")
    for rel in ("report.html", "docs/my report.html", "a/b/c.css", "données/éclair.html", raw_byte_name):
        encoded = encode_session_raw_file_path(rel)
        assert decode_session_raw_file_path(tuple(encoded.split("/"))) == rel


def test_session_file_raw_route_sandboxes_exactly_the_executable_document_types() -> None:
    # The sandbox header is what keeps a session-cwd document from scripting on
    # the authenticated origin, so it must follow the MIME table rather than a
    # hand-checked list: every type a browser can execute as a document gets
    # the CSP, and every other type does not. The executable set is spelled out
    # here independently of the implementation, so adding a script-capable type
    # to RAW_FILE_CONTENT_TYPES without sandboxing it fails this test.
    executable_document_types = {
        "text/html",
        "application/xhtml+xml",
        "image/svg+xml",
        "text/xml",
        "application/xml",
        "text/xsl",
    }
    expected: dict[str, str] = {}
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        for extension, content_type in sorted(RAW_FILE_CONTENT_TYPES.items()):
            name = f"asset.{extension}"
            (base / name).write_text("x", encoding="utf-8")
            expected[name] = content_type
        # One textual extension and one unknown extension cover the fallbacks.
        for name, content_type in (("notes.txt", "text/plain; charset=utf-8"), ("blob.bin", "application/octet-stream")):
            (base / name).write_text("x", encoding="utf-8")
            expected[name] = content_type
        manager = _FakeManager(str(base))
        deps, _responses, _inline, _attachments = _file_get_deps(
            resolve_existing_session_file=resolve_existing_session_file,
            send_inline_file_response=send_inline_file_response,
        )
        for name, content_type in expected.items():
            handler = _StreamingHandler()
            assert handle_file_get_route(
                handler,
                path=f"/api/sessions/s/file/raw/{name}",
                query="",
                manager=manager,
                deps=deps,
                match_session_route=_match_session_route,
            ) is True
            headers = dict(handler.sent_headers)
            assert handler.status == 200, (name, handler.status)
            assert headers["Content-Type"] == content_type, name
            assert headers["X-Content-Type-Options"] == "nosniff", name
            script_capable = content_type.split(";", 1)[0].strip() in executable_document_types
            assert ("Content-Security-Policy" in headers) is script_capable, (name, content_type, headers)
            if script_capable:
                assert headers["Content-Security-Policy"] == "sandbox allow-same-origin", name


def test_handle_absolute_file_preview_route_streams_previewable_blob() -> None:
    with tempfile.TemporaryDirectory() as td:
        image = Path(td) / "preview.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\nbody")
        deps, responses, inline, _attachments = _file_get_deps(
            file_kind=lambda path, prefix: ("image", "image/png") if path == image and prefix.startswith(b"\x89PNG") else ("text", None),
        )
        handled = handle_absolute_file_preview_route(
            _FakeHandler(),
            path="/api/files/blob",
            query="path=" + urllib.parse.quote(str(image)),
            deps=deps,
        )
        assert handled is True
        assert responses == []
        assert inline == [(image, "image/png")]


def test_handle_absolute_file_preview_route_maps_prefix_symlink_rejection() -> None:
    with tempfile.TemporaryDirectory() as td:
        image = Path(td) / "preview.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\nbody")
        deps, responses, inline, _attachments = _file_get_deps(
            read_regular_file_prefix=lambda _path, _byte_count: (_ for _ in ()).throw(ValueError("symlink file not supported")),
        )
        handled = handle_absolute_file_preview_route(
            _FakeHandler(),
            path="/api/files/blob",
            query="path=" + urllib.parse.quote(str(image)),
            deps=deps,
        )
        assert handled is True
        assert responses == [(400, {"error": "symlink file not supported"})]
        assert inline == []


def test_handle_absolute_file_preview_route_maps_video_preview_runtime_error() -> None:
    with tempfile.TemporaryDirectory() as td:
        video = Path(td) / "clip.mp4"
        video.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 16)
        deps, responses, inline, _attachments = _file_get_deps(
            file_kind=lambda _path, _prefix: ("video", "video/mp4"),
            ensure_video_preview=lambda _path: (_ for _ in ()).throw(RuntimeError("ffmpeg failed")),
        )
        handled = handle_absolute_file_preview_route(
            _FakeHandler(),
            path="/api/files/video_preview",
            query="path=" + urllib.parse.quote(str(video)),
            deps=deps,
        )
        assert handled is True
        assert responses == [(500, {"error": "video preview failed: ffmpeg failed"})]
        assert inline == []


def _file_write_deps(body, **overrides):
    responses: list[tuple[int, dict[str, object]]] = []

    def json_response(_handler, status: int, payload: dict[str, object]) -> None:
        responses.append((status, payload))

    deps = FileWriteRouteDeps(
        require_auth=lambda _handler: True,
        json_response=json_response,
        read_json_body=lambda _handler, **_kwargs: body,
        resolve_session_cwd=lambda raw: Path(raw),
        resolve_create_path=_resolve_create_path,
        resolve_git_existing_regular_file=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected git file")),
        file_write_lock=_null_write_lock,
    )
    for name, value in overrides.items():
        object.__setattr__(deps, name, value)
    return deps, responses


def test_handle_file_write_post_route_validates_body_before_session_lookup() -> None:
    class FailingManager:
        def refresh_session_meta(self, _session_id: str) -> None:
            raise AssertionError("session metadata should not be refreshed for invalid bodies")

        def get_session(self, _session_id: str) -> object:
            raise AssertionError("session lookup should not run for invalid bodies")

    deps, responses = _file_write_deps({"text": "new"})
    handled = handle_file_write_post_route(
        _FakeHandler(),
        path="/api/sessions/s/file/write",
        manager=FailingManager(),
        deps=deps,
        match_session_route=_match_session_route,
    )
    assert handled is True
    assert responses == [(400, {"error": "path required"})]


def test_handle_file_write_post_route_creates_file_and_records_path() -> None:
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        (base / "notes").mkdir()
        manager = _FakeManager(str(base))
        deps, responses = _file_write_deps({"path": "notes/new.md", "text": "hello\n", "create": True})
        handled = handle_file_write_post_route(
            _FakeHandler(),
            path="/api/sessions/s/file/write",
            manager=manager,
            deps=deps,
            match_session_route=_match_session_route,
        )
        target = (base / "notes" / "new.md").resolve()
        assert handled is True
        assert responses[0][0] == 200
        assert responses[0][1]["path"] == str(target)
        assert responses[0][1]["rel"] == "notes/new.md"
        assert target.read_text(encoding="utf-8") == "hello\n"
        assert manager.refreshed == ["s"]
        assert manager.recorded == [("s", str(target))]


def test_handle_file_write_post_route_create_rejects_non_empty_path_token() -> None:
    # Create with any non-empty path_token must fail loud (400): new files take
    # a UTF-8 name from ``path``, so a token is meaningless and previously a
    # bad/stale token was silently ignored. The body is validated before the
    # session is even looked up.
    class FailingManager:
        def refresh_session_meta(self, _session_id: str) -> None:
            raise AssertionError("session metadata should not be refreshed for create+token")

        def get_session(self, _session_id: str) -> object:
            raise AssertionError("session lookup should not run for create+token")

    deps, responses = _file_write_deps(
        {"path": "notes/new.md", "text": "hello\n", "create": True, "path_token": "bad-token"}
    )
    handled = handle_file_write_post_route(
        _FakeHandler(),
        path="/api/sessions/s/file/write",
        manager=FailingManager(),
        deps=deps,
        match_session_route=_match_session_route,
    )
    assert handled is True
    assert responses == [(400, {"error": "path_token is not supported for create"})]


def test_handle_file_write_post_route_update_rejects_invalid_path_token() -> None:
    # Update with an invalid path_token must remain a 400 (regression guard for
    # the create fix staying consistent with the update path).
    class FailingManager:
        def refresh_session_meta(self, _session_id: str) -> None:
            raise AssertionError("session metadata should not be refreshed for invalid token")

        def get_session(self, _session_id: str) -> object:
            raise AssertionError("session lookup should not run for invalid token")

    deps, responses = _file_write_deps(
        {"path": "notes/old.md", "text": "hello\n", "version": "v1", "path_token": "bad-token"}
    )
    handled = handle_file_write_post_route(
        _FakeHandler(),
        path="/api/sessions/s/file/write",
        manager=FailingManager(),
        deps=deps,
        match_session_route=_match_session_route,
    )
    assert handled is True
    assert responses[0][0] == 400
    assert "invalid path token" in responses[0][1]["error"]


def _global_file_deps(body, **overrides):
    responses: list[tuple[int, dict[str, object]]] = []

    def json_response(_handler, status: int, payload: dict[str, object]) -> None:
        responses.append((status, payload))

    deps = GlobalFileRouteDeps(
        require_auth=lambda _handler: True,
        json_response=json_response,
        read_json_body=lambda _handler, **_kwargs: body,
        resolve_git_client_file_view=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected git view")),
        resolve_client_file_path=lambda *, session_id, raw_path: Path(raw_path),
        read_client_file_view=lambda path: ClientFileView(kind="text", size=5, text="hello", editable=True, version="v1"),
    )
    for name, value in overrides.items():
        object.__setattr__(deps, name, value)
    return deps, responses


def test_global_file_inspect_reports_html_kind_for_html_extensions() -> None:
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        for name in ("page.html", "page.htm", "page.HTML"):
            path = base / name
            path.write_text("<p>preview</p>", encoding="utf-8")
            deps, responses = _global_file_deps(
                {"path": str(path), "session_id": "s"},
                resolve_client_file_path=lambda **_kwargs: path,
                read_client_file_view=read_client_file_view,
            )
            assert handle_global_file_post_route(_FakeHandler(), path="/api/files/inspect", manager=_FakeManager(str(base)), deps=deps)
            assert responses[0][0] == 200
            assert responses[0][1]["kind"] == "html"


def test_handle_global_file_post_route_rejects_non_string_session_id_before_resolution() -> None:
    deps, responses = _global_file_deps(
        {"path": "note.md", "session_id": 123},
        resolve_client_file_path=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("path resolution should not run")),
    )
    handled = handle_global_file_post_route(_FakeHandler(), path="/api/files/read", manager=_FakeManager("/tmp"), deps=deps)
    assert handled is True
    assert responses == [(400, {"error": "session_id must be a string"})]


def test_handle_global_file_inspect_batch_reports_missing_files_without_a_404() -> None:
    def resolve_client_file_path(*, session_id: str, raw_path: str) -> Path:
        assert session_id == "session-1"
        if raw_path == "gone.md":
            raise FileNotFoundError("gone.md")
        return Path("/repo") / raw_path

    deps, responses = _global_file_deps(
        {"session_id": "session-1", "paths": ["present.md", "gone.md"]},
        resolve_client_file_path=resolve_client_file_path,
    )
    handled = handle_global_file_post_route(_FakeHandler(), path="/api/files/inspect-batch", manager=_FakeManager("/tmp"), deps=deps)
    assert handled is True
    assert responses == [
        (
            200,
            {
                "results": [
                    {
                        "ok": True,
                        "path": "present.md",
                        "exists": True,
                        "resolved_path": "/repo/present.md",
                        "kind": "text",
                        "content_type": None,
                        "size": 5,
                        "reason": None,
                        "viewer_max_bytes": None,
                    },
                    {"path": "gone.md", "exists": False},
                ]
            },
        )
    ]


def test_handle_global_file_inspect_batch_limits_paths_before_resolution() -> None:
    deps, responses = _global_file_deps(
        {"session_id": "session-1", "paths": [f"file-{index}" for index in range(51)]},
        resolve_client_file_path=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("path resolution should not run")),
    )
    handled = handle_global_file_post_route(_FakeHandler(), path="/api/files/inspect-batch", manager=_FakeManager("/tmp"), deps=deps)
    assert handled is True
    assert responses == [(400, {"error": "at most 50 paths allowed"})]


def test_handle_global_file_post_route_preserves_whitespace_only_path_and_records_read() -> None:
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        space_file = base / " "
        space_file.write_text("space\n", encoding="utf-8")
        manager = _FakeManager(str(base))
        deps, responses = _global_file_deps(
            {"path": " ", "session_id": "s"},
            resolve_client_file_path=lambda *, session_id, raw_path: space_file if session_id == "s" and raw_path == " " else (_ for _ in ()).throw(AssertionError()),
            read_client_file_view=lambda path: ClientFileView(kind="text", size=6, text=path.read_text(encoding="utf-8"), editable=True, version="v2"),
        )
        handled = handle_global_file_post_route(_FakeHandler(), path="/api/files/read", manager=manager, deps=deps)
        assert handled is True
        assert responses == [(200, {"ok": True, "kind": "text", "path": str(space_file), "size": 6, "text": "space\n", "editable": True, "version": "v2"})]
        assert manager.recorded == [("s", str(space_file))]


def test_global_file_read_payload_keeps_git_path_media_urls_session_relative() -> None:
    payload = global_file_read_payload(
        request=GlobalFileRequest(path="assets/icon.svg", session_id="s", git_path=True),
        path_obj=Path("/repo/assets/icon.svg"),
        rel_for_url="assets/icon.svg",
        view=ClientFileView(kind="image", size=12, content_type="image/svg+xml"),
    )
    assert payload["image_url"] == "/api/sessions/s/file/blob?path=assets/icon.svg&git_path=1"
