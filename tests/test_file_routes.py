from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import urllib.parse

import pytest

from codoxear.file_routes import FileGetRouteDeps
from codoxear.file_routes import FileRouteResponse
from codoxear.file_routes import FileWriteRouteDeps
from codoxear.file_routes import SessionFileWriteRequest
from codoxear.file_routes import handle_absolute_file_preview_route
from codoxear.file_routes import handle_file_get_route
from codoxear.file_routes import handle_file_write_post_route
from codoxear.file_routes import resolve_session_write_update_path
from codoxear.file_routes import session_file_read_payload
from codoxear.file_routes import session_file_write_response
from codoxear.file_view import ClientFileView


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
        search_session_relative_files=lambda base, *, query, limit: {
            "query": query,
            "mode": "literal",
            "matches": [],
            "scanned": 0,
            "truncated": False,
        },
        list_session_relative_files=lambda _base: ["note.txt"],
        file_kind=lambda _path, _prefix: ("text", None),
        ensure_video_preview=lambda path: path.with_suffix(".preview.mp4"),
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
