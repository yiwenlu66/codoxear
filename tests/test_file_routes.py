from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import tempfile

import pytest

from codoxear.file_routes import FileRouteResponse
from codoxear.file_routes import SessionFileWriteRequest
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
