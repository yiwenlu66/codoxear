import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest
import urllib.parse
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from codoxear import file_text
from codoxear import git_ops
from codoxear import server
from codoxear.client_file_paths import resolve_client_file_path as _resolve_client_file_path_impl
from codoxear.file_routes import FileGetRouteDeps
from codoxear.file_routes import FileWriteRouteDeps
from codoxear.file_routes import GlobalFileRouteDeps
from codoxear.file_routes import handle_absolute_file_preview_route
from codoxear.file_routes import handle_file_get_route
from codoxear.file_routes import handle_file_write_post_route
from codoxear.file_routes import handle_global_file_post_route
from codoxear.file_view import ClientFileView
from codoxear.git_routes import GitRouteDeps
from codoxear.git_routes import handle_git_get_route
from codoxear.server import _current_git_branch
from codoxear.server import _download_disposition
from codoxear.server import _ensure_video_preview
from codoxear.server import _inspect_client_path
from codoxear.server import _inspect_downloadable_file
from codoxear.server import _inspect_openable_file
from codoxear.server import _read_client_file_view
from codoxear.server import _resolve_client_file_path
from codoxear.server import _resolve_existing_absolute_file
from codoxear.server import _resolve_session_cwd
from codoxear.file_text import read_text_file_for_client as _read_text_file_for_client
from codoxear.file_text import read_text_file_for_write as _read_text_file_for_write
from codoxear.file_text import write_new_text_file_atomic as _write_new_text_file_atomic
from codoxear.file_text import write_text_file_atomic as _write_text_file_atomic
from codoxear.server import _read_text_or_image
from codoxear.server import _single_byte_range
from codoxear.server_routing import match_session_route as _match_session_route


REPO_CWD = Path(__file__).resolve().parents[1]


def _safe_cwd() -> Path:
    try:
        return Path.cwd()
    except FileNotFoundError:
        return REPO_CWD


# --- Direct route-handler test infrastructure (dependency injection) ---
#
# These helpers replace the previous monkeypatch-based tests that patched
# server.MANAGER, server._require_auth, and server._json_response. All three
# seams are now satisfied by injecting a FakeManager plus capturing callables
# into the route-handler deps dataclasses, mirroring the pattern established in
# tests/test_message_routes.py and tests/test_file_routes.py. Real server
# functions (server._run_git, server._resolve_git_path, server._resolve_session_cwd,
# ...) are wired in directly so that real-git / real-filesystem behaviour is
# exercised without any module-level patching.


class _FakeHandler:
    """Minimal handler object exposing only the attribute route handlers touch."""

    def __init__(self) -> None:
        self.unauthorized = False

    def _unauthorized(self) -> None:
        self.unauthorized = True


class _FakeManager:
    """Session manager fake: holds a single session cwd and records file adds."""

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


def _capture_json() -> tuple[list[tuple[int, dict[str, object]]], object]:
    """Return (responses_list, json_response_callable) capturing every call."""
    responses: list[tuple[int, dict[str, object]]] = []

    def json_response(_handler, status: int, payload: dict[str, object]) -> None:
        responses.append((status, payload))

    return responses, json_response


def _git_view_resolvers(manager, *, resolve_git_path, read_client_file_view, require_existing_file):
    """Build resolve_git_client_file_view / resolve_git_existing_regular_file using
    the injected manager rather than the module-level MANAGER singleton.

    The server._resolve_git_client_file_view closures bake in the module MANAGER,
    which would bypass the per-test FakeManager. Wiring the impl primitives here
    keeps the manager seam explicit (dependency injection, no patching).
    """
    from codoxear.client_file_paths import resolve_git_client_file_view as _impl_view
    from codoxear.client_file_paths import resolve_git_existing_regular_file as _impl_existing

    def resolve_git_client_file_view(*, session_id, raw_path):
        return _impl_view(
            session_id=session_id,
            raw_path=raw_path,
            refresh_session_meta=manager.refresh_session_meta,
            get_session=manager.get_session,
            resolve_session_cwd=server._resolve_session_cwd,
            resolve_git_path=resolve_git_path,
            read_client_file_view=read_client_file_view,
        )

    def resolve_git_existing_regular_file(*, session_id, raw_path):
        return _impl_existing(
            session_id=session_id,
            raw_path=raw_path,
            refresh_session_meta=manager.refresh_session_meta,
            get_session=manager.get_session,
            resolve_session_cwd=server._resolve_session_cwd,
            resolve_git_path=resolve_git_path,
            require_existing_file=require_existing_file,
        )

    return resolve_git_client_file_view, resolve_git_existing_regular_file


def _file_get_deps(manager=None, **overrides):
    """FileGetRouteDeps wired with real server resolvers + capturing seams.

    By default auth always passes and json_response captures into the returned
    list. Real filesystem/git behaviour is preserved because the resolvers point
    at the actual server implementations. When ``manager`` is supplied the
    git-path view resolvers are wired to that manager (instead of the module
    MANAGER singleton) so session-aware git-path resolution honours the injected
    fake manager. Tests override individual fields (for example to inject a
    raising resolver) as needed.
    """
    responses, json_response = _capture_json()
    inline: list[tuple[Path, str]] = []
    attachments: list[tuple[Path, int, str]] = []

    def send_inline(_handler, path: Path, content_type: str) -> None:
        inline.append((path, content_type))

    def send_attachment(_handler, path: Path, *, size: int, content_disposition: str) -> None:
        attachments.append((path, size, content_disposition))

    resolve_git_client_file_view = server._resolve_git_client_file_view
    resolve_git_existing_regular_file = server._resolve_git_existing_regular_file
    if manager is not None:
        resolve_git_client_file_view, resolve_git_existing_regular_file = _git_view_resolvers(
            manager,
            resolve_git_path=server._resolve_git_path,
            read_client_file_view=server._read_client_file_view,
            require_existing_file=server._require_existing_file,
        )

    deps = FileGetRouteDeps(
        require_auth=lambda _handler: True,
        json_response=json_response,
        resolve_session_cwd=server._resolve_session_cwd,
        resolve_existing_session_file=server._resolve_existing_session_file,
        resolve_session_path=server._resolve_session_path,
        resolve_git_client_file_view=resolve_git_client_file_view,
        resolve_git_existing_regular_file=resolve_git_existing_regular_file,
        resolve_existing_absolute_file=server._resolve_existing_absolute_file,
        read_client_file_view=server._read_client_file_view,
        read_regular_file_prefix=server._read_regular_file_prefix_no_symlink,
        search_session_relative_files=server._search_session_relative_files,
        list_session_relative_files=server._list_session_relative_files,
        list_session_relative_file_entries=server._list_session_relative_file_entries,
        file_kind=server._file_kind,
        ensure_video_preview=server._ensure_video_preview,
        inspect_downloadable_file=server._inspect_downloadable_file,
        download_disposition=server._download_disposition,
        send_inline_file_response=send_inline,
        send_attachment_file_response=send_attachment,
        file_search_limit=server.FILE_SEARCH_LIMIT,
    )
    for name, value in overrides.items():
        object.__setattr__(deps, name, value)
    return deps, responses, inline, attachments


def _git_deps(**overrides):
    """GitRouteDeps wired with server git helpers + capturing seams.

    require_git_repo / resolve_git_path / git_head_blob_oid are derived from the
    chosen run_git so that a fake run_git is honoured consistently across every
    git invocation the route makes (the real server closures bake in the module
    _run_git, which would bypass an injected fake runner).
    """
    responses, json_response = _capture_json()
    run_git = overrides.pop("run_git", server._run_git)
    timeout = server.GIT_DIFF_TIMEOUT_SECONDS

    def _resolve_git_path(cwd: Path, raw_path: str):
        return git_ops.resolve_git_path(cwd, raw_path, run_git_func=run_git, timeout_s=timeout)

    def _git_head_blob_oid(cwd: Path, rel: str):
        return git_ops.git_head_blob_oid(cwd, rel, run_git_func=run_git, timeout_s=timeout)

    def _require_git_repo(cwd: Path) -> None:
        git_ops.require_git_repo(cwd, run_git_func=run_git, timeout_s=timeout)

    deps = GitRouteDeps(
        require_auth=lambda _handler: True,
        json_response=json_response,
        resolve_session_cwd=server._resolve_session_cwd,
        require_git_repo=_require_git_repo,
        split_git_nul_paths=server._split_git_nul_paths,
        run_git=run_git,
        parse_git_numstat=server._parse_git_numstat,
        resolve_git_path=_resolve_git_path,
        read_text_file_strict=server._read_text_file_strict,
        git_head_blob_oid=_git_head_blob_oid,
        git_changed_files_max=server.GIT_CHANGED_FILES_MAX,
        git_diff_timeout_seconds=timeout,
        git_diff_max_bytes=server.GIT_DIFF_MAX_BYTES,
        file_read_max_bytes=server.FILE_READ_MAX_BYTES,
    )
    for name, value in overrides.items():
        object.__setattr__(deps, name, value)
    return deps, responses


def _file_write_deps(body, manager=None, **overrides):
    """FileWriteRouteDeps wired with real server resolvers + capturing seams.

    When ``manager`` is supplied the git-path existing-file resolver is wired
    to that manager instead of the module MANAGER singleton.
    """
    responses, json_response = _capture_json()

    @contextmanager
    def null_write_lock(_path: Path):
        yield

    resolve_git_existing_regular_file = server._resolve_git_existing_regular_file
    if manager is not None:
        _resolve_git_client_file_view, resolve_git_existing_regular_file = _git_view_resolvers(
            manager,
            resolve_git_path=server._resolve_git_path,
            read_client_file_view=server._read_client_file_view,
            require_existing_file=server._require_existing_file,
        )

    deps = FileWriteRouteDeps(
        require_auth=lambda _handler: True,
        json_response=json_response,
        read_json_body=lambda _handler, **_kwargs: body,
        resolve_session_cwd=server._resolve_session_cwd,
        resolve_create_path=server._resolve_under,
        resolve_git_existing_regular_file=resolve_git_existing_regular_file,
        file_write_lock=null_write_lock,
    )
    for name, value in overrides.items():
        object.__setattr__(deps, name, value)
    return deps, responses


def _global_file_deps(body, manager=None, **overrides):
    """GlobalFileRouteDeps wired with real server resolvers + capturing seams.

    When ``manager`` is supplied the git-path view resolver and the plain
    client-file-path resolver are wired to that manager instead of the module
    MANAGER singleton (both server closures bake in the module MANAGER).
    """
    responses, json_response = _capture_json()
    resolve_git_client_file_view = server._resolve_git_client_file_view
    resolve_client_file_path = server._resolve_client_file_path
    if manager is not None:
        resolve_git_client_file_view, _resolve_git_existing_regular_file = _git_view_resolvers(
            manager,
            resolve_git_path=server._resolve_git_path,
            read_client_file_view=server._read_client_file_view,
            require_existing_file=server._require_existing_file,
        )

        def resolve_client_file_path(*, session_id, raw_path):
            return _resolve_client_file_path_impl(
                session_id=session_id,
                raw_path=raw_path,
                refresh_session_meta=manager.refresh_session_meta,
                get_session=manager.get_session,
                files_get=lambda sid: manager.files_get(sid) if hasattr(manager, "files_get") else [],
                expanduser_path=server._expanduser_path,
                resolve_session_cwd=server._resolve_session_cwd,
                run_git=server._run_git,
                git_timeout_s=server.GIT_DIFF_TIMEOUT_SECONDS,
            )
    deps = GlobalFileRouteDeps(
        require_auth=lambda _handler: True,
        json_response=json_response,
        read_json_body=lambda _handler, **_kwargs: body,
        resolve_git_client_file_view=resolve_git_client_file_view,
        resolve_client_file_path=resolve_client_file_path,
        read_client_file_view=server._read_client_file_view,
    )
    for name, value in overrides.items():
        object.__setattr__(deps, name, value)
    return deps, responses


class TestInspectOpenableFile(unittest.TestCase):
    def test_directory_is_supported_for_inspection(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "repo"
            path.mkdir()
            size, kind, image_ctype = _inspect_client_path(path)
            self.assertEqual(size, 0)
            self.assertEqual(kind, "directory")
            self.assertIsNone(image_ctype)

    def test_text_file_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "note.py"
            path.write_text("print('ok')\n", encoding="utf-8")
            raw, size, kind, image_ctype = _inspect_openable_file(path)
            self.assertEqual(kind, "text")
            self.assertIsNone(image_ctype)
            self.assertEqual(size, len(raw))

    def test_binary_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "blob.bin"
            path.write_bytes(b"\x00\x01\x02\x03")
            with self.assertRaisesRegex(ValueError, "binary file not supported"):
                _inspect_openable_file(path)

    def test_binary_file_is_download_only_for_client_view(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "blob.bin"
            path.write_bytes(b"\x00\x01\x02\x03")
            view = _read_client_file_view(path)
            self.assertEqual(view.kind, "download_only")
            self.assertEqual(view.blocked_reason, "binary")
            self.assertEqual(view.size, 4)

    def test_large_image_is_supported_for_metadata_inspection(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "large.png"
            path.write_bytes(b"\x89PNG\r\n\x1a\n" + (b"x" * (2 * 1024 * 1024)))
            size, kind, image_ctype = _inspect_client_path(path)
            self.assertGreater(size, 2 * 1024 * 1024)
            self.assertEqual(kind, "image")
            self.assertEqual(image_ctype, "image/png")

    def test_large_text_file_is_download_only_for_metadata_inspection(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "large.md"
            path.write_text("a" * (2 * 1024 * 1024 + 1), encoding="utf-8")
            size, kind, image_ctype = _inspect_client_path(path)
            self.assertGreater(size, 2 * 1024 * 1024)
            self.assertEqual(kind, "download_only")
            self.assertIsNone(image_ctype)
            view = _read_client_file_view(path)
            self.assertEqual(view.blocked_reason, "too_large")
            self.assertEqual(view.viewer_max_bytes, 2 * 1024 * 1024)

    def test_large_image_read_returns_metadata_without_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "large.png"
            path.write_bytes(b"\x89PNG\r\n\x1a\n" + (b"x" * (2 * 1024 * 1024)))
            kind, size, image_ctype, raw = _read_text_or_image(path)
            self.assertEqual(kind, "image")
            self.assertEqual(image_ctype, "image/png")
            self.assertGreater(size, 2 * 1024 * 1024)
            self.assertIsNone(raw)

    def test_text_read_returns_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "note.md"
            path.write_text("hello\n", encoding="utf-8")
            kind, size, image_ctype, raw = _read_text_or_image(path)
            self.assertEqual(kind, "markdown")
            self.assertIsNone(image_ctype)
            self.assertEqual(size, 6)
            self.assertEqual(raw, b"hello\n")

    def test_pdf_is_supported_for_metadata_and_read(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "paper.pdf"
            raw_in = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<< /Type /Catalog >>\nendobj\n%%EOF\n"
            path.write_bytes(raw_in)
            size, kind, content_type = _inspect_client_path(path)
            self.assertEqual(kind, "pdf")
            self.assertEqual(content_type, "application/pdf")
            self.assertEqual(size, len(raw_in))
            kind2, size2, content_type2, raw = _read_text_or_image(path)
            self.assertEqual(kind2, "pdf")
            self.assertEqual(size2, len(raw_in))
            self.assertEqual(content_type2, "application/pdf")
            self.assertIsNone(raw)

    def test_video_is_supported_for_metadata_and_read(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "clip.mp4"
            raw_in = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + (b"\x00" * 8192)
            path.write_bytes(raw_in)
            size, kind, content_type = _inspect_client_path(path)
            self.assertEqual(kind, "video")
            self.assertEqual(content_type, "video/mp4")
            self.assertEqual(size, len(raw_in))
            kind2, size2, content_type2, raw = _read_text_or_image(path)
            self.assertEqual(kind2, "video")
            self.assertEqual(size2, len(raw_in))
            self.assertEqual(content_type2, "video/mp4")
            self.assertIsNone(raw)

    def test_single_byte_range_supports_video_seek_shapes(self) -> None:
        self.assertEqual(_single_byte_range("bytes=10-19", 100), (10, 19))
        self.assertEqual(_single_byte_range("bytes=95-", 100), (95, 99))
        self.assertEqual(_single_byte_range("bytes=-5", 100), (95, 99))
        with self.assertRaises(ValueError):
            _single_byte_range("bytes=100-110", 100)

    @unittest.skipIf(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, "ffmpeg and ffprobe required")
    def test_video_preview_transcodes_odd_dimensions_to_browser_safe_mp4(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            src = td_path / "odd.mkv"
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc=size=161x91:rate=1",
                    "-t",
                    "0.2",
                    str(src),
                ],
                check=True,
            )
            old_dir = server.VIDEO_PREVIEW_DIR
            try:
                server.VIDEO_PREVIEW_DIR = td_path / "previews"
                preview = _ensure_video_preview(src)
                info = subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=codec_name,pix_fmt,width,height",
                        "-of",
                        "default=noprint_wrappers=1",
                        str(preview),
                    ],
                    text=True,
                )
                self.assertIn("codec_name=h264", info)
                self.assertIn("pix_fmt=yuv420p", info)
                self.assertIn("width=162", info)
                self.assertIn("height=92", info)
            finally:
                server.VIDEO_PREVIEW_DIR = old_dir

    @unittest.skipIf(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, "ffmpeg and ffprobe required")
    def test_video_preview_transcodes_to_browser_safe_mp4(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            src = td_path / "clip.mkv"
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=red:s=160x90:d=0.2",
                    "-f",
                    "lavfi",
                    "-i",
                    "anullsrc=channel_layout=stereo:sample_rate=44100",
                    "-shortest",
                    "-c:v",
                    "mpeg4",
                    "-c:a",
                    "pcm_s16le",
                    str(src),
                ],
                check=True,
            )
            old_dir = server.VIDEO_PREVIEW_DIR
            try:
                server.VIDEO_PREVIEW_DIR = td_path / "previews"
                preview = _ensure_video_preview(src)
                self.assertEqual(preview.suffix, ".mp4")
                self.assertTrue(preview.exists())
                info = subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=codec_name,pix_fmt",
                        "-of",
                        "default=noprint_wrappers=1",
                        str(preview),
                    ],
                    text=True,
                )
                self.assertIn("codec_name=h264", info)
                self.assertIn("pix_fmt=yuv420p", info)
            finally:
                server.VIDEO_PREVIEW_DIR = old_dir

    # --- Converted route-handler tests (direct handle_*_route calls) ---
    #
    # Each test below previously constructed a server.Handler via __new__ and
    # patched server.MANAGER / server._require_auth / server._json_response, then
    # drove server.Handler.do_GET/do_POST. They now call the route handlers
    # directly with injected deps, preserving the exact status/payload assertions.

    def test_unknown_session_file_resolution_does_not_fallback_to_server_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            (td_path / "note.md").write_text("cwd fallback should not win\n", encoding="utf-8")
            old_cwd = _safe_cwd()
            try:
                os.chdir(td_path)
                with self.assertRaisesRegex(FileNotFoundError, "unknown session"):
                    _resolve_client_file_path(session_id=f"missing-{uuid.uuid4().hex}", raw_path="note.md")
                with self.assertRaisesRegex(FileNotFoundError, "unknown session"):
                    _resolve_client_file_path(session_id=f"missing-{uuid.uuid4().hex}", raw_path="~definitely-no-such-codoxear-user/note.md")
            finally:
                os.chdir(old_cwd if old_cwd.exists() else REPO_CWD)

    def test_no_session_file_resolution_keeps_server_cwd_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            expected = td_path / "note.md"
            expected.write_text("local fallback still works\n", encoding="utf-8")
            old_cwd = _safe_cwd()
            try:
                os.chdir(td_path)
                self.assertEqual(_resolve_client_file_path(session_id="", raw_path="note.md"), expected.resolve())
            finally:
                os.chdir(old_cwd if old_cwd.exists() else REPO_CWD)

    def test_no_session_absolute_file_resolution_still_works(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            expected = (Path(td) / "note.md").resolve()
            expected.write_text("absolute local ref still works\n", encoding="utf-8")
            self.assertEqual(_resolve_client_file_path(session_id="", raw_path=str(expected)), expected)

    def test_bad_expanduser_path_is_bad_request_not_runtime_error(self) -> None:
        bad_path = f"~definitely-no-such-codoxear-user-{uuid.uuid4().hex}/note.md"
        with self.assertRaisesRegex(ValueError, "home directory"):
            _resolve_client_file_path(session_id="", raw_path=bad_path)
        with self.assertRaisesRegex(ValueError, "home directory"):
            _resolve_existing_absolute_file(bad_path)

    def test_bad_session_cwd_expanduser_path_is_bad_request_not_runtime_error(self) -> None:
        bad_cwd = f"~definitely-no-such-codoxear-user-{uuid.uuid4().hex}/repo"

        def _refresh(_sid: str) -> None:
            return None

        def _get_session(_sid: str):
            return SimpleNamespace(cwd=bad_cwd)

        def _files_get(_sid: str):
            return []

        with self.assertRaisesRegex(ValueError, "home directory"):
            _resolve_client_file_path_impl(
                session_id="session-with-bad-cwd",
                raw_path="note.md",
                refresh_session_meta=_refresh,
                get_session=_get_session,
                files_get=_files_get,
                expanduser_path=server._expanduser_path,
                resolve_session_cwd=server._resolve_session_cwd,
                run_git=server._run_git,
                git_timeout_s=server.GIT_DIFF_TIMEOUT_SECONDS,
            )
        with self.assertRaisesRegex(ValueError, "invalid session cwd"):
            _resolve_session_cwd("/tmp/bad\x00cwd")

    def test_bad_tracked_file_expanduser_path_is_bad_request_not_runtime_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bad_tracked = f"~definitely-no-such-codoxear-user-{uuid.uuid4().hex}/note.md"

            def _refresh(_sid: str) -> None:
                return None

            def _get_session(_sid: str):
                return SimpleNamespace(cwd=td)

            def _files_get(_sid: str):
                return [bad_tracked]

            with self.assertRaisesRegex(ValueError, "home directory"):
                _resolve_client_file_path_impl(
                    session_id="session-with-bad-tracked",
                    raw_path="note.md",
                    refresh_session_meta=_refresh,
                    get_session=_get_session,
                    files_get=_files_get,
                    expanduser_path=server._expanduser_path,
                    resolve_session_cwd=server._resolve_session_cwd,
                    run_git=server._run_git,
                    git_timeout_s=server.GIT_DIFF_TIMEOUT_SECONDS,
                )

    def test_session_file_routes_return_400_for_bad_session_cwd(self) -> None:
        bad_cwds = [f"~definitely-no-such-codoxear-user-{uuid.uuid4().hex}/repo", "/tmp/bad\x00cwd"]

        get_routes = [
            "/api/sessions/s/file/read?path=note.md",
            "/api/sessions/s/file/search?q=note",
            "/api/sessions/s/file/list",
            "/api/sessions/s/file/blob?path=note.md",
            "/api/sessions/s/file/video_preview?path=clip.mp4",
            "/api/sessions/s/file/download?path=note.md",
        ]
        git_routes = [
            "/api/sessions/s/git/changed_files",
            "/api/sessions/s/git/diff?path=note.md",
            "/api/sessions/s/git/file_versions?path=note.md",
        ]
        for bad_cwd in bad_cwds:
            for route in get_routes + git_routes:
                with self.subTest(route=route, bad_cwd=bad_cwd):
                    manager = _FakeManager(bad_cwd)
                    parsed = urllib.parse.urlparse(route)
                    if parsed.path.startswith("/api/sessions/s/git/"):
                        deps, responses = _git_deps()
                        handled = handle_git_get_route(
                            _FakeHandler(),
                            path=parsed.path,
                            query=parsed.query,
                            manager=manager,
                            deps=deps,
                            match_session_route=_match_session_route,
                        )
                    else:
                        deps, responses, _inline, _attachments = _file_get_deps()
                        handled = handle_file_get_route(
                            _FakeHandler(),
                            path=parsed.path,
                            query=parsed.query,
                            manager=manager,
                            deps=deps,
                            match_session_route=_match_session_route,
                        )
                    self.assertTrue(handled)
                    self.assertEqual(len(responses), 1)
                    status, payload = responses[0]
                    self.assertEqual(status, 400)
                    self.assertTrue(any(fragment in str(payload.get("error", "")) for fragment in ("home directory", "invalid session cwd")))

    def test_existing_file_permission_errors_are_403_not_404(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            preview = root / "preview.png"
            preview.write_bytes(b"\x89PNG\r\n\x1a\nbody")

            manager = _FakeManager(str(root))
            session_routes = [
                ("/api/sessions/s/file/blob", f"path={urllib.parse.quote(preview.name)}"),
                ("/api/sessions/s/file/video_preview", f"path={urllib.parse.quote(preview.name)}"),
            ]
            absolute_routes = [
                "/api/files/blob",
                "/api/files/video_preview",
            ]
            # Session blob/video_preview: resolve_existing_session_file raises.
            for route_path, query in session_routes:
                with self.subTest(route=route_path):
                    deps, responses, _inline, _attachments = _file_get_deps(
                        resolve_existing_session_file=lambda _base, _rel: (_ for _ in ()).throw(PermissionError("denied")),
                    )
                    handled = handle_file_get_route(
                        _FakeHandler(),
                        path=route_path,
                        query=query,
                        manager=manager,
                        deps=deps,
                        match_session_route=_match_session_route,
                    )
                    self.assertTrue(handled)
                    self.assertEqual(responses, [(403, {"error": "denied"})])
            # Absolute blob/video_preview: resolve_existing_absolute_file raises.
            for route in absolute_routes:
                with self.subTest(route=route):
                    deps, responses, _inline, _attachments = _file_get_deps(
                        resolve_existing_absolute_file=lambda _raw: (_ for _ in ()).throw(PermissionError("denied")),
                    )
                    handled = handle_absolute_file_preview_route(
                        _FakeHandler(),
                        path=route,
                        query=f"path={urllib.parse.quote(str(preview))}",
                        deps=deps,
                    )
                    self.assertTrue(handled)
                    self.assertEqual(responses, [(403, {"error": "denied"})])

    def test_preview_prefix_read_errors_are_route_local(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            preview = root / "preview.png"
            preview.write_bytes(b"\x89PNG\r\n\x1a\nbody")

            manager = _FakeManager(str(root))
            session_routes = [
                ("/api/sessions/s/file/blob", f"path={urllib.parse.quote(preview.name)}"),
                ("/api/sessions/s/file/video_preview", f"path={urllib.parse.quote(preview.name)}"),
            ]
            absolute_routes = ["/api/files/blob", "/api/files/video_preview"]
            cases = [(PermissionError("denied"), 403), (FileNotFoundError("gone"), 404)]
            for exc, expected_status in cases:
                for route_path, query in session_routes:
                    with self.subTest(route=route_path, exc=type(exc).__name__):
                        deps, responses, _inline, _attachments = _file_get_deps(
                            read_regular_file_prefix=lambda _path, _byte_count: (_ for _ in ()).throw(exc),
                        )
                        handled = handle_file_get_route(
                            _FakeHandler(),
                            path=route_path,
                            query=query,
                            manager=manager,
                            deps=deps,
                            match_session_route=_match_session_route,
                        )
                        self.assertTrue(handled)
                        self.assertEqual(len(responses), 1)
                        status, payload = responses[0]
                        self.assertEqual(status, expected_status)
                        self.assertIn(str(exc), str(payload.get("error", "")))
                for route in absolute_routes:
                    with self.subTest(route=route, exc=type(exc).__name__):
                        deps, responses, _inline, _attachments = _file_get_deps(
                            read_regular_file_prefix=lambda _path, _byte_count: (_ for _ in ()).throw(exc),
                        )
                        handled = handle_absolute_file_preview_route(
                            _FakeHandler(),
                            path=route,
                            query=f"path={urllib.parse.quote(str(preview))}",
                            deps=deps,
                        )
                        self.assertTrue(handled)
                        self.assertEqual(len(responses), 1)
                        status, payload = responses[0]
                        self.assertEqual(status, expected_status)
                        self.assertIn(str(exc), str(payload.get("error", "")))

    def test_video_preview_generation_file_errors_are_route_local(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = root / "clip.mp4"
            video.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"\x00" * 32)

            manager = _FakeManager(str(root))
            session_routes = [
                ("/api/sessions/s/file/video_preview", f"path={urllib.parse.quote(video.name)}"),
            ]
            absolute_routes = ["/api/files/video_preview"]
            cases = [(PermissionError("denied"), 403), (FileNotFoundError("gone"), 404)]
            for exc, expected_status in cases:
                for route_path, query in session_routes:
                    with self.subTest(route=route_path, exc=type(exc).__name__):
                        deps, responses, _inline, _attachments = _file_get_deps(
                            ensure_video_preview=lambda _path: (_ for _ in ()).throw(exc),
                        )
                        handled = handle_file_get_route(
                            _FakeHandler(),
                            path=route_path,
                            query=query,
                            manager=manager,
                            deps=deps,
                            match_session_route=_match_session_route,
                        )
                        self.assertTrue(handled)
                        self.assertEqual(len(responses), 1)
                        status, payload = responses[0]
                        self.assertEqual(status, expected_status)
                        self.assertIn(str(exc), str(payload.get("error", "")))
                for route in absolute_routes:
                    with self.subTest(route=route, exc=type(exc).__name__):
                        deps, responses, _inline, _attachments = _file_get_deps(
                            ensure_video_preview=lambda _path: (_ for _ in ()).throw(exc),
                        )
                        handled = handle_absolute_file_preview_route(
                            _FakeHandler(),
                            path=route,
                            query=f"path={urllib.parse.quote(str(video))}",
                            deps=deps,
                        )
                        self.assertTrue(handled)
                        self.assertEqual(len(responses), 1)
                        status, payload = responses[0]
                        self.assertEqual(status, expected_status)
                        self.assertIn(str(exc), str(payload.get("error", "")))

    @unittest.skipIf(shutil.which("git") is None, "git required")
    def test_git_changed_files_preserves_whitespace_paths(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(["git", "config", "user.email", "codoxear@example.invalid"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Codoxear Test"], cwd=repo, check=True)
            paths = [" lead.md", "trail.md ", "tab\t.md", "new\n.md", "back\\slash.md"]
            for rel in paths:
                (repo / rel).write_text("base\n", encoding="utf-8")
            subprocess.run(["git", "add", "--", *[f":(literal){rel}" for rel in paths]], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "add whitespace paths"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            for rel in paths:
                (repo / rel).write_text("current\n", encoding="utf-8")

            manager = _FakeManager(str(repo))
            deps, responses = _git_deps()
            handle_git_get_route(
                _FakeHandler(),
                path="/api/sessions/s/git/changed_files",
                query="",
                manager=manager,
                deps=deps,
                match_session_route=_match_session_route,
            )
            self.assertEqual(len(responses), 1)
            status, payload = responses[0]
            self.assertEqual(status, 200)
            self.assertCountEqual(payload["files"], paths)
            self.assertCountEqual([entry["path"] for entry in payload["entries"]], paths)

    @unittest.skipIf(shutil.which("git") is None or os.name != "posix", "git and posix surrogateescape paths required")
    def test_git_changed_files_non_utf8_path_token_round_trips_to_file_versions(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(["git", "config", "user.email", "codoxear@example.invalid"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Codoxear Test"], cwd=repo, check=True)
            rel = os.fsdecode(b"caf\xe9.py")
            path = repo / rel
            path.write_text("base\n", encoding="utf-8")
            subprocess.run(["git", "add", "--", rel], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "add nonutf path"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            path.write_text("current\n", encoding="utf-8")

            class _RecordingManager(_FakeManager):
                def __init__(self, cwd: str) -> None:
                    super().__init__(cwd)
                    self.added: list[str] = []

                def files_add(self, _session_id: str, path_value: str) -> None:
                    self.added.append(path_value)

            manager = _RecordingManager(str(repo))

            def route_get(route: str) -> tuple[int, dict]:
                parsed = urllib.parse.urlparse(route)
                if parsed.path.startswith("/api/sessions/s/git/"):
                    deps, responses = _git_deps()
                    handle_git_get_route(
                        _FakeHandler(),
                        path=parsed.path,
                        query=parsed.query,
                        manager=manager,
                        deps=deps,
                        match_session_route=_match_session_route,
                    )
                else:
                    deps, responses, _inline, _attachments = _file_get_deps(manager=manager)
                    handle_file_get_route(
                        _FakeHandler(),
                        path=parsed.path,
                        query=parsed.query,
                        manager=manager,
                        deps=deps,
                        match_session_route=_match_session_route,
                    )
                self.assertEqual(len(responses), 1)
                status, payload = responses[0]
                json.dumps(payload, ensure_ascii=False).encode("utf-8")
                return status, payload

            status, payload = route_get("/api/sessions/s/git/changed_files")
            self.assertEqual(status, 200)
            entry = payload["entries"][0]
            self.assertEqual(entry["path"], "caf\\xe9.py")
            self.assertTrue(entry["non_utf8_path"])
            self.assertEqual(git_ops.git_path_from_token(entry["api_path"]).encode("utf-8", errors="surrogateescape"), b"caf\xe9.py")
            self.assertEqual(payload["files"], ["caf\\xe9.py"])

            route = "/api/sessions/s/git/file_versions?path={}&path_token={}".format(
                urllib.parse.quote(entry["path"]),
                urllib.parse.quote(entry["api_path"]),
            )
            status, versions = route_get(route)
            self.assertEqual(status, 200)
            self.assertEqual(versions["path"], "caf\\xe9.py")
            self.assertEqual(versions["current_text"], "current\n")
            self.assertEqual(versions["base_text"], "base\n")
            self.assertEqual(git_ops.git_path_from_token(versions["api_path"]).encode("utf-8", errors="surrogateescape"), b"caf\xe9.py")

            read_route = "/api/sessions/s/file/read?path={}&path_token={}&git_path=1".format(
                urllib.parse.quote(entry["path"]),
                urllib.parse.quote(entry["api_path"]),
            )
            status, read_payload = route_get(read_route)
            self.assertEqual(status, 200)
            self.assertEqual(read_payload["rel"], "caf\\xe9.py")
            self.assertEqual(read_payload["text"], "current\n")
            self.assertTrue(read_payload["editable"])

            def route_post(route: str, body: dict) -> tuple[int, dict]:
                parsed = urllib.parse.urlparse(route)
                if parsed.path == "/api/files/inspect" or parsed.path == "/api/files/read":
                    deps, responses = _global_file_deps(body, manager=manager)
                    handle_global_file_post_route(_FakeHandler(), path=parsed.path, manager=manager, deps=deps)
                else:
                    deps, responses = _file_write_deps(body, manager=manager)
                    handle_file_write_post_route(
                        _FakeHandler(),
                        path=parsed.path,
                        manager=manager,
                        deps=deps,
                        match_session_route=_match_session_route,
                    )
                self.assertEqual(len(responses), 1)
                status, payload = responses[0]
                json.dumps(payload, ensure_ascii=False).encode("utf-8")
                return status, payload

            status, inspect_payload = route_post(
                "/api/files/inspect",
                {"path": entry["path"], "path_token": entry["api_path"], "session_id": "s", "git_path": True},
            )
            self.assertEqual(status, 200)
            self.assertEqual(inspect_payload["kind"], "text")

            status, write_payload = route_post(
                "/api/sessions/s/file/write",
                {
                    "path": entry["path"],
                    "path_token": entry["api_path"],
                    "text": "saved\n",
                    "version": read_payload["version"],
                    "git_path": True,
                },
            )
            self.assertEqual(status, 200)
            self.assertEqual(write_payload["rel"], "caf\\xe9.py")
            self.assertEqual(path.read_text(encoding="utf-8"), "saved\n")
            self.assertEqual(manager.added, [str(repo / "caf\\xe9.py"), str(repo / "caf\\xe9.py"), str(repo / "caf\\xe9.py")])

    @unittest.skipIf(shutil.which("git") is None, "git required")
    def test_git_changed_files_parses_nul_numstat_rename_record(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(["git", "config", "user.email", "codoxear@example.invalid"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Codoxear Test"], cwd=repo, check=True)
            subprocess.run(["git", "config", "diff.renames", "true"], cwd=repo, check=True)
            old = repo / "old name.md"
            old.write_text("base\n", encoding="utf-8")
            subprocess.run(["git", "add", "old name.md"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "add old"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(["git", "mv", "old name.md", "new name.md"], cwd=repo, check=True)

            manager = _FakeManager(str(repo))
            deps, responses = _git_deps()
            handle_git_get_route(
                _FakeHandler(),
                path="/api/sessions/s/git/changed_files",
                query="",
                manager=manager,
                deps=deps,
                match_session_route=_match_session_route,
            )
            self.assertEqual(len(responses), 1)
            status, payload = responses[0]
            self.assertEqual(status, 200)
            self.assertIn("new name.md", payload["files"])
            entry = next(entry for entry in payload["entries"] if entry["path"] == "new name.md")
            self.assertEqual(entry["additions"], 0)
            self.assertEqual(entry["deletions"], 0)

    def test_git_changed_files_late_git_failure_returns_409(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            manager = _FakeManager(td)

            def fake_run_git(_cwd: Path, args: list[str], **_kwargs: object) -> str:
                if args == ["rev-parse", "--is-inside-work-tree"]:
                    return "true\n"
                raise RuntimeError("git changed during refresh")

            deps, responses = _git_deps(run_git=fake_run_git)
            handle_git_get_route(
                _FakeHandler(),
                path="/api/sessions/s/git/changed_files",
                query="",
                manager=manager,
                deps=deps,
                match_session_route=_match_session_route,
            )
            self.assertEqual(responses, [(409, {"error": "git changed during refresh"})])

    def test_git_diff_resolve_git_path_failure_returns_409(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            manager = _FakeManager(str(repo))

            def fake_run_git(_cwd: Path, args: list[str], **_kwargs: object) -> str:
                if args == ["rev-parse", "--is-inside-work-tree"]:
                    return "true\n"
                if args == ["rev-parse", "--show-toplevel"]:
                    raise RuntimeError("repo vanished")
                raise AssertionError(f"unexpected git args: {args}")

            deps, responses = _git_deps(run_git=fake_run_git)
            handle_git_get_route(
                _FakeHandler(),
                path="/api/sessions/s/git/diff",
                query="path=note.md",
                manager=manager,
                deps=deps,
                match_session_route=_match_session_route,
            )
            self.assertEqual(responses, [(409, {"error": "repo vanished"})])

    def test_git_diff_oversized_output_returns_400(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / "note.md").write_text("hello\n", encoding="utf-8")
            manager = _FakeManager(str(repo))

            def fake_run_git(_cwd: Path, args: list[str], **_kwargs: object) -> str:
                if args == ["rev-parse", "--is-inside-work-tree"]:
                    return "true\n"
                if args == ["rev-parse", "--show-toplevel"]:
                    return f"{repo}\n"
                if args and args[0] == "diff":
                    raise ValueError("git output too large")
                raise AssertionError(f"unexpected git args: {args}")

            deps, responses = _git_deps(run_git=fake_run_git)
            handle_git_get_route(
                _FakeHandler(),
                path="/api/sessions/s/git/diff",
                query="path=note.md",
                manager=manager,
                deps=deps,
                match_session_route=_match_session_route,
            )
            self.assertEqual(responses, [(400, {"error": "git output too large"})])

    def test_git_file_versions_resolve_git_path_failure_returns_409(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            manager = _FakeManager(str(repo))

            def fake_run_git(_cwd: Path, args: list[str], **_kwargs: object) -> str:
                if args == ["rev-parse", "--is-inside-work-tree"]:
                    return "true\n"
                if args == ["rev-parse", "--show-toplevel"]:
                    raise RuntimeError("repo vanished")
                raise AssertionError(f"unexpected git args: {args}")

            deps, responses = _git_deps(run_git=fake_run_git)
            handle_git_get_route(
                _FakeHandler(),
                path="/api/sessions/s/git/file_versions",
                query="path=note.md",
                manager=manager,
                deps=deps,
                match_session_route=_match_session_route,
            )
            self.assertEqual(responses, [(409, {"error": "repo vanished"})])

    def test_git_file_versions_current_read_error_returns_400(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / "note.md").write_text("hello\n", encoding="utf-8")

            class _RecordingManager(_FakeManager):
                def files_add(self, _session_id: str, _path: str) -> None:
                    return None

            manager = _RecordingManager(str(repo))

            def fake_run_git(_cwd: Path, args: list[str], **_kwargs: object) -> str:
                if args == ["rev-parse", "--is-inside-work-tree"]:
                    return "true\n"
                if args == ["rev-parse", "--show-toplevel"]:
                    return f"{repo}\n"
                raise AssertionError(f"unexpected git args before current read: {args}")

            deps, responses = _git_deps(
                run_git=fake_run_git,
                read_text_file_strict=lambda *_a, **_kw: (_ for _ in ()).throw(ValueError("file too large")),
            )
            handle_git_get_route(
                _FakeHandler(),
                path="/api/sessions/s/git/file_versions",
                query="path=note.md",
                manager=manager,
                deps=deps,
                match_session_route=_match_session_route,
            )
            self.assertEqual(responses, [(400, {"error": "file too large"})])

    def test_git_file_versions_base_oversized_output_returns_400(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)

            class _RecordingManager(_FakeManager):
                def files_add(self, _session_id: str, _path: str) -> None:
                    return None

            manager = _RecordingManager(str(repo))

            def fake_run_git(_cwd: Path, args: list[str], **_kwargs: object) -> str:
                if args == ["rev-parse", "--is-inside-work-tree"]:
                    return "true\n"
                if args == ["rev-parse", "--show-toplevel"]:
                    return f"{repo}\n"
                if args == ["ls-tree", "-z", "HEAD", "--", "note.md"]:
                    return "100644 blob deadbeef\tnote.md\0"
                if args == ["cat-file", "-p", "deadbeef"]:
                    raise ValueError("git output too large")
                raise AssertionError(f"unexpected git args: {args}")

            deps, responses = _git_deps(run_git=fake_run_git)
            handle_git_get_route(
                _FakeHandler(),
                path="/api/sessions/s/git/file_versions",
                query="path=note.md",
                manager=manager,
                deps=deps,
                match_session_route=_match_session_route,
            )
            self.assertEqual(responses, [(400, {"error": "git output too large"})])

    def test_git_file_versions_missing_base_keeps_200_when_repo_still_healthy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)

            class _RecordingManager(_FakeManager):
                def files_add(self, _session_id: str, _path: str) -> None:
                    return None

            manager = _RecordingManager(str(repo))

            def fake_run_git(_cwd: Path, args: list[str], **_kwargs: object) -> str:
                if args == ["rev-parse", "--is-inside-work-tree"]:
                    return "true\n"
                if args == ["rev-parse", "--show-toplevel"]:
                    return f"{repo}\n"
                if args == ["ls-tree", "-z", "HEAD", "--", "note.md"]:
                    return ""
                raise AssertionError(f"unexpected git args: {args}")

            deps, responses = _git_deps(run_git=fake_run_git)
            handle_git_get_route(
                _FakeHandler(),
                path="/api/sessions/s/git/file_versions",
                query="path=note.md",
                manager=manager,
                deps=deps,
                match_session_route=_match_session_route,
            )
            self.assertEqual(len(responses), 1)
            status, payload = responses[0]
            self.assertEqual(status, 200)
            self.assertFalse(payload["base_exists"])
            self.assertEqual(payload["base_text"], "")

    def test_git_file_versions_base_runtime_failure_returns_409_when_head_tree_has_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)

            class _RecordingManager(_FakeManager):
                def files_add(self, _session_id: str, _path: str) -> None:
                    return None

            manager = _RecordingManager(str(repo))

            def fake_run_git(_cwd: Path, args: list[str], **_kwargs: object) -> str:
                if args == ["rev-parse", "--is-inside-work-tree"]:
                    return "true\n"
                if args == ["rev-parse", "--show-toplevel"]:
                    return f"{repo}\n"
                if args == ["ls-tree", "-z", "HEAD", "--", "note.md"]:
                    return "100644 blob deadbeef\tnote.md\0"
                if args == ["cat-file", "-p", "deadbeef"]:
                    raise RuntimeError("bad object deadbeef")
                raise AssertionError(f"unexpected git args: {args}")

            deps, responses = _git_deps(run_git=fake_run_git)
            handle_git_get_route(
                _FakeHandler(),
                path="/api/sessions/s/git/file_versions",
                query="path=note.md",
                manager=manager,
                deps=deps,
                match_session_route=_match_session_route,
            )
            self.assertEqual(responses, [(409, {"error": "bad object deadbeef"})])

    @unittest.skipIf(shutil.which("git") is None, "git required")
    def test_git_file_versions_corrupt_committed_blob_returns_409(self) -> None:
        for rel in ["note.md", " lead.md", "trail.md ", "tab\t.md", "new\n.md", "back\\slash.md", ":abc", ":(top)foo", ":(literal)bar"]:
            with self.subTest(rel=rel):
                with tempfile.TemporaryDirectory() as td:
                    repo = Path(td)
                    subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    subprocess.run(["git", "config", "user.email", "codoxear@example.invalid"], cwd=repo, check=True)
                    subprocess.run(["git", "config", "user.name", "Codoxear Test"], cwd=repo, check=True)
                    note = repo / rel
                    note.write_text("hello\n", encoding="utf-8")
                    subprocess.run(["git", "add", "--", f":(literal){rel}"], cwd=repo, check=True)
                    subprocess.run(["git", "commit", "-m", "add note"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    blob = subprocess.check_output(["git", "rev-parse", f"HEAD:{rel}"], cwd=repo, text=True).strip()
                    (repo / ".git" / "objects" / blob[:2] / blob[2:]).unlink()

                    class _RecordingManager(_FakeManager):
                        def files_add(self, _session_id: str, _path: str) -> None:
                            return None

                    manager = _RecordingManager(str(repo))
                    deps, responses = _git_deps()
                    handle_git_get_route(
                        _FakeHandler(),
                        path="/api/sessions/s/git/file_versions",
                        query=f"path={urllib.parse.quote(rel)}",
                        manager=manager,
                        deps=deps,
                        match_session_route=_match_session_route,
                    )
                    self.assertEqual(len(responses), 1)
                    status, payload = responses[0]
                    self.assertEqual(status, 409)
                    self.assertTrue(str(payload.get("error", "")).strip())

    @unittest.skipIf(shutil.which("git") is None, "git required")
    def test_git_diff_uses_repo_root_relative_path_from_subdir_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            subdir = repo / "sub"
            subdir.mkdir()
            subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(["git", "config", "user.email", "codoxear@example.invalid"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Codoxear Test"], cwd=repo, check=True)
            (repo / "root.md").write_text("ROOT base\n", encoding="utf-8")
            (subdir / "root.md").write_text("SUB base\n", encoding="utf-8")
            subprocess.run(["git", "add", "root.md", "sub/root.md"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "add notes"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (repo / "root.md").write_text("ROOT current\n", encoding="utf-8")
            (subdir / "root.md").write_text("SUB current\n", encoding="utf-8")

            manager = _FakeManager(str(subdir))
            deps, responses = _git_deps()
            handle_git_get_route(
                _FakeHandler(),
                path="/api/sessions/s/git/diff",
                query="path=root.md",
                manager=manager,
                deps=deps,
                match_session_route=_match_session_route,
            )
            self.assertEqual(len(responses), 1)
            status, payload = responses[0]
            self.assertEqual(status, 200)
            self.assertEqual(payload["path"], "root.md")
            self.assertIn("+++ b/root.md", payload["diff"])
            self.assertNotIn("+++ b/sub/root.md", payload["diff"])

    @unittest.skipIf(shutil.which("git") is None, "git required")
    def test_git_file_versions_unborn_repo_has_no_base(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            note = repo / "note.md"
            note.write_text("current\n", encoding="utf-8")

            class _RecordingManager(_FakeManager):
                def files_add(self, _session_id: str, _path: str) -> None:
                    return None

            manager = _RecordingManager(str(repo))
            deps, responses = _git_deps()
            handle_git_get_route(
                _FakeHandler(),
                path="/api/sessions/s/git/file_versions",
                query="path=note.md",
                manager=manager,
                deps=deps,
                match_session_route=_match_session_route,
            )
            self.assertEqual(len(responses), 1)
            status, payload = responses[0]
            self.assertEqual(status, 200)
            self.assertTrue(payload["current_exists"])
            self.assertFalse(payload["base_exists"])

    @unittest.skipIf(shutil.which("git") is None, "git required")
    def test_git_file_versions_whitespace_only_path_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(["git", "config", "user.email", "codoxear@example.invalid"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Codoxear Test"], cwd=repo, check=True)
            (repo / " ").write_text("space base\n", encoding="utf-8")
            subprocess.run(["git", "add", "--", ":(literal) "], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "add space"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (repo / " ").write_text("space current\n", encoding="utf-8")

            class _RecordingManager(_FakeManager):
                def files_add(self, _session_id: str, _path: str) -> None:
                    return None

            manager = _RecordingManager(str(repo))
            deps, responses = _git_deps()
            handle_git_get_route(
                _FakeHandler(),
                path="/api/sessions/s/git/file_versions",
                query="path=%20",
                manager=manager,
                deps=deps,
                match_session_route=_match_session_route,
            )
            self.assertEqual(len(responses), 1)
            status, payload = responses[0]
            self.assertEqual(status, 200)
            self.assertEqual(payload["path"], " ")
            self.assertEqual(payload["current_text"], "space current\n")
            self.assertTrue(payload["base_exists"])
            self.assertEqual(payload["base_text"], "space base\n")

    @unittest.skipIf(shutil.which("git") is None, "git required")
    def test_git_file_versions_backslash_path_is_literal(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(["git", "config", "user.email", "codoxear@example.invalid"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Codoxear Test"], cwd=repo, check=True)
            rel = "back\\slash.md"
            (repo / rel).write_text("base\n", encoding="utf-8")
            subprocess.run(["git", "add", "--", f":(literal){rel}"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "add backslash"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (repo / rel).write_text("current\n", encoding="utf-8")

            class _RecordingManager(_FakeManager):
                def files_add(self, _session_id: str, _path: str) -> None:
                    return None

            manager = _RecordingManager(str(repo))
            deps, responses = _git_deps()
            handle_git_get_route(
                _FakeHandler(),
                path="/api/sessions/s/git/file_versions",
                query=f"path={urllib.parse.quote(rel)}",
                manager=manager,
                deps=deps,
                match_session_route=_match_session_route,
            )
            self.assertEqual(len(responses), 1)
            status, payload = responses[0]
            self.assertEqual(status, 200)
            self.assertEqual(payload["path"], rel)
            self.assertEqual(payload["current_text"], "current\n")
            self.assertTrue(payload["base_exists"])
            self.assertEqual(payload["base_text"], "base\n")

    @unittest.skipIf(not hasattr(os, "symlink") or shutil.which("git") is None, "symlink and git required")
    def test_git_file_versions_symlink_replaced_dir_keeps_literal_base_without_escaping_current(self) -> None:
        with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as outside_td:
            repo = Path(td)
            outside = Path(outside_td)
            tracked_dir = repo / "d"
            tracked_dir.mkdir()
            subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(["git", "config", "user.email", "codoxear@example.invalid"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Codoxear Test"], cwd=repo, check=True)
            (tracked_dir / "f").write_text("base in repo\n", encoding="utf-8")
            subprocess.run(["git", "add", "d/f"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "add nested"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            shutil.rmtree(tracked_dir)
            (outside / "f").write_text("outside current\n", encoding="utf-8")
            os.symlink(outside, tracked_dir)

            class _RecordingManager(_FakeManager):
                def files_add(self, _session_id: str, _path: str) -> None:
                    return None

            manager = _RecordingManager(str(repo))
            deps, responses = _git_deps()
            handle_git_get_route(
                _FakeHandler(),
                path="/api/sessions/s/git/file_versions",
                query="path=d/f",
                manager=manager,
                deps=deps,
                match_session_route=_match_session_route,
            )
            self.assertEqual(len(responses), 1)
            status, payload = responses[0]
            self.assertEqual(status, 200)
            self.assertEqual(payload["path"], "d/f")
            self.assertFalse(payload["current_exists"])
            self.assertTrue(payload["base_exists"])
            self.assertEqual(payload["base_text"], "base in repo\n")

    @unittest.skipIf(not hasattr(os, "symlink") or shutil.which("git") is None, "symlink and git required")
    def test_git_file_versions_symlinked_parent_leaf_symlink_does_not_escape(self) -> None:
        with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as outside_td:
            repo = Path(td)
            outside = Path(outside_td)
            tracked_dir = repo / "d"
            tracked_dir.mkdir()
            subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(["git", "config", "user.email", "codoxear@example.invalid"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Codoxear Test"], cwd=repo, check=True)
            (tracked_dir / "link").write_text("base\n", encoding="utf-8")
            subprocess.run(["git", "add", "d/link"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "add nested"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            shutil.rmtree(tracked_dir)
            os.symlink("secret-target", outside / "link")
            os.symlink(outside, tracked_dir)

            class _RecordingManager(_FakeManager):
                def files_add(self, _session_id: str, _path: str) -> None:
                    return None

            manager = _RecordingManager(str(repo))
            deps, responses = _git_deps()
            handle_git_get_route(
                _FakeHandler(),
                path="/api/sessions/s/git/file_versions",
                query="path=d/link",
                manager=manager,
                deps=deps,
                match_session_route=_match_session_route,
            )
            self.assertEqual(len(responses), 1)
            status, payload = responses[0]
            self.assertEqual(status, 200)
            self.assertEqual(payload["path"], "d/link")
            self.assertFalse(payload["current_exists"])
            self.assertTrue(payload["base_exists"])
            self.assertEqual(payload["base_text"], "base\n")
            self.assertNotIn("secret-target", str(payload))

    @unittest.skipIf(not hasattr(os, "symlink") or shutil.which("git") is None, "symlink and git required")
    def test_git_file_versions_symlink_path_reads_link_payload(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(["git", "config", "user.email", "codoxear@example.invalid"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Codoxear Test"], cwd=repo, check=True)
            (repo / "target").write_text("TARGET CONTENT\n", encoding="utf-8")
            os.symlink("target", repo / "link")
            subprocess.run(["git", "add", "target", "link"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "add symlink"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (repo / "target").write_text("TARGET CURRENT\n", encoding="utf-8")

            class _RecordingManager(_FakeManager):
                def files_add(self, _session_id: str, _path: str) -> None:
                    return None

            manager = _RecordingManager(str(repo))
            deps, responses = _git_deps()
            handle_git_get_route(
                _FakeHandler(),
                path="/api/sessions/s/git/file_versions",
                query="path=link",
                manager=manager,
                deps=deps,
                match_session_route=_match_session_route,
            )
            self.assertEqual(len(responses), 1)
            status, payload = responses[0]
            self.assertEqual(status, 200)
            self.assertEqual(payload["path"], "link")
            self.assertEqual(payload["current_text"], "target")
            self.assertEqual(payload["current_size"], len("target"))
            self.assertTrue(payload["base_exists"])
            self.assertEqual(payload["base_text"], "target")

    @unittest.skipIf(not hasattr(os, "symlink") or shutil.which("git") is None, "symlink and git required")
    def test_git_file_versions_absolute_symlink_path_reads_link_payload(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(["git", "config", "user.email", "codoxear@example.invalid"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Codoxear Test"], cwd=repo, check=True)
            (repo / "target").write_text("TARGET BASE\n", encoding="utf-8")
            os.symlink("target", repo / "link")
            subprocess.run(["git", "add", "target", "link"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "add symlink"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (repo / "target").write_text("TARGET CURRENT\n", encoding="utf-8")

            class _RecordingManager(_FakeManager):
                def files_add(self, _session_id: str, _path: str) -> None:
                    return None

            manager = _RecordingManager(str(repo))
            encoded = urllib.parse.quote(str(repo / "link"))
            deps, responses = _git_deps()
            handle_git_get_route(
                _FakeHandler(),
                path="/api/sessions/s/git/file_versions",
                query=f"path={encoded}",
                manager=manager,
                deps=deps,
                match_session_route=_match_session_route,
            )
            self.assertEqual(len(responses), 1)
            status, payload = responses[0]
            self.assertEqual(status, 200)
            self.assertEqual(payload["path"], "link")
            self.assertEqual(payload["current_text"], "target")
            self.assertEqual(payload["base_text"], "target")

    @unittest.skipIf(not hasattr(os, "symlink") or shutil.which("git") is None, "symlink and git required")
    def test_git_file_versions_non_utf8_symlink_payload_is_json_safe(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(["git", "config", "user.email", "codoxear@example.invalid"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Codoxear Test"], cwd=repo, check=True)
            os.symlink(b"bad\xfftarget", os.fsencode(repo / "link"))
            subprocess.run(["git", "add", "link"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "add nonutf symlink"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            class _RecordingManager(_FakeManager):
                def files_add(self, _session_id: str, _path: str) -> None:
                    return None

            manager = _RecordingManager(str(repo))
            deps, responses = _git_deps()
            handle_git_get_route(
                _FakeHandler(),
                path="/api/sessions/s/git/file_versions",
                query="path=link",
                manager=manager,
                deps=deps,
                match_session_route=_match_session_route,
            )
            self.assertEqual(len(responses), 1)
            status, payload = responses[0]
            self.assertEqual(status, 200)
            self.assertEqual(payload["current_text"], "bad\ufffdtarget")
            self.assertEqual(payload["current_size"], len(b"bad\xfftarget"))
            self.assertEqual(payload["base_text"], "bad\ufffdtarget")

    @unittest.skipIf(shutil.which("git") is None, "git required")
    def test_git_file_versions_head_tree_current_file_returns_409(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            tracked_dir = repo / "d"
            tracked_dir.mkdir()
            subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(["git", "config", "user.email", "codoxear@example.invalid"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Codoxear Test"], cwd=repo, check=True)
            (tracked_dir / "base.txt").write_text("base\n", encoding="utf-8")
            subprocess.run(["git", "add", "d/base.txt"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "add dir"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            shutil.rmtree(tracked_dir)
            (repo / "d").write_text("current file replacing HEAD dir\n", encoding="utf-8")

            class _RecordingManager(_FakeManager):
                def files_add(self, _session_id: str, _path: str) -> None:
                    return None

            manager = _RecordingManager(str(repo))
            deps, responses = _git_deps()
            handle_git_get_route(
                _FakeHandler(),
                path="/api/sessions/s/git/file_versions",
                query="path=d",
                manager=manager,
                deps=deps,
                match_session_route=_match_session_route,
            )
            self.assertEqual(len(responses), 1)
            status, payload = responses[0]
            self.assertEqual(status, 409)
            self.assertIn("HEAD path is not a file", str(payload.get("error", "")))

    @unittest.skipIf(shutil.which("git") is None, "git required")
    def test_git_file_versions_base_lookup_is_repo_root_anchored_from_subdir_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            subdir = repo / "sub"
            subdir.mkdir()
            subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(["git", "config", "user.email", "codoxear@example.invalid"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Codoxear Test"], cwd=repo, check=True)
            (repo / "root.md").write_text("ROOT base\n", encoding="utf-8")
            (subdir / "root.md").write_text("SUB base\n", encoding="utf-8")
            subprocess.run(["git", "add", "root.md", "sub/root.md"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "add notes"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (repo / "root.md").write_text("ROOT current modified\n", encoding="utf-8")

            class _RecordingManager(_FakeManager):
                def files_add(self, _session_id: str, _path: str) -> None:
                    return None

            manager = _RecordingManager(str(subdir))
            deps, responses = _git_deps()
            handle_git_get_route(
                _FakeHandler(),
                path="/api/sessions/s/git/file_versions",
                query="path=root.md",
                manager=manager,
                deps=deps,
                match_session_route=_match_session_route,
            )
            self.assertEqual(len(responses), 1)
            status, payload = responses[0]
            self.assertEqual(status, 200)
            self.assertEqual(payload["path"], "root.md")
            self.assertEqual(payload["current_text"], "ROOT current modified\n")
            self.assertTrue(payload["base_exists"])
            self.assertEqual(payload["base_text"], "ROOT base\n")

    @unittest.skipIf(shutil.which("git") is None, "git required")
    def test_git_file_versions_glob_like_current_path_missing_from_head_returns_base_absent(self) -> None:
        for rel in ["*.md", "[abc].md", "?ote.md"]:
            with self.subTest(rel=rel):
                with tempfile.TemporaryDirectory() as td:
                    repo = Path(td)
                    subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    subprocess.run(["git", "config", "user.email", "codoxear@example.invalid"], cwd=repo, check=True)
                    subprocess.run(["git", "config", "user.name", "Codoxear Test"], cwd=repo, check=True)
                    (repo / "note.md").write_text("tracked\n", encoding="utf-8")
                    subprocess.run(["git", "add", "note.md"], cwd=repo, check=True)
                    subprocess.run(["git", "commit", "-m", "add note"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    (repo / rel).write_text("current only\n", encoding="utf-8")

                    class _RecordingManager(_FakeManager):
                        def files_add(self, _session_id: str, _path: str) -> None:
                            return None

                    manager = _RecordingManager(str(repo))
                    deps, responses = _git_deps()
                    handle_git_get_route(
                        _FakeHandler(),
                        path="/api/sessions/s/git/file_versions",
                        query=f"path={urllib.parse.quote(rel)}",
                        manager=manager,
                        deps=deps,
                        match_session_route=_match_session_route,
                    )
                    self.assertEqual(len(responses), 1)
                    status, payload = responses[0]
                    self.assertEqual(status, 200)
                    self.assertTrue(payload["current_exists"])
                    self.assertFalse(payload["base_exists"])
                    self.assertEqual(payload["base_text"], "")

    @unittest.skipIf(shutil.which("git") is None, "git required")
    def test_git_file_versions_corrupt_blob_ignores_literal_pathspec_env(self) -> None:
        # Retained patch: patch.dict(os.environ, GIT_*_PATHSPECS) is a
        # process-environment boundary patch. It proves the server's run_git
        # honours literal_pathspecs=True regardless of inherited environment
        # variables that would otherwise alter git's pathspec parsing.
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(["git", "config", "user.email", "codoxear@example.invalid"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Codoxear Test"], cwd=repo, check=True)
            note = repo / "note.md"
            note.write_text("hello\n", encoding="utf-8")
            subprocess.run(["git", "add", "note.md"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "add note"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            blob = subprocess.check_output(["git", "rev-parse", "HEAD:note.md"], cwd=repo, text=True).strip()
            (repo / ".git" / "objects" / blob[:2] / blob[2:]).unlink()

            class _RecordingManager(_FakeManager):
                def files_add(self, _session_id: str, _path: str) -> None:
                    return None

            manager = _RecordingManager(str(repo))
            deps, responses = _git_deps()
            with patch.dict(os.environ, {"GIT_LITERAL_PATHSPECS": "1", "GIT_GLOB_PATHSPECS": "1", "GIT_ICASE_PATHSPECS": "1"}):
                handle_git_get_route(
                    _FakeHandler(),
                    path="/api/sessions/s/git/file_versions",
                    query="path=note.md",
                    manager=manager,
                    deps=deps,
                    match_session_route=_match_session_route,
                )
            self.assertEqual(len(responses), 1)
            status, payload = responses[0]
            self.assertEqual(status, 409)
            self.assertTrue(str(payload.get("error", "")).strip())

    def test_file_write_update_rejects_invalid_path_without_500(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            manager = _FakeManager(td)
            deps, responses = _file_write_deps({"path": "bad\x00name", "text": "new", "version": "old"})
            handled = handle_file_write_post_route(
                _FakeHandler(),
                path="/api/sessions/s/file/write",
                manager=manager,
                deps=deps,
                match_session_route=_match_session_route,
            )
            self.assertTrue(handled)
            self.assertEqual(responses, [(400, {"error": "invalid path"})])

    def test_file_write_validation_runs_before_unknown_session_lookup(self) -> None:
        class FailingManager:
            def refresh_session_meta(self, _session_id: str) -> None:
                raise AssertionError("session metadata should not be refreshed for invalid write bodies")

            def get_session(self, _session_id: str) -> object | None:
                raise AssertionError("session lookup should not run for invalid write bodies")

        deps, responses = _file_write_deps({"text": "new"})
        handled = handle_file_write_post_route(
            _FakeHandler(),
            path="/api/sessions/missing/file/write",
            manager=FailingManager(),
            deps=deps,
            match_session_route=_match_session_route,
        )
        self.assertTrue(handled)
        self.assertEqual(responses, [(400, {"error": "path required"})])

    def test_file_write_create_allows_root_cwd_descendant(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as td:
            target = Path(td) / "created-from-root.txt"
            rel_from_root = str(target.relative_to(Path("/")))

            class _RecordingManager(_FakeManager):
                def __init__(self, cwd: str) -> None:
                    super().__init__(cwd)
                    self.added: list[str] = []

                def files_add(self, _session_id: str, path: str) -> None:
                    self.added.append(path)

            manager = _RecordingManager("/")
            deps, responses = _file_write_deps({"path": rel_from_root, "text": "root cwd create\n", "create": True})
            handled = handle_file_write_post_route(
                _FakeHandler(),
                path="/api/sessions/s/file/write",
                manager=manager,
                deps=deps,
                match_session_route=_match_session_route,
            )
            self.assertTrue(handled)
            self.assertEqual(len(responses), 1)
            status, payload = responses[0]
            self.assertEqual(status, 200)
            self.assertEqual(payload["path"], str(target.resolve()))
            self.assertEqual(target.read_text(encoding="utf-8"), "root cwd create\n")
            self.assertEqual(manager.added, [str(target.resolve())])

    @unittest.skipIf(not hasattr(os, "symlink"), "symlink required")
    def test_file_write_update_rejects_relative_symlink_parent_escape(self) -> None:
        with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as outside_td:
            base = Path(td)
            outside = Path(outside_td)
            target = outside / "note.py"
            target.write_text("old outside\n", encoding="utf-8")
            os.symlink(outside, base / "link")
            version = hashlib.sha256(b"old outside\n").hexdigest()

            class _RecordingManager(_FakeManager):
                def __init__(self, cwd: str) -> None:
                    super().__init__(cwd)
                    self.added: list[str] = []

                def files_add(self, _session_id: str, path: str) -> None:
                    self.added.append(path)

            manager = _RecordingManager(str(base))
            deps, responses = _file_write_deps({"path": "link/note.py", "text": "new outside\n", "version": version})
            handled = handle_file_write_post_route(
                _FakeHandler(),
                path="/api/sessions/s/file/write",
                manager=manager,
                deps=deps,
                match_session_route=_match_session_route,
            )
            self.assertTrue(handled)
            self.assertEqual(len(responses), 1)
            status, payload = responses[0]
            self.assertEqual(status, 400)
            self.assertIn("escapes session cwd", str(payload.get("error", "")))
            self.assertEqual(target.read_text(encoding="utf-8"), "old outside\n")
            self.assertEqual(manager.added, [])

    def test_git_branch_probe_tolerates_file_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cwd_file = Path(td) / "not-a-directory"
            cwd_file.write_text("not a cwd", encoding="utf-8")
            self.assertIsNone(_current_git_branch(cwd_file))

    @unittest.skipIf(shutil.which("git") is None, "git required")
    def test_global_file_routes_git_path_use_repo_root_from_subdir_cwd(self) -> None:
        for route in ("/api/files/read", "/api/files/inspect"):
            with self.subTest(route=route), tempfile.TemporaryDirectory() as td:
                repo = Path(td)
                subdir = repo / "sub"
                target = repo / "d" / "f.md"
                subdir.mkdir()
                target.parent.mkdir()
                target.write_text("root file\n", encoding="utf-8")
                subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                class _RecordingManager(_FakeManager):
                    def files_add(self, _session_id: str, _path: str) -> None:
                        return None

                manager = _RecordingManager(str(subdir))
                deps, responses = _global_file_deps({"path": "d/f.md", "session_id": "s", "git_path": True}, manager=manager)
                handled = handle_global_file_post_route(_FakeHandler(), path=route, manager=manager, deps=deps)
                self.assertTrue(handled)
                self.assertEqual(len(responses), 1)
                status, payload = responses[0]
                self.assertEqual(status, 200)
                self.assertEqual(payload["path"], str(target))
                if route.endswith("/read"):
                    self.assertEqual(payload["text"], "root file\n")
                else:
                    self.assertEqual(payload["kind"], "markdown")

    @unittest.skipIf(shutil.which("git") is None, "git required")
    def test_session_file_read_git_path_uses_repo_root_from_subdir_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            subdir = repo / "sub"
            target = repo / "d" / "f.md"
            subdir.mkdir()
            target.parent.mkdir()
            target.write_text("root file\n", encoding="utf-8")
            subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            class _RecordingManager(_FakeManager):
                def files_add(self, _session_id: str, _path: str) -> None:
                    return None

            manager = _RecordingManager(str(subdir))
            deps, responses, _inline, _attachments = _file_get_deps(manager=manager)
            handled = handle_file_get_route(
                _FakeHandler(),
                path="/api/sessions/s/file/read",
                query="path=d/f.md&git_path=1",
                manager=manager,
                deps=deps,
                match_session_route=_match_session_route,
            )
            self.assertTrue(handled)
            self.assertEqual(len(responses), 1)
            status, payload = responses[0]
            self.assertEqual(status, 200)
            self.assertEqual(payload["path"], str(target))
            self.assertEqual(payload["rel"], "d/f.md")
            self.assertEqual(payload["text"], "root file\n")

    @unittest.skipIf(shutil.which("git") is None, "git required")
    def test_global_file_read_git_path_media_urls_stay_session_git_relative(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            subdir = repo / "sub"
            image = repo / "assets" / "icon.svg"
            subdir.mkdir()
            image.parent.mkdir()
            image.write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>\n', encoding="utf-8")
            subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            class _RecordingManager(_FakeManager):
                def files_add(self, _session_id: str, _path: str) -> None:
                    return None

            manager = _RecordingManager(str(subdir))
            deps, responses = _global_file_deps({"path": "assets/icon.svg", "session_id": "s", "git_path": True}, manager=manager)
            handled = handle_global_file_post_route(_FakeHandler(), path="/api/files/read", manager=manager, deps=deps)
            self.assertTrue(handled)
            self.assertEqual(len(responses), 1)
            status, payload = responses[0]
            self.assertEqual(status, 200)
            self.assertEqual(payload["kind"], "image")
            self.assertEqual(payload["image_url"], "/api/sessions/s/file/blob?path=assets/icon.svg&git_path=1")

    @unittest.skipIf(not hasattr(os, "symlink") or shutil.which("git") is None, "symlink and git required")
    def test_session_file_read_git_path_does_not_follow_symlinked_parent_leaf_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as outside_td:
            repo = Path(td)
            outside = Path(outside_td)
            tracked_dir = repo / "d"
            tracked_dir.mkdir()
            subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (tracked_dir / "link").write_text("base\n", encoding="utf-8")
            shutil.rmtree(tracked_dir)
            os.symlink("secret-target", outside / "link")
            os.symlink(outside, tracked_dir)

            manager = _FakeManager(str(repo))
            deps, responses, _inline, _attachments = _file_get_deps(manager=manager)
            handled = handle_file_get_route(
                _FakeHandler(),
                path="/api/sessions/s/file/read",
                query="path=d/link&git_path=1",
                manager=manager,
                deps=deps,
                match_session_route=_match_session_route,
            )
            self.assertTrue(handled)
            self.assertEqual(len(responses), 1)
            status, payload = responses[0]
            self.assertEqual(status, 404)
            self.assertNotIn("secret-target", str(payload))

    def test_global_file_routes_allow_whitespace_only_filename(self) -> None:
        for route in ("/api/files/read", "/api/files/inspect"):
            with self.subTest(route=route), tempfile.TemporaryDirectory() as td:
                base = Path(td)
                (base / " ").write_text("space file\n", encoding="utf-8")

                manager = _FakeManager(str(base))
                deps, responses = _global_file_deps({"path": " ", "session_id": "s"}, manager=manager)
                handled = handle_global_file_post_route(_FakeHandler(), path=route, manager=manager, deps=deps)
                self.assertTrue(handled)
                self.assertEqual(len(responses), 1)
                status, payload = responses[0]
                self.assertEqual(status, 200)
                self.assertEqual(payload["path"], str(base / " "))
                if route.endswith("/read"):
                    self.assertEqual(payload["text"], "space file\n")
                else:
                    self.assertEqual(payload["kind"], "text")

    def test_global_file_routes_reject_non_string_session_id(self) -> None:
        for route in ("/api/files/read", "/api/files/inspect"):
            with self.subTest(route=route):
                deps, responses = _global_file_deps({"path": "note.md", "session_id": 123})
                handled = handle_global_file_post_route(_FakeHandler(), path=route, manager=_FakeManager("/tmp"), deps=deps)
                self.assertTrue(handled)
                self.assertEqual(responses, [(400, {"error": "session_id must be a string"})])
    def test_text_file_for_client_marks_utf8_as_editable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "note.md"
            raw = b"hello\n"
            path.write_bytes(raw)
            text, size, editable, version = _read_text_file_for_client(path, max_bytes=1024)
            self.assertEqual(text, "hello\n")
            self.assertEqual(size, len(raw))
            self.assertTrue(editable)
            self.assertEqual(version, hashlib.sha256(raw).hexdigest())

    def test_text_file_for_client_marks_invalid_utf8_as_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "note.txt"
            raw = b"broken:\xff\n"
            path.write_bytes(raw)
            text, size, editable, version = _read_text_file_for_client(path, max_bytes=1024)
            self.assertEqual(size, len(raw))
            self.assertFalse(editable)
            self.assertIn("broken:", text)
            self.assertIn("\ufffd", text)
            self.assertEqual(version, hashlib.sha256(raw).hexdigest())

    def test_text_file_for_write_rejects_invalid_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "note.txt"
            path.write_bytes(b"broken:\xff\n")
            with self.assertRaisesRegex(ValueError, "utf-8 text"):
                _read_text_file_for_write(path, max_bytes=1024)

    def test_write_text_file_atomic_updates_contents(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "note.py"
            path.write_text("print('old')\n", encoding="utf-8")
            size, version = _write_text_file_atomic(path, text="print('new')\n")
            raw = b"print('new')\n"
            self.assertEqual(path.read_text(encoding="utf-8"), "print('new')\n")
            self.assertEqual(size, len(raw))
            self.assertEqual(version, hashlib.sha256(raw).hexdigest())

    def test_write_new_text_file_atomic_creates_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "note.py"
            size, version = _write_new_text_file_atomic(path, text="print('new')\n")
            raw = b"print('new')\n"
            self.assertEqual(path.read_text(encoding="utf-8"), "print('new')\n")
            self.assertEqual(size, len(raw))
            self.assertEqual(version, hashlib.sha256(raw).hexdigest())

    def test_write_new_text_file_atomic_rejects_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "note.py"
            path.write_text("print('old')\n", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                _write_new_text_file_atomic(path, text="print('new')\n")

    def test_write_new_text_file_atomic_rejects_missing_parent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "nested" / "note.py"
            with self.assertRaisesRegex(FileNotFoundError, "parent directory not found"):
                _write_new_text_file_atomic(path, text="print('new')\n")

    @unittest.skipIf(not hasattr(os, "symlink"), "symlink required")
    def test_text_file_for_write_rejects_symlink_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as outside_td:
            base = Path(td)
            outside = Path(outside_td)
            target = outside / "note.py"
            target.write_text("outside\n", encoding="utf-8")
            os.symlink(outside, base / "link")
            with self.assertRaisesRegex(ValueError, "symlink parent"):
                _read_text_file_for_write(base / "link" / "note.py", max_bytes=1024)

    @unittest.skipIf(not hasattr(os, "symlink"), "symlink required")
    def test_read_text_file_for_write_does_not_reopen_path_after_parent_check(self) -> None:
        with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as outside_td:
            base = Path(td)
            parent = base / "dir"
            parent.mkdir()
            path = parent / "note.py"
            path.write_text("old\n", encoding="utf-8")
            outside = Path(outside_td)
            outside_parent = outside / "dir"
            outside_parent.mkdir()
            outside_target = outside_parent / "note.py"
            outside_target.write_text("outside\n", encoding="utf-8")
            moved_parent = base / "dir-original"
            original_read_bytes = Path.read_bytes
            swapped = False

            def racing_read_bytes(self):
                nonlocal swapped
                if self == path and not swapped:
                    parent.rename(moved_parent)
                    os.symlink(outside_parent, parent)
                    swapped = True
                return original_read_bytes(self)

            with patch.object(Path, "read_bytes", racing_read_bytes):
                text, size, version = _read_text_file_for_write(path, max_bytes=1024)

            self.assertFalse(swapped)
            self.assertEqual(text, "old\n")
            self.assertEqual(size, len(b"old\n"))
            self.assertEqual(version, hashlib.sha256(b"old\n").hexdigest())
            self.assertEqual(outside_target.read_text(encoding="utf-8"), "outside\n")

    @unittest.skipIf(not hasattr(os, "symlink"), "symlink required")
    def test_write_text_file_atomic_rejects_symlink_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as outside_td:
            base = Path(td)
            outside = Path(outside_td)
            target = outside / "note.py"
            target.write_text("outside\n", encoding="utf-8")
            os.symlink(outside, base / "link")
            with self.assertRaisesRegex(ValueError, "symlink parent"):
                _write_text_file_atomic(base / "link" / "note.py", text="new\n")
            self.assertEqual(target.read_text(encoding="utf-8"), "outside\n")

    @unittest.skipIf(not hasattr(os, "symlink"), "symlink required")
    def test_write_text_file_atomic_parent_swap_does_not_replace_symlink_target(self) -> None:
        with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as outside_td:
            base = Path(td)
            parent = base / "dir"
            parent.mkdir()
            path = parent / "note.py"
            path.write_text("old\n", encoding="utf-8")
            outside = Path(outside_td)
            outside_parent = outside / "dir"
            outside_parent.mkdir()
            outside_target = outside_parent / "note.py"
            outside_target.write_text("outside\n", encoding="utf-8")
            (outside_parent / ".note.py.codoxear-tmp-race").write_text("attacker tmp\n", encoding="utf-8")
            moved_parent = base / "dir-original"
            real_replace = os.replace
            swapped = False

            def racing_replace(src, dst, *args, **kwargs):
                nonlocal swapped
                if not swapped:
                    parent.rename(moved_parent)
                    os.symlink(outside_parent, parent)
                    swapped = True
                return real_replace(src, dst, *args, **kwargs)

            with patch.object(file_text.secrets, "token_hex", return_value="race"), patch.object(file_text.os, "replace", side_effect=racing_replace):
                size, version = _write_text_file_atomic(path, text="new\n")

            self.assertTrue(swapped)
            self.assertEqual(size, len(b"new\n"))
            self.assertEqual(version, hashlib.sha256(b"new\n").hexdigest())
            self.assertEqual((moved_parent / "note.py").read_text(encoding="utf-8"), "new\n")
            self.assertEqual(outside_target.read_text(encoding="utf-8"), "outside\n")

    @unittest.skipIf(not hasattr(os, "symlink"), "symlink required")
    def test_write_new_text_file_atomic_parent_swap_does_not_create_in_symlink_target(self) -> None:
        with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as outside_td:
            base = Path(td)
            parent = base / "dir"
            parent.mkdir()
            path = parent / "note.py"
            outside = Path(outside_td)
            outside_parent = outside / "dir"
            outside_parent.mkdir()
            outside_target = outside_parent / "note.py"
            outside_tmp = outside_parent / ".note.py.codoxear-tmp-race"
            outside_tmp.write_text("attacker tmp\n", encoding="utf-8")
            moved_parent = base / "dir-original"
            real_link = os.link
            swapped = False

            def racing_link(src, dst, *args, **kwargs):
                nonlocal swapped
                if not swapped:
                    parent.rename(moved_parent)
                    os.symlink(outside_parent, parent)
                    swapped = True
                return real_link(src, dst, *args, **kwargs)

            with patch.object(file_text.secrets, "token_hex", return_value="race"), patch.object(file_text.os, "link", side_effect=racing_link):
                size, version = _write_new_text_file_atomic(path, text="new\n")

            self.assertTrue(swapped)
            self.assertEqual(size, len(b"new\n"))
            self.assertEqual(version, hashlib.sha256(b"new\n").hexdigest())
            self.assertEqual((moved_parent / "note.py").read_text(encoding="utf-8"), "new\n")
            self.assertFalse(outside_target.exists())
            self.assertEqual(outside_tmp.read_text(encoding="utf-8"), "attacker tmp\n")

    def test_binary_download_inspection_returns_size_without_buffering(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "blob.bin"
            raw_in = b"\x00\x01\x02\x03"
            path.write_bytes(raw_in)
            original_read_bytes = Path.read_bytes
            try:
                Path.read_bytes = lambda self: (_ for _ in ()).throw(AssertionError("download inspection must not buffer file bytes"))  # type: ignore[assignment]
                size = _inspect_downloadable_file(path)
            finally:
                Path.read_bytes = original_read_bytes  # type: ignore[assignment]
            self.assertEqual(size, len(raw_in))

    def test_download_disposition_uses_utf8_filename(self) -> None:
        path = Path("/tmp/report 1.py")
        self.assertEqual(_download_disposition(path), "attachment; filename*=UTF-8''report%201.py")


if __name__ == "__main__":
    unittest.main()
