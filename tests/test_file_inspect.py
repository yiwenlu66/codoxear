import hashlib
import os
import shutil
import subprocess
import tempfile
import unittest
import urllib.parse
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from codoxear import server
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
from codoxear.server import _read_text_file_for_client
from codoxear.server import _read_text_file_for_write
from codoxear.server import _read_text_or_image
from codoxear.server import _single_byte_range
from codoxear.server import _write_new_text_file_atomic
from codoxear.server import _write_text_file_atomic


REPO_CWD = Path(__file__).resolve().parents[1]


def _safe_cwd() -> Path:
    try:
        return Path.cwd()
    except FileNotFoundError:
        return REPO_CWD


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

        class FakeManager:
            def refresh_session_meta(self, _session_id: str) -> None:
                return None

            def get_session(self, _session_id: str) -> object:
                return SimpleNamespace(cwd=bad_cwd)

        with patch.object(server, "MANAGER", FakeManager()):
            with self.assertRaisesRegex(ValueError, "home directory"):
                _resolve_client_file_path(session_id="session-with-bad-cwd", raw_path="note.md")
        with self.assertRaisesRegex(ValueError, "invalid session cwd"):
            _resolve_session_cwd("/tmp/bad\x00cwd")

    def test_bad_tracked_file_expanduser_path_is_bad_request_not_runtime_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bad_tracked = f"~definitely-no-such-codoxear-user-{uuid.uuid4().hex}/note.md"

            class FakeManager:
                def refresh_session_meta(self, _session_id: str) -> None:
                    return None

                def get_session(self, _session_id: str) -> object:
                    return SimpleNamespace(cwd=td)

                def files_get(self, _session_id: str) -> list[str]:
                    return [bad_tracked]

            with patch.object(server, "MANAGER", FakeManager()):
                with self.assertRaisesRegex(ValueError, "home directory"):
                    _resolve_client_file_path(session_id="session-with-bad-tracked", raw_path="note.md")

    def test_session_file_routes_return_400_for_bad_session_cwd(self) -> None:
        bad_cwds = [f"~definitely-no-such-codoxear-user-{uuid.uuid4().hex}/repo", "/tmp/bad\x00cwd"]

        routes = [
            "/api/sessions/s/file/read?path=note.md",
            "/api/sessions/s/file/search?q=note",
            "/api/sessions/s/file/list",
            "/api/sessions/s/file/blob?path=note.md",
            "/api/sessions/s/file/video_preview?path=clip.mp4",
            "/api/sessions/s/file/download?path=note.md",
            "/api/sessions/s/git/changed_files",
            "/api/sessions/s/git/diff?path=note.md",
            "/api/sessions/s/git/file_versions?path=note.md",
        ]
        for bad_cwd in bad_cwds:
            class FakeManager:
                def refresh_session_meta(self, _session_id: str) -> None:
                    return None

                def get_session(self, _session_id: str) -> object:
                    return SimpleNamespace(cwd=bad_cwd)

            for route in routes:
                with self.subTest(route=route, bad_cwd=bad_cwd):
                    parsed = urllib.parse.urlparse(route)
                    handler = server.Handler.__new__(server.Handler)
                    handler._parse_prefixed_request_path = lambda parsed=parsed: (parsed, parsed.path)  # type: ignore[attr-defined]
                    handler._handle_static_get = lambda _path: False  # type: ignore[attr-defined]
                    handler._handle_voice_get = lambda _path, _query: False  # type: ignore[attr-defined]
                    responses = []
                    with patch.object(server, "MANAGER", FakeManager()), patch.object(server, "_require_auth", return_value=True), patch.object(
                        server, "_json_response", side_effect=lambda _handler, status, obj: responses.append((status, obj))
                    ):
                        server.Handler.do_GET(handler)
                    self.assertEqual(len(responses), 1)
                    status, payload = responses[0]
                    self.assertEqual(status, 400)
                    self.assertTrue(any(fragment in str(payload.get("error", "")) for fragment in ("home directory", "invalid session cwd")))

    def test_existing_file_permission_errors_are_403_not_404(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            preview = root / "preview.png"
            preview.write_bytes(b"\x89PNG\r\n\x1a\nbody")

            class FakeManager:
                def refresh_session_meta(self, _session_id: str) -> None:
                    return None

                def get_session(self, _session_id: str) -> object:
                    return SimpleNamespace(cwd=str(root))

            routes = [
                f"/api/sessions/s/file/blob?path={urllib.parse.quote(preview.name)}",
                f"/api/sessions/s/file/video_preview?path={urllib.parse.quote(preview.name)}",
                f"/api/files/blob?path={urllib.parse.quote(str(preview))}",
                f"/api/files/video_preview?path={urllib.parse.quote(str(preview))}",
            ]
            for route in routes:
                with self.subTest(route=route):
                    parsed = urllib.parse.urlparse(route)
                    handler = server.Handler.__new__(server.Handler)
                    handler._parse_prefixed_request_path = lambda parsed=parsed: (parsed, parsed.path)  # type: ignore[attr-defined]
                    handler._handle_static_get = lambda _path: False  # type: ignore[attr-defined]
                    handler._handle_voice_get = lambda _path, _query: False  # type: ignore[attr-defined]
                    responses = []
                    with patch.object(server, "MANAGER", FakeManager()), patch.object(server, "_require_auth", return_value=True), patch.object(
                        Path, "stat", side_effect=PermissionError("denied")
                    ), patch.object(server, "_json_response", side_effect=lambda _handler, status, obj: responses.append((status, obj))):
                        server.Handler.do_GET(handler)
                    self.assertEqual(responses, [(403, {"error": "denied"})])

    def test_preview_prefix_read_errors_are_route_local(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            preview = root / "preview.png"
            preview.write_bytes(b"\x89PNG\r\n\x1a\nbody")

            class FakeManager:
                def refresh_session_meta(self, _session_id: str) -> None:
                    return None

                def get_session(self, _session_id: str) -> object:
                    return SimpleNamespace(cwd=str(root))

            routes = [
                f"/api/sessions/s/file/blob?path={urllib.parse.quote(preview.name)}",
                f"/api/sessions/s/file/video_preview?path={urllib.parse.quote(preview.name)}",
                f"/api/files/blob?path={urllib.parse.quote(str(preview))}",
                f"/api/files/video_preview?path={urllib.parse.quote(str(preview))}",
            ]
            cases = [(PermissionError("denied"), 403), (FileNotFoundError("gone"), 404)]
            for exc, expected_status in cases:
                for route in routes:
                    with self.subTest(route=route, exc=type(exc).__name__):
                        parsed = urllib.parse.urlparse(route)
                        handler = server.Handler.__new__(server.Handler)
                        handler._parse_prefixed_request_path = lambda parsed=parsed: (parsed, parsed.path)  # type: ignore[attr-defined]
                        handler._handle_static_get = lambda _path: False  # type: ignore[attr-defined]
                        handler._handle_voice_get = lambda _path, _query: False  # type: ignore[attr-defined]
                        responses = []
                        with patch.object(server, "MANAGER", FakeManager()), patch.object(server, "_require_auth", return_value=True), patch.object(
                            Path, "open", side_effect=exc
                        ), patch.object(server, "_json_response", side_effect=lambda _handler, status, obj: responses.append((status, obj))):
                            server.Handler.do_GET(handler)
                        self.assertEqual(len(responses), 1)
                        status, payload = responses[0]
                        self.assertEqual(status, expected_status)
                        self.assertIn(str(exc), str(payload.get("error", "")))

    def test_video_preview_generation_file_errors_are_route_local(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = root / "clip.mp4"
            video.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"\x00" * 32)

            class FakeManager:
                def refresh_session_meta(self, _session_id: str) -> None:
                    return None

                def get_session(self, _session_id: str) -> object:
                    return SimpleNamespace(cwd=str(root))

            routes = [
                f"/api/sessions/s/file/video_preview?path={urllib.parse.quote(video.name)}",
                f"/api/files/video_preview?path={urllib.parse.quote(str(video))}",
            ]
            cases = [(PermissionError("denied"), 403), (FileNotFoundError("gone"), 404)]
            for exc, expected_status in cases:
                for route in routes:
                    with self.subTest(route=route, exc=type(exc).__name__):
                        parsed = urllib.parse.urlparse(route)
                        handler = server.Handler.__new__(server.Handler)
                        handler._parse_prefixed_request_path = lambda parsed=parsed: (parsed, parsed.path)  # type: ignore[attr-defined]
                        handler._handle_static_get = lambda _path: False  # type: ignore[attr-defined]
                        handler._handle_voice_get = lambda _path, _query: False  # type: ignore[attr-defined]
                        responses = []
                        with patch.object(server, "MANAGER", FakeManager()), patch.object(server, "_require_auth", return_value=True), patch.object(
                            server, "_ensure_video_preview", side_effect=exc
                        ), patch.object(server, "_json_response", side_effect=lambda _handler, status, obj: responses.append((status, obj))):
                            server.Handler.do_GET(handler)
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

            class FakeManager:
                def refresh_session_meta(self, _session_id: str) -> None:
                    return None

                def get_session(self, _session_id: str) -> object:
                    return SimpleNamespace(cwd=str(repo))

            parsed = urllib.parse.urlparse("/api/sessions/s/git/changed_files")
            handler = server.Handler.__new__(server.Handler)
            handler._parse_prefixed_request_path = lambda parsed=parsed: (parsed, parsed.path)  # type: ignore[attr-defined]
            handler._handle_static_get = lambda _path: False  # type: ignore[attr-defined]
            handler._handle_voice_get = lambda _path, _query: False  # type: ignore[attr-defined]
            responses = []
            with patch.object(server, "MANAGER", FakeManager()), patch.object(server, "_require_auth", return_value=True), patch.object(
                server, "_json_response", side_effect=lambda _handler, status, obj: responses.append((status, obj))
            ):
                server.Handler.do_GET(handler)
            self.assertEqual(len(responses), 1)
            status, payload = responses[0]
            self.assertEqual(status, 200)
            self.assertCountEqual(payload["files"], paths)
            self.assertCountEqual([entry["path"] for entry in payload["entries"]], paths)

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

            class FakeManager:
                def refresh_session_meta(self, _session_id: str) -> None:
                    return None

                def get_session(self, _session_id: str) -> object:
                    return SimpleNamespace(cwd=str(repo))

            parsed = urllib.parse.urlparse("/api/sessions/s/git/changed_files")
            handler = server.Handler.__new__(server.Handler)
            handler._parse_prefixed_request_path = lambda parsed=parsed: (parsed, parsed.path)  # type: ignore[attr-defined]
            handler._handle_static_get = lambda _path: False  # type: ignore[attr-defined]
            handler._handle_voice_get = lambda _path, _query: False  # type: ignore[attr-defined]
            responses = []
            with patch.object(server, "MANAGER", FakeManager()), patch.object(server, "_require_auth", return_value=True), patch.object(
                server, "_json_response", side_effect=lambda _handler, status, obj: responses.append((status, obj))
            ):
                server.Handler.do_GET(handler)
            self.assertEqual(len(responses), 1)
            status, payload = responses[0]
            self.assertEqual(status, 200)
            self.assertIn("new name.md", payload["files"])
            entry = next(entry for entry in payload["entries"] if entry["path"] == "new name.md")
            self.assertEqual(entry["additions"], 0)
            self.assertEqual(entry["deletions"], 0)

    def test_git_changed_files_late_git_failure_returns_409(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            class FakeManager:
                def refresh_session_meta(self, _session_id: str) -> None:
                    return None

                def get_session(self, _session_id: str) -> object:
                    return SimpleNamespace(cwd=td)

            def fake_run_git(_cwd: Path, args: list[str], **_kwargs: object) -> str:
                if args == ["rev-parse", "--is-inside-work-tree"]:
                    return "true\n"
                raise RuntimeError("git changed during refresh")

            parsed = urllib.parse.urlparse("/api/sessions/s/git/changed_files")
            handler = server.Handler.__new__(server.Handler)
            handler._parse_prefixed_request_path = lambda parsed=parsed: (parsed, parsed.path)  # type: ignore[attr-defined]
            handler._handle_static_get = lambda _path: False  # type: ignore[attr-defined]
            handler._handle_voice_get = lambda _path, _query: False  # type: ignore[attr-defined]
            responses = []
            with patch.object(server, "MANAGER", FakeManager()), patch.object(server, "_require_auth", return_value=True), patch.object(
                server, "_run_git", side_effect=fake_run_git
            ), patch.object(server, "_json_response", side_effect=lambda _handler, status, obj: responses.append((status, obj))):
                server.Handler.do_GET(handler)
            self.assertEqual(responses, [(409, {"error": "git changed during refresh"})])

    def test_git_diff_resolve_git_path_failure_returns_409(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)

            class FakeManager:
                def refresh_session_meta(self, _session_id: str) -> None:
                    return None

                def get_session(self, _session_id: str) -> object:
                    return SimpleNamespace(cwd=str(repo))

            def fake_run_git(_cwd: Path, args: list[str], **_kwargs: object) -> str:
                if args == ["rev-parse", "--is-inside-work-tree"]:
                    return "true\n"
                if args == ["rev-parse", "--show-toplevel"]:
                    raise RuntimeError("repo vanished")
                raise AssertionError(f"unexpected git args: {args}")

            parsed = urllib.parse.urlparse("/api/sessions/s/git/diff?path=note.md")
            handler = server.Handler.__new__(server.Handler)
            handler._parse_prefixed_request_path = lambda parsed=parsed: (parsed, parsed.path)  # type: ignore[attr-defined]
            handler._handle_static_get = lambda _path: False  # type: ignore[attr-defined]
            handler._handle_voice_get = lambda _path, _query: False  # type: ignore[attr-defined]
            responses = []
            with patch.object(server, "MANAGER", FakeManager()), patch.object(server, "_require_auth", return_value=True), patch.object(
                server, "_run_git", side_effect=fake_run_git
            ), patch.object(server, "_json_response", side_effect=lambda _handler, status, obj: responses.append((status, obj))):
                server.Handler.do_GET(handler)
            self.assertEqual(responses, [(409, {"error": "repo vanished"})])

    def test_git_diff_oversized_output_returns_400(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / "note.md").write_text("hello\n", encoding="utf-8")

            class FakeManager:
                def refresh_session_meta(self, _session_id: str) -> None:
                    return None

                def get_session(self, _session_id: str) -> object:
                    return SimpleNamespace(cwd=str(repo))

            def fake_run_git(_cwd: Path, args: list[str], **_kwargs: object) -> str:
                if args == ["rev-parse", "--is-inside-work-tree"]:
                    return "true\n"
                if args == ["rev-parse", "--show-toplevel"]:
                    return f"{repo}\n"
                if args and args[0] == "diff":
                    raise ValueError("git output too large")
                raise AssertionError(f"unexpected git args: {args}")

            parsed = urllib.parse.urlparse("/api/sessions/s/git/diff?path=note.md")
            handler = server.Handler.__new__(server.Handler)
            handler._parse_prefixed_request_path = lambda parsed=parsed: (parsed, parsed.path)  # type: ignore[attr-defined]
            handler._handle_static_get = lambda _path: False  # type: ignore[attr-defined]
            handler._handle_voice_get = lambda _path, _query: False  # type: ignore[attr-defined]
            responses = []
            with patch.object(server, "MANAGER", FakeManager()), patch.object(server, "_require_auth", return_value=True), patch.object(
                server, "_run_git", side_effect=fake_run_git
            ), patch.object(server, "_json_response", side_effect=lambda _handler, status, obj: responses.append((status, obj))):
                server.Handler.do_GET(handler)
            self.assertEqual(responses, [(400, {"error": "git output too large"})])

    def test_git_file_versions_resolve_git_path_failure_returns_409(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)

            class FakeManager:
                def refresh_session_meta(self, _session_id: str) -> None:
                    return None

                def get_session(self, _session_id: str) -> object:
                    return SimpleNamespace(cwd=str(repo))

            def fake_run_git(_cwd: Path, args: list[str], **_kwargs: object) -> str:
                if args == ["rev-parse", "--is-inside-work-tree"]:
                    return "true\n"
                if args == ["rev-parse", "--show-toplevel"]:
                    raise RuntimeError("repo vanished")
                raise AssertionError(f"unexpected git args: {args}")

            parsed = urllib.parse.urlparse("/api/sessions/s/git/file_versions?path=note.md")
            handler = server.Handler.__new__(server.Handler)
            handler._parse_prefixed_request_path = lambda parsed=parsed: (parsed, parsed.path)  # type: ignore[attr-defined]
            handler._handle_static_get = lambda _path: False  # type: ignore[attr-defined]
            handler._handle_voice_get = lambda _path, _query: False  # type: ignore[attr-defined]
            responses = []
            with patch.object(server, "MANAGER", FakeManager()), patch.object(server, "_require_auth", return_value=True), patch.object(
                server, "_run_git", side_effect=fake_run_git
            ), patch.object(server, "_json_response", side_effect=lambda _handler, status, obj: responses.append((status, obj))):
                server.Handler.do_GET(handler)
            self.assertEqual(responses, [(409, {"error": "repo vanished"})])

    def test_git_file_versions_current_read_error_returns_400(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / "note.md").write_text("hello\n", encoding="utf-8")

            class FakeManager:
                def refresh_session_meta(self, _session_id: str) -> None:
                    return None

                def get_session(self, _session_id: str) -> object:
                    return SimpleNamespace(cwd=str(repo))

                def files_add(self, _session_id: str, _path: str) -> None:
                    return None

            def fake_run_git(_cwd: Path, args: list[str], **_kwargs: object) -> str:
                if args == ["rev-parse", "--is-inside-work-tree"]:
                    return "true\n"
                if args == ["rev-parse", "--show-toplevel"]:
                    return f"{repo}\n"
                raise AssertionError(f"unexpected git args before current read: {args}")

            parsed = urllib.parse.urlparse("/api/sessions/s/git/file_versions?path=note.md")
            handler = server.Handler.__new__(server.Handler)
            handler._parse_prefixed_request_path = lambda parsed=parsed: (parsed, parsed.path)  # type: ignore[attr-defined]
            handler._handle_static_get = lambda _path: False  # type: ignore[attr-defined]
            handler._handle_voice_get = lambda _path, _query: False  # type: ignore[attr-defined]
            responses = []
            with patch.object(server, "MANAGER", FakeManager()), patch.object(server, "_require_auth", return_value=True), patch.object(
                server, "_run_git", side_effect=fake_run_git
            ), patch.object(server, "_read_text_file_strict", side_effect=ValueError("file too large")), patch.object(
                server, "_json_response", side_effect=lambda _handler, status, obj: responses.append((status, obj))
            ):
                server.Handler.do_GET(handler)
            self.assertEqual(responses, [(400, {"error": "file too large"})])

    def test_git_file_versions_base_oversized_output_returns_400(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)

            class FakeManager:
                def refresh_session_meta(self, _session_id: str) -> None:
                    return None

                def get_session(self, _session_id: str) -> object:
                    return SimpleNamespace(cwd=str(repo))

                def files_add(self, _session_id: str, _path: str) -> None:
                    return None

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

            parsed = urllib.parse.urlparse("/api/sessions/s/git/file_versions?path=note.md")
            handler = server.Handler.__new__(server.Handler)
            handler._parse_prefixed_request_path = lambda parsed=parsed: (parsed, parsed.path)  # type: ignore[attr-defined]
            handler._handle_static_get = lambda _path: False  # type: ignore[attr-defined]
            handler._handle_voice_get = lambda _path, _query: False  # type: ignore[attr-defined]
            responses = []
            with patch.object(server, "MANAGER", FakeManager()), patch.object(server, "_require_auth", return_value=True), patch.object(
                server, "_run_git", side_effect=fake_run_git
            ), patch.object(server, "_json_response", side_effect=lambda _handler, status, obj: responses.append((status, obj))):
                server.Handler.do_GET(handler)
            self.assertEqual(responses, [(400, {"error": "git output too large"})])

    def test_git_file_versions_missing_base_keeps_200_when_repo_still_healthy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)

            class FakeManager:
                def refresh_session_meta(self, _session_id: str) -> None:
                    return None

                def get_session(self, _session_id: str) -> object:
                    return SimpleNamespace(cwd=str(repo))

                def files_add(self, _session_id: str, _path: str) -> None:
                    return None

            def fake_run_git(_cwd: Path, args: list[str], **_kwargs: object) -> str:
                if args == ["rev-parse", "--is-inside-work-tree"]:
                    return "true\n"
                if args == ["rev-parse", "--show-toplevel"]:
                    return f"{repo}\n"
                if args == ["ls-tree", "-z", "HEAD", "--", "note.md"]:
                    return ""
                raise AssertionError(f"unexpected git args: {args}")

            parsed = urllib.parse.urlparse("/api/sessions/s/git/file_versions?path=note.md")
            handler = server.Handler.__new__(server.Handler)
            handler._parse_prefixed_request_path = lambda parsed=parsed: (parsed, parsed.path)  # type: ignore[attr-defined]
            handler._handle_static_get = lambda _path: False  # type: ignore[attr-defined]
            handler._handle_voice_get = lambda _path, _query: False  # type: ignore[attr-defined]
            responses = []
            with patch.object(server, "MANAGER", FakeManager()), patch.object(server, "_require_auth", return_value=True), patch.object(
                server, "_run_git", side_effect=fake_run_git
            ), patch.object(server, "_json_response", side_effect=lambda _handler, status, obj: responses.append((status, obj))):
                server.Handler.do_GET(handler)
            self.assertEqual(len(responses), 1)
            status, payload = responses[0]
            self.assertEqual(status, 200)
            self.assertFalse(payload["base_exists"])
            self.assertEqual(payload["base_text"], "")

    def test_git_file_versions_base_runtime_failure_returns_409_when_head_tree_has_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)

            class FakeManager:
                def refresh_session_meta(self, _session_id: str) -> None:
                    return None

                def get_session(self, _session_id: str) -> object:
                    return SimpleNamespace(cwd=str(repo))

                def files_add(self, _session_id: str, _path: str) -> None:
                    return None

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

            parsed = urllib.parse.urlparse("/api/sessions/s/git/file_versions?path=note.md")
            handler = server.Handler.__new__(server.Handler)
            handler._parse_prefixed_request_path = lambda parsed=parsed: (parsed, parsed.path)  # type: ignore[attr-defined]
            handler._handle_static_get = lambda _path: False  # type: ignore[attr-defined]
            handler._handle_voice_get = lambda _path, _query: False  # type: ignore[attr-defined]
            responses = []
            with patch.object(server, "MANAGER", FakeManager()), patch.object(server, "_require_auth", return_value=True), patch.object(
                server, "_run_git", side_effect=fake_run_git
            ), patch.object(server, "_json_response", side_effect=lambda _handler, status, obj: responses.append((status, obj))):
                server.Handler.do_GET(handler)
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

                    class FakeManager:
                        def refresh_session_meta(self, _session_id: str) -> None:
                            return None

                        def get_session(self, _session_id: str) -> object:
                            return SimpleNamespace(cwd=str(repo))

                        def files_add(self, _session_id: str, _path: str) -> None:
                            return None

                    parsed = urllib.parse.urlparse(f"/api/sessions/s/git/file_versions?path={urllib.parse.quote(rel)}")
                    handler = server.Handler.__new__(server.Handler)
                    handler._parse_prefixed_request_path = lambda parsed=parsed: (parsed, parsed.path)  # type: ignore[attr-defined]
                    handler._handle_static_get = lambda _path: False  # type: ignore[attr-defined]
                    handler._handle_voice_get = lambda _path, _query: False  # type: ignore[attr-defined]
                    responses = []
                    with patch.object(server, "MANAGER", FakeManager()), patch.object(server, "_require_auth", return_value=True), patch.object(
                        server, "_json_response", side_effect=lambda _handler, status, obj: responses.append((status, obj))
                    ):
                        server.Handler.do_GET(handler)
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

            class FakeManager:
                def refresh_session_meta(self, _session_id: str) -> None:
                    return None

                def get_session(self, _session_id: str) -> object:
                    return SimpleNamespace(cwd=str(subdir))

            parsed = urllib.parse.urlparse("/api/sessions/s/git/diff?path=root.md")
            handler = server.Handler.__new__(server.Handler)
            handler._parse_prefixed_request_path = lambda parsed=parsed: (parsed, parsed.path)  # type: ignore[attr-defined]
            handler._handle_static_get = lambda _path: False  # type: ignore[attr-defined]
            handler._handle_voice_get = lambda _path, _query: False  # type: ignore[attr-defined]
            responses = []
            with patch.object(server, "MANAGER", FakeManager()), patch.object(server, "_require_auth", return_value=True), patch.object(
                server, "_json_response", side_effect=lambda _handler, status, obj: responses.append((status, obj))
            ):
                server.Handler.do_GET(handler)
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

            class FakeManager:
                def refresh_session_meta(self, _session_id: str) -> None:
                    return None

                def get_session(self, _session_id: str) -> object:
                    return SimpleNamespace(cwd=str(repo))

                def files_add(self, _session_id: str, _path: str) -> None:
                    return None

            parsed = urllib.parse.urlparse("/api/sessions/s/git/file_versions?path=note.md")
            handler = server.Handler.__new__(server.Handler)
            handler._parse_prefixed_request_path = lambda parsed=parsed: (parsed, parsed.path)  # type: ignore[attr-defined]
            handler._handle_static_get = lambda _path: False  # type: ignore[attr-defined]
            handler._handle_voice_get = lambda _path, _query: False  # type: ignore[attr-defined]
            responses = []
            with patch.object(server, "MANAGER", FakeManager()), patch.object(server, "_require_auth", return_value=True), patch.object(
                server, "_json_response", side_effect=lambda _handler, status, obj: responses.append((status, obj))
            ):
                server.Handler.do_GET(handler)
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

            class FakeManager:
                def refresh_session_meta(self, _session_id: str) -> None:
                    return None

                def get_session(self, _session_id: str) -> object:
                    return SimpleNamespace(cwd=str(repo))

                def files_add(self, _session_id: str, _path: str) -> None:
                    return None

            parsed = urllib.parse.urlparse("/api/sessions/s/git/file_versions?path=%20")
            handler = server.Handler.__new__(server.Handler)
            handler._parse_prefixed_request_path = lambda parsed=parsed: (parsed, parsed.path)  # type: ignore[attr-defined]
            handler._handle_static_get = lambda _path: False  # type: ignore[attr-defined]
            handler._handle_voice_get = lambda _path, _query: False  # type: ignore[attr-defined]
            responses = []
            with patch.object(server, "MANAGER", FakeManager()), patch.object(server, "_require_auth", return_value=True), patch.object(
                server, "_json_response", side_effect=lambda _handler, status, obj: responses.append((status, obj))
            ):
                server.Handler.do_GET(handler)
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

            class FakeManager:
                def refresh_session_meta(self, _session_id: str) -> None:
                    return None

                def get_session(self, _session_id: str) -> object:
                    return SimpleNamespace(cwd=str(repo))

                def files_add(self, _session_id: str, _path: str) -> None:
                    return None

            parsed = urllib.parse.urlparse(f"/api/sessions/s/git/file_versions?path={urllib.parse.quote(rel)}")
            handler = server.Handler.__new__(server.Handler)
            handler._parse_prefixed_request_path = lambda parsed=parsed: (parsed, parsed.path)  # type: ignore[attr-defined]
            handler._handle_static_get = lambda _path: False  # type: ignore[attr-defined]
            handler._handle_voice_get = lambda _path, _query: False  # type: ignore[attr-defined]
            responses = []
            with patch.object(server, "MANAGER", FakeManager()), patch.object(server, "_require_auth", return_value=True), patch.object(
                server, "_json_response", side_effect=lambda _handler, status, obj: responses.append((status, obj))
            ):
                server.Handler.do_GET(handler)
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

            class FakeManager:
                def refresh_session_meta(self, _session_id: str) -> None:
                    return None

                def get_session(self, _session_id: str) -> object:
                    return SimpleNamespace(cwd=str(repo))

                def files_add(self, _session_id: str, _path: str) -> None:
                    return None

            parsed = urllib.parse.urlparse("/api/sessions/s/git/file_versions?path=d/f")
            handler = server.Handler.__new__(server.Handler)
            handler._parse_prefixed_request_path = lambda parsed=parsed: (parsed, parsed.path)  # type: ignore[attr-defined]
            handler._handle_static_get = lambda _path: False  # type: ignore[attr-defined]
            handler._handle_voice_get = lambda _path, _query: False  # type: ignore[attr-defined]
            responses = []
            with patch.object(server, "MANAGER", FakeManager()), patch.object(server, "_require_auth", return_value=True), patch.object(
                server, "_json_response", side_effect=lambda _handler, status, obj: responses.append((status, obj))
            ):
                server.Handler.do_GET(handler)
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

            class FakeManager:
                def refresh_session_meta(self, _session_id: str) -> None:
                    return None

                def get_session(self, _session_id: str) -> object:
                    return SimpleNamespace(cwd=str(repo))

                def files_add(self, _session_id: str, _path: str) -> None:
                    return None

            parsed = urllib.parse.urlparse("/api/sessions/s/git/file_versions?path=d/link")
            handler = server.Handler.__new__(server.Handler)
            handler._parse_prefixed_request_path = lambda parsed=parsed: (parsed, parsed.path)  # type: ignore[attr-defined]
            handler._handle_static_get = lambda _path: False  # type: ignore[attr-defined]
            handler._handle_voice_get = lambda _path, _query: False  # type: ignore[attr-defined]
            responses = []
            with patch.object(server, "MANAGER", FakeManager()), patch.object(server, "_require_auth", return_value=True), patch.object(
                server, "_json_response", side_effect=lambda _handler, status, obj: responses.append((status, obj))
            ):
                server.Handler.do_GET(handler)
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

            class FakeManager:
                def refresh_session_meta(self, _session_id: str) -> None:
                    return None

                def get_session(self, _session_id: str) -> object:
                    return SimpleNamespace(cwd=str(repo))

                def files_add(self, _session_id: str, _path: str) -> None:
                    return None

            parsed = urllib.parse.urlparse("/api/sessions/s/git/file_versions?path=link")
            handler = server.Handler.__new__(server.Handler)
            handler._parse_prefixed_request_path = lambda parsed=parsed: (parsed, parsed.path)  # type: ignore[attr-defined]
            handler._handle_static_get = lambda _path: False  # type: ignore[attr-defined]
            handler._handle_voice_get = lambda _path, _query: False  # type: ignore[attr-defined]
            responses = []
            with patch.object(server, "MANAGER", FakeManager()), patch.object(server, "_require_auth", return_value=True), patch.object(
                server, "_json_response", side_effect=lambda _handler, status, obj: responses.append((status, obj))
            ):
                server.Handler.do_GET(handler)
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

            class FakeManager:
                def refresh_session_meta(self, _session_id: str) -> None:
                    return None

                def get_session(self, _session_id: str) -> object:
                    return SimpleNamespace(cwd=str(repo))

                def files_add(self, _session_id: str, _path: str) -> None:
                    return None

            encoded = urllib.parse.quote(str(repo / "link"))
            parsed = urllib.parse.urlparse(f"/api/sessions/s/git/file_versions?path={encoded}")
            handler = server.Handler.__new__(server.Handler)
            handler._parse_prefixed_request_path = lambda parsed=parsed: (parsed, parsed.path)  # type: ignore[attr-defined]
            handler._handle_static_get = lambda _path: False  # type: ignore[attr-defined]
            handler._handle_voice_get = lambda _path, _query: False  # type: ignore[attr-defined]
            responses = []
            with patch.object(server, "MANAGER", FakeManager()), patch.object(server, "_require_auth", return_value=True), patch.object(
                server, "_json_response", side_effect=lambda _handler, status, obj: responses.append((status, obj))
            ):
                server.Handler.do_GET(handler)
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

            class FakeManager:
                def refresh_session_meta(self, _session_id: str) -> None:
                    return None

                def get_session(self, _session_id: str) -> object:
                    return SimpleNamespace(cwd=str(repo))

                def files_add(self, _session_id: str, _path: str) -> None:
                    return None

            parsed = urllib.parse.urlparse("/api/sessions/s/git/file_versions?path=link")
            handler = server.Handler.__new__(server.Handler)
            handler._parse_prefixed_request_path = lambda parsed=parsed: (parsed, parsed.path)  # type: ignore[attr-defined]
            handler._handle_static_get = lambda _path: False  # type: ignore[attr-defined]
            handler._handle_voice_get = lambda _path, _query: False  # type: ignore[attr-defined]
            responses = []
            with patch.object(server, "MANAGER", FakeManager()), patch.object(server, "_require_auth", return_value=True), patch.object(
                server, "_json_response", side_effect=lambda _handler, status, obj: responses.append((status, obj))
            ):
                server.Handler.do_GET(handler)
            self.assertEqual(len(responses), 1)
            status, payload = responses[0]
            self.assertEqual(status, 200)
            self.assertEqual(payload["current_text"], "bad�target")
            self.assertEqual(payload["current_size"], len(b"bad\xfftarget"))
            self.assertEqual(payload["base_text"], "bad�target")

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

            class FakeManager:
                def refresh_session_meta(self, _session_id: str) -> None:
                    return None

                def get_session(self, _session_id: str) -> object:
                    return SimpleNamespace(cwd=str(repo))

                def files_add(self, _session_id: str, _path: str) -> None:
                    return None

            parsed = urllib.parse.urlparse("/api/sessions/s/git/file_versions?path=d")
            handler = server.Handler.__new__(server.Handler)
            handler._parse_prefixed_request_path = lambda parsed=parsed: (parsed, parsed.path)  # type: ignore[attr-defined]
            handler._handle_static_get = lambda _path: False  # type: ignore[attr-defined]
            handler._handle_voice_get = lambda _path, _query: False  # type: ignore[attr-defined]
            responses = []
            with patch.object(server, "MANAGER", FakeManager()), patch.object(server, "_require_auth", return_value=True), patch.object(
                server, "_json_response", side_effect=lambda _handler, status, obj: responses.append((status, obj))
            ):
                server.Handler.do_GET(handler)
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

            class FakeManager:
                def refresh_session_meta(self, _session_id: str) -> None:
                    return None

                def get_session(self, _session_id: str) -> object:
                    return SimpleNamespace(cwd=str(subdir))

                def files_add(self, _session_id: str, _path: str) -> None:
                    return None

            parsed = urllib.parse.urlparse("/api/sessions/s/git/file_versions?path=root.md")
            handler = server.Handler.__new__(server.Handler)
            handler._parse_prefixed_request_path = lambda parsed=parsed: (parsed, parsed.path)  # type: ignore[attr-defined]
            handler._handle_static_get = lambda _path: False  # type: ignore[attr-defined]
            handler._handle_voice_get = lambda _path, _query: False  # type: ignore[attr-defined]
            responses = []
            with patch.object(server, "MANAGER", FakeManager()), patch.object(server, "_require_auth", return_value=True), patch.object(
                server, "_json_response", side_effect=lambda _handler, status, obj: responses.append((status, obj))
            ):
                server.Handler.do_GET(handler)
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

                    class FakeManager:
                        def refresh_session_meta(self, _session_id: str) -> None:
                            return None

                        def get_session(self, _session_id: str) -> object:
                            return SimpleNamespace(cwd=str(repo))

                        def files_add(self, _session_id: str, _path: str) -> None:
                            return None

                    parsed = urllib.parse.urlparse(f"/api/sessions/s/git/file_versions?path={urllib.parse.quote(rel)}")
                    handler = server.Handler.__new__(server.Handler)
                    handler._parse_prefixed_request_path = lambda parsed=parsed: (parsed, parsed.path)  # type: ignore[attr-defined]
                    handler._handle_static_get = lambda _path: False  # type: ignore[attr-defined]
                    handler._handle_voice_get = lambda _path, _query: False  # type: ignore[attr-defined]
                    responses = []
                    with patch.object(server, "MANAGER", FakeManager()), patch.object(server, "_require_auth", return_value=True), patch.object(
                        server, "_json_response", side_effect=lambda _handler, status, obj: responses.append((status, obj))
                    ):
                        server.Handler.do_GET(handler)
                    self.assertEqual(len(responses), 1)
                    status, payload = responses[0]
                    self.assertEqual(status, 200)
                    self.assertTrue(payload["current_exists"])
                    self.assertFalse(payload["base_exists"])
                    self.assertEqual(payload["base_text"], "")

    @unittest.skipIf(shutil.which("git") is None, "git required")
    def test_git_file_versions_corrupt_blob_ignores_literal_pathspec_env(self) -> None:
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

            class FakeManager:
                def refresh_session_meta(self, _session_id: str) -> None:
                    return None

                def get_session(self, _session_id: str) -> object:
                    return SimpleNamespace(cwd=str(repo))

                def files_add(self, _session_id: str, _path: str) -> None:
                    return None

            parsed = urllib.parse.urlparse("/api/sessions/s/git/file_versions?path=note.md")
            handler = server.Handler.__new__(server.Handler)
            handler._parse_prefixed_request_path = lambda parsed=parsed: (parsed, parsed.path)  # type: ignore[attr-defined]
            handler._handle_static_get = lambda _path: False  # type: ignore[attr-defined]
            handler._handle_voice_get = lambda _path, _query: False  # type: ignore[attr-defined]
            responses = []
            with patch.dict(os.environ, {"GIT_LITERAL_PATHSPECS": "1", "GIT_GLOB_PATHSPECS": "1", "GIT_ICASE_PATHSPECS": "1"}), patch.object(server, "MANAGER", FakeManager()), patch.object(
                server, "_require_auth", return_value=True
            ), patch.object(server, "_json_response", side_effect=lambda _handler, status, obj: responses.append((status, obj))):
                server.Handler.do_GET(handler)
            self.assertEqual(len(responses), 1)
            status, payload = responses[0]
            self.assertEqual(status, 409)
            self.assertTrue(str(payload.get("error", "")).strip())

    def test_file_write_update_rejects_invalid_path_without_500(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            class FakeManager:
                def refresh_session_meta(self, _session_id: str) -> None:
                    return None

                def get_session(self, _session_id: str) -> object:
                    return SimpleNamespace(cwd=td)

            parsed = urllib.parse.urlparse("/api/sessions/s/file/write")
            handler = server.Handler.__new__(server.Handler)
            handler._parse_prefixed_request_path = lambda parsed=parsed: (parsed, parsed.path)  # type: ignore[attr-defined]
            handler._handle_voice_post = lambda _path: False  # type: ignore[attr-defined]
            handler._read_json_body = lambda **_kwargs: {"path": "bad\x00name", "text": "new", "version": "old"}  # type: ignore[attr-defined]
            responses = []
            with patch.object(server, "MANAGER", FakeManager()), patch.object(server, "_require_auth", return_value=True), patch.object(
                server, "_json_response", side_effect=lambda _handler, status, obj: responses.append((status, obj))
            ):
                server.Handler.do_POST(handler)
            self.assertEqual(responses, [(400, {"error": "invalid path"})])

    def test_file_write_create_allows_root_cwd_descendant(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as td:
            target = Path(td) / "created-from-root.txt"
            rel_from_root = str(target.relative_to(Path("/")))
            added: list[str] = []

            class FakeManager:
                def refresh_session_meta(self, _session_id: str) -> None:
                    return None

                def get_session(self, _session_id: str) -> object:
                    return SimpleNamespace(cwd="/")

                def files_add(self, _session_id: str, path: str) -> None:
                    added.append(path)

            parsed = urllib.parse.urlparse("/api/sessions/s/file/write")
            handler = server.Handler.__new__(server.Handler)
            handler._parse_prefixed_request_path = lambda parsed=parsed: (parsed, parsed.path)  # type: ignore[attr-defined]
            handler._handle_voice_post = lambda _path: False  # type: ignore[attr-defined]
            handler._read_json_body = lambda **_kwargs: {"path": rel_from_root, "text": "root cwd create\n", "create": True}  # type: ignore[attr-defined]
            responses = []
            with patch.object(server, "MANAGER", FakeManager()), patch.object(server, "_require_auth", return_value=True), patch.object(
                server, "_json_response", side_effect=lambda _handler, status, obj: responses.append((status, obj))
            ):
                server.Handler.do_POST(handler)
            self.assertEqual(len(responses), 1)
            status, payload = responses[0]
            self.assertEqual(status, 200)
            self.assertEqual(payload["path"], str(target.resolve()))
            self.assertEqual(target.read_text(encoding="utf-8"), "root cwd create\n")
            self.assertEqual(added, [str(target.resolve())])

    @unittest.skipIf(not hasattr(os, "symlink"), "symlink required")
    def test_file_write_update_rejects_relative_symlink_parent_escape(self) -> None:
        with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as outside_td:
            base = Path(td)
            outside = Path(outside_td)
            target = outside / "note.py"
            target.write_text("old outside\n", encoding="utf-8")
            os.symlink(outside, base / "link")
            version = hashlib.sha256(b"old outside\n").hexdigest()
            added: list[str] = []

            class FakeManager:
                def refresh_session_meta(self, _session_id: str) -> None:
                    return None

                def get_session(self, _session_id: str) -> object:
                    return SimpleNamespace(cwd=str(base))

                def files_add(self, _session_id: str, path: str) -> None:
                    added.append(path)

            parsed = urllib.parse.urlparse("/api/sessions/s/file/write")
            handler = server.Handler.__new__(server.Handler)
            handler._parse_prefixed_request_path = lambda parsed=parsed: (parsed, parsed.path)  # type: ignore[attr-defined]
            handler._handle_voice_post = lambda _path: False  # type: ignore[attr-defined]
            handler._read_json_body = lambda **_kwargs: {"path": "link/note.py", "text": "new outside\n", "version": version}  # type: ignore[attr-defined]
            responses = []
            with patch.object(server, "MANAGER", FakeManager()), patch.object(server, "_require_auth", return_value=True), patch.object(
                server, "_json_response", side_effect=lambda _handler, status, obj: responses.append((status, obj))
            ):
                server.Handler.do_POST(handler)
            self.assertEqual(len(responses), 1)
            status, payload = responses[0]
            self.assertEqual(status, 400)
            self.assertIn("escapes session cwd", str(payload.get("error", "")))
            self.assertEqual(target.read_text(encoding="utf-8"), "old outside\n")
            self.assertEqual(added, [])

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

                class FakeManager:
                    def refresh_session_meta(self, _session_id: str) -> None:
                        return None

                    def get_session(self, _session_id: str) -> object:
                        return SimpleNamespace(cwd=str(subdir))

                    def files_add(self, _session_id: str, _path: str) -> None:
                        return None

                parsed = urllib.parse.urlparse(route)
                handler = server.Handler.__new__(server.Handler)
                handler._parse_prefixed_request_path = lambda parsed=parsed: (parsed, parsed.path)  # type: ignore[attr-defined]
                handler._handle_voice_post = lambda _path: False  # type: ignore[attr-defined]
                handler._read_json_body = lambda **_kwargs: {"path": "d/f.md", "session_id": "s", "git_path": True}  # type: ignore[attr-defined]
                responses = []
                with patch.object(server, "MANAGER", FakeManager()), patch.object(server, "_require_auth", return_value=True), patch.object(
                    server, "_json_response", side_effect=lambda _handler, status, obj: responses.append((status, obj))
                ):
                    server.Handler.do_POST(handler)
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

            class FakeManager:
                def refresh_session_meta(self, _session_id: str) -> None:
                    return None

                def get_session(self, _session_id: str) -> object:
                    return SimpleNamespace(cwd=str(subdir))

                def files_add(self, _session_id: str, _path: str) -> None:
                    return None

            parsed = urllib.parse.urlparse("/api/sessions/s/file/read?path=d/f.md&git_path=1")
            handler = server.Handler.__new__(server.Handler)
            handler._parse_prefixed_request_path = lambda parsed=parsed: (parsed, parsed.path)  # type: ignore[attr-defined]
            handler._handle_static_get = lambda _path: False  # type: ignore[attr-defined]
            handler._handle_voice_get = lambda _path, _query: False  # type: ignore[attr-defined]
            responses = []
            with patch.object(server, "MANAGER", FakeManager()), patch.object(server, "_require_auth", return_value=True), patch.object(
                server, "_json_response", side_effect=lambda _handler, status, obj: responses.append((status, obj))
            ):
                server.Handler.do_GET(handler)
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

            class FakeManager:
                def refresh_session_meta(self, _session_id: str) -> None:
                    return None

                def get_session(self, _session_id: str) -> object:
                    return SimpleNamespace(cwd=str(subdir))

                def files_add(self, _session_id: str, _path: str) -> None:
                    return None

            parsed = urllib.parse.urlparse("/api/files/read")
            handler = server.Handler.__new__(server.Handler)
            handler._parse_prefixed_request_path = lambda parsed=parsed: (parsed, parsed.path)  # type: ignore[attr-defined]
            handler._handle_voice_post = lambda _path: False  # type: ignore[attr-defined]
            handler._read_json_body = lambda **_kwargs: {"path": "assets/icon.svg", "session_id": "s", "git_path": True}  # type: ignore[attr-defined]
            responses = []
            with patch.object(server, "MANAGER", FakeManager()), patch.object(server, "_require_auth", return_value=True), patch.object(
                server, "_json_response", side_effect=lambda _handler, status, obj: responses.append((status, obj))
            ):
                server.Handler.do_POST(handler)
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

            class FakeManager:
                def refresh_session_meta(self, _session_id: str) -> None:
                    return None

                def get_session(self, _session_id: str) -> object:
                    return SimpleNamespace(cwd=str(repo))

            parsed = urllib.parse.urlparse("/api/sessions/s/file/read?path=d/link&git_path=1")
            handler = server.Handler.__new__(server.Handler)
            handler._parse_prefixed_request_path = lambda parsed=parsed: (parsed, parsed.path)  # type: ignore[attr-defined]
            handler._handle_static_get = lambda _path: False  # type: ignore[attr-defined]
            handler._handle_voice_get = lambda _path, _query: False  # type: ignore[attr-defined]
            responses = []
            with patch.object(server, "MANAGER", FakeManager()), patch.object(server, "_require_auth", return_value=True), patch.object(
                server, "_json_response", side_effect=lambda _handler, status, obj: responses.append((status, obj))
            ):
                server.Handler.do_GET(handler)
            self.assertEqual(len(responses), 1)
            status, payload = responses[0]
            self.assertEqual(status, 404)
            self.assertNotIn("secret-target", str(payload))

    def test_global_file_routes_allow_whitespace_only_filename(self) -> None:
        for route in ("/api/files/read", "/api/files/inspect"):
            with self.subTest(route=route), tempfile.TemporaryDirectory() as td:
                base = Path(td)
                (base / " ").write_text("space file\n", encoding="utf-8")

                class FakeManager:
                    def refresh_session_meta(self, _session_id: str) -> None:
                        return None

                    def get_session(self, _session_id: str) -> object:
                        return SimpleNamespace(cwd=str(base))

                    def files_add(self, _session_id: str, _path: str) -> None:
                        return None

                parsed = urllib.parse.urlparse(route)
                handler = server.Handler.__new__(server.Handler)
                handler._parse_prefixed_request_path = lambda parsed=parsed: (parsed, parsed.path)  # type: ignore[attr-defined]
                handler._handle_voice_post = lambda _path: False  # type: ignore[attr-defined]
                handler._read_json_body = lambda **_kwargs: {"path": " ", "session_id": "s"}  # type: ignore[attr-defined]
                responses = []
                with patch.object(server, "MANAGER", FakeManager()), patch.object(server, "_require_auth", return_value=True), patch.object(
                    server, "_json_response", side_effect=lambda _handler, status, obj: responses.append((status, obj))
                ):
                    server.Handler.do_POST(handler)
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
                parsed = urllib.parse.urlparse(route)
                handler = server.Handler.__new__(server.Handler)
                handler._parse_prefixed_request_path = lambda parsed=parsed: (parsed, parsed.path)  # type: ignore[attr-defined]
                handler._handle_voice_post = lambda _path: False  # type: ignore[attr-defined]
                handler._read_json_body = lambda **_kwargs: {"path": "note.md", "session_id": 123}  # type: ignore[attr-defined]
                responses = []
                with patch.object(server, "_require_auth", return_value=True), patch.object(
                    server, "_json_response", side_effect=lambda _handler, status, obj: responses.append((status, obj))
                ):
                    server.Handler.do_POST(handler)
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
