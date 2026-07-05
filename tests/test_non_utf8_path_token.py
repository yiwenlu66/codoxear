"""Roundtrip tests for the plain-file reversible path-token channel.

A raw-byte (non-UTF-8) filename surfaced by ``file/list`` and walk-mode
``file/search`` is JSON-safe only as an escaped display string. These tests
prove the display string is JSON-encodable and that the accompanying
``api_path`` token carried by the list/search response round-trips through the
plain (non-git) file routes ``read``/``blob``/``download``/``write`` to open
the actual on-disk file.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
import urllib.parse
from pathlib import Path
from types import SimpleNamespace

from codoxear import git_ops
from codoxear import server
from codoxear.file_search import search_session_relative_files
from codoxear.file_routes import FileGetRouteDeps
from codoxear.file_routes import FileWriteRouteDeps
from codoxear.file_routes import handle_file_get_route
from codoxear.file_routes import handle_file_write_post_route
from codoxear.server_routing import match_session_route as _match_session_route


RAW_NAME_BYTES = b"bad\xffname.txt"


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

    def files_add(self, session_id: str, path: str, api_path: str | None = None) -> None:
        self.recorded.append((session_id, path, api_path))


def _capture_json():
    responses: list[tuple[int, dict[str, object]]] = []

    def json_response(_handler, status: int, payload: dict[str, object]) -> None:
        responses.append((status, payload))

    return responses, json_response


def _file_get_deps(manager: _FakeManager):
    responses, json_response = _capture_json()
    inline: list[tuple[Path, str]] = []
    attachments: list[tuple[Path, int, str]] = []

    def send_inline(_handler, path: Path, content_type: str) -> None:
        inline.append((path, content_type))

    def send_attachment(_handler, path: Path, *, size: int, content_disposition: str) -> None:
        attachments.append((path, size, content_disposition))

    deps = FileGetRouteDeps(
        require_auth=lambda _handler: True,
        json_response=json_response,
        resolve_session_cwd=server._resolve_session_cwd,
        resolve_existing_session_file=server._resolve_existing_session_file,
        resolve_session_path=server._resolve_session_path,
        resolve_git_client_file_view=server._resolve_git_client_file_view,
        resolve_git_existing_regular_file=server._resolve_git_existing_regular_file,
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
    return deps, responses, inline, attachments


def _file_write_deps(manager: _FakeManager):
    responses, json_response = _capture_json()

    def read_json_body(handler, **_kwargs):
        return json.loads(handler._body.read())

    deps = FileWriteRouteDeps(
        require_auth=lambda _handler: True,
        json_response=json_response,
        read_json_body=read_json_body,
        resolve_session_cwd=server._resolve_session_cwd,
        resolve_create_path=server._resolve_under,
        resolve_git_existing_regular_file=server._resolve_git_existing_regular_file,
        file_write_lock=server._file_write_lock,
    )
    return deps, responses


def _make_bad_file(root: Path, contents: bytes = b"raw-bytes\n") -> None:
    """Create ``bad<0xff>name.txt`` (a real raw-byte filename) under root."""
    with open(os.fsencode(root) + b"/" + RAW_NAME_BYTES, "wb") as fh:
        fh.write(contents)


def _list_entry_for_bad_file(root: Path) -> tuple[str, str]:
    """Drive ``file/list`` and return (display_path, api_path_token) for the raw file."""
    manager = _FakeManager(str(root))
    deps, responses, _inline, _attachments = _file_get_deps(manager)
    handle_file_get_route(
        _FakeHandler(),
        path="/api/sessions/s/file/list",
        query="",
        manager=manager,
        deps=deps,
        match_session_route=_match_session_route,
    )
    (status, payload) = responses[0]
    assert status == 200, payload
    entries = [e for e in payload["entries"] if e.get("non_utf8_path")]
    assert len(entries) == 1, payload["entries"]
    entry = entries[0]
    return str(entry["path"]), str(entry["api_path"])


class TestNonUtf8PathTokenRoundtrip(unittest.TestCase):
    def test_list_and_search_expose_reversible_token_for_raw_byte_name(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_bad_file(root)
            manager = _FakeManager(str(root))

            # file/list: legacy ``files`` stays JSON-safe display; additive
            # ``entries`` carries api_path + non_utf8_path for the raw-byte name.
            deps, responses, _inline, _attachments = _file_get_deps(manager)
            handle_file_get_route(
                _FakeHandler(),
                path="/api/sessions/s/file/list",
                query="",
                manager=manager,
                deps=deps,
                match_session_route=_match_session_route,
            )
            (status, payload) = responses[0]
            self.assertEqual(status, 200)
            entry = next(e for e in payload["entries"] if e.get("non_utf8_path"))
            display_rel = str(entry["path"])
            token = str(entry["api_path"])
            # Display is the backslashreplace form, JSON/UTF-8 safe.
            self.assertEqual(display_rel, r"bad\xffname.txt")
            self.assertIn(display_rel, payload["files"])
            # The whole response body is JSON/UTF-8 encodable.
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.assertIn(rb"bad\\xffname.txt", body)
            # Token decodes back to the raw byte sequence.
            self.assertEqual(
                git_ops.git_path_from_token(token).encode("utf-8", errors="surrogateescape"),
                RAW_NAME_BYTES,
            )

            # file/search (walk mode, non-git cwd): matches carry the same token.
            deps2, responses2, _inline2, _attachments2 = _file_get_deps(manager)
            handle_file_get_route(
                _FakeHandler(),
                path="/api/sessions/s/file/search",
                query=urllib.parse.urlencode({"q": "name", "limit": "80"}),
                manager=manager,
                deps=deps2,
                match_session_route=_match_session_route,
            )
            (status2, payload2) = responses2[0]
            self.assertEqual(status2, 200)
            self.assertEqual(payload2["mode"], "walk")
            match = next(m for m in payload2["matches"] if m["path"] == display_rel)
            self.assertTrue(match["non_utf8_path"])
            self.assertEqual(match["api_path"], token)

    def test_file_list_serializes_non_utf8_cwd_too(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            parent = Path(td)
            raw_root = os.fsencode(parent) + b"/cwd\xffdir"
            os.mkdir(raw_root)
            root = Path(os.fsdecode(raw_root))
            (root / "ok.txt").write_text("ok\n", encoding="utf-8")
            manager = _FakeManager(str(root))

            deps, responses, _inline, _attachments = _file_get_deps(manager)
            handle_file_get_route(
                _FakeHandler(),
                path="/api/sessions/s/file/list",
                query="",
                manager=manager,
                deps=deps,
                match_session_route=_match_session_route,
            )
            (status, payload) = responses[0]
            self.assertEqual(status, 200)
            self.assertIn(r"cwd\xffdir", str(payload["cwd"]))
            json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def test_git_mode_search_surfaces_reversible_token_for_raw_byte_name(self) -> None:
        # git ls-files stores raw-byte filenames; the search must decode them
        # with surrogateescape (not errors=replace) and emit path_response_fields
        # so the raw-byte git path carries a reversible api_path token.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            subprocess.run(["git", "init"], cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(["git", "config", "user.email", "a@b.c"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True)
            (root / "ok.txt").write_text("ok\n", encoding="utf-8")
            _make_bad_file(root, contents=b"raw-bytes\n")
            subprocess.run(["git", "add", "ok.txt"], cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(
                ["git", "add", "--", os.fsdecode(RAW_NAME_BYTES)],
                cwd=root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            result = search_session_relative_files(root, query="name")
            self.assertEqual(result["mode"], "git")
            match = next(m for m in result["matches"] if m.get("non_utf8_path"))
            display_rel = str(match["path"])
            token = str(match["api_path"])
            # Display is JSON/UTF-8 safe (backslashreplace), not a replacement char.
            self.assertEqual(display_rel, r"bad\xffname.txt")
            self.assertNotIn("\ufffd", display_rel)
            json.dumps(result, ensure_ascii=False).encode("utf-8")
            # Token round-trips to the raw byte sequence.
            self.assertEqual(
                git_ops.git_path_from_token(token).encode("utf-8", errors="surrogateescape"),
                RAW_NAME_BYTES,
            )

            # The token opens the real raw-byte file through the plain read route.
            manager = _FakeManager(str(root))
            deps, responses, _inline, _attachments = _file_get_deps(manager)
            handle_file_get_route(
                _FakeHandler(),
                path="/api/sessions/s/file/read",
                query=urllib.parse.urlencode({"path": display_rel, "path_token": token}),
                manager=manager,
                deps=deps,
                match_session_route=_match_session_route,
            )
            (status, payload) = responses[0]
            self.assertEqual(status, 200)
            self.assertEqual(payload["text"], "raw-bytes\n")
            self.assertEqual(payload["api_path"], token)

            # The read also recorded the file in session history WITH its token,
            # so a raw-byte git file is reopenable from recent entries.
            recorded = manager.recorded[-1]
            self.assertEqual(recorded[0], "s")
            self.assertTrue(isinstance(recorded[1], str) and recorded[1].endswith(display_rel))
            self.assertEqual(recorded[2], token)

    def test_plain_file_read_round_trips_token_to_open_real_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_bad_file(root)
            display_rel, token = _list_entry_for_bad_file(root)
            manager = _FakeManager(str(root))

            deps, responses, _inline, _attachments = _file_get_deps(manager)
            handle_file_get_route(
                _FakeHandler(),
                path="/api/sessions/s/file/read",
                query=urllib.parse.urlencode({"path": display_rel, "path_token": token}),
                manager=manager,
                deps=deps,
                match_session_route=_match_session_route,
            )
            (status, payload) = responses[0]
            self.assertEqual(status, 200)
            # The token channel decoded back to the real bytes and read the file.
            self.assertEqual(payload["text"], "raw-bytes\n")
            self.assertEqual(payload["rel"], display_rel)
            self.assertTrue(payload["path"].endswith(display_rel))
            self.assertTrue(payload["non_utf8_path"])
            self.assertEqual(payload["api_path"], token)

    def test_plain_file_download_round_trips_token_to_real_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_bad_file(root)
            display_rel, token = _list_entry_for_bad_file(root)
            manager = _FakeManager(str(root))

            deps, responses, _inline, attachments = _file_get_deps(manager)
            handle_file_get_route(
                _FakeHandler(),
                path="/api/sessions/s/file/download",
                query=urllib.parse.urlencode({"path": display_rel, "path_token": token}),
                manager=manager,
                deps=deps,
                match_session_route=_match_session_route,
            )
            self.assertEqual(responses, [])
            (path_obj, size, _disposition) = attachments[0]
            self.assertEqual(path_obj.name.encode("utf-8", errors="surrogateescape"), RAW_NAME_BYTES)
            self.assertEqual(size, len(b"raw-bytes\n"))

    def test_plain_file_blob_resolver_finds_real_file_via_token(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_bad_file(root)
            display_rel, token = _list_entry_for_bad_file(root)
            manager = _FakeManager(str(root))

            deps, responses, _inline, _attachments = _file_get_deps(manager)
            handle_file_get_route(
                _FakeHandler(),
                path="/api/sessions/s/file/blob",
                query=urllib.parse.urlencode({"path": display_rel, "path_token": token}),
                manager=manager,
                deps=deps,
                match_session_route=_match_session_route,
            )
            # blob of a text file is not inline-previewable -> 400, but the
            # resolver must have found the real raw-byte file (no 404), proving
            # the token decoded to the correct bytes.
            (status, payload) = responses[0]
            self.assertEqual(status, 400)
            self.assertEqual(payload["error"], "file is not previewable inline")

    def test_plain_file_write_round_trips_token_to_update_real_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_bad_file(root)
            display_rel, token = _list_entry_for_bad_file(root)
            manager = _FakeManager(str(root))

            # Read first to obtain the current version hash via the same token.
            read_deps, read_responses, _inline, _attachments = _file_get_deps(manager)
            handle_file_get_route(
                _FakeHandler(),
                path="/api/sessions/s/file/read",
                query=urllib.parse.urlencode({"path": display_rel, "path_token": token}),
                manager=manager,
                deps=read_deps,
                match_session_route=_match_session_route,
            )
            version = read_responses[0][1]["version"]

            deps, responses = _file_write_deps(manager)
            handler = _FakeHandler()
            handler._body = SimpleNamespace(
                read=lambda *_a, **_k: json.dumps(
                    {
                        "path": display_rel,
                        "path_token": token,
                        "text": "updated contents\n",
                        "version": version,
                    }
                ).encode("utf-8")
            )
            handled = handle_file_write_post_route(
                handler,
                path="/api/sessions/s/file/write",
                manager=manager,
                deps=deps,
                match_session_route=_match_session_route,
            )
            self.assertTrue(handled)
            (status, payload) = responses[0]
            self.assertEqual(status, 200)
            self.assertEqual(payload["rel"], display_rel)

            # The real raw-byte file on disk must now hold the new contents.
            with open(os.fsencode(root) + b"/" + RAW_NAME_BYTES, "rb") as fh:
                self.assertEqual(fh.read(), b"updated contents\n")


if __name__ == "__main__":
    unittest.main()
