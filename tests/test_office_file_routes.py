from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit, urlencode

import pytest

from codoxear import server
from codoxear.file_get_routes import handle_file_get_route
from codoxear.file_global_routes import GlobalFileRequest, global_file_read_payload, handle_global_file_post_route
from codoxear.file_types import OFFICE_CONTENT_TYPES
from codoxear.file_view import read_client_file_view
from codoxear.server_routing import match_session_route


class Handler:
    unauthorized = False

    def _unauthorized(self):
        self.unauthorized = True


def route_context(tmp_path, converter=None):
    responses, inline, attachments, calls = [], [], [], []
    manager = SimpleNamespace(
        refresh_session_meta=lambda sid: None,
        get_session=lambda sid: SimpleNamespace(cwd=str(tmp_path)) if sid == "s" else None,
        files_add=lambda *args, **kwargs: None,
    )

    def convert(path):
        calls.append(path)
        if converter:
            return converter(path)
        return tmp_path / "cached.pdf"

    deps = replace(
        server._route_deps_factory().file_get_route_deps(),
        require_auth=lambda handler: True,
        json_response=lambda handler, status, payload: responses.append((status, payload)),
        resolve_git_existing_regular_file=lambda **kwargs: (tmp_path / kwargs["raw_path"], kwargs["raw_path"]),
        ensure_office_preview=convert,
        send_inline_file_response=lambda handler, path, mime: inline.append((path, mime)),
        send_attachment_file_response=lambda handler, path, **kwargs: attachments.append((path, kwargs)),
    )
    return deps, manager, responses, inline, attachments, calls


def dispatch(handler, deps, manager, url):
    split = urlsplit(url)
    assert handle_file_get_route(handler, path=split.path, query=split.query, manager=manager,
                                 deps=deps, match_session_route=match_session_route)


@pytest.mark.parametrize("absolute", [False, True])
def test_office_preview_routes_authenticate_before_path_access(tmp_path, absolute):
    deps, manager, responses, inline, attachments, calls = route_context(tmp_path)
    deps = replace(deps, require_auth=lambda handler: False,
                   resolve_existing_absolute_file=lambda path: pytest.fail("path read before auth"),
                   resolve_session_cwd=lambda path: pytest.fail("path read before auth"))
    handler = Handler()
    prefix = "/api/files" if absolute else "/api/sessions/s/file"
    dispatch(handler, deps, manager, prefix + "/office_preview?path=report.docx")
    assert handler.unauthorized
    assert not responses and not inline and not calls


@pytest.mark.parametrize("absolute", [False, True])
def test_office_preview_routes_serve_pdf_while_blob_and_download_remain_original(tmp_path, absolute):
    source = tmp_path / "原稿 report.docx"
    source.write_bytes(b"original Office bytes")
    deps, manager, responses, inline, attachments, calls = route_context(tmp_path)
    prefix = "/api/files" if absolute else "/api/sessions/s/file"
    query = urlencode({"path": str(source) if absolute else source.name})
    dispatch(Handler(), deps, manager, prefix + "/office_preview?" + query)
    assert inline == [(tmp_path / "cached.pdf", "application/pdf")]
    assert calls == [source]
    dispatch(Handler(), deps, manager, prefix + "/blob?" + query)
    assert inline[-1] == (source, OFFICE_CONTENT_TYPES[".docx"])
    if not absolute:
        dispatch(Handler(), deps, manager, prefix + "/download?" + query)
        assert attachments[0][0] == source
        assert attachments[0][1]["size"] == len(b"original Office bytes")
    assert not responses
    assert source.read_bytes() == b"original Office bytes"


def test_session_and_global_reads_are_lazy_and_keep_original_metadata(tmp_path):
    source = tmp_path / "report.xlsx"
    source.write_bytes(b"original")
    deps, manager, responses, inline, attachments, calls = route_context(tmp_path)
    dispatch(Handler(), deps, manager, "/api/sessions/s/file/read?path=report.xlsx")
    status, payload = responses[-1]
    assert status == 200
    assert payload["kind"] == "office"
    assert payload["path"] == str(source)
    assert payload["size"] == 8
    assert payload["content_type"] == OFFICE_CONTENT_TYPES[".xlsx"]
    assert payload["source_content_type"] == payload["content_type"]
    assert payload["preview_content_type"] == "application/pdf"
    assert payload["source_name"] == "report.xlsx"
    assert payload["preview_url"].startswith("/api/sessions/s/file/office_preview?")
    assert payload["source_url"].startswith("/api/sessions/s/file/blob?")
    global_deps = replace(server._route_deps_factory().global_file_route_deps(),
                          require_auth=lambda handler: True,
                          json_response=deps.json_response,
                          read_json_body=lambda handler: {"path": str(source)})
    for operation in ("read", "inspect"):
        assert handle_global_file_post_route(Handler(), path="/api/files/" + operation,
                                              manager=manager, deps=global_deps)
        assert responses[-1][0] == 200
        assert responses[-1][1]["kind"] == "office"
    assert not calls and not inline
    read_payload = responses[-2][1]
    assert read_payload["preview_url"].startswith("/api/files/office_preview?")
    assert parse_qs(urlsplit(read_payload["preview_url"]).query)["path"] == [str(source)]
    assert read_payload["source_url"].startswith("/api/files/blob?")


def test_global_git_office_payload_uses_session_authority(tmp_path):
    source = tmp_path / "doc.ppt"
    source.write_bytes(b"original")
    payload = global_file_read_payload(request=GlobalFileRequest(path="doc.ppt", session_id="s", git_path=True),
                                       path_obj=source, rel_for_url="doc.ppt", view=read_client_file_view(source))
    assert payload["preview_url"] == "/api/sessions/s/file/office_preview?path=doc.ppt&git_path=1"
    assert payload["source_url"] == "/api/sessions/s/file/blob?path=doc.ppt&git_path=1"


@pytest.mark.parametrize("absolute", [False, True])
@pytest.mark.parametrize("failure,status", [(FileNotFoundError("gone"), 404), (PermissionError("denied"), 403),
                                           (ValueError("invalid source"), 400), (RuntimeError("LibreOffice missing"), 500),
                                           (OSError("disk full"), 500)])
def test_converter_failures_are_route_local_readable_errors(tmp_path, absolute, failure, status):
    source = tmp_path / "report.doc"
    source.write_bytes(b"original")

    def fail(path):
        raise failure

    deps, manager, responses, inline, attachments, calls = route_context(tmp_path, converter=fail)
    prefix = "/api/files" if absolute else "/api/sessions/s/file"
    query = urlencode({"path": str(source) if absolute else source.name})
    dispatch(Handler(), deps, manager, prefix + "/office_preview?" + query)
    assert responses[-1][0] == status
    assert str(failure) in responses[-1][1]["error"]
    assert not inline


@pytest.mark.parametrize("absolute", [False, True])
def test_office_routes_reject_wrong_kind_and_symlink(tmp_path, absolute):
    source = tmp_path / "report.txt"
    source.write_text("ordinary text")
    deps, manager, responses, inline, attachments, calls = route_context(tmp_path)
    prefix = "/api/files" if absolute else "/api/sessions/s/file"
    for path in [source, tmp_path / "alias.docx"]:
        if path != source:
            path.symlink_to(source)
        query = urlencode({"path": str(path) if absolute else path.name})
        dispatch(Handler(), deps, manager, prefix + "/office_preview?" + query)
        assert responses[-1][0] == 400
    assert not calls and not inline


def test_session_office_route_accepts_git_path_and_rejects_unknown_session(tmp_path):
    source = tmp_path / "report.docx"
    source.write_bytes(b"original")
    deps, manager, responses, inline, attachments, calls = route_context(tmp_path)
    dispatch(Handler(), deps, manager, "/api/sessions/s/file/office_preview?path=report.docx&git_path=1")
    assert calls == [source]
    dispatch(Handler(), deps, manager, "/api/sessions/unknown/file/office_preview?path=report.docx")
    assert responses[-1] == (404, {"error": "unknown session"})
