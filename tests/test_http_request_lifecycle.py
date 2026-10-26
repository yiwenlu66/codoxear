from __future__ import annotations

import http.server
import socket
import threading
from types import SimpleNamespace

import pytest

from codoxear import server
from codoxear.server_handler import make_server_handler
from codoxear.server_http import read_body
from codoxear.auth import sign_cookie


_TEST_SECRET = b"http-request-lifecycle-test-key"
_AUTH_HEADER = f"Cookie: {server.COOKIE_NAME}={sign_cookie({'v': 1}, secret=_TEST_SECRET)}\r\n".encode()


@pytest.fixture
def http_address(monkeypatch, request):
    # Real route dispatch, JSON/body helpers and HTTP parser; no session runtime.
    monkeypatch.setattr(server, "URL_PREFIX", getattr(request, "param", ""))
    monkeypatch.setattr(server, "HMAC_SECRET", _TEST_SECRET)
    monkeypatch.setattr(server, "MANAGER", SimpleNamespace(
        _voice_push=SimpleNamespace(listener_heartbeat=lambda **kwargs: kwargs),
        upload_status=lambda session_id, upload_id: {
            "ok": True, "upload_id": upload_id, "state": "absent",
            "attachments": [], "pending_attachment": None, "attachments_revision": 0,
        },
    ))
    base_handler = make_server_handler(server)

    class Handler(base_handler):
        def do_POST(self):
            if self.path != "/raw":
                return super().do_POST()
            try:
                body = read_body(self, limit=16)
                self.deps.json_response(self, 200, {"bytes": len(body)})
            except Exception as exc:
                self.deps.handle_route_exception(self, exc)

    instance = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    try:
        yield instance.server_address
    finally:
        instance.shutdown()
        instance.server_close()
        thread.join(timeout=3)


def _request(method="GET", path="/api/me", headers=_AUTH_HEADER, body=b""):
    return f"{method} {path} HTTP/1.1\r\nHost: test\r\n".encode() + headers + b"\r\n" + body


def _response(stream):
    status_line = stream.readline()
    assert status_line, "connection closed before response"
    status = int(status_line.split()[1])
    headers = {}
    while (line := stream.readline()) != b"\r\n":
        assert line, "connection closed during headers"
        key, value = line.decode().split(":", 1)
        headers[key.lower()] = value.strip()
    body = stream.read(int(headers.get("content-length", "0")))
    assert len(body) == int(headers.get("content-length", "0"))
    return status, headers, body


def _assert_closed_rejection(address, request, status, *, eof=False):
    with socket.create_connection(address, timeout=3) as connection:
        connection.sendall(request + (b"" if eof else _request()))
        if eof:
            connection.shutdown(socket.SHUT_WR)
        with connection.makefile("rb") as stream:
            actual_status, headers, body = _response(stream)
            assert actual_status == status, body
            assert headers["connection"] == "close"
            assert stream.read(1) == b"", "rejected body was returned to HTTP parser"


def test_unauthorized_listener_closes_before_pipelined_get(http_address):
    body = b'{"client_id":"phone","enabled":false}'
    _assert_closed_rejection(http_address, _request(
        "POST", "/api/audio/listener", f"Content-Length: {len(body)}\r\n".encode(), body,
    ), 401)


@pytest.mark.parametrize("headers", [
    b"Content-Length: -1\r\n",
    b"Content-Length: +1\r\n",
    b"Content-Length: junk\r\n",
    b"Content-Length: \r\n",
    b"Content-Length: 1, 1\r\n",
    b"Content-Length: 1\r\nContent-Length: 1\r\n",
    b"Content-Length: 1\r\nContent-Length: 2\r\n",
    b"Transfer-Encoding: chunked\r\n",
    b"Transfer-Encoding: identity\r\n",
    b"Transfer-Encoding: chunked\r\nContent-Length: 1\r\n",
])
def test_unsupported_or_invalid_framing_closes(http_address, headers):
    _assert_closed_rejection(http_address, _request("POST", "/api/audio/listener", headers, b"x"), 400)


def test_oversized_body_is_rejected_without_waiting_for_payload(http_address):
    _assert_closed_rejection(http_address, _request(
        "POST", "/api/audio/listener", _AUTH_HEADER + b"Content-Length: 2097153\r\n",
    ), 413)


def test_premature_eof_is_explicit_bad_request(http_address):
    _assert_closed_rejection(http_address, _request(
        "POST", "/api/audio/listener", _AUTH_HEADER + b"Content-Length: 50\r\n", b"{}",
    ), 400, eof=True)


@pytest.mark.parametrize("body, expected", [
    (b'{"client_id":"phone","enabled":false}', 200),
    (b"invalid-json", 400),
])
def test_fully_consumed_json_reuses_same_socket(http_address, body, expected):
    with socket.create_connection(http_address, timeout=3) as connection:
        connection.sendall(_request("POST", "/api/audio/listener",
            _AUTH_HEADER + f"Content-Length: {len(body)}\r\n".encode(), body) + _request())
        with connection.makefile("rb") as stream:
            status, headers, _body = _response(stream)
            assert status == expected
            assert headers.get("connection") != "close"
            assert _response(stream)[0] == 200


def test_shared_raw_reader_accounts_for_binary_body(http_address):
    with socket.create_connection(http_address, timeout=3) as connection:
        connection.sendall(_request("POST", "/raw", b"Content-Length: 4\r\n", b"\x00\xff\x01\xfe") + _request())
        with connection.makefile("rb") as stream:
            status, headers, body = _response(stream)
            assert status == 200
            assert body == b'{"bytes": 4}'
            assert headers.get("connection") != "close"
            assert _response(stream)[0] == 200


def test_get_with_unread_body_closes(http_address):
    _assert_closed_rejection(http_address, _request(
        headers=_AUTH_HEADER + b"Content-Length: 2\r\n", body=b"{}",
    ), 200)


@pytest.mark.parametrize("http_address", ["/prefix"], indirect=True)
def test_prefix_redirect_with_unread_body_closes(http_address):
    _assert_closed_rejection(http_address, _request(
        "POST", "/prefix", b"Content-Length: 2\r\n", b"{}",
    ), 308)


def test_bodyless_unauthorized_request_is_safe_to_reuse(http_address):
    with socket.create_connection(http_address, timeout=3) as connection:
        connection.sendall(_request(headers=b"") + _request())
        with connection.makefile("rb") as stream:
            assert _response(stream)[0] == 401
            assert _response(stream)[0] == 200


def test_route_that_ignores_body_closes_even_on_success(http_address):
    _assert_closed_rejection(http_address, _request(
        "POST", "/api/logout", _AUTH_HEADER + b"Content-Length: 2\r\n", b"{}",
    ), 200)


def test_raw_size_rejection_and_eof_close(http_address):
    _assert_closed_rejection(http_address, _request(
        "POST", "/raw", b"Content-Length: 17\r\n",
    ), 413)
    _assert_closed_rejection(http_address, _request(
        "POST", "/raw", b"Content-Length: 4\r\n", b"\x00\xff",
    ), 400, eof=True)


def test_expect_continue_oversized_request_still_closes(http_address):
    with socket.create_connection(http_address, timeout=3) as connection:
        connection.sendall(_request("POST", "/api/audio/listener",
            _AUTH_HEADER + b"Expect: 100-continue\r\nContent-Length: 2097153\r\n"))
        with connection.makefile("rb") as stream:
            assert _response(stream)[0] == 100
            status, headers, _body = _response(stream)
            assert status == 413
            assert headers["connection"] == "close"
            assert stream.read(1) == b""


def test_upload_status_is_no_store_only_for_its_request(http_address):
    with socket.create_connection(http_address, timeout=3) as connection:
        connection.sendall(_request(path="/api/sessions/test/attachments/upload-status?upload_id=upload-test") + _request())
        with connection.makefile("rb") as stream:
            status, headers, body = _response(stream)
            assert status == 200, body
            assert headers["cache-control"] == "no-store"
            status, headers, _body = _response(stream)
            assert status == 200
            assert headers.get("cache-control") != "no-store"


def test_invalid_upload_status_is_also_no_store(http_address):
    with socket.create_connection(http_address, timeout=3) as connection:
        connection.sendall(_request(path="/api/sessions/test/attachments/upload-status?upload_id=bad!"))
        with connection.makefile("rb") as stream:
            status, headers, _body = _response(stream)
            assert status == 400
            assert headers["cache-control"] == "no-store"
