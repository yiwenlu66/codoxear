from __future__ import annotations

import gzip
import errno
import http.server
import json
import os
import traceback
from typing import Any, Callable


CLIENT_DISCONNECT_ERRNOS = {errno.EPIPE, errno.ECONNRESET, errno.ECONNABORTED}
CLIENT_DISCONNECT_ERRORS = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)
GZIP_MIN_BYTES = 1024


def gzip_response_body(handler: http.server.BaseHTTPRequestHandler, body: bytes) -> tuple[bytes, bool]:
    accept_encoding = str(handler.headers.get("Accept-Encoding") or "")
    accepts_gzip = False
    for raw_coding in accept_encoding.split(","):
        parts = [part.strip() for part in raw_coding.split(";")]
        if not parts or parts[0].lower() != "gzip":
            continue
        quality = next((part.split("=", 1)[1] for part in parts[1:] if part.lower().startswith("q=") and "=" in part), "1")
        try:
            accepts_gzip = float(quality) > 0
        except ValueError:
            accepts_gzip = False
        break
    if len(body) < GZIP_MIN_BYTES or not accepts_gzip:
        return body, False
    return gzip.compress(body), True


class BadRequestError(Exception):
    """Client request body or shape was invalid."""


class RequestPayloadTooLargeError(Exception):
    """Client request body exceeded the configured size limit."""


def is_client_disconnect(exc: BaseException) -> bool:
    if isinstance(exc, CLIENT_DISCONNECT_ERRORS):
        return True
    return isinstance(exc, OSError) and getattr(exc, "errno", None) in CLIENT_DISCONNECT_ERRNOS


def handle_route_exception(
    handler: http.server.BaseHTTPRequestHandler,
    exc: BaseException,
    *,
    json_response: Callable[[http.server.BaseHTTPRequestHandler, int, dict[str, Any]], None],
) -> None:
    if is_client_disconnect(exc):
        return
    if isinstance(exc, BadRequestError):
        json_response(handler, 400, {"error": str(exc)})
        return
    if isinstance(exc, RequestPayloadTooLargeError):
        json_response(handler, 413, {"error": str(exc)})
        return
    traceback.print_exc()
    payload = {"error": str(exc)}
    if os.environ.get("CODEX_WEB_DEBUG_ERRORS") == "1":
        payload["trace"] = traceback.format_exc()
    json_response(handler, 500, payload)


def json_response(
    handler: http.server.BaseHTTPRequestHandler,
    status: int,
    obj: Any,
    *,
    set_auth_cookie: Callable[[http.server.BaseHTTPRequestHandler], None],
) -> None:
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    body, gzip_encoded = gzip_response_body(handler, body)
    handler.send_response(status)
    if getattr(handler, "_codoxear_refresh_auth_cookie", False):
        set_auth_cookie(handler)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    if gzip_encoded:
        handler.send_header("Content-Encoding", "gzip")
        handler.send_header("Vary", "Accept-Encoding")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def if_none_match_contains(header_value: str | None, etag: str) -> bool:
    if header_value is None:
        return False
    values = [part.strip() for part in str(header_value).split(",")]
    return "*" in values or etag in values


def json_response_with_etag(
    handler: http.server.BaseHTTPRequestHandler,
    obj: Any,
    *,
    sha256_hex: Callable[[bytes], str],
    set_auth_cookie: Callable[[http.server.BaseHTTPRequestHandler], None],
) -> None:
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    body, gzip_encoded = gzip_response_body(handler, body)
    etag = '"' + sha256_hex(body) + '"'
    if if_none_match_contains(handler.headers.get("If-None-Match"), etag):
        handler.send_response(304)
        if getattr(handler, "_codoxear_refresh_auth_cookie", False):
            set_auth_cookie(handler)
        handler.send_header("ETag", etag)
        handler.send_header("Cache-Control", "private, no-cache")
        if gzip_encoded:
            handler.send_header("Content-Encoding", "gzip")
            handler.send_header("Vary", "Accept-Encoding")
        handler.send_header("Content-Length", "0")
        handler.end_headers()
        return
    handler.send_response(200)
    if getattr(handler, "_codoxear_refresh_auth_cookie", False):
        set_auth_cookie(handler)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    if gzip_encoded:
        handler.send_header("Content-Encoding", "gzip")
        handler.send_header("Vary", "Accept-Encoding")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("ETag", etag)
    handler.send_header("Cache-Control", "private, no-cache")
    handler.end_headers()
    handler.wfile.write(body)


def read_body(handler: http.server.BaseHTTPRequestHandler, limit: int = 2 * 1024 * 1024) -> bytes:
    content_length = handler.headers.get("Content-Length")
    if content_length is None:
        content_length = "0"
    cleaned = str(content_length).strip()
    if not cleaned:
        cleaned = "0"
    try:
        size = int(cleaned)
    except (TypeError, ValueError) as exc:
        raise BadRequestError("invalid content-length") from exc
    if size < 0:
        raise BadRequestError(f"invalid content-length: {size}")
    if size > limit:
        raise RequestPayloadTooLargeError(f"request body too large (max {limit} bytes)")
    return handler.rfile.read(size)
