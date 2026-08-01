from __future__ import annotations

import gzip
from io import BytesIO

from codoxear.server_handler import CodoxearHandler
from codoxear.server_http import json_response
from codoxear.server_http import json_response_with_etag


class _Handler:
    def __init__(self, accept_encoding: str = "") -> None:
        self.headers = {"Accept-Encoding": accept_encoding}
        self.status: int | None = None
        self.sent: list[tuple[str, str]] = []
        self.ended = False
        self.wfile = BytesIO()

    def send_response(self, status: int) -> None:
        self.status = status

    def send_header(self, name: str, value: str) -> None:
        self.sent.append((name, value))

    def end_headers(self) -> None:
        self.ended = True


def _no_cookie(_handler) -> None:
    pass


def test_json_response_gzips_large_gzip_accepted_bodies() -> None:
    handler = _Handler("br, gzip;q=0.8")
    json_response(handler, 200, {"value": "x" * 2048}, set_auth_cookie=_no_cookie)

    assert handler.status == 200
    assert ("Content-Encoding", "gzip") in handler.sent
    assert ("Vary", "Accept-Encoding") in handler.sent
    assert gzip.decompress(handler.wfile.getvalue()) == b'{"value": "' + b"x" * 2048 + b'"}'
    assert ("Content-Length", str(len(handler.wfile.getvalue()))) in handler.sent


def test_json_response_respects_gzip_opt_out_and_etag_variants() -> None:
    identity = _Handler("gzip;q=0")
    json_response(identity, 200, {"value": "x" * 2048}, set_auth_cookie=_no_cookie)
    assert ("Content-Encoding", "gzip") not in identity.sent

    compressed = _Handler("gzip")
    json_response_with_etag(compressed, {"value": "x" * 2048}, sha256_hex=lambda data: data.hex(), set_auth_cookie=_no_cookie)
    assert ("Content-Encoding", "gzip") in compressed.sent
    compressed_etag = next(value for name, value in compressed.sent if name == "ETag")

    identity_etag = _Handler("")
    json_response_with_etag(identity_etag, {"value": "x" * 2048}, sha256_hex=lambda data: data.hex(), set_auth_cookie=_no_cookie)
    identity_etag_value = next(value for name, value in identity_etag.sent if name == "ETag")
    assert compressed_etag != identity_etag_value


def test_handler_uses_http_1_1() -> None:
    assert CodoxearHandler.protocol_version == "HTTP/1.1"
