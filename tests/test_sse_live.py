from __future__ import annotations

import socketserver

from codoxear.server_main import ThreadingHTTPServer


def test_http_server_remains_threaded_for_persistent_sse_handlers() -> None:
    assert issubclass(ThreadingHTTPServer, socketserver.ThreadingMixIn)
    assert ThreadingHTTPServer.daemon_threads is True
