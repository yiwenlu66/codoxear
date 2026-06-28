from __future__ import annotations

import http.server
import os
import signal
import socket
import socketserver
import sys
import threading
from typing import Any


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


class ThreadingHTTPServerV6(ThreadingHTTPServer):
    address_family = socket.AF_INET6

    def server_bind(self) -> None:
        v6only = getattr(socket, "IPV6_V6ONLY", None)
        if v6only is not None:
            self.socket.setsockopt(socket.IPPROTO_IPV6, v6only, 0)
        super().server_bind()


def run_main(server: Any) -> None:
    os.makedirs(server.APP_DIR, exist_ok=True)
    os.makedirs(server.UPLOAD_DIR, exist_ok=True)
    try:
        server._require_password()
    except Exception as exc:
        sys.stderr.write(f"error: {exc}\n")
        raise SystemExit(2)

    host = server.DEFAULT_HOST
    httpd: ThreadingHTTPServer
    if ":" in host:
        httpd = ThreadingHTTPServerV6((host, server.DEFAULT_PORT), server.Handler)
    else:
        httpd = ThreadingHTTPServer((host, server.DEFAULT_PORT), server.Handler)

    def _sigterm(_signo: int, _frame: Any) -> None:
        server.MANAGER.stop()
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    httpd.serve_forever()
