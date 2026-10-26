#!/usr/bin/env python3
"""Loopback delay proxy for slow-link behavioral verification.

Forwards 127.0.0.1:<listen> to 127.0.0.1:<upstream>, inserting a per-request
delay read from a control file (default 0) before the response is delivered.
SSE streams are relayed undelayed so the live channel stays honest; every
other response is buffered, delayed, then delivered — the shape of a slow
connection, not a dead one.
"""
import http.client
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

listen_port = int(sys.argv[1])
upstream_port = int(sys.argv[2])
delay_file = sys.argv[3]


def current_delay() -> float:
    try:
        return float(open(delay_file).read().strip())
    except Exception:
        return 0.0


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass

    def _relay(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        conn = http.client.HTTPConnection("127.0.0.1", upstream_port, timeout=30)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length", "connection")}
        conn.request(self.command, self.path, body=body, headers=headers)
        upstream = conn.getresponse()
        content_type = upstream.getheader("Content-Type") or ""
        if "text/event-stream" in content_type:
            self.send_response(upstream.status)
            for key, value in upstream.getheaders():
                if key.lower() in ("content-length", "connection", "transfer-encoding"):
                    continue
                self.send_header(key, value)
            self.end_headers()
            self.wfile.flush()
            try:
                while True:
                    chunk = upstream.read1(4096)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        payload = upstream.read()
        delay = current_delay()
        if delay > 0:
            time.sleep(delay)
        self.send_response(upstream.status)
        for key, value in upstream.getheaders():
            if key.lower() in ("content-length", "connection", "transfer-encoding"):
                continue
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
        conn.close()

    do_GET = _relay
    do_POST = _relay
    do_DELETE = _relay
    do_PUT = _relay


server = ThreadingHTTPServer(("127.0.0.1", listen_port), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
print(f"delay-proxy listening on {listen_port} -> {upstream_port}", flush=True)
threading.Event().wait()
