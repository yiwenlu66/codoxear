"""Container-only HTTP fault proxy: real origin commits, no UI mocks.

Control endpoints and metadata are test-only, never mounted in the application.
Finite responses are read completely before faults; live SSE is disabled so the
browser uses its supported polling fallback. No file contents enter diagnostics.
"""
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import threading
import time
from urllib.parse import urlsplit

lock = threading.Lock()
state = {"lose_next": False, "fail_status": False, "cut_next": False,
         "stale_catalog": False, "delay": 0, "uploads": [], "status": [],
         "active_catalog": 0, "max_catalog": 0, "catalog_reads": 0}
catalog = None


class Proxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def reply(self, status, body, headers=()):
        self.send_response(status)
        for key, value in headers:
            if key.lower() not in {"content-length", "connection", "transfer-encoding", "content-encoding"}:
                self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.forward()

    def do_POST(self):
        self.forward()

    def forward(self):
        global catalog
        path = urlsplit(self.path).path
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        if path == "/__fault":
            if self.command == "POST":
                with lock:
                    state.update(json.loads(body))
            with lock:
                result = json.dumps(state).encode()
            self.reply(200, result)
            return
        if path.endswith("/live"):
            self.reply(503, b'{}')
            return
        is_catalog = path == "/api/sessions" and self.command == "GET"
        is_upload = path.endswith("/inject_file")
        is_status = path.endswith("/attachments/upload-status")
        with lock:
            fail_status = state["fail_status"]
            lose = is_upload and state["lose_next"]
            cut = is_upload and state["cut_next"]
            if is_upload:
                state["lose_next"] = state["cut_next"] = False
                state["uploads"].append({"id": self.headers.get("X-Codoxear-Upload-Id"),
                    "bytes": len(body), "declared": self.headers.get("Content-Length"),
                    "type": self.headers.get("Content-Type"), "sha256": hashlib.sha256(body).hexdigest(),
                    "cut": cut, "lost_response": lose})
            delay = state["delay"] if is_catalog else 0
            stale = state["stale_catalog"] and is_catalog and catalog is not None
            if is_catalog:
                state["catalog_reads"] += 1
                state["active_catalog"] += 1
                state["max_catalog"] = max(state["max_catalog"], state["active_catalog"])
        try:
            if is_status and fail_status:
                self.reply(503, b'{"error":"test status unavailable"}')
                return
            if stale:
                self.reply(200, catalog, [("Content-Type", "application/json"), ("X-Codoxear-Stale", "timeout")])
                return
            upstream = http.client.HTTPConnection("127.0.0.1", 19882, timeout=25)
            headers = {k: v for k, v in self.headers.items() if k.lower() not in {"host", "connection", "accept-encoding"}}
            headers["Connection"] = "close"
            if cut:
                upstream.putrequest(self.command, self.path)
                for k, v in headers.items():
                    upstream.putheader(k, v)
                upstream.endheaders(body[:max(1, len(body) // 2)])
                upstream.sock.shutdown(socket.SHUT_WR)
            else:
                upstream.request(self.command, self.path, body, headers)
            response = upstream.getresponse()
            payload = response.read()
            response_headers = response.getheaders()
            status = response.status
            upstream.close()
            with lock:
                if is_catalog and catalog is None and status == 200:
                    catalog = payload
                if is_status:
                    state["status"].append({"path": self.path, "cache_control": dict(response_headers).get("Cache-Control"), "status": status})
                if is_upload:
                    state["uploads"][-1]["origin_status"] = status
            if delay:
                time.sleep(delay)
            if lose or cut:
                # A bare pre-header TCP reset lets Chromium transparently replay
                # a POST. Deliver headers plus a truncated body to force an
                # observable lost response, after the real origin commit.
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload[:1])
                self.wfile.flush()
                self.close_connection = True
                self.connection.shutdown(socket.SHUT_RDWR)
                return
            self.reply(status, payload, response_headers)
        finally:
            if is_catalog:
                with lock:
                    state["active_catalog"] -= 1


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 19881), Proxy).serve_forever()
