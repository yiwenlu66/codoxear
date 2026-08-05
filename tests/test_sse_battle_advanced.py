"""Socket-level SSE failure simulations against a broker-shaped native log.

The test server exposes the production ``handle_messages_live_stream`` handler
through a real HTTP/1.1 socket.  The log writer emits the same Codex JSONL rows
that a broker binds, so reconnect recovery proves the production cursor and
normalization path rather than a synthetic in-memory event bus.
"""
from __future__ import annotations

import http.client
import json
import socket
import socketserver
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.parse
from io import BytesIO
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from types import SimpleNamespace

from codoxear.message_cursor import decode_message_cursor
from codoxear.message_cursor import encode_message_cursor
from codoxear.message_routes import MessageRouteDeps
from codoxear.message_routes import handle_messages_live_stream
from codoxear.message_routes import handle_messages_tail
from codoxear.server_main import ThreadingHTTPServer
from codoxear.session_model import Session


ROOT = Path(__file__).resolve().parents[1]
SSE_CONTROLLER = ROOT / "codoxear" / "static" / "app_sse.js"
MESSAGE_FLOW_CONTROLLER = ROOT / "codoxear" / "static" / "app_message_flow.js"
POLLING_HELPERS = ROOT / "codoxear" / "static" / "app_polling.js"
TRANSCRIPT_HELPERS = ROOT / "codoxear" / "static" / "app_transcript.js"
_CURSOR_SECRET = b"sse-battle-test-cursor-secret"


class _BrokerSession:
    """A broker-bound Codex transcript producer with a durable JSONL log."""

    def __init__(self, root: Path) -> None:
        self.log_path = root / "rollout-sse-battle.jsonl"
        self.log_path.touch()
        self.session = Session(
            session_id="sse-battle",
            thread_id="sse-battle-thread",
            broker_pid=1,
            codex_pid=1,
            agent_backend="codex",
            owned=False,
            start_ts=time.time(),
            cwd=str(root),
            log_path=self.log_path,
            sock_path=root / "sse-battle.sock",
        )

    def publish_assistant(self, text: str) -> None:
        row = {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
                "phase": "final_answer",
            },
            "ts": time.time(),
        }
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            stream.flush()


class _LiveManager:
    def __init__(self, broker: _BrokerSession) -> None:
        self.broker = broker
        self.deltas: list[int] = []

    def refresh_session_meta(self, _session_id: str) -> None:
        return None

    def get_session(self, session_id: str) -> Session | None:
        return self.broker.session if session_id == self.broker.session.session_id else None

    def mark_log_delta(self, _session_id: str, *, objs, new_off: int) -> None:
        del objs
        self.deltas.append(new_off)

    def _attach_notification_texts(self, events):
        return events


def _route_deps() -> MessageRouteDeps:
    def encode_cursor(*, kind: str, session: Session, pos: int) -> str:
        return encode_message_cursor(kind=kind, session=session, pos=pos, secret=_CURSOR_SECRET)

    def decode_cursor(token: str, *, kind: str, session: Session) -> int:
        return decode_message_cursor(token, kind=kind, session=session, secret=_CURSOR_SECRET)

    return MessageRouteDeps(
        require_auth=lambda _handler: True,
        set_auth_cookie=lambda _handler: None,
        json_response=lambda handler, status, payload: _json_response(handler, status, payload),
        launch_attempt_transcript_for_session_id=lambda _session_id: None,
        transcript_export_max_bytes=1024 * 1024,
        transcript_search_max_line_bytes=64 * 1024,
        decode_message_cursor=decode_cursor,
        encode_message_cursor=encode_cursor,
        record_metric=lambda _name, _value: None,
        message_runtime_snapshot=lambda _session_id, session, **_kwargs: ({}, False, 0, session.token),
    )


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: object) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class _BattleHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self) -> None:
        super().setup()
        with self.server.live_lock:
            self.server.live_handlers.add(self)

    def finish(self) -> None:
        with self.server.live_lock:
            self.server.live_handlers.discard(self)
        super().finish()

    def do_GET(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path != "/api/sessions/sse-battle/live":
            self.send_error(404)
            return
        handle_messages_live_stream(
            self,
            session_id="sse-battle",
            query=parsed.query,
            manager=self.server.manager,
            deps=self.server.deps,
        )

    def log_message(self, _format: str, *_args: object) -> None:
        return


class _BattleServer:
    def __init__(self, manager: _LiveManager) -> None:
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _BattleHandler)
        self.httpd.manager = manager
        self.httpd.deps = _route_deps()
        self.httpd.live_handlers = set()
        self.httpd.live_lock = threading.Lock()
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def port(self) -> int:
        return int(self.httpd.server_address[1])

    def drop_live_connections(self) -> None:
        """Abruptly close server-owned sockets, as a server restart would."""
        with self.httpd.live_lock:
            handlers = list(self.httpd.live_handlers)
        for handler in handlers:
            try:
                handler.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                handler.connection.close()
            except OSError:
                pass

    def close(self) -> None:
        self.drop_live_connections()
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)


class _SseTranscriptClient:
    """Small EventSource-equivalent parser retaining the production cursor."""

    def __init__(self) -> None:
        self.connection: http.client.HTTPConnection | None = None
        self.response: http.client.HTTPResponse | None = None
        self.live_cursor: str | None = None
        self.transcript: list[str] = []

    def open(self, port: int) -> None:
        query = ""
        if self.live_cursor:
            query = "?" + urllib.parse.urlencode({"cursor": self.live_cursor})
        self.close()
        self.connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        self.connection.request("GET", "/api/sessions/sse-battle/live" + query)
        self.response = self.connection.getresponse()
        if self.response.status != 200:
            raise AssertionError(f"unexpected SSE status {self.response.status}")
        if self.response.getheader("Content-Type") != "text/event-stream":
            raise AssertionError("SSE endpoint did not return text/event-stream")

    def read_event(self, timeout: float = 10.0) -> tuple[str, dict]:
        if self.connection is None or self.response is None:
            raise AssertionError("SSE connection is not open")
        stream_socket = getattr(getattr(getattr(self.response, "fp", None), "raw", None), "_sock", None)
        if stream_socket is None:
            raise AssertionError("SSE response has no readable socket")
        stream_socket.settimeout(timeout)
        name = "message"
        data_lines: list[str] = []
        while True:
            raw = self.response.fp.readline()
            if not raw:
                raise ConnectionError("SSE stream closed before a complete event")
            line = raw.decode("utf-8").rstrip("\r\n")
            if not line:
                payload = json.loads("\n".join(data_lines)) if data_lines else {}
                if name == "message":
                    cursor = payload.get("live_cursor")
                    if isinstance(cursor, str) and cursor:
                        self.live_cursor = cursor
                    self.transcript.extend(
                        str(event["text"])
                        for event in payload.get("events", [])
                        if isinstance(event, dict) and isinstance(event.get("text"), str)
                    )
                return name, payload
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data_lines.append(line[6:])

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
        self.connection = None
        self.response = None


class _VirtualHeartbeatHandler:
    """In-memory transport that disconnects only after three heartbeats."""

    def __init__(self) -> None:
        self.status: int | None = None
        self.headers: list[tuple[str, str]] = []
        self.unauthorized = False
        self.wfile = _HeartbeatDisconnectingStream()

    def _unauthorized(self) -> None:
        self.unauthorized = True

    def send_response(self, status: int) -> None:
        self.status = status

    def send_header(self, name: str, value: str) -> None:
        self.headers.append((name, value))

    def end_headers(self) -> None:
        return None


class _HeartbeatDisconnectingStream(BytesIO):
    def flush(self) -> None:
        super().flush()
        if self.getvalue().count(b"event: heartbeat") >= 3:
            raise BrokenPipeError()


class _VirtualClock:
    """Advances the live handler through a long quiet period without wall time."""

    def __init__(self) -> None:
        self.value = 100.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


class _SlowDisconnectRelay(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, upstream_port: int, *, drop_after_bytes: int) -> None:
        self.upstream_port = upstream_port
        self.drop_after_bytes = drop_after_bytes
        super().__init__(("127.0.0.1", 0), _SlowDisconnectRelayHandler)


class _SlowDisconnectRelayHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        request = bytearray()
        while b"\r\n\r\n" not in request:
            part = self.request.recv(4096)
            if not part:
                return
            request.extend(part)
        with socket.create_connection(("127.0.0.1", self.server.upstream_port), timeout=5) as upstream:
            upstream.sendall(request)
            forwarded = 0
            while True:
                block = upstream.recv(17)
                if not block:
                    return
                try:
                    self.request.sendall(block)
                except OSError:
                    return
                forwarded += len(block)
                time.sleep(0.003)
                if forwarded >= self.server.drop_after_bytes:
                    return


class TestSseBattleAdvanced(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.broker = _BrokerSession(Path(self.tmp.name))
        self.manager = _LiveManager(self.broker)
        self.server = _BattleServer(self.manager)

    def tearDown(self) -> None:
        self.server.close()
        self.tmp.cleanup()

    def _open_at_eof(self) -> _SseTranscriptClient:
        client = _SseTranscriptClient()
        client.open(self.server.port)
        name, payload = client.read_event()
        self.assertEqual(name, "message")
        self.assertEqual(payload["events"], [])
        self.assertTrue(client.live_cursor)
        return client

    def test_server_disconnect_reconnects_from_cursor_without_message_loss(self) -> None:
        client = self._open_at_eof()
        try:
            self.broker.publish_assistant("before-disconnect")
            name, _payload = client.read_event()
            self.assertEqual(name, "message")
            self.server.drop_live_connections()
            with self.assertRaises((ConnectionError, OSError, http.client.HTTPException)):
                client.read_event(timeout=2)
            self.broker.publish_assistant("after-disconnect")
            started = time.monotonic()
            client.open(self.server.port)
            name, _payload = client.read_event()
            elapsed_ms = (time.monotonic() - started) * 1000
            self.assertEqual(name, "message")
            self.assertLess(elapsed_ms, 1000, "cursor reconnect did not establish within 1 second")
            self.assertEqual(client.transcript, ["before-disconnect", "after-disconnect"])
        finally:
            client.close()

    def test_idle_stream_remains_open_past_sixty_seconds_with_heartbeats(self) -> None:
        """Virtual time proves the production heartbeat loop through 75 idle seconds.

        A real 60-second wall-clock test would make every local and CI run wait
        a minute while testing only the handler's monotonic-time branch.  The
        HTTP socket tests above still prove that these unbuffered writes reach a
        connected client; this clock advances that same handler past the phone
        idle threshold without slowing the suite.
        """
        import codoxear.message_routes as message_routes

        clock = _VirtualClock()
        handler = _VirtualHeartbeatHandler()
        original_monotonic = message_routes.time.monotonic
        original_sleep = message_routes.time.sleep
        message_routes.time.monotonic = clock.monotonic
        message_routes.time.sleep = clock.sleep
        try:
            handle_messages_live_stream(
                handler,
                session_id="sse-battle",
                query="",
                manager=self.manager,
                deps=_route_deps(),
            )
        finally:
            message_routes.time.monotonic = original_monotonic
            message_routes.time.sleep = original_sleep

        self.assertEqual(handler.status, 200)
        self.assertGreaterEqual(clock.value - 100.0, 60.0)
        self.assertEqual(handler.wfile.getvalue().count(b"event: heartbeat"), 3)

    def test_tail_fallback_rehydrates_the_log_with_a_resume_cursor(self) -> None:
        self.broker.publish_assistant("tail-fallback-one")
        self.broker.publish_assistant("tail-fallback-two")
        handler = _VirtualHeartbeatHandler()

        handle_messages_tail(
            handler,
            session_id="sse-battle",
            query="limit=20",
            manager=self.manager,
            deps=_route_deps(),
        )

        payload = json.loads(handler.wfile.getvalue())
        self.assertEqual(handler.status, 200)
        self.assertEqual(
            [event["text"] for event in payload["events"]],
            ["tail-fallback-one", "tail-fallback-two"],
        )
        self.assertIsInstance(payload["live_cursor"], str)
        self.assertTrue(payload["live_cursor"])

    def test_rapid_native_log_events_arrive_once_and_in_order(self) -> None:
        client = self._open_at_eof()
        try:
            expected = [f"rapid-{index:04d}" for index in range(1000)]
            for text in expected:
                self.broker.publish_assistant(text)
            name, payload = client.read_event(timeout=10)
            self.assertEqual(name, "message")
            self.assertEqual([event["text"] for event in payload["events"]], expected)
            self.assertEqual(client.transcript, expected)
            self.assertEqual(len(set(client.transcript)), 1000)
        finally:
            client.close()

    def test_fragmented_network_failure_replays_only_unconfirmed_delta(self) -> None:
        client = self._open_at_eof()
        relay = _SlowDisconnectRelay(self.server.port, drop_after_bytes=220)
        relay_thread = threading.Thread(target=relay.serve_forever, daemon=True)
        relay_thread.start()
        try:
            self.broker.publish_assistant("replay-after-fragmented-network-error")
            client.open(int(relay.server_address[1]))
            with self.assertRaises((ConnectionError, OSError, http.client.HTTPException)):
                client.read_event(timeout=5)
            client.open(self.server.port)
            name, _payload = client.read_event(timeout=5)
            self.assertEqual(name, "message")
            self.assertEqual(client.transcript, ["replay-after-fragmented-network-error"])
        finally:
            client.close()
            relay.shutdown()
            relay.server_close()
            relay_thread.join(timeout=2)

    def test_visible_resume_reopens_the_actual_message_flow_sse_after_hidden_retry(self) -> None:
        """The flow called by app.js must defer reconnects while hidden."""
        sources = [
            POLLING_HELPERS.read_text(encoding="utf-8"),
            TRANSCRIPT_HELPERS.read_text(encoding="utf-8"),
            MESSAGE_FLOW_CONTROLLER.read_text(encoding="utf-8"),
        ]
        js = f"""
const vm = require("vm");
const sources = [];
const timers = [];
let visibility = "visible";
const noop = () => {{}};
class FakeEventSource {{
  constructor(url) {{ this.url = url; this.listeners = {{}}; sources.push(this); }}
  addEventListener(type, callback) {{ this.listeners[type] = callback; }}
  close() {{ this.closed = true; }}
}}
const ctx = {{ window: {{}}, console, Date, URL, encodeURIComponent }};
vm.createContext(ctx);
{''.join(f'vm.runInContext({json.dumps(source)}, ctx);' for source in sources)}
const options = new Proxy({{
  getSelected: () => "sse-battle", getGeneration: () => 7, isAppDisposed: () => false,
  getTurnOpen: () => false, setTurnOpen: noop, getSessionInfo: () => ({{ session_id: "sse-battle" }}),
  activeTranscriptSnapshot: () => ({{ state: "bound", liveCursor: "cursor-7" }}),
  resolveAppUrl: (path) => "https://phone.tailnet.example" + path,
  visibilityState: () => visibility, EventSource: FakeEventSource,
  typingRowRuntime: {{ snapshot: () => ({{ stats: {{}} }}), updateTypingStats: noop, updateSubagentGauge: noop, resetTypingStats: noop }},
  setTimeout: (callback, delay) => {{ const timer = {{ callback, delay, cleared: false }}; timers.push(timer); return timer; }},
  clearTimeout: (timer) => {{ if (timer) timer.cleared = true; }}, now: () => 1000,
}}, {{ get: (target, key) => key in target ? target[key] : noop }});
const controller = ctx.window.CodoxearMessageFlow.createMessageFlowController(options);
controller.openMessageEventSource("sse-battle", 7);
sources[0].onopen();
sources[0].listeners.error();
const retry = timers.find((timer) => timer.delay === 6000);
visibility = "hidden";
retry.callback();
const whileHidden = sources.length;
visibility = "visible";
controller.resumeLiveDelivery();
process.stdout.write(JSON.stringify({{ retryDelay: retry.delay, whileHidden, afterVisible: sources.length, resumeUrl: sources[1].url }}));
"""
        result = subprocess.run(["node", "-e", js], check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(result.stdout), {
            "retryDelay": 6000,
            "whileHidden": 1,
            "afterVisible": 2,
            "resumeUrl": "https://phone.tailnet.example/api/sessions/sse-battle/live?cursor=cursor-7",
        })

    def test_malformed_event_surfaces_error_and_next_event_is_delivered(self) -> None:
        source = SSE_CONTROLLER.read_text(encoding="utf-8")
        js = f"""
const vm = require("vm");
const sources = [];
class FakeEventSource {{
  constructor(url) {{ this.url = url; this.listeners = {{}}; sources.push(this); }}
  addEventListener(type, callback) {{ this.listeners[type] = callback; }}
  close() {{ this.closed = true; }}
  emit(type, data) {{
    if (type === "open") return this.onopen();
    return this.listeners[type]({{ data }});
  }}
}}
const ctx = {{ window: {{}}, EventSource: FakeEventSource, setTimeout, clearTimeout }};
vm.createContext(ctx);
vm.runInContext({json.dumps(source)}, ctx);
const malformed = [];
const delivered = [];
const controller = ctx.window.CodoxearSse.createMessageEventSourceController({{
  EventSourceImpl: FakeEventSource,
  resolveUrl: (path) => "https://phone.tailnet.example" + path,
  getSnapshot: () => ({{ state: "bound", liveCursor: "cursor-1" }}),
  isActive: () => true,
  onStateChange: () => {{}},
  onOpen: () => {{}},
  onMessage: (_sid, _gen, payload) => delivered.push(payload.text),
  onFallback: () => {{}},
  onMalformedMessage: (error) => malformed.push(error.name),
}});
controller.open("sse-battle", 9);
sources[0].emit("open");
sources[0].emit("message", "{{not-json");
sources[0].emit("message", JSON.stringify({{ text: "after-malformed" }}));
process.stdout.write(JSON.stringify({{ sourceCount: sources.length, malformed, delivered, closed: Boolean(sources[0].closed) }}));
"""
        result = subprocess.run(["node", "-e", js], check=True, capture_output=True, text=True)
        outcome = json.loads(result.stdout)
        self.assertEqual(outcome, {
            "sourceCount": 1,
            "malformed": ["SyntaxError"],
            "delivered": ["after-malformed"],
            "closed": False,
        })

    def test_browser_controller_retries_errors_and_visibility_resume_bypasses_backoff(self) -> None:
        source = SSE_CONTROLLER.read_text(encoding="utf-8")
        js = f"""
const vm = require("vm");
const sources = [];
const timers = [];
class FakeEventSource {{
  constructor(url) {{ this.url = url; this.listeners = {{}}; this.closed = false; sources.push(this); }}
  addEventListener(type, callback) {{ this.listeners[type] = callback; }}
  close() {{ this.closed = true; }}
  emit(type, data) {{
    if (type === "open") return this.onopen();
    return this.listeners[type]({{ data }});
  }}
}}
const ctx = {{ window: {{}}, EventSource: FakeEventSource, setTimeout: (fn, delay) => {{ const timer = {{ fn, delay, cancelled: false }}; timers.push(timer); return timer; }}, clearTimeout: (timer) => {{ timer.cancelled = true; }} }};
vm.createContext(ctx);
vm.runInContext({json.dumps(source)}, ctx);
const transitions = [];
const fallbacks = [];
const messages = [];
let active = true;
const controller = ctx.window.CodoxearSse.createMessageEventSourceController({{
  EventSourceImpl: FakeEventSource,
  resolveUrl: (path) => "https://phone.tailnet.example" + path,
  getSnapshot: () => ({{ state: "bound", liveCursor: "cursor-1" }}),
  isActive: () => active,
  onStateChange: (open) => transitions.push(open),
  onOpen: () => transitions.push("opened"),
  onMessage: (_sid, _gen, payload) => messages.push(payload.text),
  onFallback: () => fallbacks.push("fallback"),
  retryMs: 1000,
}});
controller.open("sse-battle", 9);
sources[0].emit("open");
sources[0].emit("error");
const timedRetry = {{ delay: timers[0].delay, cancelled: timers[0].cancelled }};
timers[0].fn();
sources[1].emit("open");
sources[1].emit("message", JSON.stringify({{ text: "delivered-after-retry" }}));
sources[1].emit("error");
controller.resume("sse-battle", 9); // visibilityState changed back to visible
sources[2].emit("open");
process.stdout.write(JSON.stringify({{ sourceCount: sources.length, timedRetry, transitions, fallbacks, messages, urls: sources.map((item) => item.url) }}));
"""
        result = subprocess.run(["node", "-e", js], check=True, capture_output=True, text=True)
        outcome = json.loads(result.stdout)
        self.assertEqual(outcome["timedRetry"], {"delay": 1000, "cancelled": False})
        self.assertEqual(outcome["sourceCount"], 3)
        self.assertEqual(outcome["fallbacks"], ["fallback", "fallback"])
        self.assertEqual(outcome["messages"], ["delivered-after-retry"])
        self.assertTrue(all("cursor=cursor-1" in url for url in outcome["urls"]))
        self.assertEqual(outcome["transitions"], [False, True, "opened", False, False, True, "opened", False, False, True, "opened"])


if __name__ == "__main__":
    unittest.main()
