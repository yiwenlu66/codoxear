"""Federation hub: accept node connections, mirror sessions, relay control.

The hub listens on ``CODEX_WEB_FEDERATION_LISTEN`` for node connections
speaking the length-prefixed JSON frame protocol from ``federation_proto``.
For every session a node registers, the hub mirrors its sidecar metadata into
the local socks dir (with pid/log rewrites, see docs/federation.md) and
creates a relay unix socket that forwards the broker control RPC to the node.
The ordinary discovery/watchdog/listing pipeline then treats the mirrored
session like a local one; content routes are reverse-proxied over HTTP.

Without the env vars no hub is started and nothing in this module runs.
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import threading
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, MutableMapping

from . import federation_proto
from .socket_json import send_socket_json_line

RELAY_RPC_TIMEOUT_SECONDS = 15.0
LAUNCH_TIMEOUT_SECONDS = 30.0

ENV_FEDERATION_LISTEN = "CODEX_WEB_FEDERATION_LISTEN"
ENV_FEDERATION_TOKEN = "CODEX_WEB_FEDERATION_TOKEN"

# Hop-by-hop headers never forwarded in either direction by the content proxy.
_PROXY_DROP_REQUEST_HEADERS = frozenset({
    "host", "cookie", "connection", "content-length", "keep-alive",
    "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade",
})
_PROXY_DROP_RESPONSE_HEADERS = frozenset({
    "connection", "content-length", "keep-alive",
    "te", "trailer", "transfer-encoding", "upgrade",
})


@dataclass(frozen=True)
class FederationHubConfig:
    listen_host: str
    listen_port: int
    secret: str
    sock_dir: Path


def _split_host_port(raw: str, *, default_port: int) -> tuple[str, int]:
    text = raw.strip()
    if not text:
        raise ValueError("empty host:port")
    if text.startswith("["):  # [::1]:8744
        host, _, rest = text[1:].partition("]")
        if not rest.startswith(":"):
            return host, default_port
        return host, int(rest[1:])
    if ":" in text:
        host, _, port_text = text.rpartition(":")
        return host, int(port_text)
    return text, default_port


def hub_config_from_env(
    environ: MutableMapping[str, str] | None = None,
    *,
    sock_dir: Path,
) -> FederationHubConfig | None:
    """Return the hub config when CODEX_WEB_FEDERATION_LISTEN+TOKEN are set."""
    env = os.environ if environ is None else environ
    listen = (env.get(ENV_FEDERATION_LISTEN) or "").strip()
    secret = (env.get(ENV_FEDERATION_TOKEN) or "").strip()
    if not listen or not secret:
        return None
    host, port = _split_host_port(listen, default_port=8744)
    return FederationHubConfig(listen_host=host, listen_port=port, secret=secret, sock_dir=sock_dir)


class _PendingCall:
    def __init__(self) -> None:
        self.event = threading.Event()
        self.message: dict[str, Any] | None = None


class _NodeConnection:
    """One authenticated node connection and its pending RPC calls."""

    def __init__(self, hub: FederationHub, conn: socket.socket, *, name: str, addr: Any, http_port: int) -> None:
        self.hub = hub
        self.conn = conn
        self.name = name
        self.addr = addr
        self.http_port = int(http_port)
        self.sessions: set[str] = set()
        self._send_lock = threading.Lock()
        self._pending: dict[int, _PendingCall] = {}
        self._next_rid = 0
        self.closed = threading.Event()

    @property
    def http_host(self) -> str:
        host = self.addr[0] if isinstance(self.addr, tuple) and self.addr else str(self.addr)
        # A node that connected via a wildcard/unspecified peer cannot be
        # reached there; fall back to loopback for same-host tests.
        return host

    def send(self, payload: dict[str, Any]) -> None:
        with self._send_lock:
            federation_proto.send_frame(self.conn, payload)

    def _issue(self, payload_factory: Callable[[int], dict[str, Any]], *, timeout_s: float) -> dict[str, Any]:
        with self._send_lock:
            self._next_rid += 1
            rid = self._next_rid
            call = _PendingCall()
            self._pending[rid] = call
            try:
                federation_proto.send_frame(self.conn, payload_factory(rid))
            except Exception:
                self._pending.pop(rid, None)
                raise
        if not call.event.wait(timeout_s):
            with self._send_lock:
                self._pending.pop(rid, None)
            raise TimeoutError(f"federation node {self.name} did not answer rid {rid} within {timeout_s:.1f}s")
        message = call.message
        if message is None:
            raise ConnectionError(f"federation node {self.name} disconnected")
        return message

    def call_sock_rpc(self, session_id: str, request: dict[str, Any], *, timeout_s: float = RELAY_RPC_TIMEOUT_SECONDS) -> dict[str, Any]:
        message = self._issue(
            lambda rid: federation_proto.sock_rpc_message(rid=rid, session_id=session_id, request=request),
            timeout_s=timeout_s,
        )
        if message.get("type") != federation_proto.SOCK_RESULT:
            raise RuntimeError(f"unexpected federation reply type {message.get('type')!r}")
        if not message.get("ok"):
            raise RuntimeError(str(message.get("error") or "node sock rpc failed"))
        response = message.get("response")
        return dict(response) if isinstance(response, dict) else {}

    def launch(self, params: dict[str, Any], *, timeout_s: float = LAUNCH_TIMEOUT_SECONDS) -> dict[str, Any]:
        message = self._issue(
            lambda rid: {"type": federation_proto.LAUNCH, "rid": rid, "params": dict(params)},
            timeout_s=timeout_s,
        )
        if message.get("type") != federation_proto.LAUNCH_RESULT:
            raise RuntimeError(f"unexpected federation reply type {message.get('type')!r}")
        if not message.get("ok"):
            raise RuntimeError(str(message.get("error") or "node launch failed"))
        result = message.get("result")
        return dict(result) if isinstance(result, dict) else {}

    def _deliver_result(self, message: dict[str, Any]) -> None:
        rid = message.get("rid")
        if not isinstance(rid, int):
            return
        with self._send_lock:
            call = self._pending.pop(rid, None)
        if call is not None:
            call.message = message
            call.event.set()

    def _fail_all_pending(self) -> None:
        with self._send_lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for call in pending:
            call.event.set()

    def close(self) -> None:
        self.closed.set()
        try:
            self.conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.conn.close()
        except OSError:
            pass
        self._fail_all_pending()


class _RelayListener:
    """Unix-socket relay for one mirrored session on the hub.

    Each accepted connection speaks the broker control protocol (one JSON
    line request, one JSON line response); the request is forwarded to the
    owning node as a sock_rpc frame.
    """

    def __init__(self, hub: FederationHub, session_id: str, node: _NodeConnection) -> None:
        self.hub = hub
        self.session_id = session_id
        self.node = node
        self.sock_path = hub.config.sock_dir / f"{session_id}.sock"
        self._stop = threading.Event()
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.sock_path.unlink()
        except FileNotFoundError:
            pass
        listener.bind(str(self.sock_path))
        listener.listen(16)
        listener.settimeout(0.5)
        self._listener = listener
        self._thread = threading.Thread(
            target=self._accept_loop,
            name=f"federation-relay-{self.session_id[:8]}",
            daemon=True,
        )
        self._thread.start()

    def _accept_loop(self) -> None:
        listener = self._listener
        assert listener is not None
        while not self._stop.is_set():
            try:
                conn, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        try:
            conn.settimeout(RELAY_RPC_TIMEOUT_SECONDS + 5.0)
            line = self._read_json_line(conn)
            if line is None:
                return
            try:
                response = self.node.call_sock_rpc(self.session_id, line)
            except Exception as exc:
                response = {"ok": False, "error": f"federation relay: {exc}"}
            send_socket_json_line(conn, response)
        except OSError:
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    @staticmethod
    def _read_json_line(conn: socket.socket) -> dict[str, Any] | None:
        buf = bytearray()
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                return None
            buf.extend(chunk)
            newline = buf.find(b"\n")
            if newline >= 0:
                raw = bytes(buf[:newline])
                try:
                    obj = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    return None
                return obj if isinstance(obj, dict) else None

    def stop(self) -> None:
        self._stop.set()
        listener = self._listener
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            self.sock_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


class FederationHub:
    def __init__(
        self,
        config: FederationHubConfig,
        *,
        drop_sessions: Callable[[list[str]], None] | None = None,
    ) -> None:
        self.config = config
        self._drop_sessions = drop_sessions
        self._lock = threading.Lock()
        self._nodes: dict[str, _NodeConnection] = {}
        self._session_nodes: dict[str, _NodeConnection] = {}
        self._relays: dict[str, _RelayListener] = {}
        self._stop = threading.Event()
        self._listener: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._handler_threads: list[threading.Thread] = []

    @property
    def port(self) -> int:
        listener = self._listener
        if listener is None:
            return self.config.listen_port
        return int(listener.getsockname()[1])

    # ------------------------------------------------------------------ #
    # Lifecycle

    def start(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.config.listen_host, self.config.listen_port))
        listener.listen(16)
        listener.settimeout(0.5)
        self._listener = listener
        self._accept_thread = threading.Thread(target=self._accept_loop, name="federation-hub-accept", daemon=True)
        self._accept_thread.start()

    def stop(self) -> None:
        self._stop.set()
        listener = self._listener
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=2.0)
        with self._lock:
            nodes = list(self._nodes.values())
        for node in nodes:
            node.close()
        for thread in list(self._handler_threads):
            thread.join(timeout=2.0)
        with self._lock:
            relays = list(self._relays.values())
            self._relays.clear()
            self._session_nodes.clear()
            self._nodes.clear()
        for relay in relays:
            relay.stop()

    # ------------------------------------------------------------------ #
    # Registry accessors (used by server integration)

    def node_names(self) -> list[str]:
        with self._lock:
            return sorted(self._nodes)

    def node_for_session(self, session_id: str) -> _NodeConnection | None:
        with self._lock:
            return self._session_nodes.get(session_id)

    def node_named(self, name: str) -> _NodeConnection | None:
        with self._lock:
            return self._nodes.get(name)

    def node_name_for_session(self, session_id: str) -> str | None:
        node = self.node_for_session(session_id)
        return node.name if node is not None else None

    def is_mirrored_session(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._session_nodes

    def launch_on_node(self, node_name: str, params: dict[str, Any], *, timeout_s: float = LAUNCH_TIMEOUT_SECONDS) -> dict[str, Any]:
        node = self.node_named(node_name)
        if node is None:
            raise KeyError(f"unknown federation node {node_name!r}")
        return node.launch(params, timeout_s=timeout_s)

    # ------------------------------------------------------------------ #
    # Connection handling

    def _accept_loop(self) -> None:
        listener = self._listener
        assert listener is not None
        while not self._stop.is_set():
            try:
                conn, addr = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            thread = threading.Thread(target=self._handle_connection, args=(conn, addr), daemon=True)
            self._handler_threads.append(thread)
            thread.start()

    def _handle_connection(self, conn: socket.socket, addr: Any) -> None:
        conn.settimeout(None)
        node: _NodeConnection | None = None
        try:
            hello = federation_proto.recv_frame(conn)
            if hello.get("type") != federation_proto.HELLO:
                return
            name = hello.get("name")
            if not isinstance(name, str) or not name.strip():
                return
            name = name.strip()
            if not federation_proto.token_matches(self.config.secret, name, hello.get("token")):
                return
            http_port = hello.get("http_port")
            if not isinstance(http_port, int) or isinstance(http_port, bool) or http_port <= 0:
                return
            node = _NodeConnection(self, conn, name=name, addr=addr, http_port=http_port)
            with self._lock:
                previous = self._nodes.get(name)
                self._nodes[name] = node
            if previous is not None:
                previous.close()
                self._drop_node_sessions(previous)
            federation_proto.send_frame(conn, {"type": federation_proto.WELCOME})
            while True:
                message = federation_proto.recv_frame(conn)
                msg_type = message.get("type")
                if msg_type == federation_proto.REGISTER:
                    self._handle_register(node, message)
                elif msg_type == federation_proto.UNREGISTER:
                    session_id = message.get("session_id")
                    if isinstance(session_id, str) and session_id:
                        self._remove_mirror(session_id, node=node)
                elif msg_type in (federation_proto.SOCK_RESULT, federation_proto.LAUNCH_RESULT):
                    node._deliver_result(message)
        except (ConnectionError, federation_proto.FederationProtocolError, OSError):
            pass
        finally:
            if node is not None:
                node.close()
                with self._lock:
                    if self._nodes.get(node.name) is node:
                        del self._nodes[node.name]
                self._drop_node_sessions(node)
            else:
                try:
                    conn.close()
                except OSError:
                    pass

    # ------------------------------------------------------------------ #
    # Session mirroring

    def _rewrite_mirror_meta(self, meta: dict[str, Any]) -> dict[str, Any]:
        mirrored = dict(meta)
        mirrored["broker_pid"] = os.getpid()
        mirrored["codex_pid"] = os.getpid()
        mirrored["log_path"] = None
        return mirrored

    def _handle_register(self, node: _NodeConnection, message: dict[str, Any]) -> None:
        meta = message.get("session")
        if not isinstance(meta, dict):
            return
        session_id = meta.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            return
        mirrored = self._rewrite_mirror_meta(meta)
        meta_path = self.config.sock_dir / f"{session_id}.json"
        # Write the sidecar before the relay socket exists: discovery scans
        # *.sock, so a json without sock is invisible, while a sock without
        # json would be pruned as stale.
        tmp_path = meta_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(mirrored, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp_path, meta_path)
        with self._lock:
            self._session_nodes[session_id] = node
            node.sessions.add(session_id)
            relay = self._relays.get(session_id)
            if relay is None or relay.node is not node:
                if relay is not None:
                    relay.stop()
                relay = _RelayListener(self, session_id, node)
                self._relays[session_id] = relay
                relay.start()

    def _remove_mirror(self, session_id: str, *, node: _NodeConnection | None = None) -> None:
        with self._lock:
            owner = self._session_nodes.get(session_id)
            if owner is None or (node is not None and owner is not node):
                return
            del self._session_nodes[session_id]
            owner.sessions.discard(session_id)
            relay = self._relays.pop(session_id, None)
        if relay is not None:
            relay.stop()
        try:
            (self.config.sock_dir / f"{session_id}.json").unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass
        if self._drop_sessions is not None:
            self._drop_sessions([session_id])

    def _drop_node_sessions(self, node: _NodeConnection) -> None:
        with self._lock:
            session_ids = [sid for sid, owner in self._session_nodes.items() if owner is node]
        for session_id in session_ids:
            self._remove_mirror(session_id, node=node)


# ---------------------------------------------------------------------- #
# Process-wide active hub (started from server main when configured)

_active_hub: FederationHub | None = None
_active_hub_lock = threading.Lock()


def set_active_hub(hub: FederationHub | None) -> None:
    global _active_hub
    with _active_hub_lock:
        _active_hub = hub


def active_hub() -> FederationHub | None:
    with _active_hub_lock:
        return _active_hub


def node_name_for_session(session_id: str) -> str | None:
    hub = active_hub()
    return hub.node_name_for_session(session_id) if hub is not None else None


def is_mirrored_session(session_id: str) -> bool:
    hub = active_hub()
    return hub.is_mirrored_session(session_id) if hub is not None else False


def remote_session_id_for_path(path: str) -> str | None:
    """``/api/sessions/<id>/...`` -> *id* when it names a federation-remote session."""
    parts = path.split("/")
    if len(parts) < 5 or parts[:3] != ["", "api", "sessions"]:
        return None
    session_id = urllib.parse.unquote(parts[3])
    if not session_id:
        return None
    hub = active_hub()
    if hub is None or hub.node_for_session(session_id) is None:
        return None
    return session_id


def proxy_request_to_node(
    handler: Any,
    *,
    session_id: str,
    path: str,
    query: str,
    read_body: Callable[..., bytes],
) -> bool:
    """Reverse-proxy one HTTP request to the owning node server.

    Streams the response body in chunks (SSE-safe) and closes the client
    connection afterwards, so no content-length negotiation is needed.
    Returns False (without touching the response) when the session is not or
    no longer remote.
    """
    hub = active_hub()
    node = hub.node_for_session(session_id) if hub is not None else None
    if hub is None or node is None:
        return False
    body: bytes | None = None
    if handler.command in ("POST", "PUT", "PATCH", "DELETE"):
        body = read_body(handler, limit=64 * 1024 * 1024)
    target = path + ("?" + query if query else "")
    headers = {
        key: value
        for key, value in handler.headers.items()
        if key.lower() not in _PROXY_DROP_REQUEST_HEADERS
    }
    headers[federation_proto.FEDERATION_AUTH_HEADER] = federation_proto.federation_token(hub.config.secret, node.name)
    conn = http.client.HTTPConnection(node.http_host, node.http_port, timeout=30)
    try:
        conn.request(handler.command, target, body=body, headers=headers)
        response = conn.getresponse()
    except OSError as exc:
        handler.deps.json_response(handler, 502, {"error": f"federation node unreachable: {exc}"})
        return True
    handler.send_response(response.status)
    for key, value in response.getheaders():
        if key.lower() in _PROXY_DROP_RESPONSE_HEADERS:
            continue
        handler.send_header(key, value)
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler.close_connection = True
    try:
        while True:
            chunk = response.read(65536)
            if not chunk:
                break
            handler.wfile.write(chunk)
            handler.wfile.flush()
    finally:
        conn.close()
    return True
