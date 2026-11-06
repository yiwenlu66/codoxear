"""Federation node client: publish local sessions to a hub.

When ``CODEX_WEB_FEDERATION_HUB`` + ``CODEX_WEB_FEDERATION_TOKEN`` are set, a
normal codoxear-server doubles as a federation node: it connects to the hub,
watches its own socks dir for broker sidecars, pushes register/unregister/
meta-update frames, and executes hub-initiated sock_rpc / launch frames
against the local broker sockets and session launcher.

Without the env vars nothing here runs and the server stays standalone.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, MutableMapping

from . import federation_proto
from .control_socket import call_control_socket

ENV_FEDERATION_HUB = "CODEX_WEB_FEDERATION_HUB"
ENV_FEDERATION_TOKEN = "CODEX_WEB_FEDERATION_TOKEN"
ENV_NODE_NAME = "CODEX_WEB_NODE_NAME"

DEFAULT_HUB_PORT = 8744
WATCH_POLL_SECONDS = 1.0
RECONNECT_MIN_SECONDS = 0.5
RECONNECT_MAX_SECONDS = 5.0
SOCK_RPC_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class FederationNodeConfig:
    hub_host: str
    hub_port: int
    secret: str
    node_name: str
    http_port: int
    sock_dir: Path
    watch_poll_s: float = WATCH_POLL_SECONDS


def _split_host_port(raw: str) -> tuple[str, int]:
    text = raw.strip()
    if not text:
        raise ValueError("empty hub address")
    if text.startswith("["):
        host, _, rest = text[1:].partition("]")
        if not rest.startswith(":"):
            return host, DEFAULT_HUB_PORT
        return host, int(rest[1:])
    if ":" in text:
        host, _, port_text = text.rpartition(":")
        return host, int(port_text)
    return text, DEFAULT_HUB_PORT


def node_config_from_env(
    environ: MutableMapping[str, str] | None = None,
    *,
    http_port: int,
    sock_dir: Path,
) -> FederationNodeConfig | None:
    """Return the node config when CODEX_WEB_FEDERATION_HUB+TOKEN are set."""
    env = os.environ if environ is None else environ
    hub = (env.get(ENV_FEDERATION_HUB) or "").strip()
    secret = (env.get(ENV_FEDERATION_TOKEN) or "").strip()
    if not hub or not secret:
        return None
    host, port = _split_host_port(hub)
    name = (env.get(ENV_NODE_NAME) or "").strip() or socket.gethostname()
    return FederationNodeConfig(
        hub_host=host,
        hub_port=port,
        secret=secret,
        node_name=name,
        http_port=int(http_port),
        sock_dir=sock_dir,
    )


def expected_incoming_token(environ: MutableMapping[str, str] | None = None) -> str | None:
    """Token a hub must present on proxied HTTP requests to this node.

    Returns None when this server is not configured as a federation node.
    """
    env = os.environ if environ is None else environ
    hub = (env.get(ENV_FEDERATION_HUB) or "").strip()
    secret = (env.get(ENV_FEDERATION_TOKEN) or "").strip()
    if not hub or not secret:
        return None
    name = (env.get(ENV_NODE_NAME) or "").strip() or socket.gethostname()
    return federation_proto.federation_token(secret, name)


def _meta_fingerprint(meta: dict[str, Any]) -> str:
    canonical = json.dumps(meta, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_sidecar_meta(meta_path: Path) -> dict[str, Any] | None:
    try:
        obj = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return obj if isinstance(obj, dict) else None


class FederationNode:
    def __init__(
        self,
        config: FederationNodeConfig,
        *,
        launch_session: Callable[[dict[str, Any]], dict[str, Any]],
        is_mirrored_session: Callable[[str], bool] | None = None,
    ) -> None:
        self.config = config
        self._launch_session = launch_session
        # When this process is also a hub, never re-announce mirrored
        # sidecars upstream (that would chain-federate another node's rows).
        self._is_mirrored_session = is_mirrored_session or (lambda _sid: False)
        self._stop = threading.Event()
        self._connected = threading.Event()
        self._send_lock = threading.Lock()
        self._conn: socket.socket | None = None
        self._announced: dict[str, str] = {}
        self._stems: dict[str, str] = {}
        self._stems_lock = threading.Lock()
        self._connect_thread: threading.Thread | None = None
        self._watch_thread: threading.Thread | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle

    def start(self) -> None:
        self._watch_thread = threading.Thread(target=self._watch_loop, name="federation-node-watch", daemon=True)
        self._watch_thread.start()
        self._connect_thread = threading.Thread(target=self._connect_loop, name="federation-node-connect", daemon=True)
        self._connect_thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._close_conn()
        for thread in (self._connect_thread, self._watch_thread):
            if thread is not None:
                thread.join(timeout=3.0)

    def _close_conn(self) -> None:
        with self._send_lock:
            conn = self._conn
            self._conn = None
            self._connected.clear()
        if conn is not None:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                conn.close()
            except OSError:
                pass

    def _send(self, payload: dict[str, Any]) -> bool:
        with self._send_lock:
            conn = self._conn
            if conn is None or not self._connected.is_set():
                return False
            try:
                federation_proto.send_frame(conn, payload)
            except OSError:
                return False
            return True

    # ------------------------------------------------------------------ #
    # Connection management

    def _connect_loop(self) -> None:
        backoff = RECONNECT_MIN_SECONDS
        while not self._stop.is_set():
            try:
                self._run_connection()
            except (ConnectionError, federation_proto.FederationProtocolError, OSError):
                pass
            finally:
                self._close_conn()
            if self._stop.is_set():
                break
            time.sleep(backoff)
            backoff = min(RECONNECT_MAX_SECONDS, backoff * 2.0)

    def _run_connection(self) -> None:
        conn = socket.create_connection((self.config.hub_host, self.config.hub_port), timeout=10.0)
        conn.settimeout(None)
        try:
            federation_proto.send_frame(
                conn,
                federation_proto.hello_message(
                    node_name=self.config.node_name,
                    secret=self.config.secret,
                    http_port=self.config.http_port,
                ),
            )
            reply = federation_proto.recv_frame(conn)
            if reply.get("type") != federation_proto.WELCOME:
                raise ConnectionError("federation hub rejected the hello handshake")
            with self._send_lock:
                self._conn = conn
                self._connected.set()
                # Force a full re-announce on the next watcher pass.
                self._announced.clear()
            while not self._stop.is_set():
                message = federation_proto.recv_frame(conn)
                msg_type = message.get("type")
                if msg_type == federation_proto.SOCK_RPC:
                    threading.Thread(target=self._handle_sock_rpc, args=(message,), daemon=True).start()
                elif msg_type == federation_proto.LAUNCH:
                    threading.Thread(target=self._handle_launch, args=(message,), daemon=True).start()
        finally:
            self._close_conn()

    # ------------------------------------------------------------------ #
    # Hub-initiated frames

    def _handle_sock_rpc(self, message: dict[str, Any]) -> None:
        rid = message.get("rid")
        session_id = message.get("session_id")
        request = message.get("request")
        if not isinstance(rid, int) or not isinstance(session_id, str) or not isinstance(request, dict):
            return
        sock_path = self._sock_for_session(session_id)
        try:
            response = call_control_socket(sock_path, request, timeout_s=SOCK_RPC_TIMEOUT_SECONDS)
            reply = federation_proto.sock_result_message(rid=rid, ok=True, response=response)
        except Exception as exc:
            reply = federation_proto.sock_result_message(rid=rid, ok=False, error=f"{type(exc).__name__}: {exc}")
        self._send(reply)

    def _sock_for_session(self, session_id: str) -> Path:
        """Resolve the local sock for a federation key.

        The federation key is the bound session id once known, but the sock
        file keeps the launch-time stem (e.g. broker-<pid>), so map through
        the latest scan; fall back to the literal name for never-renamed
        sessions.
        """
        with self._stems_lock:
            stem = self._stems.get(session_id)
        if stem:
            candidate = self.config.sock_dir / f"{stem}.sock"
            if candidate.exists():
                return candidate
        return self.config.sock_dir / f"{session_id}.sock"

    def _handle_launch(self, message: dict[str, Any]) -> None:
        rid = message.get("rid")
        params = message.get("params")
        if not isinstance(rid, int) or not isinstance(params, dict):
            return
        try:
            result = self._launch_session(params)
            reply = {"type": federation_proto.LAUNCH_RESULT, "rid": rid, "ok": True, "result": dict(result)}
        except Exception as exc:
            reply = {
                "type": federation_proto.LAUNCH_RESULT,
                "rid": rid,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        self._send(reply)

    # ------------------------------------------------------------------ #
    # Local sidecar watcher

    def _scan_sessions(self) -> dict[str, tuple[str, dict[str, Any]]]:
        """stem -> (fingerprint, meta) for every live local sidecar.

        The federation key is the sidecar/sock FILE STEM (e.g. broker-<pid>),
        matching how the server keys sessions (session_id; the meta's
        "session_id" field is the backend thread id and changes at bind time).
        Metas are announced unchanged, including a null session_id pre-bind.
        """
        out: dict[str, tuple[str, dict[str, Any]]] = {}
        stems: dict[str, str] = {}
        try:
            meta_paths = sorted(self.config.sock_dir.glob("*.json"))
        except OSError:
            return out
        for meta_path in meta_paths:
            if meta_path.name.endswith(".json.tmp"):
                continue
            meta = _read_sidecar_meta(meta_path)
            if meta is None:
                continue
            stem = meta_path.stem
            if self._is_mirrored_session(stem):
                continue
            sock_path = self.config.sock_dir / f"{stem}.sock"
            if not sock_path.exists():
                continue
            out[stem] = (_meta_fingerprint(meta), meta)
            stems[stem] = stem
        with self._stems_lock:
            self._stems = stems
        return out

    def _watch_pass(self) -> None:
        current = self._scan_sessions()
        announced = self._announced
        for session_id, (fingerprint, meta) in current.items():
            if announced.get(session_id) == fingerprint:
                continue
            if not self._connected.is_set():
                continue
            if self._send({"type": federation_proto.REGISTER, "session": meta, "key": session_id}):
                announced[session_id] = fingerprint
        for session_id in list(announced):
            if session_id in current:
                continue
            if not self._connected.is_set():
                # Not gone, just unknown until the next connected pass.
                del announced[session_id]
                continue
            if self._send({"type": federation_proto.UNREGISTER, "session_id": session_id}):
                del announced[session_id]

    def _watch_loop(self) -> None:
        while not self._stop.is_set():
            if self._connected.is_set():
                try:
                    self._watch_pass()
                except Exception:
                    # A watcher failure must never kill the connection loop;
                    # the next pass retries the diff from scratch.
                    pass
            self._stop.wait(self.config.watch_poll_s)
