import base64
import gzip
import io
import json
import os
import socket
import tempfile
import threading
import urllib.parse
import unittest
import uuid
from pathlib import Path
from typing import Any
from typing import cast
from unittest.mock import ANY
from unittest.mock import Mock
from unittest.mock import patch

from codoxear import pi_messages
from codoxear.server import Handler
from codoxear.server import Session
from codoxear.server import SessionManager
from codoxear.server import _parse_create_session_request
from codoxear.server import _provider_choice_for_backend
from codoxear.server import _json_response
from codoxear.server import _session_details_payload
from codoxear.server import _session_live_payload
from codoxear.server import _session_workspace_payload
from codoxear.server import _ui_requests_version
from tests.pi_fixtures import pi_persisted_session_file
from tests.pi_fixtures import pi_runtime_session_file


def _pi_message_entry(
    role: str, text: str, *, block_type: str = "output_text"
) -> dict[str, object]:
    return {
        "type": "message",
        "payload": {
            "type": "message",
            "role": role,
            "content": [{"type": block_type, "text": text}],
        },
    }


def _codex_user_message_entry(text: str) -> dict[str, object]:
    return {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
        },
    }


def _write_jsonl(path: Path, entries: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8"
    )


def _make_manager() -> SessionManager:
    mgr = SessionManager.__new__(SessionManager)
    mgr._lock = threading.Lock()
    mgr._bad_sidecars = {}
    mgr._sessions = {}
    mgr._harness = {}
    mgr._aliases = {}
    mgr._sidebar_meta = {}
    mgr._hidden_sessions = set()
    mgr._files = {}
    mgr._queues = {}
    mgr._bridge_events = {}
    mgr._bridge_event_offsets = {}
    mgr._outbound_requests = {}
    mgr._queue_wakeup = threading.Event()
    mgr._pi_commands_cache = {}
    mgr._recent_cwds = {}
    mgr._last_discover_ts = 0.0
    mgr._discover_existing_if_stale = lambda *args, **kwargs: None  # type: ignore[method-assign]
    mgr._prune_dead_sessions = lambda *args, **kwargs: None  # type: ignore[method-assign]
    mgr._update_meta_counters = lambda *args, **kwargs: None  # type: ignore[method-assign]
    mgr._save_recent_cwds = lambda *args, **kwargs: None  # type: ignore[method-assign]
    mgr.refresh_session_meta = lambda *args, **kwargs: None  # type: ignore[method-assign]
    return mgr


class _Proc:
    pid = 2468
    stderr = None

    def wait(self) -> int:
        return 0


class _HandlerHarness:
    def __init__(self, path: str, body: bytes = b"") -> None:
        self.path = path
        self.headers = {"Content-Length": str(len(body))}
        self.rfile = io.BytesIO(body)
        self.wfile = io.BytesIO()
        self.status: int | None = None
        self.sent_headers: list[tuple[str, str]] = []

    def send_response(self, status: int) -> None:
        self.status = status

    def send_header(self, key: str, value: str) -> None:
        self.sent_headers.append((key, value))

    def end_headers(self) -> None:
        return


class _ResettingWriter:
    def write(self, _data: bytes) -> int:
        raise ConnectionResetError(54, "Connection reset by peer")


class TestPiBackendRouting(unittest.TestCase):
    def test_json_response_uses_compact_encoding(self) -> None:
        handler = _HandlerHarness("/api/test")

        _json_response(cast(Any, handler), 200, {"ok": True, "nested": {"value": 1}})

        self.assertEqual(handler.status, 200)
        self.assertEqual(
            handler.wfile.getvalue().decode("utf-8"),
            '{"ok":true,"nested":{"value":1}}',
        )

    def test_json_response_gzips_when_client_accepts_it(self) -> None:
        handler = _HandlerHarness("/api/test")
        handler.headers["Accept-Encoding"] = "br, gzip"

        _json_response(cast(Any, handler), 200, {"ok": True})

        headers = dict(handler.sent_headers)
        self.assertEqual(headers.get("Content-Encoding"), "gzip")
        body = gzip.decompress(handler.wfile.getvalue()).decode("utf-8")
        self.assertEqual(body, '{"ok":true}')

    def test_delete_route_decodes_historical_session_id(self) -> None:
        mgr = _make_manager()
        mgr.delete_session = lambda session_id: session_id == "history:pi:resume-1"  # type: ignore[method-assign]
        encoded = urllib.parse.quote("history:pi:resume-1", safe="")
        handler = _HandlerHarness(f"/api/sessions/{encoded}/delete", body=b"{}")

        with (
            patch("codoxear.server._require_auth", return_value=True),
            patch("codoxear.server.MANAGER", mgr),
        ):
            Handler.do_POST(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(handler.status, 200)
        self.assertEqual(payload, {"ok": True})

    def test_get_session_commands_caches_pi_command_results(self) -> None:
        mgr = _make_manager()
        sock = Path("/tmp/pi-session.sock")
        session_path = Path("/tmp/pi-session.jsonl")
        mgr._sessions["pi-session"] = Session(
            session_id="pi-session",
            thread_id="pi-thread-001",
            agent_backend="pi",
            backend="pi",
            broker_pid=3333,
            codex_pid=4444,
            owned=True,
            start_ts=123.0,
            cwd="/tmp/pi-cwd",
            log_path=None,
            sock_path=sock,
            session_path=session_path,
        )

        with (
            patch.object(
                mgr,
                "_sock_call",
                return_value={
                    "commands": [
                        {"name": "reload", "description": "Reload Pi"},
                        {"name": "resume", "description": "Resume session"},
                    ]
                },
            ) as sock_call,
            patch("codoxear.server.time.time", side_effect=[100.0, 100.0, 110.0]),
        ):
            first = mgr.get_session_commands("pi-session")
            second = mgr.get_session_commands("pi-session")

        self.assertEqual(
            first,
            {
                "commands": [
                    {"name": "reload", "description": "Reload Pi"},
                    {"name": "resume", "description": "Resume session"},
                ]
            },
        )
        self.assertEqual(second, first)
        sock_call.assert_called_once_with(sock, {"cmd": "commands"}, timeout_s=2.0)

    def test_commands_endpoint_returns_pi_commands(self) -> None:
        handler = _HandlerHarness("/api/sessions/pi-session/commands")

        with (
            patch("codoxear.server.MANAGER") as manager,
            patch("codoxear.server._require_auth", return_value=True),
        ):
            manager.get_session_commands.return_value = {
                "commands": [{"name": "reload", "description": "Reload Pi"}]
            }

            Handler.do_GET(handler)  # type: ignore[arg-type]

        self.assertEqual(handler.status, 200)
        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(
            payload,
            {"commands": [{"name": "reload", "description": "Reload Pi"}]},
        )
        manager.get_session_commands.assert_called_once_with("pi-session")

    def test_get_session_commands_legacy_broker_unknown_cmd_falls_back_to_empty(
        self,
    ) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            sock = Path(td) / "pi.sock"
            sock.touch()
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=123.0,
                cwd="/tmp",
                log_path=None,
                sock_path=sock,
                session_path=Path(td) / "pi-session.jsonl",
                transport="pi-rpc",
                supports_live_ui=True,
                ui_protocol_version=1,
            )

            def _sock_call(
                _sock: Path, req: dict[str, object], timeout_s: float = 0.0
            ) -> dict[str, object]:
                self.assertEqual(req, {"cmd": "commands"})
                self.assertEqual(timeout_s, 2.0)
                return {"error": "unknown cmd"}

            mgr._sock_call = _sock_call  # type: ignore[method-assign]

            payload = mgr.get_session_commands("pi-session")

        self.assertEqual(payload, {"commands": []})

    def test_commands_endpoint_legacy_broker_unknown_cmd_returns_empty(self) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            sock = Path(td) / "pi.sock"
            sock.touch()
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=123.0,
                cwd="/tmp",
                log_path=None,
                sock_path=sock,
                session_path=Path(td) / "pi-session.jsonl",
                transport="pi-rpc",
                supports_live_ui=True,
                ui_protocol_version=1,
            )

            def _sock_call(
                _sock: Path, req: dict[str, object], timeout_s: float = 0.0
            ) -> dict[str, object]:
                self.assertEqual(req, {"cmd": "commands"})
                self.assertEqual(timeout_s, 2.0)
                return {"error": "unknown cmd"}

            mgr._sock_call = _sock_call  # type: ignore[method-assign]
            handler = _HandlerHarness("/api/sessions/pi-session/commands")

            with (
                patch("codoxear.server._require_auth", return_value=True),
                patch("codoxear.server.MANAGER", mgr),
            ):
                Handler.do_GET(handler)  # type: ignore[arg-type]

        self.assertEqual(handler.status, 200)
        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(payload, {"commands": []})

    def test_discover_existing_registers_live_pi_bootstrap_sidecars_on_slow_state_probe(
        self,
    ) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            sock_dir = Path(td)
            sock = sock_dir / "pi-session.sock"
            sock.touch()
            session_path = sock_dir / "pi-session.jsonl"
            meta_path = sock_dir / "pi-session.json"
            meta_path.write_text(
                json.dumps(
                    {
                        "session_id": "pi-thread-001",
                        "backend": "pi",
                        "owner": "web",
                        "broker_pid": 3333,
                        "codex_pid": 4444,
                        "cwd": "/tmp/pi-cwd",
                        "start_ts": 123.0,
                        "session_path": str(session_path),
                        "sock_path": str(sock),
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch("codoxear.server.SOCK_DIR", sock_dir),
                patch.object(
                    mgr, "_sock_call", side_effect=socket.timeout("slow bootstrap")
                ),
                patch("codoxear.server._pid_alive", return_value=True),
                patch("codoxear.server.time.time", return_value=133.0),
            ):
                mgr._discover_existing(force=True)

            self.assertTrue(sock.exists())
            self.assertTrue(meta_path.exists())

        session = mgr.get_session("pi-session")
        assert session is not None
        self.assertFalse(session.busy)

    def test_discover_existing_registers_refused_socket_sidecar_when_pids_alive(
        self,
    ) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            sock_dir = Path(td)
            sock = sock_dir / "pi-session.sock"
            sock.touch()
            session_path = sock_dir / "pi-session.jsonl"
            meta_path = sock_dir / "pi-session.json"
            meta_path.write_text(
                json.dumps(
                    {
                        "session_id": "pi-thread-001",
                        "backend": "pi",
                        "owner": "web",
                        "broker_pid": 3333,
                        "codex_pid": 4444,
                        "cwd": "/tmp/pi-cwd",
                        "start_ts": 123.0,
                        "session_path": str(session_path),
                        "sock_path": str(sock),
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch("codoxear.server.SOCK_DIR", sock_dir),
                patch.object(
                    mgr,
                    "_sock_call",
                    side_effect=ConnectionRefusedError("stale listener"),
                ),
                patch("codoxear.server._pid_alive", return_value=True),
            ):
                mgr._discover_existing(force=True, skip_invalid_sidecars=True)

            self.assertTrue(sock.exists())
            self.assertTrue(meta_path.exists())
            self.assertFalse(mgr._sidecar_is_quarantined(sock))

        session = mgr.get_session("pi-session")
        assert session is not None
        self.assertFalse(session.busy)

    def test_discover_existing_ignores_socket_without_metadata_sidecar(self) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            sock_dir = Path(td)
            sock = sock_dir / "pi-session.sock"
            sock.touch()

            with (
                patch("codoxear.server.SOCK_DIR", sock_dir),
                patch("codoxear.server.sys.stderr.write") as stderr_write,
                patch("codoxear.server.sys.stderr.flush") as stderr_flush,
            ):
                mgr._discover_existing(force=True, skip_invalid_sidecars=True)

            self.assertTrue(sock.exists())
            self.assertFalse(mgr._sidecar_is_quarantined(sock))
            stderr_write.assert_not_called()
            stderr_flush.assert_not_called()

        self.assertIsNone(mgr.get_session("pi-session"))

    def test_discover_existing_refreshes_session_path_for_tracked_pi_session(
        self,
    ) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            sock_dir = Path(td)
            sock = sock_dir / "pi-session.sock"
            sock.touch()
            old_session_path = sock_dir / "old-pi-session.jsonl"
            new_session_path = sock_dir / "new-pi-session.jsonl"
            old_session_path.write_text(
                '{"session_id":"pi-thread-001"}\n', encoding="utf-8"
            )
            new_session_path.write_text(
                '{"session_id":"pi-thread-002"}\n', encoding="utf-8"
            )
            (sock_dir / "pi-session.json").write_text(
                json.dumps(
                    {
                        "session_id": "pi-thread-002",
                        "backend": "pi",
                        "owner": "web",
                        "broker_pid": 3333,
                        "codex_pid": 4444,
                        "cwd": "/tmp/pi-cwd",
                        "start_ts": 123.0,
                        "session_path": str(new_session_path),
                        "sock_path": str(sock),
                    }
                ),
                encoding="utf-8",
            )
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=123.0,
                cwd=td,
                log_path=None,
                sock_path=sock,
                session_path=old_session_path,
            )

            with (
                patch("codoxear.server.SOCK_DIR", sock_dir),
                patch.object(
                    mgr,
                    "_sock_call",
                    return_value={"busy": False, "queue_len": 0, "token": None},
                ),
                patch("codoxear.server._pid_alive", return_value=True),
            ):
                mgr._discover_existing(force=True)

        session = mgr.get_session("pi-session")
        self.assertIsNotNone(session)
        assert session is not None
        self.assertEqual(session.thread_id, "pi-thread-002")
        self.assertEqual(session.session_path, new_session_path)

    def test_discover_existing_prefers_exact_pi_thread_match_over_same_cwd_discovery(
        self,
    ) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            sock_dir = Path(td)
            sock = sock_dir / "pi-session.sock"
            sock.touch()
            exact_session_path = sock_dir / "exact-pi-session.jsonl"
            wrong_session_path = sock_dir / "wrong-pi-session.jsonl"
            exact_session_path.write_text(
                '{"type":"session","id":"pi-thread-exact","cwd":"/tmp/pi-cwd"}\n',
                encoding="utf-8",
            )
            wrong_session_path.write_text(
                '{"type":"session","id":"pi-thread-wrong","cwd":"/tmp/pi-cwd"}\n',
                encoding="utf-8",
            )
            (sock_dir / "pi-session.json").write_text(
                json.dumps(
                    {
                        "session_id": "pi-thread-exact",
                        "backend": "pi",
                        "owner": "web",
                        "broker_pid": 3333,
                        "codex_pid": 4444,
                        "cwd": "/tmp/pi-cwd",
                        "start_ts": 123.0,
                        "sock_path": str(sock),
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch("codoxear.server.SOCK_DIR", sock_dir),
                patch.object(
                    mgr,
                    "_sock_call",
                    return_value={"busy": False, "queue_len": 0, "token": None},
                ),
                patch("codoxear.server._pid_alive", return_value=True),
                patch(
                    "codoxear.server._find_session_log_for_session_id",
                    return_value=exact_session_path,
                ) as find_exact,
                patch(
                    "codoxear.server._discover_pi_session_for_cwd",
                    return_value=wrong_session_path,
                ) as discover,
                patch("codoxear.server._patch_metadata_session_path") as patch_meta,
            ):
                mgr._discover_existing(force=True)

        session = mgr.get_session("pi-session")
        self.assertIsNotNone(session)
        assert session is not None
        self.assertEqual(session.thread_id, "pi-thread-exact")
        self.assertEqual(session.session_path, exact_session_path)
        find_exact.assert_called()
        discover.assert_not_called()
        patch_meta.assert_called_once()
        self.assertEqual(patch_meta.call_args.args[:2], (sock, exact_session_path))

    def test_discover_existing_keeps_pi_session_file_out_of_log_path(self) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            sock_dir = Path(td)
            sock = sock_dir / "pi-session.sock"
            sock.touch()
            pi_session = sock_dir / "pi-session.jsonl"
            pi_session.write_text('{"session_id":"pi-thread-001"}\n', encoding="utf-8")
            (sock_dir / "pi-session.json").write_text(
                json.dumps(
                    {
                        "session_id": "pi-thread-001",
                        "backend": "pi",
                        "owner": "web",
                        "broker_pid": 3333,
                        "codex_pid": 4444,
                        "cwd": "/tmp/pi-cwd",
                        "start_ts": 123.0,
                        "session_path": str(pi_session),
                        "sock_path": str(sock),
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch("codoxear.server.SOCK_DIR", sock_dir),
                patch.object(
                    mgr,
                    "_sock_call",
                    return_value={"busy": False, "queue_len": 0, "token": None},
                ),
                patch("codoxear.server._pid_alive", return_value=True),
                patch(
                    "codoxear.server._coerce_main_thread_log",
                    side_effect=AssertionError(
                        "pi sidecars must not trigger rollout coercion"
                    ),
                ),
                patch(
                    "codoxear.server._read_run_settings_from_log",
                    side_effect=AssertionError(
                        "pi sidecars must not trigger rollout parsing"
                    ),
                ),
            ):
                mgr._discover_existing(force=True)

        session = mgr.get_session("pi-session")
        self.assertIsNotNone(session)
        assert session is not None
        self.assertIsNone(session.log_path)

    def test_discover_existing_reads_backend_from_sidecar(self) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            sock_dir = Path(td)
            sock = sock_dir / "pi-session.sock"
            sock.touch()
            session_path = sock_dir / "pi-session.jsonl"
            (sock_dir / "pi-session.json").write_text(
                json.dumps(
                    {
                        "session_id": "pi-thread-001",
                        "backend": "pi",
                        "owner": "web",
                        "broker_pid": 3333,
                        "codex_pid": 4444,
                        "cwd": "/tmp/pi-cwd",
                        "start_ts": 123.0,
                        "session_path": str(session_path),
                        "log_path": None,
                        "sock_path": str(sock),
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch("codoxear.server.SOCK_DIR", sock_dir),
                patch.object(
                    mgr,
                    "_sock_call",
                    return_value={"busy": False, "queue_len": 0, "token": None},
                ),
                patch("codoxear.server._pid_alive", return_value=True),
            ):
                mgr._discover_existing(force=True)

        session = mgr.get_session("pi-session")
        self.assertIsNotNone(session)
        assert session is not None
        self.assertEqual(session.backend, "pi")
        self.assertEqual(session.codex_pid, 4444)

        rows = mgr.list_sessions()
        self.assertEqual(rows[0]["backend"], "pi")
        self.assertEqual(rows[0]["pid"], 4444)

    def test_discover_existing_tolerates_missing_pi_session_path(self) -> None:
        """PTY-wrapped pi sessions (piox) have no session_path in sidecar.

        The discover loop should not raise and should attempt auto-discovery
        via _discover_pi_session_for_cwd instead.
        """
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            sock_dir = Path(td)
            sock = sock_dir / "pi-session.sock"
            sock.touch()
            (sock_dir / "pi-session.json").write_text(
                json.dumps(
                    {
                        "session_id": "pi-thread-001",
                        "backend": "pi",
                        "owner": "web",
                        "broker_pid": os.getpid(),
                        "codex_pid": os.getpid(),
                        "cwd": "/tmp/pi-cwd",
                        "start_ts": 123.0,
                        "sock_path": str(sock),
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch("codoxear.server.SOCK_DIR", sock_dir),
                patch.object(
                    mgr, "_sock_call", return_value={"busy": False, "queue_len": 0}
                ),
                patch(
                    "codoxear.server._discover_pi_session_for_cwd", return_value=None
                ),
            ):
                mgr._discover_existing(force=True)
                session = mgr.get_session("pi-session")
                self.assertIsNotNone(session)
                assert session is not None
                self.assertEqual(session.backend, "pi")
                self.assertIsNone(session.session_path)

    def test_discover_existing_recovers_pi_backend_from_open_pi_log_when_sidecar_says_codex(
        self,
    ) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            sock_dir = Path(td)
            sock = sock_dir / "pi-session.sock"
            sock.touch()
            session_path = sock_dir / "pi-session.jsonl"
            session_path.write_text(
                '{"type":"session","id":"pi-thread-001","cwd":"/tmp/pi-cwd"}\n',
                encoding="utf-8",
            )
            meta_path = sock_dir / "pi-session.json"
            meta_path.write_text(
                json.dumps(
                    {
                        "session_id": "pi-thread-001",
                        "backend": "codex",
                        "agent_backend": "codex",
                        "owner": "",
                        "supports_web_control": True,
                        "broker_pid": os.getpid(),
                        "codex_pid": os.getpid(),
                        "cwd": "/tmp/pi-cwd",
                        "start_ts": 123.0,
                        "log_path": None,
                        "sock_path": str(sock),
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch("codoxear.server.SOCK_DIR", sock_dir),
                patch.object(
                    mgr, "_sock_call", return_value={"busy": False, "queue_len": 0}
                ),
                patch("codoxear.server._pid_alive", return_value=True),
                patch(
                    "codoxear.server._proc_find_open_rollout_log",
                    return_value=session_path,
                ),
            ):
                mgr._discover_existing(force=True)

            session = mgr.get_session("pi-session")
            self.assertIsNotNone(session)
            assert session is not None
            self.assertEqual(session.backend, "pi")
            self.assertEqual(session.agent_backend, "pi")
            self.assertEqual(session.session_path, session_path)

            persisted = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["backend"], "pi")
            self.assertEqual(persisted["agent_backend"], "pi")
            self.assertEqual(persisted["session_path"], str(session_path))

    def test_list_sessions_includes_supported_pi_sidecar_without_session_path(
        self,
    ) -> None:
        """A brokered pi sidecar without session_path is valid (PTY-wrapped piox)."""
        mgr = _make_manager()
        mgr._discover_existing_if_stale = (
            SessionManager._discover_existing_if_stale.__get__(mgr, SessionManager)
        )  # type: ignore[method-assign]
        with tempfile.TemporaryDirectory() as td:
            sock_dir = Path(td)
            good_sock = sock_dir / "good-session.sock"
            pty_sock = sock_dir / "pty-session.sock"
            good_sock.touch()
            pty_sock.touch()
            good_session_path = sock_dir / "good-session.jsonl"
            good_session_path.write_text(
                '{"session_id":"pi-thread-good"}\n', encoding="utf-8"
            )
            (sock_dir / "good-session.json").write_text(
                json.dumps(
                    {
                        "session_id": "pi-thread-good",
                        "backend": "pi",
                        "owner": "web",
                        "broker_pid": os.getpid(),
                        "codex_pid": os.getpid(),
                        "cwd": "/tmp/pi-cwd",
                        "start_ts": 123.0,
                        "session_path": str(good_session_path),
                        "sock_path": str(good_sock),
                    }
                ),
                encoding="utf-8",
            )
            # PTY-wrapped pi session: no session_path in sidecar
            (sock_dir / "pty-session.json").write_text(
                json.dumps(
                    {
                        "session_id": "pi-thread-pty",
                        "backend": "pi",
                        "owner": "",
                        "supports_web_control": True,
                        "broker_pid": os.getpid(),
                        "codex_pid": os.getpid(),
                        "cwd": "/tmp/pi-cwd",
                        "start_ts": 456.0,
                        "sock_path": str(pty_sock),
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch("codoxear.server.SOCK_DIR", sock_dir),
                patch.object(
                    mgr,
                    "_sock_call",
                    return_value={"busy": False, "queue_len": 0, "token": None},
                ),
                patch(
                    "codoxear.server._discover_pi_session_for_cwd", return_value=None
                ),
            ):
                rows = mgr.list_sessions()

            sids = sorted(row["session_id"] for row in rows)
            self.assertEqual(sids, ["pi-thread-good", "pi-thread-pty"])

    def test_list_sessions_uses_attention_timestamps_not_pi_file_mtime(self) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            older_path = Path(td) / "older.jsonl"
            newer_path = Path(td) / "newer.jsonl"
            older_path.write_text('{"type":"session","id":"older"}\n', encoding="utf-8")
            newer_path.write_text('{"type":"session","id":"newer"}\n', encoding="utf-8")
            os.utime(older_path, (10_000.0, 10_000.0))
            os.utime(newer_path, (9_000.0, 9_000.0))
            mgr._sessions = {
                "older-runtime": Session(
                    session_id="older-runtime",
                    thread_id="older-session",
                    agent_backend="pi",
                    backend="pi",
                    broker_pid=os.getpid(),
                    codex_pid=os.getpid(),
                    owned=True,
                    start_ts=100.0,
                    cwd="/tmp/older",
                    log_path=None,
                    sock_path=Path(td) / "older.sock",
                    session_path=older_path,
                    last_chat_ts=100.0,
                    last_chat_history_scanned=True,
                ),
                "newer-runtime": Session(
                    session_id="newer-runtime",
                    thread_id="newer-session",
                    agent_backend="pi",
                    backend="pi",
                    broker_pid=os.getpid(),
                    codex_pid=os.getpid(),
                    owned=True,
                    start_ts=200.0,
                    cwd="/tmp/newer",
                    log_path=None,
                    sock_path=Path(td) / "newer.sock",
                    session_path=newer_path,
                    last_chat_ts=200.0,
                    last_chat_history_scanned=True,
                ),
            }

            with patch("codoxear.server._current_git_branch", return_value=None):
                rows = mgr.list_sessions()

        self.assertEqual([row["session_id"] for row in rows[:2]], ["newer-session", "older-session"])

    def test_list_sessions_hides_unsupported_terminal_pi_sidecar(self) -> None:
        mgr = _make_manager()
        mgr._discover_existing_if_stale = (
            SessionManager._discover_existing_if_stale.__get__(mgr, SessionManager)
        )  # type: ignore[method-assign]
        with tempfile.TemporaryDirectory() as td:
            sock_dir = Path(td)
            pty_sock = sock_dir / "pty-session.sock"
            pty_sock.touch()
            (sock_dir / "pty-session.json").write_text(
                json.dumps(
                    {
                        "session_id": "pi-thread-pty",
                        "backend": "pi",
                        "owner": "",
                        "broker_pid": os.getpid(),
                        "codex_pid": os.getpid(),
                        "cwd": "/tmp/pi-cwd",
                        "start_ts": 456.0,
                        "sock_path": str(pty_sock),
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch("codoxear.server.SOCK_DIR", sock_dir),
                patch.object(
                    mgr,
                    "_sock_call",
                    return_value={"busy": False, "queue_len": 0, "token": None},
                ),
                patch(
                    "codoxear.server._discover_pi_session_for_cwd", return_value=None
                ),
            ):
                rows = mgr.list_sessions()

            self.assertEqual(rows, [])

    def test_discover_existing_skips_invalid_service_tier_sidecar_when_requested(
        self,
    ) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            sock_dir = Path(td)
            good_sock = sock_dir / "good-session.sock"
            bad_sock = sock_dir / "bad-session.sock"
            good_sock.touch()
            bad_sock.touch()
            good_session_path = sock_dir / "good-session.jsonl"
            bad_session_path = sock_dir / "bad-session.jsonl"
            good_session_path.write_text(
                '{"session_id":"pi-thread-good"}\n', encoding="utf-8"
            )
            bad_session_path.write_text(
                '{"session_id":"pi-thread-bad"}\n', encoding="utf-8"
            )
            (sock_dir / "good-session.json").write_text(
                json.dumps(
                    {
                        "session_id": "pi-thread-good",
                        "backend": "pi",
                        "owner": "web",
                        "broker_pid": 3333,
                        "codex_pid": 4444,
                        "cwd": td,
                        "start_ts": 123.0,
                        "session_path": str(good_session_path),
                        "service_tier": "fast",
                        "sock_path": str(good_sock),
                    }
                ),
                encoding="utf-8",
            )
            (sock_dir / "bad-session.json").write_text(
                json.dumps(
                    {
                        "session_id": "pi-thread-bad",
                        "backend": "pi",
                        "owner": "web",
                        "broker_pid": 5555,
                        "codex_pid": 6666,
                        "cwd": td,
                        "start_ts": 456.0,
                        "session_path": str(bad_session_path),
                        "service_tier": "ultra",
                        "sock_path": str(bad_sock),
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch("codoxear.server.SOCK_DIR", sock_dir),
                patch.object(
                    mgr,
                    "_sock_call",
                    return_value={"busy": False, "queue_len": 0, "token": None},
                ),
                patch("codoxear.server._pid_alive", return_value=True),
            ):
                mgr._discover_existing(force=True, skip_invalid_sidecars=True)

            rows = mgr.list_sessions()
            self.assertEqual([row["session_id"] for row in rows], ["pi-thread-good"])
            self.assertTrue(mgr._sidecar_is_quarantined(bad_sock))

            with (
                patch("codoxear.server.SOCK_DIR", sock_dir),
                patch("codoxear.server._pid_alive", return_value=True),
            ):
                with self.assertRaisesRegex(
                    ValueError, "service_tier must be one of fast, flex"
                ):
                    mgr._discover_existing(force=True)

    def test_discover_existing_skips_invalid_run_settings_sidecar_when_requested(
        self,
    ) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            sock_dir = Path(td)
            good_sock = sock_dir / "good-session.sock"
            bad_sock = sock_dir / "bad-session.sock"
            good_sock.touch()
            bad_sock.touch()
            good_session_path = sock_dir / "good-session.jsonl"
            bad_session_path = sock_dir / "bad-session.jsonl"
            good_session_path.write_text(
                '{"session_id":"pi-thread-good"}\n', encoding="utf-8"
            )
            bad_session_path.write_text(
                '{"session_id":"pi-thread-bad"}\n', encoding="utf-8"
            )
            (sock_dir / "good-session.json").write_text(
                json.dumps(
                    {
                        "session_id": "pi-thread-good",
                        "backend": "pi",
                        "owner": "web",
                        "broker_pid": 3333,
                        "codex_pid": 4444,
                        "cwd": td,
                        "start_ts": 123.0,
                        "session_path": str(good_session_path),
                        "preferred_auth_method": "apikey",
                        "sock_path": str(good_sock),
                    }
                ),
                encoding="utf-8",
            )
            (sock_dir / "bad-session.json").write_text(
                json.dumps(
                    {
                        "session_id": "pi-thread-bad",
                        "backend": "pi",
                        "owner": "web",
                        "broker_pid": 5555,
                        "codex_pid": 6666,
                        "cwd": td,
                        "start_ts": 456.0,
                        "session_path": str(bad_session_path),
                        "preferred_auth_method": "oauth",
                        "sock_path": str(bad_sock),
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch("codoxear.server.SOCK_DIR", sock_dir),
                patch.object(
                    mgr,
                    "_sock_call",
                    return_value={"busy": False, "queue_len": 0, "token": None},
                ),
                patch("codoxear.server._pid_alive", return_value=True),
            ):
                mgr._discover_existing(force=True, skip_invalid_sidecars=True)

            rows = mgr.list_sessions()
            self.assertEqual([row["session_id"] for row in rows], ["pi-thread-good"])
            self.assertTrue(mgr._sidecar_is_quarantined(bad_sock))

            with (
                patch("codoxear.server.SOCK_DIR", sock_dir),
                patch("codoxear.server._pid_alive", return_value=True),
            ):
                with self.assertRaisesRegex(
                    ValueError, "preferred_auth_method must be one of chatgpt, apikey"
                ):
                    mgr._discover_existing(force=True)

    def test_prune_dead_sessions_keeps_live_session_on_stale_probe_error(self) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            sock = Path(td) / "pi-session.sock"
            sock.touch()
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=123.0,
                cwd=td,
                log_path=None,
                sock_path=sock,
                session_path=Path(td) / "pi-session.jsonl",
            )
            mgr._clear_deleted_session_state = lambda _sid: None  # type: ignore[method-assign]

            with (
                patch.object(
                    mgr,
                    "_refresh_session_state",
                    return_value=(False, ConnectionRefusedError("socket not ready")),
                ),
                patch("codoxear.server._pid_alive", return_value=True),
            ):
                mgr._prune_dead_sessions()

            self.assertTrue(sock.exists())

        self.assertIn("pi-session", mgr._sessions)

    def test_pi_session_rows_do_not_invent_provider_choice(self) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            sock = Path(td) / "pi.sock"
            sock.touch()
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                backend="pi",
                agent_backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=123.0,
                cwd=td,
                log_path=None,
                sock_path=sock,
            )

            row = mgr.list_sessions()[0]

        self.assertIsNone(row["provider_choice"])
        self.assertIsNone(
            _provider_choice_for_backend(
                backend="pi", model_provider=None, preferred_auth_method=None
            )
        )

    def test_spawn_web_session_dispatches_to_pi_backend(self) -> None:
        manager = SessionManager.__new__(SessionManager)

        session_uuid = uuid.UUID("11111111-2222-3333-4444-555555555555")
        with (
            tempfile.TemporaryDirectory() as td,
            patch(
                "codoxear.server.subprocess.Popen", return_value=_Proc()
            ) as popen_mock,
            patch("codoxear.server._now", return_value=1774708716.099),
            patch("codoxear.server.uuid.uuid4", return_value=session_uuid),
            patch.object(SessionManager, "_persist_durable_session_record", lambda *_args, **_kwargs: None),
            patch.object(threading.Thread, "start", lambda self: None),
        ):
            result = SessionManager.spawn_web_session(manager, cwd=td, backend="pi")

        argv = popen_mock.call_args.args[0]
        expected_session_path = (
            Path.home()
            / ".pi"
            / "agent"
            / "sessions"
            / f"--{str(Path(td).resolve()).strip('/').replace('/', '-')}--"
            / "2026-03-28T14-38-36-099Z_11111111-2222-3333-4444-555555555555.jsonl"
        )
        self.assertEqual(argv[:4], [ANY, "-m", "codoxear.pi_broker", "--cwd"])
        self.assertEqual(Path(argv[4]).resolve(), Path(td).resolve())
        self.assertEqual(argv[5:7], ["--session-file", str(expected_session_path)])
        self.assertEqual(argv[7], "--")
        self.assertEqual(argv[8], "-e")
        self.assertEqual(
            result,
            {
                "backend": "pi",
                "session_id": str(session_uuid),
                "runtime_id": None,
                "pending_startup": True,
            },
        )

    def test_spawn_web_session_adds_codoxear_ask_user_bridge_extension_for_pi(
        self,
    ) -> None:
        manager = SessionManager.__new__(SessionManager)

        session_uuid = uuid.UUID("11111111-2222-3333-4444-555555555555")
        with (
            tempfile.TemporaryDirectory() as td,
            patch(
                "codoxear.server.subprocess.Popen", return_value=_Proc()
            ) as popen_mock,
            patch("codoxear.server._now", return_value=1774708716.099),
            patch("codoxear.server.uuid.uuid4", return_value=session_uuid),
            patch.object(SessionManager, "_persist_durable_session_record", lambda *_args, **_kwargs: None),
            patch.object(threading.Thread, "start", lambda self: None),
        ):
            SessionManager.spawn_web_session(manager, cwd=td, backend="pi")

        argv = popen_mock.call_args.args[0]
        self.assertIn("-e", argv)
        extension_index = argv.index("-e")
        self.assertEqual(
            Path(argv[extension_index + 1]).resolve(),
            (
                Path(__file__).resolve().parents[1]
                / "codoxear"
                / "pi_extensions"
                / "ask_user_bridge.ts"
            ).resolve(),
        )

    def test_create_session_defaults_to_codex_backend(self) -> None:
        payload = _parse_create_session_request({"cwd": "/tmp/codex-cwd"})

        self.assertEqual(payload["backend"], "codex")
        self.assertEqual(payload["cwd"], "/tmp/codex-cwd")
        self.assertIsNone(payload["name"])

    def test_create_session_preserves_optional_name(self) -> None:
        payload = _parse_create_session_request(
            {"cwd": "/tmp/codex-cwd", "name": "Inbox cleanup"}
        )

        self.assertEqual(payload["name"], "Inbox cleanup")

    def test_create_session_preserves_pi_fields_and_ignores_codex_only_fields(
        self,
    ) -> None:
        payload = _parse_create_session_request(
            {
                "cwd": "/tmp/pi-cwd",
                "backend": "pi",
                "model_provider": "openai",
                "preferred_auth_method": "oauth",
                "reasoning_effort": "high",
                "service_tier": "ultra",
                "create_in_tmux": True,
                "resume_session_id": "resume-me",
            }
        )

        self.assertEqual(payload["backend"], "pi")
        self.assertEqual(payload["cwd"], "/tmp/pi-cwd")
        self.assertEqual(payload["model_provider"], "openai")
        self.assertIsNone(payload["preferred_auth_method"])
        self.assertEqual(payload["reasoning_effort"], "high")
        self.assertIsNone(payload["service_tier"])
        self.assertTrue(payload["create_in_tmux"])
        self.assertEqual(payload["resume_session_id"], "resume-me")

    def test_create_session_rejects_non_string_resume_session_id_for_pi_backend(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "resume_session_id must be a string"):
            _parse_create_session_request(
                {
                    "cwd": "/tmp/pi-cwd",
                    "backend": "pi",
                    "resume_session_id": 123,
                }
            )

    def test_create_session_route_sets_alias_atomically(self) -> None:
        mgr = _make_manager()
        mgr.spawn_web_session = Mock(
            return_value={
                "session_id": "pending-pi-session",
                "runtime_id": None,
                "backend": "pi",
                "pending_startup": True,
            }
        )
        mgr.set_created_session_name = Mock(return_value="Inbox cleanup")
        handler = _HandlerHarness(
            "/api/sessions",
            body=json.dumps(
                {
                    "cwd": "/tmp/project",
                    "backend": "pi",
                    "name": "Inbox cleanup",
                }
            ).encode("utf-8"),
        )

        with (
            patch("codoxear.server._require_auth", return_value=True),
            patch("codoxear.server.MANAGER", mgr),
        ):
            Handler.do_POST(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(handler.status, 200)
        self.assertEqual(payload["alias"], "Inbox cleanup")
        mgr.set_created_session_name.assert_called_once_with(
            session_id="pending-pi-session",
            runtime_id=None,
            backend="pi",
            name="Inbox cleanup",
        )

    def test_delete_session_uses_shutdown_for_pi_backend(self) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            sock = Path(td) / "pi.sock"
            sock.touch()
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                backend="pi",
                agent_backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=123.0,
                cwd="/tmp/pi-cwd",
                log_path=None,
                sock_path=sock,
            )
            mgr.files_clear = lambda _sid: None  # type: ignore[method-assign]
            mgr._clear_deleted_session_state = lambda _sid: None  # type: ignore[method-assign]
            mgr._sock_call = lambda *_args, **_kwargs: {"ok": True}  # type: ignore[method-assign]

            self.assertTrue(mgr.delete_session("pi-session"))

    def test_spawn_web_session_can_start_pi_in_tmux(self) -> None:
        manager = SessionManager.__new__(SessionManager)

        with (
            tempfile.TemporaryDirectory() as td,
            patch("codoxear.server.shutil.which", return_value="/usr/bin/tmux"),
            patch(
                "codoxear.server.subprocess.run",
                side_effect=[
                    __import__("subprocess").CompletedProcess(
                        ["/usr/bin/tmux", "has-session", "-t", "codoxear"],
                        1,
                        stdout="",
                        stderr="",
                    ),
                    __import__("subprocess").CompletedProcess(
                        ["/usr/bin/tmux", "new-session"],
                        0,
                        stdout="%8\n",
                        stderr="",
                    ),
                ],
            ) as run_mock,
            patch("codoxear.server.uuid.uuid4", return_value=uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")),
            patch.object(SessionManager, "_persist_durable_session_record", lambda *_args, **_kwargs: None),
            patch.object(threading.Thread, "start", lambda self: None),
        ):
            result = SessionManager.spawn_web_session(
                manager,
                cwd=td,
                backend="pi",
                create_in_tmux=True,
            )

        self.assertEqual(
            result,
            {
                "backend": "pi",
                "session_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                "runtime_id": None,
                "pending_startup": True,
                "tmux_session": "codoxear",
                "tmux_window": ANY,
            },
        )
        shell_cmd = run_mock.call_args_list[1].args[0][-1]
        self.assertIn("CODEX_WEB_AGENT_BACKEND=pi", shell_cmd)
        self.assertIn("CODEX_WEB_TRANSPORT=tmux", shell_cmd)
        self.assertIn("CODEX_WEB_TMUX_SESSION=codoxear", shell_cmd)
        self.assertIn("CODEX_WEB_TMUX_WINDOW=", shell_cmd)
        self.assertIn("codoxear.pi_broker", shell_cmd)
        self.assertNotIn("broker_pid", result)

    def test_interrupt_routes_through_keys_for_pi_backend(self) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            sock = Path(td) / "pi.sock"
            sock.touch()
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=123.0,
                cwd="/tmp/pi-cwd",
                log_path=None,
                sock_path=sock,
            )
            mgr._sock_call = lambda _sock, req, timeout_s=0.0: (
                {"ok": True, "queued": False, "n": 1} if req["cmd"] == "keys" else {}
            )  # type: ignore[method-assign]

            resp = mgr.inject_keys("pi-session", "\\x1b")

        self.assertEqual(resp, {"ok": True, "queued": False, "n": 1})

    def test_interrupt_raises_when_pi_broker_rejects_abort(self) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            sock = Path(td) / "pi.sock"
            sock.touch()
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=123.0,
                cwd="/tmp/pi-cwd",
                log_path=None,
                sock_path=sock,
            )
            mgr._sock_call = lambda _sock, req, timeout_s=0.0: (
                {"error": "abort failed"} if req["cmd"] == "keys" else {}
            )  # type: ignore[method-assign]

            with self.assertRaisesRegex(ValueError, "abort failed"):
                mgr.inject_keys("pi-session", "\\x1b")

    def test_interrupt_endpoint_returns_broker_key_error(self) -> None:
        handler = _HandlerHarness("/api/sessions/pi-session/interrupt")

        with (
            patch("codoxear.server._require_auth", return_value=True),
            patch("codoxear.server.MANAGER") as manager,
        ):
            manager.inject_keys.side_effect = ValueError("abort failed")
            Handler.do_POST(handler)  # type: ignore[arg-type]

        body = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(handler.status, 502)
        self.assertEqual(body, {"error": "abort failed"})

    def test_send_endpoint_accepts_immediately_and_worker_surfaces_broker_error(self) -> None:
        body = json.dumps({"text": "hello pi"}).encode("utf-8")
        handler = _HandlerHarness("/api/sessions/pi-session/send", body=body)
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            sock = Path(td) / "pi.sock"
            sock.touch()
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=123.0,
                cwd="/tmp/pi-cwd",
                log_path=None,
                sock_path=sock,
            )
            mgr._sock_call = lambda _sock, req, timeout_s=0.0: (
                {"error": "prompt rejected"} if req["cmd"] == "send" else {"busy": False, "queue_len": 0, "token": None}
            )  # type: ignore[method-assign]

            with (
                patch("codoxear.server._require_auth", return_value=True),
                patch("codoxear.server.MANAGER", mgr),
            ):
                Handler.do_POST(handler)  # type: ignore[arg-type]

        body = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(handler.status, 200)
        self.assertTrue(body["accepted"])
        self.assertEqual(body["delivery_state"], "queued")
        mgr._maybe_drain_outbound_request("pi-session")
        bridge_events = mgr._bridge_events.get("pi-thread-001", [])
        self.assertTrue(bridge_events)
        self.assertEqual(bridge_events[-1]["event"]["request_state"], "failed")

    def test_send_accepts_immediately_and_worker_touches_pi_session_file(self) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            session_path = Path(td) / "pi-session.jsonl"
            sock = Path(td) / "pi.sock"
            sock.touch()
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=os.getpid(),
                codex_pid=os.getpid(),
                owned=True,
                start_ts=123.0,
                cwd=td,
                log_path=None,
                sock_path=sock,
                session_path=session_path,
                busy=False,
            )
            mgr._sock_call = lambda _sock, req, timeout_s=0.0: (
                {"busy": True, "queue_len": 0, "token": None}
                if req["cmd"] == "send"
                else {"busy": False, "queue_len": 0, "token": None}
            )  # type: ignore[method-assign]

            res = mgr.send("pi-session", "hello pi")

            self.assertTrue(bool(res.get("accepted")))
            self.assertEqual(res.get("delivery_state"), "queued")
            self.assertEqual(len(mgr._outbound_requests["pi-session"]), 1)

            drained = mgr._maybe_drain_outbound_request("pi-session")

            self.assertTrue(drained)
            self.assertTrue(session_path.exists())
            s = mgr._sessions["pi-session"]
            self.assertTrue(isinstance(s.pi_busy_activity_floor, float))

    def test_send_failure_surfaces_bridge_event_without_silence(self) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            session_path = Path(td) / "pi-session.jsonl"
            sock = Path(td) / "pi.sock"
            sock.touch()
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=os.getpid(),
                codex_pid=os.getpid(),
                owned=True,
                start_ts=123.0,
                cwd=td,
                log_path=None,
                sock_path=sock,
                session_path=session_path,
                busy=False,
            )

            def _failing_sock_call(
                _sock: Path, req: dict[str, object], timeout_s: float = 0.0
            ) -> dict[str, object]:
                if req.get("cmd") == "send":
                    raise ConnectionRefusedError("socket not ready")
                return {"busy": False, "queue_len": 0, "token": None}

            mgr._sock_call = _failing_sock_call  # type: ignore[method-assign]

            res = mgr.send("pi-session", "hello queued")

            self.assertTrue(bool(res.get("accepted")))
            self.assertEqual(res.get("delivery_state"), "queued")
            self.assertEqual(len(mgr._outbound_requests["pi-session"]), 1)

            mgr._maybe_drain_outbound_request("pi-session")
            mgr._maybe_drain_outbound_request("pi-session")
            mgr._maybe_drain_outbound_request("pi-session")

            bridge_events = mgr._bridge_events.get("pi-thread-001", [])
            self.assertEqual(len(mgr._outbound_requests.get("pi-session", [])), 0)
            self.assertTrue(bridge_events)
            self.assertEqual(bridge_events[-1]["event"]["request_state"], "failed")
            self.assertIn("Original prompt", bridge_events[-1]["event"]["text"])

    def test_send_resumes_sqlite_recovered_historical_pi_session(self) -> None:
        mgr = _make_manager()
        mgr.spawn_web_session = Mock(return_value={
            "session_id": "live-session",
            "runtime_id": "live-runtime",
            "backend": "pi",
        })
        mgr.list_sessions = Mock(return_value=[
            {
                "session_id": "history:pi:resume-1",
                "resume_session_id": "resume-1",
                "cwd": "/tmp/project",
                "agent_backend": "pi",
                "backend": "pi",
                "historical": True,
            }
        ])
        mgr.send = SessionManager.send.__get__(mgr, SessionManager)
        mgr._discover_existing = Mock()
        mgr._runtime_session_id_for_identifier = Mock(side_effect=lambda session_id: session_id if session_id == "live-runtime" else None)
        mgr._sessions = {
            "live-runtime": Session(
                session_id="live-runtime",
                thread_id="live-session",
                agent_backend="pi",
                backend="pi",
                broker_pid=os.getpid(),
                codex_pid=os.getpid(),
                owned=True,
                start_ts=123.0,
                cwd="/tmp/project",
                log_path=None,
                sock_path=Path("/tmp/live-runtime.sock"),
                session_path=Path("/tmp/live-session.jsonl"),
                busy=False,
            )
        }
        mgr._sock_call = Mock(return_value={"busy": True, "queue_len": 0, "token": None})

        res = mgr.send("history:pi:resume-1", "resume me")

        mgr.spawn_web_session.assert_called_once_with(cwd="/tmp/project", backend="pi", resume_session_id="resume-1")
        mgr._discover_existing.assert_called_once_with(force=True, skip_invalid_sidecars=True)
        self.assertEqual(res["session_id"], "live-session")
        self.assertEqual(res["runtime_id"], "live-runtime")
        self.assertEqual(res["backend"], "pi")
        self.assertTrue(bool(res.get("accepted")))

    def test_send_route_url_decodes_historical_session_id(self) -> None:
        mgr = _make_manager()
        mgr.send = Mock(return_value={"ok": True, "accepted": True})
        encoded = urllib.parse.quote("history:pi:resume-1", safe="")
        handler = _HandlerHarness(
            f"/api/sessions/{encoded}/send",
            body=json.dumps({"text": "resume me"}).encode("utf-8"),
        )

        with (
            patch("codoxear.server._require_auth", return_value=True),
            patch("codoxear.server.MANAGER", mgr),
        ):
            Handler.do_POST(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(handler.status, 200)
        self.assertEqual(payload, {"ok": True, "accepted": True})
        mgr.send.assert_called_once_with("history:pi:resume-1", "resume me")

    def test_update_pi_last_chat_ts_ignores_tool_only_events(self) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            session_path = Path(td) / "pi-session.jsonl"
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=os.getpid(),
                codex_pid=os.getpid(),
                owned=True,
                start_ts=123.0,
                cwd="/tmp/project",
                log_path=None,
                sock_path=Path(td) / "pi.sock",
                session_path=session_path,
                last_chat_ts=100.0,
            )

            SessionManager._update_pi_last_chat_ts(
                mgr,
                "pi-session",
                [{"type": "tool", "ts": 999.0, "text": "bash"}],
                session_path=session_path,
            )

            self.assertEqual(mgr._sessions["pi-session"].last_chat_ts, 100.0)

            SessionManager._update_pi_last_chat_ts(
                mgr,
                "pi-session",
                [{"type": "tool_result", "ts": 120.0, "text": "bash failed", "is_error": True}],
                session_path=session_path,
            )

            self.assertEqual(mgr._sessions["pi-session"].last_chat_ts, 120.0)

    def test_get_state_preserves_cached_token_when_broker_returns_none(self) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            sock = Path(td) / "pi.sock"
            sock.touch()
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=os.getpid(),
                codex_pid=os.getpid(),
                owned=True,
                start_ts=123.0,
                cwd="/tmp/project",
                log_path=None,
                sock_path=sock,
                session_path=Path("/tmp/pi-session.jsonl"),
                token={"tokens_in_context": 123},
            )
            mgr._sock_call = Mock(return_value={"busy": False, "queue_len": 0, "token": None})

            state = SessionManager.get_state(mgr, "pi-session")

        self.assertEqual(state, {"busy": False, "queue_len": 0, "token": None})
        self.assertEqual(mgr._sessions["pi-session"].token, {"tokens_in_context": 123})

    def test_ui_state_route_returns_pending_requests_for_pi_session(self) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            sock = Path(td) / "pi.sock"
            sock.touch()
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=123.0,
                cwd="/tmp",
                log_path=None,
                sock_path=sock,
                session_path=Path("/tmp/pi-session.jsonl"),
            )
            mgr._sock_call = lambda _sock, req, timeout_s=0.0: (
                {"requests": [{"id": "ui-req-1", "method": "select"}]}
                if req["cmd"] == "ui_state"
                else {}
            )  # type: ignore[method-assign]
            handler = _HandlerHarness("/api/sessions/pi-session/ui_state")

            with (
                patch("codoxear.server._require_auth", return_value=True),
                patch("codoxear.server.MANAGER", mgr),
            ):
                Handler.do_GET(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(handler.status, 200)
        self.assertEqual(payload["requests"][0]["id"], "ui-req-1")

    def test_ui_state_route_filters_fire_and_forget_ui_requests(self) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            sock = Path(td) / "pi.sock"
            sock.touch()
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=123.0,
                cwd="/tmp",
                log_path=None,
                sock_path=sock,
                session_path=Path("/tmp/pi-session.jsonl"),
            )
            mgr._sock_call = lambda _sock, req, timeout_s=0.0: (
                {
                    "requests": [
                        {"id": "ui-notify-1", "method": "notify", "message": "indexed"},
                        {"id": "ui-widget-1", "method": "setWidget", "message": None},
                        {
                            "id": "ui-req-1",
                            "method": "select",
                            "question": "Pick one",
                            "options": ["Details", "Sidebar"],
                        },
                    ]
                }
                if req["cmd"] == "ui_state"
                else {}
            )  # type: ignore[method-assign]
            handler = _HandlerHarness("/api/sessions/pi-session/ui_state")

            with (
                patch("codoxear.server._require_auth", return_value=True),
                patch("codoxear.server.MANAGER", mgr),
            ):
                Handler.do_GET(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(handler.status, 200)
        self.assertEqual(
            payload,
            {
                "requests": [
                    {
                        "id": "ui-req-1",
                        "method": "select",
                        "question": "Pick one",
                        "options": ["Details", "Sidebar"],
                    }
                ]
            },
        )

    def test_ui_state_route_returns_unknown_session_for_dead_pi_broker(self) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            sock = Path(td) / "pi.sock"
            sock.touch()
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=123.0,
                cwd="/tmp",
                log_path=None,
                sock_path=sock,
                session_path=Path("/tmp/pi-session.jsonl"),
            )
            mgr._sock_call = lambda _sock, req, timeout_s=0.0: (_ for _ in ()).throw(
                BrokenPipeError("broker unavailable")
            )  # type: ignore[method-assign]
            handler = _HandlerHarness("/api/sessions/pi-session/ui_state")

            with (
                patch("codoxear.server._require_auth", return_value=True),
                patch("codoxear.server.MANAGER", mgr),
                patch("codoxear.server._pid_alive", return_value=False),
            ):
                Handler.do_GET(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(handler.status, 404)
        self.assertEqual(payload, {"error": "unknown session"})
        self.assertNotIn("pi-session", mgr._sessions)

    def test_messages_route_allows_historical_pi_session_without_live_runtime(self) -> None:
        mgr = _make_manager()
        mgr.refresh_session_meta = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
        mgr.get_session = lambda _sid: None  # type: ignore[method-assign]
        mgr.get_messages_page = lambda _sid, **_kwargs: {
            "events": [{"type": "assistant", "text": "hello"}],
            "diag": {},
            "offset": 0,
            "busy": False,
            "queue_len": 0,
            "token": None,
            "has_older": False,
            "next_before": 0,
        }  # type: ignore[method-assign]
        handler = _HandlerHarness("/api/sessions/history:pi:resume-1/messages?init=1&limit=20")

        with (
            patch("codoxear.server._require_auth", return_value=True),
            patch("codoxear.server.MANAGER", mgr),
            patch(
                "codoxear.server._historical_session_row",
                return_value={"agent_backend": "pi", "backend": "pi", "resume_session_id": "resume-1"},
            ),
        ):
            Handler.do_GET(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(handler.status, 200)
        self.assertEqual(payload["events"], [{"type": "assistant", "text": "hello"}])

    def test_session_details_payload_falls_back_to_historical_row(self) -> None:
        mgr = _make_manager()
        mgr.list_sessions = lambda *args, **kwargs: []  # type: ignore[method-assign]

        with patch(
            "codoxear.server._historical_session_row",
            return_value={"session_id": "history:pi:resume-1", "agent_backend": "pi", "backend": "pi", "cwd": "/tmp/hist", "historical": True},
        ):
            payload = _session_details_payload(mgr, "history:pi:resume-1")

        self.assertEqual(payload["session"]["session_id"], "history:pi:resume-1")
        self.assertTrue(payload["session"]["historical"])

    def test_session_workspace_payload_allows_historical_pi_session_without_live_runtime(self) -> None:
        mgr = _make_manager()
        mgr.get_session = lambda _sid: None  # type: ignore[method-assign]

        with patch(
            "codoxear.server._historical_session_row",
            return_value={"session_id": "history:pi:resume-1", "agent_backend": "pi", "backend": "pi", "cwd": "/tmp/hist", "historical": True},
        ):
            payload = _session_workspace_payload(mgr, "history:pi:resume-1")

        self.assertEqual(payload["session_id"], "history:pi:resume-1")
        self.assertIsNone(payload["runtime_id"])
        self.assertEqual(payload["queue"], {"items": []})

    def test_session_live_payload_allows_historical_pi_session_without_live_runtime(self) -> None:
        mgr = _make_manager()
        mgr.get_session = lambda _sid: None  # type: ignore[method-assign]
        mgr.get_messages_page = lambda _sid, **_kwargs: {
            "events": [{"role": "assistant", "text": "historical"}],
            "offset": 4,
            "has_older": True,
            "next_before": 123,
        }  # type: ignore[method-assign]

        with patch(
            "codoxear.server._historical_session_row",
            return_value={"session_id": "history:pi:resume-1", "agent_backend": "pi", "backend": "pi", "cwd": "/tmp/hist", "historical": True},
        ):
            payload = _session_live_payload(mgr, "history:pi:resume-1", offset=4)

        self.assertEqual(payload["session_id"], "history:pi:resume-1")
        self.assertIsNone(payload["runtime_id"])
        self.assertEqual(payload["events"], [{"role": "assistant", "text": "historical"}])
        self.assertEqual(payload["requests"], [])
        self.assertFalse(payload["busy"])

    def test_session_live_payload_falls_back_to_listed_stale_session_row(self) -> None:
        mgr = _make_manager()
        mgr.get_session = lambda _sid: None  # type: ignore[method-assign]
        mgr.list_sessions = lambda *args, **kwargs: [  # type: ignore[method-assign]
            {"session_id": "history:pi:resume-1", "agent_backend": "pi", "backend": "pi", "historical": True}
        ]
        mgr.get_messages_page = lambda _sid, **_kwargs: (_ for _ in ()).throw(KeyError("unknown session"))  # type: ignore[method-assign]

        with patch("codoxear.server._historical_session_row", return_value=None):
            payload = _session_live_payload(mgr, "history:pi:resume-1", offset=2)

        self.assertEqual(payload["session_id"], "history:pi:resume-1")
        self.assertEqual(payload["events"], [])
        self.assertEqual(payload["offset"], 2)
        self.assertFalse(payload["busy"])

    def test_live_route_returns_messages_busy_and_requests_for_pi_session(self) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            session_path = Path(td) / "pi-session.jsonl"
            _write_jsonl(
                session_path,
                [
                    {"type": "session", "session_id": "pi-session-001"},
                    _pi_message_entry("user", "first prompt"),
                    _pi_message_entry("assistant", "first reply"),
                ],
            )
            sock = Path(td) / "pi.sock"
            sock.touch()
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=123.0,
                cwd=td,
                log_path=None,
                sock_path=sock,
                session_path=session_path,
            )

            def _sock_call(
                _sock: Path, req: dict[str, object], timeout_s: float = 0.0
            ) -> dict[str, object]:
                if req["cmd"] == "state":
                    return {"busy": True, "queue_len": 2, "token": None}
                if req["cmd"] == "ui_state":
                    return {
                        "requests": [
                            {
                                "id": "ask-1",
                                "method": "select",
                                "question": "Pick one",
                            }
                        ]
                    }
                raise AssertionError(f"unexpected broker command: {req['cmd']!r}")

            mgr._sock_call = _sock_call  # type: ignore[method-assign]
            handler = _HandlerHarness("/api/sessions/pi-session/live?offset=0")

            with (
                patch("codoxear.server._require_auth", return_value=True),
                patch("codoxear.server.MANAGER", mgr),
            ):
                Handler.do_GET(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(handler.status, 200)
        self.assertEqual(payload["busy"], False)
        self.assertEqual(
            payload["requests"],
            [{"id": "ask-1", "method": "select", "question": "Pick one"}],
        )
        self.assertEqual(
            [event["role"] for event in payload["events"]],
            ["user", "assistant"],
        )
        self.assertNotIn("diagnostics", payload)
        self.assertNotIn("queue", payload)

    def test_live_route_omits_unchanged_requests_when_client_has_current_version(
        self,
    ) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            session_path = Path(td) / "pi-session.jsonl"
            _write_jsonl(
                session_path,
                [
                    {"type": "session", "session_id": "pi-session-001"},
                    _pi_message_entry("user", "first prompt"),
                    _pi_message_entry("assistant", "first reply"),
                ],
            )
            sock = Path(td) / "pi.sock"
            sock.touch()
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=123.0,
                cwd=td,
                log_path=None,
                sock_path=sock,
                session_path=session_path,
            )

            def _sock_call(
                _sock: Path, req: dict[str, object], timeout_s: float = 0.0
            ) -> dict[str, object]:
                if req["cmd"] == "state":
                    return {"busy": True, "queue_len": 2, "token": None}
                if req["cmd"] == "ui_state":
                    return {
                        "requests": [
                            {
                                "id": "ask-1",
                                "method": "select",
                                "question": "Pick one",
                            }
                        ]
                    }
                raise AssertionError(f"unexpected broker command: {req['cmd']!r}")

            mgr._sock_call = _sock_call  # type: ignore[method-assign]
            current_version = _ui_requests_version(
                [{"id": "ask-1", "method": "select", "question": "Pick one"}]
            )
            handler = _HandlerHarness(
                f"/api/sessions/pi-session/live?offset=2&requests_version={current_version}"
            )

            with (
                patch("codoxear.server._require_auth", return_value=True),
                patch("codoxear.server.MANAGER", mgr),
            ):
                Handler.do_GET(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(handler.status, 200)
        self.assertEqual(payload["requests_version"], current_version)
        self.assertNotIn("requests", payload)

    def test_session_live_payload_appends_streaming_pi_rpc_assistant_event(
        self,
    ) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            sock = Path(td) / "pi.sock"
            sock.touch()
            session_path = Path(td) / "pi-session.jsonl"
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=123.0,
                cwd=td,
                log_path=None,
                sock_path=sock,
                session_path=session_path,
                transport="pi-rpc",
                supports_live_ui=True,
                ui_protocol_version=1,
            )

            def _sock_call(
                _sock: Path, req: dict[str, object], timeout_s: float = 0.0
            ) -> dict[str, object]:
                if req["cmd"] == "state":
                    return {"busy": True, "queue_len": 0, "token": None}
                if req["cmd"] == "ui_state":
                    return {"requests": []}
                if req["cmd"] == "live_messages":
                    self.assertEqual(req, {"cmd": "live_messages", "offset": 0})
                    self.assertEqual(timeout_s, 1.5)
                    return {
                        "offset": 2,
                        "events": [
                            {
                                "role": "assistant",
                                "text": "partial reply",
                                "streaming": True,
                                "stream_id": "pi-stream:turn-001",
                                "turn_id": "turn-001",
                                "ts": 2.0,
                            }
                        ],
                    }
                raise AssertionError(f"unexpected broker command: {req['cmd']!r}")

            mgr._sock_call = _sock_call  # type: ignore[method-assign]
            mgr.get_messages_page = lambda *_args, **_kwargs: {
                "thread_id": "pi-thread-001",
                "log_path": str(session_path),
                "offset": 1,
                "events": [{"role": "user", "text": "hello", "ts": 1.0}],
                "busy": True,
                "queue_len": 0,
                "token": None,
            }  # type: ignore[method-assign]

            payload = _session_live_payload(mgr, "pi-session", offset=0)

        self.assertEqual(
            payload["events"],
            [
                {"role": "user", "text": "hello", "ts": 1.0},
                {
                    "role": "assistant",
                    "text": "partial reply",
                    "streaming": True,
                    "stream_id": "pi-stream:turn-001",
                    "turn_id": "turn-001",
                    "ts": 2.0,
                },
            ],
        )

    def test_session_live_payload_omits_streaming_event_once_durable_assistant_exists(
        self,
    ) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            sock = Path(td) / "pi.sock"
            sock.touch()
            session_path = Path(td) / "pi-session.jsonl"
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=123.0,
                cwd=td,
                log_path=None,
                sock_path=sock,
                session_path=session_path,
                transport="pi-rpc",
                supports_live_ui=True,
                ui_protocol_version=1,
            )

            def _sock_call(
                _sock: Path, req: dict[str, object], timeout_s: float = 0.0
            ) -> dict[str, object]:
                if req["cmd"] == "state":
                    return {"busy": True, "queue_len": 0, "token": None}
                if req["cmd"] == "ui_state":
                    return {"requests": []}
                if req["cmd"] == "live_messages":
                    return {
                        "offset": 2,
                        "events": [
                            {
                                "role": "assistant",
                                "text": "done",
                                "streaming": True,
                                "completed": True,
                                "stream_id": "pi-stream:turn-001",
                                "turn_id": "turn-001",
                                "ts": 2.0,
                            }
                        ],
                    }
                raise AssertionError(f"unexpected broker command: {req['cmd']!r}")

            mgr._sock_call = _sock_call  # type: ignore[method-assign]
            mgr.get_messages_page = lambda *_args, **_kwargs: {
                "thread_id": "pi-thread-001",
                "log_path": str(session_path),
                "offset": 3,
                "events": [
                    {"role": "user", "text": "hello", "ts": 1.0},
                    {
                        "role": "assistant",
                        "text": "done",
                        "turn_id": "turn-001",
                        "ts": 3.0,
                    },
                ],
                "busy": False,
                "queue_len": 0,
                "token": None,
            }  # type: ignore[method-assign]

            payload = _session_live_payload(mgr, "pi-session", offset=0)

        self.assertEqual(
            payload["events"],
            [
                {"role": "user", "text": "hello", "ts": 1.0},
                {"role": "assistant", "text": "done", "turn_id": "turn-001", "ts": 3.0},
            ],
        )

    def test_session_live_payload_uses_separate_live_offset_for_pi_rpc_streams(
        self,
    ) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            sock = Path(td) / "pi.sock"
            sock.touch()
            session_path = Path(td) / "pi-session.jsonl"
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=123.0,
                cwd=td,
                log_path=None,
                sock_path=sock,
                session_path=session_path,
                transport="pi-rpc",
                supports_live_ui=True,
                ui_protocol_version=1,
            )

            def _sock_call(
                _sock: Path, req: dict[str, object], timeout_s: float = 0.0
            ) -> dict[str, object]:
                if req["cmd"] == "state":
                    return {"busy": True, "queue_len": 0, "token": None}
                if req["cmd"] == "ui_state":
                    return {"requests": []}
                if req["cmd"] == "live_messages":
                    self.assertEqual(req, {"cmd": "live_messages", "offset": 7})
                    return {"offset": 8, "events": []}
                raise AssertionError(f"unexpected broker command: {req['cmd']!r}")

            mgr._sock_call = _sock_call  # type: ignore[method-assign]
            mgr.get_messages_page = lambda *_args, **_kwargs: {
                "thread_id": "pi-thread-001",
                "log_path": str(session_path),
                "offset": 100,
                "events": [{"role": "user", "text": "hello", "ts": 1.0}],
                "busy": True,
                "queue_len": 0,
                "token": None,
            }  # type: ignore[method-assign]

            mgr._append_bridge_event(
                "pi-thread-001",
                {
                    "type": "pi_event",
                    "summary": "Bridge send failed",
                    "event_id": "bridge:test-1",
                    "request_state": "failed",
                },
            )

            payload = _session_live_payload(
                mgr, "pi-session", offset=100, live_offset=7, bridge_offset=0
            )

        self.assertEqual(payload["offset"], 100)
        self.assertEqual(payload["live_offset"], 8)
        self.assertEqual(payload["bridge_offset"], 1)
        self.assertEqual(payload["transport_state"], "unknown")
        self.assertEqual(payload["events"][-1]["event_id"], "bridge:test-1")

    def test_session_live_payload_keeps_streaming_reply_when_old_history_has_same_text(
        self,
    ) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            sock = Path(td) / "pi.sock"
            sock.touch()
            session_path = Path(td) / "pi-session.jsonl"
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=123.0,
                cwd=td,
                log_path=None,
                sock_path=sock,
                session_path=session_path,
                transport="pi-rpc",
                supports_live_ui=True,
                ui_protocol_version=1,
            )

            def _sock_call(
                _sock: Path, req: dict[str, object], timeout_s: float = 0.0
            ) -> dict[str, object]:
                if req["cmd"] == "state":
                    return {"busy": True, "queue_len": 0, "token": None}
                if req["cmd"] == "ui_state":
                    return {"requests": []}
                if req["cmd"] == "live_messages":
                    return {
                        "offset": 2,
                        "events": [
                            {
                                "role": "assistant",
                                "text": "Done",
                                "streaming": True,
                                "stream_id": "pi-stream:turn-002",
                                "turn_id": "turn-002",
                                "ts": 4.0,
                            }
                        ],
                    }
                raise AssertionError(f"unexpected broker command: {req['cmd']!r}")

            mgr._sock_call = _sock_call  # type: ignore[method-assign]
            mgr.get_messages_page = lambda *_args, **_kwargs: {
                "thread_id": "pi-thread-001",
                "log_path": str(session_path),
                "offset": 3,
                "events": [
                    {"role": "user", "text": "first", "ts": 1.0},
                    {
                        "role": "assistant",
                        "text": "Done",
                        "turn_id": "turn-001",
                        "ts": 2.0,
                    },
                    {"role": "user", "text": "second", "ts": 3.0},
                ],
                "busy": True,
                "queue_len": 0,
                "token": None,
            }  # type: ignore[method-assign]

            payload = _session_live_payload(mgr, "pi-session", offset=3, live_offset=0)

        self.assertEqual(
            payload["events"][-1],
            {
                "role": "assistant",
                "text": "Done",
                "streaming": True,
                "stream_id": "pi-stream:turn-002",
                "turn_id": "turn-002",
                "ts": 4.0,
            },
        )

    def test_workspace_route_returns_diagnostics_and_queue_only(self) -> None:
        handler = _HandlerHarness("/api/sessions/pi-session/workspace")
        session = Session(
            session_id="pi-session",
            thread_id="pi-thread-001",
            agent_backend="pi",
            backend="pi",
            broker_pid=3333,
            codex_pid=4444,
            owned=True,
            start_ts=123.0,
            cwd="/tmp",
            log_path=None,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
        )

        with (
            patch("codoxear.server._require_auth", return_value=True),
            patch("codoxear.server.MANAGER") as manager,
            patch("codoxear.server._current_git_branch", return_value=None),
            patch(
                "codoxear.server._pi_messages.read_latest_pi_todo_snapshot",
                return_value=None,
            ),
        ):
            manager.refresh_session_meta.return_value = None
            manager.get_session.return_value = session
            manager.get_state.return_value = {
                "busy": False,
                "queue_len": 1,
                "token": None,
            }
            manager._queue_len.return_value = 1
            manager.queue_list.return_value = ["Queued follow-up"]
            manager.sidebar_meta_get.return_value = {
                "priority_offset": 0.0,
                "snooze_until": None,
                "dependency_session_id": None,
            }

            Handler.do_GET(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(handler.status, 200)
        self.assertEqual(payload["ok"], True)
        self.assertEqual(payload["session_id"], "pi-thread-001")
        self.assertEqual(payload["runtime_id"], "pi-session")
        self.assertIn("diagnostics", payload)
        self.assertEqual(payload["diagnostics"]["busy"], False)
        self.assertEqual(
            payload["diagnostics"]["todo_snapshot"],
            {"available": False, "error": False, "items": []},
        )
        self.assertEqual(payload["queue"], {"items": ["Queued follow-up"]})
        self.assertNotIn("events", payload)
        self.assertNotIn("requests", payload)

    def test_pi_session_messages_and_ui_state_can_coexist_for_ask_user(self) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            session_path = Path(td) / "pi-session.jsonl"
            _write_jsonl(
                session_path,
                [
                    {"type": "session", "id": "pi-session-001"},
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "toolCall",
                                    "id": "ask-1",
                                    "name": "ask_user",
                                    "arguments": {
                                        "context": "Need a placement decision.",
                                        "question": "Where should the todo live?",
                                        "options": ["Details", "Sidebar"],
                                        "allowFreeform": True,
                                    },
                                }
                            ],
                        },
                    },
                ],
            )
            sock = Path(td) / "pi.sock"
            sock.touch()
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=123.0,
                cwd=td,
                log_path=None,
                sock_path=sock,
                session_path=session_path,
            )

            def _sock_call(
                _sock: Path, req: dict[str, object], timeout_s: float = 0.0
            ) -> dict[str, object]:
                if req["cmd"] == "state":
                    return {"busy": False, "queue_len": 0, "token": None}
                if req["cmd"] == "ui_state":
                    return {
                        "requests": [
                            {
                                "id": "ask-1",
                                "method": "select",
                                "question": "Where should the todo live?",
                                "context": "Need a placement decision.",
                                "options": ["Details", "Sidebar"],
                                "allow_freeform": True,
                                "allow_multiple": False,
                                "status": "pending",
                            }
                        ]
                    }
                raise AssertionError(f"unexpected broker command: {req['cmd']!r}")

            mgr._sock_call = _sock_call  # type: ignore[method-assign]
            messages_handler = _HandlerHarness(
                "/api/sessions/pi-session/messages?init=1&limit=20"
            )
            ui_handler = _HandlerHarness("/api/sessions/pi-session/ui_state")

            with (
                patch("codoxear.server._require_auth", return_value=True),
                patch("codoxear.server.MANAGER", mgr),
            ):
                Handler.do_GET(messages_handler)  # type: ignore[arg-type]
                Handler.do_GET(ui_handler)  # type: ignore[arg-type]

        messages_payload = json.loads(messages_handler.wfile.getvalue().decode("utf-8"))
        ui_payload = json.loads(ui_handler.wfile.getvalue().decode("utf-8"))

        self.assertEqual(messages_handler.status, 200)
        self.assertEqual(ui_handler.status, 200)
        self.assertEqual(
            messages_payload["events"],
            [
                {
                    "type": "ask_user",
                    "tool_call_id": "ask-1",
                    "question": "Where should the todo live?",
                    "context": "Need a placement decision.",
                    "options": ["Details", "Sidebar"],
                    "allow_freeform": True,
                    "allow_multiple": False,
                    "timeout_ms": None,
                    "resolved": False,
                    "ts": 0.0,
                }
            ],
        )
        self.assertEqual(
            ui_payload,
            {
                "requests": [
                    {
                        "id": "ask-1",
                        "method": "select",
                        "question": "Where should the todo live?",
                        "context": "Need a placement decision.",
                        "options": ["Details", "Sidebar"],
                        "allow_freeform": True,
                        "allow_multiple": False,
                        "status": "pending",
                    }
                ]
            },
        )

    def test_session_write_routes_return_json_404_for_malformed_paths(self) -> None:
        cases = [
            ("/api/sessions//rename", b'{"name":"alias"}'),
            ("/api/sessions//focus", b'{"focused":true}'),
            ("/api/sessions//send", b'{"text":"hello"}'),
            ("/api/sessions//ui_response", b'{"id":"ui-1","value":"x"}'),
            ("/api/sessions//enqueue", b'{"text":"later"}'),
        ]
        with patch("codoxear.server._require_auth", return_value=True):
            for raw_path, body in cases:
                handler = _HandlerHarness(raw_path, body=body)
                handler._unauthorized = lambda: None  # type: ignore[attr-defined]
                handler.send_error = lambda code: setattr(handler, "status", code)  # type: ignore[attr-defined]

                Handler.do_POST(handler)  # type: ignore[arg-type]

                self.assertEqual(handler.status, 404, raw_path)
                self.assertEqual(
                    handler.wfile.getvalue().decode("utf-8"),
                    '{"error":"unknown session"}',
                )

    def test_ui_response_route_does_not_fallback_to_send_for_live_pi_rpc_session(
        self,
    ) -> None:
        mgr = _make_manager()
        body = json.dumps(
            {
                "id": "ui-req-1",
                "value": "Details",
                "cmd": "send",
                "text": "ignore me",
                "extra": "drop me",
            }
        ).encode("utf-8")
        forwarded: dict[str, object] = {}
        with tempfile.TemporaryDirectory() as td:
            sock = Path(td) / "pi.sock"
            sock.touch()
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=123.0,
                cwd="/tmp",
                log_path=None,
                sock_path=sock,
                session_path=Path("/tmp/pi-session.jsonl"),
                transport="pi-rpc",
                supports_live_ui=True,
                ui_protocol_version=1,
            )

            def _sock_call(
                _sock: Path, req: dict[str, object], timeout_s: float = 0.0
            ) -> dict[str, object]:
                forwarded.update(req)
                return (
                    {"status": "accepted", "forwarded": dict(req)}
                    if req["cmd"] == "ui_response"
                    else {}
                )

            mgr._sock_call = _sock_call  # type: ignore[method-assign]
            handler = _HandlerHarness("/api/sessions/pi-session/ui_response", body=body)

            with (
                patch("codoxear.server._require_auth", return_value=True),
                patch("codoxear.server.MANAGER", mgr),
            ):
                Handler.do_POST(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(handler.status, 200)
        self.assertEqual(
            forwarded,
            {
                "cmd": "ui_response",
                "id": "ui-req-1",
                "value": "Details",
            },
        )
        self.assertEqual(payload, {"ok": True})

    def test_ui_response_route_rejects_live_pi_rpc_unknown_cmd_without_send_fallback(
        self,
    ) -> None:
        mgr = _make_manager()
        body = json.dumps({"id": "ui-req-1", "value": "Details"}).encode("utf-8")
        forwarded: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as td:
            sock = Path(td) / "pi.sock"
            sock.touch()
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=123.0,
                cwd="/tmp",
                log_path=None,
                sock_path=sock,
                session_path=Path("/tmp/pi-session.jsonl"),
                transport="pi-rpc",
                supports_live_ui=True,
                ui_protocol_version=1,
            )

            def _sock_call(
                _sock: Path, req: dict[str, object], timeout_s: float = 0.0
            ) -> dict[str, object]:
                forwarded.append(dict(req))
                if req["cmd"] == "ui_response":
                    return {"error": "unknown cmd"}
                if req["cmd"] == "send":
                    raise AssertionError(
                        "live pi-rpc session must not use send fallback"
                    )
                return {}

            mgr._sock_call = _sock_call  # type: ignore[method-assign]
            handler = _HandlerHarness("/api/sessions/pi-session/ui_response", body=body)

            with (
                patch("codoxear.server._require_auth", return_value=True),
                patch("codoxear.server.MANAGER", mgr),
            ):
                Handler.do_POST(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(handler.status, 502)
        self.assertEqual(
            payload, {"error": "live ui responses are unavailable for this pi session"}
        )
        self.assertEqual(
            forwarded, [{"cmd": "ui_response", "id": "ui-req-1", "value": "Details"}]
        )

    def test_ui_state_route_rejects_live_pi_rpc_unknown_cmd_without_silent_fallback(
        self,
    ) -> None:
        mgr = _make_manager()
        forwarded: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as td:
            sock = Path(td) / "pi.sock"
            sock.touch()
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=123.0,
                cwd="/tmp",
                log_path=None,
                sock_path=sock,
                session_path=Path("/tmp/pi-session.jsonl"),
                transport="pi-rpc",
                supports_live_ui=True,
                ui_protocol_version=1,
            )

            def _sock_call(
                _sock: Path, req: dict[str, object], timeout_s: float = 0.0
            ) -> dict[str, object]:
                forwarded.append(dict(req))
                if req["cmd"] == "ui_state":
                    return {"error": "unknown cmd"}
                raise AssertionError(f"unexpected broker command: {req['cmd']!r}")

            mgr._sock_call = _sock_call  # type: ignore[method-assign]
            handler = _HandlerHarness("/api/sessions/pi-session/ui_state")

            with (
                patch("codoxear.server._require_auth", return_value=True),
                patch("codoxear.server.MANAGER", mgr),
            ):
                Handler.do_GET(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(handler.status, 502)
        self.assertEqual(
            payload,
            {"error": "live ui interactions are unavailable for this pi session"},
        )
        self.assertEqual(forwarded, [{"cmd": "ui_state"}])

    def test_ui_response_route_legacy_pi_session_still_uses_send_fallback(self) -> None:
        mgr = _make_manager()
        body = json.dumps({"id": "ui-req-1", "value": "Details"}).encode("utf-8")
        forwarded: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as td:
            sock = Path(td) / "pi.sock"
            sock.touch()
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=os.getpid(),
                codex_pid=os.getpid(),
                owned=True,
                start_ts=123.0,
                cwd="/tmp",
                log_path=None,
                sock_path=sock,
                session_path=Path("/tmp/pi-session.jsonl"),
                transport="pty",
            )

            def _sock_call(
                _sock: Path, req: dict[str, object], timeout_s: float = 0.0
            ) -> dict[str, object]:
                forwarded.append(dict(req))
                if req["cmd"] == "ui_response":
                    return {"error": "unknown cmd"}
                if req["cmd"] == "send":
                    return {"queued": False, "queue_len": 0}
                if req["cmd"] == "state":
                    return {"busy": False, "queue_len": 0, "token": None}
                return {}

            mgr._sock_call = _sock_call  # type: ignore[method-assign]
            handler = _HandlerHarness("/api/sessions/pi-session/ui_response", body=body)

            with (
                patch("codoxear.server._require_auth", return_value=True),
                patch("codoxear.server.MANAGER", mgr),
            ):
                Handler.do_POST(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(handler.status, 200)
        self.assertEqual(payload, {"ok": True})
        self.assertEqual(
            forwarded,
            [
                {"cmd": "ui_response", "id": "ui-req-1", "value": "Details"},
                {"cmd": "state"},
            ],
        )

    def test_ui_response_route_refreshes_stale_live_capabilities_before_fallback_decision(
        self,
    ) -> None:
        mgr = _make_manager()
        mgr.refresh_session_meta = SessionManager.refresh_session_meta.__get__(
            mgr, SessionManager
        )  # type: ignore[method-assign]
        body = json.dumps({"id": "ui-req-1", "value": "Details"}).encode("utf-8")
        forwarded: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as td:
            sock = Path(td) / "pi.sock"
            sock.touch()
            session_path = Path(td) / "pi-session.jsonl"
            session_path.write_text(
                '{"type":"session","session_id":"pi-thread-001"}\n', encoding="utf-8"
            )
            sock.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "session_id": "pi-thread-001",
                        "backend": "pi",
                        "owner": "web",
                        "broker_pid": os.getpid(),
                        "codex_pid": os.getpid(),
                        "cwd": td,
                        "start_ts": 123.0,
                        "session_path": str(session_path),
                        "sock_path": str(sock),
                        "transport": "pty",
                    }
                ),
                encoding="utf-8",
            )
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=os.getpid(),
                codex_pid=os.getpid(),
                owned=True,
                start_ts=123.0,
                cwd=td,
                log_path=None,
                sock_path=sock,
                session_path=session_path,
                transport="pi-rpc",
                supports_live_ui=True,
                ui_protocol_version=1,
            )

            def _sock_call(
                _sock: Path, req: dict[str, object], timeout_s: float = 0.0
            ) -> dict[str, object]:
                forwarded.append(dict(req))
                if req["cmd"] == "ui_response":
                    return {"error": "unknown cmd"}
                if req["cmd"] == "send":
                    return {"queued": False, "queue_len": 0}
                if req["cmd"] == "state":
                    return {"busy": False, "queue_len": 0, "token": None}
                raise AssertionError(f"unexpected broker command: {req['cmd']!r}")

            mgr._sock_call = _sock_call  # type: ignore[method-assign]
            handler = _HandlerHarness("/api/sessions/pi-session/ui_response", body=body)

            with (
                patch("codoxear.server._require_auth", return_value=True),
                patch("codoxear.server.MANAGER", mgr),
            ):
                Handler.do_POST(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(handler.status, 200)
        self.assertEqual(payload, {"ok": True})
        self.assertEqual(
            forwarded,
            [
                {"cmd": "ui_response", "id": "ui-req-1", "value": "Details"},
                {"cmd": "state"},
            ],
        )

    def test_ui_response_route_returns_unknown_session_for_dead_pi_broker(self) -> None:
        mgr = _make_manager()
        body = json.dumps({"id": "ui-req-1", "value": "Details"}).encode("utf-8")
        with tempfile.TemporaryDirectory() as td:
            sock = Path(td) / "pi.sock"
            sock.touch()
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=123.0,
                cwd="/tmp",
                log_path=None,
                sock_path=sock,
                session_path=Path("/tmp/pi-session.jsonl"),
            )
            mgr._sock_call = lambda _sock, req, timeout_s=0.0: (_ for _ in ()).throw(
                BrokenPipeError("broker unavailable")
            )  # type: ignore[method-assign]
            handler = _HandlerHarness("/api/sessions/pi-session/ui_response", body=body)

            with (
                patch("codoxear.server._require_auth", return_value=True),
                patch("codoxear.server.MANAGER", mgr),
                patch("codoxear.server._pid_alive", return_value=False),
            ):
                Handler.do_POST(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(handler.status, 404)
        self.assertEqual(payload, {"error": "unknown session"})
        self.assertNotIn("pi-session", mgr._sessions)

    def test_attachment_inject_endpoints_reject_pi_backend_without_broker_keys(
        self,
    ) -> None:
        body = json.dumps(
            {
                "filename": "note.txt",
                "attachment_index": 1,
                "data_b64": base64.b64encode(b"hello").decode("ascii"),
            }
        ).encode("utf-8")
        pi_session = Session(
            session_id="pi-session",
            thread_id="pi-thread-001",
            agent_backend="pi",
            backend="pi",
            broker_pid=3333,
            codex_pid=4444,
            owned=True,
            start_ts=123.0,
            cwd="/tmp/pi-cwd",
            log_path=None,
            sock_path=Path("/tmp/pi.sock"),
            session_path=Path("/tmp/pi-session.jsonl"),
        )

        for path in (
            "/api/sessions/pi-session/inject_file",
            "/api/sessions/pi-session/inject_image",
        ):
            with self.subTest(path=path):
                handler = _HandlerHarness(path, body)
                with (
                    patch("codoxear.server._require_auth", return_value=True),
                    patch("codoxear.server.MANAGER") as manager,
                ):
                    manager.get_session.return_value = pi_session

                    Handler.do_POST(handler)  # type: ignore[arg-type]

                payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
                self.assertEqual(handler.status, 409)
                self.assertEqual(
                    payload,
                    {
                        "error": "attachment injection is not supported for Pi sessions",
                        "backend": "pi",
                        "operation": "attachment_injection",
                    },
                )
                manager.refresh_session_meta.assert_called_once_with(
                    "pi-session", strict=False
                )
                manager.get_session.assert_called_once_with("pi-session")
                manager.inject_keys.assert_not_called()

    def test_sessions_bootstrap_returns_cwd_groups(self) -> None:
        handler = _HandlerHarness("/api/sessions/bootstrap")
        cwd_groups = {"/tmp": {"label": "Temp", "collapsed": True}}
        with (
            patch("codoxear.server._require_auth", return_value=True),
            patch("codoxear.server.MANAGER") as manager,
            patch(
                "codoxear.server._read_new_session_defaults",
                return_value={"default_backend": "pi"},
            ),
            patch("codoxear.server._tmux_available", return_value=True),
        ):
            manager.recent_cwds.return_value = ["/tmp"]
            manager.cwd_groups_get.return_value = cwd_groups

            Handler.do_GET(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(handler.status, 200)
        self.assertEqual(payload["recent_cwds"], ["/tmp"])
        self.assertEqual(payload["cwd_groups"], cwd_groups)
        self.assertEqual(payload["new_session_defaults"], {"default_backend": "pi"})
        self.assertEqual(payload["tmux_available"], True)
        manager.list_sessions.assert_not_called()
        manager.cwd_groups_get.assert_called_once()

    def test_list_sessions_returns_only_lightweight_rows(self) -> None:
        handler = _HandlerHarness("/api/sessions")
        expected_cwd = str(Path("/tmp/project").resolve(strict=False))
        with (
            patch("codoxear.server._require_auth", return_value=True),
            patch("codoxear.server.MANAGER") as manager,
        ):
            manager.list_sessions.return_value = [
                {
                    "session_id": "sess-1",
                    "thread_id": "thread-1",
                    "cwd": "/tmp/project",
                    "agent_backend": "pi",
                    "broker_pid": 2468,
                    "busy": True,
                    "queue_len": 2,
                    "alias": "Active",
                    "files": ["large", "unused"],
                    "log_path": "/tmp/session.jsonl",
                    "token": {"used": 123},
                    "thinking": 4,
                    "harness_enabled": True,
                    "tmux_window": "0",
                    "model_provider": "unused-provider-internal",
                    "model": "gpt-5.4",
                    "provider_choice": "openai-api",
                    "reasoning_effort": "high",
                    "service_tier": "fast",
                    "priority_offset": 0.25,
                    "snooze_until": 1234,
                    "dependency_session_id": "sess-0",
                    "resume_session_id": "resume-1",
                    "time_priority": 0.5,
                }
            ]

            Handler.do_GET(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(handler.status, 200)
        self.assertEqual(
            payload,
            {
                "sessions": [
                    {
                        "session_id": "sess-1",
                        "thread_id": "thread-1",
                        "display_name": "Active",
                        "cwd": expected_cwd,
                        "agent_backend": "pi",
                        "busy": True,
                        "queue_len": 2,
                        "alias": "Active",
                    }
                ],
            },
        )
        self.assertNotIn("model", payload["sessions"][0])
        self.assertNotIn("provider_choice", payload["sessions"][0])
        self.assertNotIn("reasoning_effort", payload["sessions"][0])
        self.assertNotIn("service_tier", payload["sessions"][0])
        self.assertNotIn("priority_offset", payload["sessions"][0])
        self.assertNotIn("snooze_until", payload["sessions"][0])
        self.assertNotIn("dependency_session_id", payload["sessions"][0])
        self.assertNotIn("resume_session_id", payload["sessions"][0])
        self.assertNotIn("broker_pid", payload["sessions"][0])
        manager.recent_cwds.assert_not_called()
        manager.cwd_groups_get.assert_not_called()

    def test_session_details_returns_launch_and_edit_fields_removed_from_list(
        self,
    ) -> None:
        handler = _HandlerHarness("/api/sessions/sess-1/details")
        with (
            patch("codoxear.server._require_auth", return_value=True),
            patch("codoxear.server.MANAGER") as manager,
        ):
            manager.list_sessions.return_value = [
                {
                    "session_id": "sess-1",
                    "cwd": "/tmp/project",
                    "agent_backend": "codex",
                    "model": "gpt-5.4",
                    "provider_choice": "openai-api",
                    "reasoning_effort": "high",
                    "service_tier": "fast",
                    "transport": "tmux",
                    "priority_offset": 0.25,
                    "snooze_until": 1234,
                    "dependency_session_id": "sess-0",
                }
            ]

            Handler.do_GET(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(handler.status, 200)
        self.assertEqual(payload["session"]["model"], "gpt-5.4")
        self.assertEqual(payload["session"]["provider_choice"], "openai-api")
        self.assertEqual(payload["session"]["priority_offset"], 0.25)

    def test_list_sessions_paginates_flat_rows_and_reports_remaining_count(
        self,
    ) -> None:
        handler = _HandlerHarness("/api/sessions?limit=5")
        docs_cwd = str(Path("/work/docs").resolve(strict=False))
        with (
            patch("codoxear.server._require_auth", return_value=True),
            patch("codoxear.server.MANAGER") as manager,
        ):
            manager.list_sessions.return_value = [
                {
                    "session_id": f"docs-{index}",
                    "runtime_id": f"docs-runtime-{index}",
                    "alias": f"Docs {index}",
                    "cwd": docs_cwd,
                    "agent_backend": "pi",
                }
                for index in range(1, 8)
            ] + [
                {
                    "session_id": "api-1",
                    "runtime_id": "api-runtime-1",
                    "alias": "API 1",
                    "cwd": "/work/api",
                    "agent_backend": "codex",
                }
            ]

            Handler.do_GET(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(handler.status, 200)
        self.assertEqual(
            [row["session_id"] for row in payload["sessions"]],
            ["docs-1", "docs-2", "docs-3", "docs-4", "docs-5"],
        )
        self.assertEqual(payload["sessions"][0]["runtime_id"], "docs-runtime-1")
        self.assertEqual(payload["remaining_count"], 3)

    def test_list_sessions_supports_flat_offset_pagination(self) -> None:
        docs_cwd = str(Path("/work/docs").resolve(strict=False))
        handler = _HandlerHarness("/api/sessions?offset=5&limit=5")
        with (
            patch("codoxear.server._require_auth", return_value=True),
            patch("codoxear.server.MANAGER") as manager,
        ):
            manager.list_sessions.return_value = [
                {
                    "session_id": f"docs-{index}",
                    "alias": f"Docs {index}",
                    "cwd": docs_cwd,
                    "agent_backend": "pi",
                }
                for index in range(1, 8)
            ]

            Handler.do_GET(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(handler.status, 200)
        self.assertEqual(
            [row["session_id"] for row in payload["sessions"]], ["docs-6", "docs-7"]
        )
        self.assertNotIn("remaining_count", payload)

    def test_list_sessions_preserves_manager_order_in_flat_payload(self) -> None:
        handler = _HandlerHarness("/api/sessions")
        cwds = [
            str(Path(f"/work/group-{index}").resolve(strict=False))
            for index in range(1, 6)
        ]
        with (
            patch("codoxear.server._require_auth", return_value=True),
            patch("codoxear.server.MANAGER") as manager,
        ):
            manager.list_sessions.return_value = [
                {
                    "session_id": "recent-1",
                    "cwd": cwds[0],
                    "agent_backend": "pi",
                    "busy": False,
                },
                {
                    "session_id": "recent-2",
                    "cwd": cwds[1],
                    "agent_backend": "pi",
                    "busy": False,
                },
                {
                    "session_id": "recent-3",
                    "cwd": cwds[2],
                    "agent_backend": "pi",
                    "busy": False,
                },
                {
                    "session_id": "old-idle",
                    "cwd": cwds[3],
                    "agent_backend": "pi",
                    "busy": False,
                },
                {
                    "session_id": "old-busy",
                    "cwd": cwds[4],
                    "agent_backend": "pi",
                    "busy": True,
                },
            ]

            Handler.do_GET(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(handler.status, 200)
        self.assertEqual(
            [row["session_id"] for row in payload["sessions"]],
            ["recent-1", "recent-2", "recent-3", "old-idle", "old-busy"],
        )
        self.assertNotIn("omitted_group_count", payload)

    def test_list_sessions_keeps_flat_order_for_busy_and_recent_rows(self) -> None:
        handler = _HandlerHarness("/api/sessions")
        group_old = str(Path("/work/old").resolve(strict=False))
        group_busy = str(Path("/work/busy").resolve(strict=False))
        group_recent = str(Path("/work/recent").resolve(strict=False))
        with (
            patch("codoxear.server._require_auth", return_value=True),
            patch("codoxear.server.MANAGER") as manager,
        ):
            manager.list_sessions.return_value = [
                {
                    "session_id": "old-1",
                    "cwd": group_old,
                    "agent_backend": "pi",
                    "busy": False,
                    "updated_ts": 100,
                },
                {
                    "session_id": "busy-1",
                    "cwd": group_busy,
                    "agent_backend": "pi",
                    "busy": True,
                    "updated_ts": 50,
                },
                {
                    "session_id": "recent-1",
                    "cwd": group_recent,
                    "agent_backend": "pi",
                    "busy": False,
                    "updated_ts": 200,
                },
            ]

            Handler.do_GET(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(handler.status, 200)
        self.assertEqual(
            [row["session_id"] for row in payload["sessions"]],
            ["old-1", "busy-1", "recent-1"],
        )

    def test_list_sessions_supports_loading_more_omitted_groups(self) -> None:
        handler = _HandlerHarness("/api/sessions?group_offset=4&group_limit=2")
        cwds = [
            str(Path(f"/work/group-{index}").resolve(strict=False))
            for index in range(1, 6)
        ]
        with (
            patch("codoxear.server._require_auth", return_value=True),
            patch("codoxear.server.MANAGER") as manager,
        ):
            manager.list_sessions.return_value = [
                {
                    "session_id": f"group-{index}",
                    "cwd": cwd,
                    "agent_backend": "pi",
                    "busy": index == 5,
                }
                for index, cwd in enumerate(cwds, start=1)
            ]

            Handler.do_GET(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(handler.status, 200)
        self.assertEqual(
            [row["session_id"] for row in payload["sessions"]], ["group-4"]
        )
        self.assertEqual(payload["omitted_group_count"], 0)

    def test_json_response_ignores_connection_reset_during_write(self) -> None:
        handler = _HandlerHarness("/api/sessions")
        handler.wfile = cast(io.BytesIO, _ResettingWriter())

        _json_response(cast(Any, handler), 200, {"ok": True})

        self.assertEqual(handler.status, 200)

    def test_list_sessions_normalizes_session_cwds(self) -> None:
        handler = _HandlerHarness("/api/sessions")
        raw_cwd = "  /tmp/project/../project/docs  "
        expected_cwd = str(Path(raw_cwd.strip()).resolve(strict=False))
        with (
            patch("codoxear.server._require_auth", return_value=True),
            patch("codoxear.server.MANAGER") as manager,
        ):
            manager.list_sessions.return_value = [
                {"session_id": "sess-1", "cwd": raw_cwd}
            ]

            Handler.do_GET(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(handler.status, 200)
        self.assertEqual(payload["sessions"][0]["cwd"], expected_cwd)
        self.assertNotIn("cwd_groups", payload)

    def test_edit_cwd_group_updates_metadata(self) -> None:
        body = json.dumps(
            {"cwd": "/tmp", "label": "New Label", "collapsed": True}
        ).encode("utf-8")
        handler = _HandlerHarness("/api/cwd_groups/edit", body)

        with (
            patch("codoxear.server._require_auth", return_value=True),
            patch("codoxear.server.MANAGER") as manager,
        ):
            manager.cwd_group_set.return_value = (
                "/tmp",
                {"label": "New Label", "collapsed": True},
            )

            Handler.do_POST(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(handler.status, 200)
        self.assertEqual(
            payload,
            {"ok": True, "cwd": "/tmp", "label": "New Label", "collapsed": True},
        )
        manager.cwd_group_set.assert_called_once_with(
            cwd="/tmp", label="New Label", collapsed=True
        )

    def test_edit_cwd_group_returns_400_on_value_error(self) -> None:
        body = json.dumps({"cwd": "/tmp", "collapsed": "not-a-bool"}).encode("utf-8")
        handler = _HandlerHarness("/api/cwd_groups/edit", body)

        with (
            patch("codoxear.server._require_auth", return_value=True),
            patch("codoxear.server.MANAGER") as manager,
        ):
            manager.cwd_group_set.side_effect = ValueError(
                "collapsed must be a boolean"
            )

            Handler.do_POST(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(handler.status, 400)
        self.assertEqual(payload, {"error": "collapsed must be a boolean"})

    def test_edit_cwd_group_returns_400_for_non_string_label(self) -> None:
        body = json.dumps({"cwd": "/tmp", "label": 123}).encode("utf-8")
        handler = _HandlerHarness("/api/cwd_groups/edit", body)

        with (
            patch("codoxear.server._require_auth", return_value=True),
            patch("codoxear.server.MANAGER") as manager,
        ):
            manager.cwd_group_set.side_effect = ValueError("label must be a string")

            Handler.do_POST(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(handler.status, 400)
        self.assertEqual(payload, {"error": "label must be a string"})

    def test_edit_cwd_group_returns_400_for_unknown_cwd(self) -> None:
        body = json.dumps({"cwd": "/tmp/unknown", "label": "Unknown"}).encode("utf-8")
        handler = _HandlerHarness("/api/cwd_groups/edit", body)

        with (
            patch("codoxear.server._require_auth", return_value=True),
            patch("codoxear.server.MANAGER") as manager,
        ):
            manager.cwd_group_set.side_effect = ValueError(
                "cwd is not a known session working directory"
            )

            Handler.do_POST(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(handler.status, 400)
        self.assertEqual(
            payload, {"error": "cwd is not a known session working directory"}
        )

    def test_edit_cwd_group_returns_400_on_empty_body(self) -> None:
        handler = _HandlerHarness("/api/cwd_groups/edit")

        with (
            patch("codoxear.server._require_auth", return_value=True),
            patch("codoxear.server.MANAGER") as manager,
        ):
            Handler.do_POST(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(handler.status, 400)
        self.assertEqual(payload, {"error": "empty request body"})
        manager.cwd_group_set.assert_not_called()

    def test_edit_cwd_group_returns_400_on_malformed_json(self) -> None:
        handler = _HandlerHarness("/api/cwd_groups/edit", b"{")

        with (
            patch("codoxear.server._require_auth", return_value=True),
            patch("codoxear.server.MANAGER") as manager,
        ):
            Handler.do_POST(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(handler.status, 400)
        self.assertIn(
            "Expecting property name enclosed in double quotes", payload["error"]
        )
        manager.cwd_group_set.assert_not_called()

    def test_edit_cwd_group_returns_400_on_non_object_json(self) -> None:
        handler = _HandlerHarness(
            "/api/cwd_groups/edit", json.dumps(["/tmp"]).encode("utf-8")
        )

        with (
            patch("codoxear.server._require_auth", return_value=True),
            patch("codoxear.server.MANAGER") as manager,
        ):
            Handler.do_POST(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(handler.status, 400)
        self.assertEqual(payload, {"error": "invalid json body (expected object)"})
        manager.cwd_group_set.assert_not_called()

    def test_notifications_test_push_returns_payload(self) -> None:
        handler = _HandlerHarness("/api/notifications/test_push", b"{}")

        with (
            patch("codoxear.server._require_auth", return_value=True),
            patch("codoxear.server.MANAGER") as manager,
        ):
            manager._voice_push.send_test_push_notification.return_value = {
                "sent_count": 1,
                "failed_count": 0,
                "target_count": 1,
                "notification_text": "回复完成",
            }

            Handler.do_POST(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(handler.status, 200)
        self.assertEqual(payload["sent_count"], 1)
        manager._voice_push.send_test_push_notification.assert_called_once_with(
            session_display_name="Codoxear test"
        )

    def test_notifications_test_push_returns_400_for_missing_mobile_subscriptions(
        self,
    ) -> None:
        handler = _HandlerHarness("/api/notifications/test_push", b"{}")

        with (
            patch("codoxear.server._require_auth", return_value=True),
            patch("codoxear.server.MANAGER") as manager,
        ):
            manager._voice_push.send_test_push_notification.side_effect = ValueError(
                "no enabled mobile subscriptions"
            )

            Handler.do_POST(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(handler.status, 400)
        self.assertEqual(payload, {"error": "no enabled mobile subscriptions"})


class TestPiMessageNormalization(unittest.TestCase):
    def test_prompt_and_turn_events_emit_user_and_assistant_rows(self) -> None:
        entries = [
            *pi_persisted_session_file(),
            {"type": "tool.started", "tool_name": "read", "turn_id": "turn-001"},
            {"type": "bashExecution", "command": "pwd", "turn_id": "turn-001"},
        ]

        events, _meta, flags, diag = pi_messages.normalize_pi_entries(entries)

        # User and assistant messages plus tool markers from tool.started / bashExecution
        self.assertEqual(
            events[0],
            {
                "role": "user",
                "text": "Summarize the current repository state.",
                "ts": 0.0,
            },
        )
        self.assertEqual(events[1]["role"], "assistant")
        self.assertEqual(
            events[1]["text"], "Codoxear serves a browser UI for Codex-style sessions."
        )
        self.assertEqual(events[1]["ts"], 1.0)
        self.assertEqual(events[1]["message_class"], "final_response")
        self.assertIsInstance(events[1]["message_id"], str)
        self.assertEqual(
            events[2:],
            [
                {"type": "tool", "name": "read", "ts": 2.0},
                {"type": "tool", "name": "bashExecution", "ts": 2.1},
            ],
        )
        self.assertTrue(flags["turn_start"])
        self.assertTrue(flags["turn_end"])
        self.assertFalse(flags["turn_aborted"])
        self.assertEqual(diag["tool_names"], ["read", "bashExecution"])
        self.assertEqual(diag["last_tool"], "bashExecution")

    def test_runtime_pi_session_shape_normalizes_into_chat_rows(self) -> None:
        events, _meta, flags, diag = pi_messages.normalize_pi_entries(
            pi_runtime_session_file()
        )

        # Millisecond timestamps are converted to seconds by _entry_ts()
        self.assertEqual(
            events[0],
            {
                "role": "user",
                "text": "Summarize the current repository state.",
                "ts": 1774708707.28,
            },
        )
        self.assertEqual(events[1]["role"], "assistant")
        self.assertEqual(
            events[1]["text"], "Codoxear serves a browser UI for Codex-style sessions."
        )
        self.assertEqual(events[1]["ts"], 1774708716.099)
        self.assertEqual(events[1]["message_class"], "final_response")
        self.assertIsInstance(events[1]["message_id"], str)
        self.assertFalse(flags["turn_start"])
        self.assertFalse(flags["turn_end"])
        self.assertEqual(diag["tool_names"], [])

    def test_session_file_replay_restores_history_after_restart(self) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            session_path = Path(td) / "pi-session.jsonl"
            session_path.write_text(
                "".join(
                    json.dumps(entry) + "\n" for entry in pi_persisted_session_file()
                ),
                encoding="utf-8",
            )
            session_size = session_path.stat().st_size
            sock = Path(td) / "pi.sock"
            sock.touch()
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                backend="pi",
                agent_backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=123.0,
                cwd=td,
                log_path=None,
                sock_path=sock,
                session_path=session_path,
            )
            mgr._sock_call = lambda *_args, **_kwargs: {
                "busy": False,
                "queue_len": 0,
                "token": None,
            }  # type: ignore[method-assign]

            first = mgr.get_messages_page(
                "pi-session", offset=0, init=True, limit=20, before=0
            )

            restarted = _make_manager()
            restarted._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=123.0,
                cwd=td,
                log_path=None,
                sock_path=sock,
                session_path=session_path,
            )
            restarted._sock_call = lambda *_args, **_kwargs: {
                "busy": False,
                "queue_len": 0,
                "token": None,
            }  # type: ignore[method-assign]

            second = restarted.get_messages_page(
                "pi-session", offset=0, init=True, limit=20, before=0
            )

        self.assertEqual(first["events"], second["events"])
        self.assertEqual(
            [event["role"] for event in second["events"]], ["user", "assistant"]
        )
        self.assertEqual(second["offset"], session_size)
        self.assertFalse(second["has_older"])
        self.assertEqual(second["next_before"], 0)

    def test_pi_messages_refresh_session_listing_recency_on_replay_and_poll(
        self,
    ) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            session_path = Path(td) / "pi-session.jsonl"
            _write_jsonl(
                session_path,
                [
                    {"type": "session", "session_id": "pi-session-001"},
                    _pi_message_entry("user", "first prompt"),
                    _pi_message_entry("assistant", "first reply"),
                ],
            )
            replay_mtime = 1_700_000_100.0
            poll_mtime = replay_mtime + 60.0
            import os

            os.utime(session_path, (replay_mtime, replay_mtime))
            sock = Path(td) / "pi.sock"
            sock.touch()
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=123.0,
                cwd=td,
                log_path=None,
                sock_path=sock,
                session_path=session_path,
            )
            mgr._sock_call = lambda *_args, **_kwargs: {
                "busy": False,
                "queue_len": 0,
                "token": None,
            }  # type: ignore[method-assign]

            replay = mgr.get_messages_page(
                "pi-session", offset=0, init=True, limit=20, before=0
            )
            rows_after_replay = mgr.list_sessions()

            with session_path.open("a", encoding="utf-8") as f:
                for entry in (
                    _pi_message_entry("user", "follow-up prompt"),
                    _pi_message_entry("assistant", "follow-up reply"),
                ):
                    f.write(json.dumps(entry) + "\n")
            os.utime(session_path, (poll_mtime, poll_mtime))

            poll = mgr.get_messages_page(
                "pi-session", offset=replay["offset"], init=False, limit=20, before=0
            )
            rows_after_poll = mgr.list_sessions()

        self.assertEqual([event["ts"] for event in replay["events"]], [0.0, 1.0])
        self.assertEqual(rows_after_replay[0]["updated_ts"], replay_mtime)
        self.assertGreater(poll["events"][0]["ts"], replay["events"][-1]["ts"])
        self.assertEqual(
            [event["text"] for event in poll["events"]],
            ["follow-up prompt", "follow-up reply"],
        )
        self.assertEqual(rows_after_poll[0]["updated_ts"], poll_mtime)

    def test_list_sessions_refreshes_pi_recency_without_messages_poll(self) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            session_path = Path(td) / "pi-session.jsonl"
            _write_jsonl(
                session_path,
                [
                    {"type": "session", "session_id": "pi-session-001"},
                    _pi_message_entry("user", "first prompt"),
                    _pi_message_entry("assistant", "first reply"),
                ],
            )
            initial_mtime = 1_700_000_100.0
            refreshed_mtime = initial_mtime + 120.0
            import os

            os.utime(session_path, (initial_mtime, initial_mtime))
            sock = Path(td) / "pi.sock"
            sock.touch()
            mgr._sessions["other-session"] = Session(
                session_id="other-session",
                thread_id="thread-other",
                agent_backend="codex",
                backend="codex",
                broker_pid=1111,
                codex_pid=2222,
                owned=True,
                start_ts=123.0,
                cwd=td,
                log_path=None,
                sock_path=sock,
                last_chat_ts=initial_mtime + 30.0,
            )
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=123.0,
                cwd=td,
                log_path=None,
                sock_path=sock,
                session_path=session_path,
            )
            mgr._sock_call = lambda *_args, **_kwargs: {
                "busy": False,
                "queue_len": 0,
                "token": None,
            }  # type: ignore[method-assign]

            rows_before_refresh = mgr.list_sessions()

            with session_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(_pi_message_entry("assistant", "fresh reply")) + "\n"
                )
            os.utime(session_path, (refreshed_mtime, refreshed_mtime))

            rows_after_refresh = mgr.list_sessions()

        self.assertEqual(
            [row["session_id"] for row in rows_before_refresh[:2]],
            ["thread-other", "pi-thread-001"],
        )
        self.assertEqual(
            [row["session_id"] for row in rows_after_refresh[:2]],
            ["pi-thread-001", "thread-other"],
        )
        self.assertEqual(rows_after_refresh[0]["updated_ts"], refreshed_mtime)

    def test_list_sessions_retries_first_user_message_preview_for_new_codex_sessions(
        self,
    ) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "codex-session.jsonl"
            log_path.write_text("", encoding="utf-8")
            sock = Path(td) / "codex.sock"
            sock.touch()
            mgr._sessions["codex-session"] = Session(
                session_id="codex-session",
                thread_id="codex-thread-001",
                agent_backend="codex",
                backend="codex",
                broker_pid=1111,
                codex_pid=2222,
                owned=True,
                start_ts=123.0,
                cwd=td,
                log_path=log_path,
                sock_path=sock,
            )
            mgr._sock_call = lambda *_args, **_kwargs: {
                "busy": False,
                "queue_len": 0,
                "token": None,
            }  # type: ignore[method-assign]
            mgr.idle_from_log = lambda *_args, **_kwargs: True  # type: ignore[method-assign]

            first_rows = mgr.list_sessions()
            self.assertIsNone(mgr._sessions["codex-session"].first_user_message)

            with log_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(_codex_user_message_entry("draft a migration plan"))
                    + "\n"
                )

            second_rows = mgr.list_sessions()

        self.assertEqual(first_rows[0]["first_user_message"], "")
        self.assertEqual(second_rows[0]["first_user_message"], "draft a migration plan")

    def test_list_sessions_retries_first_user_message_preview_for_new_pi_sessions(
        self,
    ) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            session_path = Path(td) / "pi-session.jsonl"
            _write_jsonl(
                session_path,
                [{"type": "session", "session_id": "pi-session-001"}],
            )
            sock = Path(td) / "pi.sock"
            sock.touch()
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=123.0,
                cwd=td,
                log_path=None,
                sock_path=sock,
                session_path=session_path,
            )
            mgr._sock_call = lambda *_args, **_kwargs: {
                "busy": False,
                "queue_len": 0,
                "token": None,
            }  # type: ignore[method-assign]

            first_rows = mgr.list_sessions()
            self.assertIsNone(mgr._sessions["pi-session"].first_user_message)

            with session_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        _pi_message_entry("user", "investigate websocket reconnects")
                    )
                    + "\n"
                )

            second_rows = mgr.list_sessions()

        self.assertEqual(first_rows[0]["first_user_message"], "")
        self.assertEqual(
            second_rows[0]["first_user_message"],
            "investigate websocket reconnects",
        )

    def test_list_sessions_keeps_pi_busy_when_session_file_has_not_advanced(
        self,
    ) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            session_path = Path(td) / "pi-session.jsonl"
            _write_jsonl(
                session_path,
                [
                    {"type": "session", "session_id": "pi-session-001"},
                    _pi_message_entry("user", "first prompt"),
                    _pi_message_entry("assistant", "first reply"),
                ],
            )
            unchanged_mtime = 1_700_000_100.0
            os.utime(session_path, (unchanged_mtime, unchanged_mtime))
            sock = Path(td) / "pi.sock"
            sock.touch()
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=123.0,
                cwd=td,
                log_path=None,
                sock_path=sock,
                session_path=session_path,
                busy=True,
                last_chat_ts=unchanged_mtime,
                pi_busy_activity_floor=unchanged_mtime,
            )
            mgr._sock_call = lambda *_args, **_kwargs: {
                "busy": True,
                "queue_len": 0,
                "token": None,
            }  # type: ignore[method-assign]

            rows = mgr.list_sessions()

        self.assertTrue(rows[0]["busy"])

    def test_refresh_session_meta_resets_pi_caches_when_session_path_changes(
        self,
    ) -> None:
        mgr = _make_manager()
        mgr.refresh_session_meta = SessionManager.refresh_session_meta.__get__(
            mgr, SessionManager
        )  # type: ignore[method-assign]
        with tempfile.TemporaryDirectory() as td:
            sock = Path(td) / "pi.sock"
            sock.touch()
            old_session_path = Path(td) / "old-session.jsonl"
            new_session_path = Path(td) / "new-session.jsonl"
            old_session_path.write_text(
                '{"session_id":"pi-thread-001"}\n', encoding="utf-8"
            )
            new_session_path.write_text(
                '{"session_id":"pi-thread-002"}\n', encoding="utf-8"
            )
            sock.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "session_id": "pi-thread-002",
                        "backend": "pi",
                        "owner": "web",
                        "broker_pid": 3333,
                        "codex_pid": 4444,
                        "cwd": td,
                        "start_ts": 123.0,
                        "session_path": str(new_session_path),
                        "sock_path": str(sock),
                    }
                ),
                encoding="utf-8",
            )
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=123.0,
                cwd=td,
                log_path=None,
                sock_path=sock,
                session_path=old_session_path,
                last_chat_ts=999.0,
                meta_thinking=4,
                meta_tools=5,
                meta_system=6,
                meta_log_off=77,
                chat_index_events=[{"role": "assistant", "text": "stale", "ts": 999.0}],
                chat_index_scan_bytes=88,
                chat_index_scan_complete=True,
                chat_index_log_off=99,
                idle_cache_log_off=111,
                idle_cache_value=True,
                queue_idle_since=222.0,
            )

            mgr.refresh_session_meta("pi-session")

        session = mgr.get_session("pi-session")
        self.assertIsNotNone(session)
        assert session is not None
        self.assertEqual(session.thread_id, "pi-thread-002")
        self.assertEqual(session.session_path, new_session_path)
        self.assertIsNone(session.last_chat_ts)
        self.assertEqual(session.meta_thinking, 0)
        self.assertEqual(session.meta_tools, 0)
        self.assertEqual(session.meta_system, 0)
        self.assertEqual(session.meta_log_off, 0)
        self.assertEqual(session.chat_index_events, [])
        self.assertEqual(session.chat_index_scan_bytes, 0)
        self.assertFalse(session.chat_index_scan_complete)
        self.assertEqual(session.chat_index_log_off, 0)
        self.assertEqual(session.idle_cache_log_off, -1)
        self.assertIsNone(session.idle_cache_value)
        self.assertIsNone(session.queue_idle_since)

    def test_refresh_session_meta_requires_pi_session_path_metadata(self) -> None:
        mgr = _make_manager()
        mgr.refresh_session_meta = SessionManager.refresh_session_meta.__get__(
            mgr, SessionManager
        )  # type: ignore[method-assign]
        with tempfile.TemporaryDirectory() as td:
            sock = Path(td) / "pi.sock"
            sock.touch()
            sock.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "session_id": "pi-thread-001",
                        "backend": "pi",
                        "owner": "web",
                        "broker_pid": 3333,
                        "codex_pid": 4444,
                        "cwd": td,
                        "start_ts": 123.0,
                        "sock_path": str(sock),
                    }
                ),
                encoding="utf-8",
            )
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=123.0,
                cwd=td,
                log_path=None,
                sock_path=sock,
                session_path=Path(td) / "pi-session.jsonl",
            )

            with self.assertRaisesRegex(ValueError, "missing session_path"):
                mgr.refresh_session_meta("pi-session")

    def test_refresh_session_meta_non_strict_recovers_missing_pi_session_path_via_discovery(
        self,
    ) -> None:
        mgr = _make_manager()
        mgr.refresh_session_meta = SessionManager.refresh_session_meta.__get__(
            mgr, SessionManager
        )  # type: ignore[method-assign]
        with tempfile.TemporaryDirectory() as td:
            sock = Path(td) / "pi.sock"
            sock.touch()
            recovered_session_path = Path(td) / "recovered-session.jsonl"
            recovered_session_path.write_text(
                '{"type":"session","session_id":"pi-thread-001"}\n', encoding="utf-8"
            )
            sock.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "session_id": "pi-thread-001",
                        "backend": "pi",
                        "owner": "web",
                        "broker_pid": 3333,
                        "codex_pid": 4444,
                        "cwd": td,
                        "start_ts": 123.0,
                        "sock_path": str(sock),
                    }
                ),
                encoding="utf-8",
            )
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=123.0,
                cwd=td,
                log_path=None,
                sock_path=sock,
                session_path=None,
            )

            with patch(
                "codoxear.server._discover_pi_session_for_cwd",
                return_value=recovered_session_path,
            ):
                mgr.refresh_session_meta("pi-session", strict=False)

        session = mgr.get_session("pi-session")
        self.assertIsNotNone(session)
        assert session is not None
        self.assertEqual(session.session_path, recovered_session_path)
        self.assertFalse(mgr._sidecar_is_quarantined(sock))

    def test_refresh_session_meta_non_strict_prefers_exact_thread_match_over_same_cwd_discovery(
        self,
    ) -> None:
        mgr = _make_manager()
        mgr.refresh_session_meta = SessionManager.refresh_session_meta.__get__(
            mgr, SessionManager
        )  # type: ignore[method-assign]
        with tempfile.TemporaryDirectory() as td:
            sock = Path(td) / "pi.sock"
            sock.touch()
            exact_session_path = Path(td) / "exact-session.jsonl"
            wrong_session_path = Path(td) / "wrong-session.jsonl"
            exact_session_path.write_text(
                '{"type":"session","id":"pi-thread-001","cwd":"/tmp/pi-cwd"}\n',
                encoding="utf-8",
            )
            wrong_session_path.write_text(
                '{"type":"session","id":"pi-thread-wrong","cwd":"/tmp/pi-cwd"}\n',
                encoding="utf-8",
            )
            sock.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "session_id": "pi-thread-001",
                        "backend": "pi",
                        "owner": "web",
                        "broker_pid": 3333,
                        "codex_pid": 4444,
                        "cwd": "/tmp/pi-cwd",
                        "start_ts": 123.0,
                        "sock_path": str(sock),
                    }
                ),
                encoding="utf-8",
            )
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=123.0,
                cwd=td,
                log_path=None,
                sock_path=sock,
                session_path=None,
            )

            with (
                patch(
                    "codoxear.server._find_session_log_for_session_id",
                    return_value=exact_session_path,
                ) as find_exact,
                patch(
                    "codoxear.server._discover_pi_session_for_cwd",
                    return_value=wrong_session_path,
                ) as discover,
                patch("codoxear.server._patch_metadata_session_path") as patch_meta,
            ):
                mgr.refresh_session_meta("pi-session", strict=False)

        session = mgr.get_session("pi-session")
        self.assertIsNotNone(session)
        assert session is not None
        self.assertEqual(session.session_path, exact_session_path)
        find_exact.assert_called()
        discover.assert_not_called()
        patch_meta.assert_called_once()
        self.assertEqual(patch_meta.call_args.args[:2], (sock, exact_session_path))

    def test_refresh_session_meta_reuses_matching_pi_session_path_without_global_scan(
        self,
    ) -> None:
        mgr = _make_manager()
        mgr.refresh_session_meta = SessionManager.refresh_session_meta.__get__(
            mgr, SessionManager
        )  # type: ignore[method-assign]
        with tempfile.TemporaryDirectory() as td:
            sock = Path(td) / "pi.sock"
            sock.touch()
            session_path = Path(td) / "exact-session.jsonl"
            session_path.write_text(
                '{"type":"session","id":"pi-thread-001","cwd":"/tmp/pi-cwd"}\n',
                encoding="utf-8",
            )
            sock.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "session_id": "pi-thread-001",
                        "backend": "pi",
                        "owner": "web",
                        "broker_pid": 3333,
                        "codex_pid": 4444,
                        "cwd": "/tmp/pi-cwd",
                        "start_ts": 123.0,
                        "sock_path": str(sock),
                        "session_path": str(session_path),
                    }
                ),
                encoding="utf-8",
            )
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=123.0,
                cwd=td,
                log_path=None,
                sock_path=sock,
                session_path=session_path,
            )

            with (
                patch(
                    "codoxear.server._find_session_log_for_session_id",
                    side_effect=AssertionError("global scan should not run"),
                ),
                patch(
                    "codoxear.server._discover_pi_session_for_cwd",
                    side_effect=AssertionError("cwd discovery should not run"),
                ),
            ):
                mgr.refresh_session_meta("pi-session")

        session = mgr.get_session("pi-session")
        self.assertIsNotNone(session)
        assert session is not None
        self.assertEqual(session.session_path, session_path)

    def test_refresh_session_meta_non_strict_reuses_in_memory_pi_session_path_cache(
        self,
    ) -> None:
        mgr = _make_manager()
        mgr.refresh_session_meta = SessionManager.refresh_session_meta.__get__(
            mgr, SessionManager
        )  # type: ignore[method-assign]
        with tempfile.TemporaryDirectory() as td:
            sock = Path(td) / "pi.sock"
            sock.touch()
            session_path = Path(td) / "cached-session.jsonl"
            session_path.write_text(
                '{"type":"session","id":"pi-thread-001","cwd":"/tmp/pi-cwd"}\n',
                encoding="utf-8",
            )
            sock.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "session_id": "pi-thread-001",
                        "backend": "pi",
                        "owner": "web",
                        "broker_pid": 3333,
                        "codex_pid": 4444,
                        "cwd": "/tmp/pi-cwd",
                        "start_ts": 123.0,
                        "sock_path": str(sock),
                    }
                ),
                encoding="utf-8",
            )
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=123.0,
                cwd=td,
                log_path=None,
                sock_path=sock,
                session_path=session_path,
            )

            with (
                patch(
                    "codoxear.server._find_session_log_for_session_id",
                ) as find_exact,
                patch(
                    "codoxear.server._discover_pi_session_for_cwd",
                ) as discover,
            ):
                mgr.refresh_session_meta("pi-session", strict=False)

        find_exact.assert_not_called()
        discover.assert_not_called()

        session = mgr.get_session("pi-session")
        self.assertIsNotNone(session)
        assert session is not None
        self.assertEqual(session.session_path, session_path)

    def test_refresh_session_meta_updates_live_pi_ui_capabilities_from_sidecar(
        self,
    ) -> None:
        mgr = _make_manager()
        mgr.refresh_session_meta = SessionManager.refresh_session_meta.__get__(
            mgr, SessionManager
        )  # type: ignore[method-assign]
        with tempfile.TemporaryDirectory() as td:
            sock = Path(td) / "pi.sock"
            sock.touch()
            session_path = Path(td) / "pi-session.jsonl"
            session_path.write_text(
                '{"type":"session","session_id":"pi-thread-001"}\n', encoding="utf-8"
            )
            sock.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "session_id": "pi-thread-001",
                        "backend": "pi",
                        "owner": "web",
                        "broker_pid": 3333,
                        "codex_pid": 4444,
                        "cwd": td,
                        "start_ts": 123.0,
                        "session_path": str(session_path),
                        "sock_path": str(sock),
                        "transport": "pi-rpc",
                        "supports_live_ui": True,
                        "ui_protocol_version": 1,
                    }
                ),
                encoding="utf-8",
            )
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=123.0,
                cwd=td,
                log_path=None,
                sock_path=sock,
                session_path=session_path,
                transport="pi-rpc",
                supports_live_ui=None,
                ui_protocol_version=None,
            )

            mgr.refresh_session_meta("pi-session")

        session = mgr.get_session("pi-session")
        self.assertIsNotNone(session)
        assert session is not None
        self.assertTrue(session.supports_live_ui)
        self.assertEqual(session.ui_protocol_version, 1)

    def test_refresh_session_meta_clears_missing_live_pi_ui_capabilities(self) -> None:
        mgr = _make_manager()
        mgr.refresh_session_meta = SessionManager.refresh_session_meta.__get__(
            mgr, SessionManager
        )  # type: ignore[method-assign]
        with tempfile.TemporaryDirectory() as td:
            sock = Path(td) / "pi.sock"
            sock.touch()
            session_path = Path(td) / "pi-session.jsonl"
            session_path.write_text(
                '{"type":"session","session_id":"pi-thread-001"}\n', encoding="utf-8"
            )
            sock.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "session_id": "pi-thread-001",
                        "backend": "pi",
                        "owner": "web",
                        "broker_pid": 3333,
                        "codex_pid": 4444,
                        "cwd": td,
                        "start_ts": 123.0,
                        "session_path": str(session_path),
                        "sock_path": str(sock),
                        "transport": "pi-rpc",
                    }
                ),
                encoding="utf-8",
            )
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=123.0,
                cwd=td,
                log_path=None,
                sock_path=sock,
                session_path=session_path,
                transport="pi-rpc",
                supports_live_ui=True,
                ui_protocol_version=1,
            )

            mgr.refresh_session_meta("pi-session")

        session = mgr.get_session("pi-session")
        self.assertIsNotNone(session)
        assert session is not None
        self.assertIsNone(session.supports_live_ui)
        self.assertIsNone(session.ui_protocol_version)

    def test_refresh_session_meta_non_strict_quarantines_invalid_pi_sidecar(
        self,
    ) -> None:
        mgr = _make_manager()
        mgr.refresh_session_meta = SessionManager.refresh_session_meta.__get__(
            mgr, SessionManager
        )  # type: ignore[method-assign]
        with tempfile.TemporaryDirectory() as td:
            sock = Path(td) / "pi.sock"
            sock.touch()
            known_session_path = Path(td) / "known-session.jsonl"
            known_session_path.write_text(
                '{"type":"session","session_id":"pi-thread-001"}\n', encoding="utf-8"
            )
            sock.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "session_id": "pi-thread-001",
                        "backend": "pi",
                        "owner": "web",
                        "broker_pid": 3333,
                        "codex_pid": 4444,
                        "cwd": td,
                        "start_ts": 123.0,
                        "session_path": "   ",
                        "sock_path": str(sock),
                    }
                ),
                encoding="utf-8",
            )
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=123.0,
                cwd=td,
                log_path=None,
                sock_path=sock,
                session_path=known_session_path,
            )

            mgr.refresh_session_meta("pi-session", strict=False)

            self.assertTrue(mgr._sidecar_is_quarantined(sock))

        session = mgr.get_session("pi-session")
        self.assertIsNotNone(session)
        assert session is not None
        self.assertEqual(session.session_path, known_session_path)

    def test_get_messages_page_tolerates_malformed_sidecar_after_discovery(
        self,
    ) -> None:
        mgr = _make_manager()
        mgr.refresh_session_meta = SessionManager.refresh_session_meta.__get__(
            mgr, SessionManager
        )  # type: ignore[method-assign]
        with tempfile.TemporaryDirectory() as td:
            session_path = Path(td) / "pi-session.jsonl"
            _write_jsonl(
                session_path,
                [
                    {"type": "session", "session_id": "pi-thread-001"},
                    _pi_message_entry("user", "hello"),
                    _pi_message_entry("assistant", "world"),
                ],
            )
            sock = Path(td) / "pi.sock"
            sock.touch()
            sock.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "session_id": "pi-thread-001",
                        "backend": "pi",
                        "owner": "web",
                        "broker_pid": 3333,
                        "codex_pid": 4444,
                        "cwd": td,
                        "start_ts": 123.0,
                        "sock_path": str(sock),
                    }
                ),
                encoding="utf-8",
            )
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=123.0,
                cwd=td,
                log_path=None,
                sock_path=sock,
                session_path=session_path,
            )
            mgr._sock_call = lambda *_args, **_kwargs: {
                "busy": False,
                "queue_len": 0,
                "token": None,
            }  # type: ignore[method-assign]

            payload = mgr.get_messages_page(
                "pi-session", offset=0, init=True, limit=20, before=0
            )
            expected_offset = session_path.stat().st_size

        self.assertEqual(
            [event["role"] for event in payload["events"]], ["user", "assistant"]
        )
        self.assertEqual(payload["offset"], expected_offset)

    def test_removed_event_stream_view_no_longer_includes_system_events(self) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            session_path = Path(td) / "pi-session.jsonl"
            _write_jsonl(
                session_path,
                [
                    {"type": "session", "session_id": "pi-thread-001"},
                    {
                        "type": "model_change",
                        "timestamp": "2026-04-03T01:02:03.000Z",
                        "provider": "macaron",
                        "modelId": "gpt-5.4",
                    },
                    {
                        "type": "custom_event",
                        "timestamp": "2026-04-03T01:02:05.000Z",
                        "payload": {"phase": "queued", "count": 3},
                    },
                    _pi_message_entry("user", "hello"),
                    _pi_message_entry("assistant", "world"),
                ],
            )
            sock = Path(td) / "pi.sock"
            sock.touch()
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                backend="pi",
                agent_backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=123.0,
                cwd=td,
                log_path=None,
                sock_path=sock,
                session_path=session_path,
            )
            mgr._sock_call = lambda *_args, **_kwargs: {
                "busy": False,
                "queue_len": 0,
                "token": None,
            }  # type: ignore[method-assign]

            payload = mgr.get_messages_page(
                "pi-session", offset=0, init=True, limit=20, before=0, view="events"
            )

        self.assertEqual(
            [event["role"] for event in payload["events"]], ["user", "assistant"]
        )
        self.assertFalse(
            any(
                event.get("type")
                in {"pi_event", "pi_model_change", "pi_thinking_level_change"}
                for event in payload["events"]
            )
        )

    def test_delta_polling_keeps_synthetic_timestamps_monotonic(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            session_path = Path(td) / "pi-session.jsonl"
            initial_entries = [
                {"type": "session", "session_id": "pi-session-001"},
                _pi_message_entry("user", "first prompt"),
                _pi_message_entry("assistant", "first reply"),
            ]
            _write_jsonl(session_path, initial_entries)

            initial_events, initial_off, _has_older, _next_before, _diag = (
                pi_messages.read_pi_message_page(session_path, limit=20, before=0)
            )

            with session_path.open("a", encoding="utf-8") as f:
                for entry in (
                    _pi_message_entry("user", "follow-up prompt"),
                    _pi_message_entry("assistant", "follow-up reply"),
                ):
                    f.write(json.dumps(entry) + "\n")

            read_offsets: list[int] = []
            real_read = pi_messages._read_jsonl_from_offset

            def _spy_read(path: Path, offset: int, max_bytes: int = 2 * 1024 * 1024):
                read_offsets.append(int(offset))
                return real_read(path, offset, max_bytes=max_bytes)

            with patch.object(
                pi_messages, "_read_jsonl_from_offset", side_effect=_spy_read
            ):
                delta_events, _new_off, _meta, _flags, _diag = (
                    pi_messages.read_pi_message_delta(session_path, offset=initial_off)
                )

        self.assertEqual([event["ts"] for event in initial_events], [0.0, 1.0])
        self.assertEqual(read_offsets[0], initial_off)
        self.assertTrue(all(offset >= initial_off for offset in read_offsets))
        self.assertEqual(
            sorted(event["ts"] for event in delta_events),
            [event["ts"] for event in delta_events],
        )
        self.assertGreater(delta_events[0]["ts"], initial_events[-1]["ts"])
        self.assertEqual(
            [event["text"] for event in delta_events],
            ["follow-up prompt", "follow-up reply"],
        )

    def test_large_session_replay_reads_past_sixty_four_chunks(self) -> None:
        with (
            tempfile.TemporaryDirectory() as td,
            patch.object(pi_messages, "_PI_READ_MAX_BYTES", 256),
        ):
            session_path = Path(td) / "pi-session.jsonl"
            entries: list[dict[str, object]] = [
                {"type": "session", "session_id": "pi-session-001"}
            ]
            for idx in range(200):
                entries.append(
                    _pi_message_entry("user", f"message {idx:03d} {'.' * 40}")
                )
            _write_jsonl(session_path, entries)

            events, new_off, has_older, next_before, _diag = (
                pi_messages.read_pi_message_page(session_path, limit=500, before=0)
            )

            self.assertEqual(len(events), 200)
            self.assertEqual(
                events[-1]["text"],
                "message 199 ........................................",
            )
            self.assertEqual(new_off, session_path.stat().st_size)
            self.assertFalse(has_older)
            self.assertEqual(next_before, 0)

    def test_replay_and_older_pages_cap_diag_tool_names(self) -> None:
        with (
            tempfile.TemporaryDirectory() as td,
            patch.object(pi_messages, "_PI_DIAG_TOOL_NAMES_LIMIT", 3),
        ):
            session_path = Path(td) / "pi-session.jsonl"
            entries: list[dict[str, object]] = [
                {"type": "session", "session_id": "pi-session-001"},
                _pi_message_entry("user", "first prompt"),
                _pi_message_entry("assistant", "first reply"),
            ]
            for idx in range(6):
                entries.append(
                    {
                        "type": "tool.started",
                        "tool_name": f"tool-{idx}",
                        "turn_id": "turn-001",
                    }
                )
            _write_jsonl(session_path, entries)

            _events, _new_off, _has_older, _next_before, init_diag = (
                pi_messages.read_pi_message_page(session_path, limit=20, before=0)
            )
            (
                _older_events,
                _older_off,
                _older_has_older,
                _older_next_before,
                older_diag,
            ) = pi_messages.read_pi_message_page(
                session_path,
                limit=1,
                before=1,
            )

        self.assertEqual(init_diag["tool_names"], ["tool-3", "tool-4", "tool-5"])
        self.assertEqual(init_diag["last_tool"], "tool-5")
        self.assertEqual(older_diag["tool_names"], ["tool-3", "tool-4", "tool-5"])
        self.assertEqual(older_diag["last_tool"], "tool-5")

    def test_normalize_accepts_input_text_and_text_blocks(self) -> None:
        entries = [
            _pi_message_entry("user", "typed prompt", block_type="input_text"),
            _pi_message_entry("assistant", "plain reply", block_type="text"),
        ]

        events, _meta, _flags, _diag = pi_messages.normalize_pi_entries(
            entries, include_system=True
        )

        self.assertEqual(events[0], {"role": "user", "text": "typed prompt", "ts": 0.0})
        self.assertEqual(events[1]["role"], "assistant")
        self.assertEqual(events[1]["text"], "plain reply")
        self.assertEqual(events[1]["ts"], 1.0)
        self.assertEqual(events[1]["message_class"], "final_response")
        self.assertIsInstance(events[1]["message_id"], str)

    def test_normalize_marks_final_assistant_text_with_message_metadata(self) -> None:
        entries = [
            _pi_message_entry("user", "typed prompt", block_type="input_text"),
            _pi_message_entry("assistant", "plain reply", block_type="text"),
        ]

        events, _meta, _flags, _diag = pi_messages.normalize_pi_entries(
            entries, include_system=True
        )

        self.assertEqual(events[1]["role"], "assistant")
        self.assertEqual(events[1]["message_class"], "final_response")
        self.assertIsInstance(events[1]["message_id"], str)

    def test_normalize_marks_assistant_text_before_tool_call_as_narration(self) -> None:
        entries = [
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Working on it"},
                        {
                            "type": "toolCall",
                            "id": "call-1",
                            "name": "bash",
                            "arguments": {"command": "pwd"},
                        },
                    ],
                },
            },
        ]

        events, _meta, _flags, _diag = pi_messages.normalize_pi_entries(
            entries, include_system=True
        )

        self.assertEqual(events[0]["role"], "assistant")
        self.assertEqual(events[0]["message_class"], "narration")
        self.assertIsInstance(events[0]["message_id"], str)

    def test_subagent_call_and_result_produce_subagent_event(self) -> None:
        entries = [
            _pi_message_entry("user", "do something"),
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "toolCall",
                            "id": "call_abc",
                            "name": "subagent",
                            "arguments": {
                                "agent": "reviewer",
                                "task": "Review the plan",
                            },
                        }
                    ],
                },
            },
            {
                "type": "message",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "call_abc",
                    "toolName": "subagent",
                    "content": [{"type": "text", "text": "APPROVED"}],
                },
            },
        ]
        events, _meta, _flags, _diag = pi_messages.normalize_pi_entries(
            entries, include_system=True
        )
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0], {"role": "user", "text": "do something", "ts": 0.0})
        sa = events[1]
        self.assertEqual(sa["type"], "subagent")
        self.assertEqual(sa["agent"], "reviewer")
        self.assertEqual(sa["task"], "Review the plan")
        self.assertEqual(sa["output"], "APPROVED")

    def test_subagent_call_without_result_emits_pending(self) -> None:
        entries = [
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "toolCall",
                            "id": "call_xyz",
                            "name": "subagent",
                            "arguments": {
                                "agent": "worker",
                                "task": "Implement feature",
                            },
                        }
                    ],
                },
            },
        ]
        events, _meta, _flags, _diag = pi_messages.normalize_pi_entries(
            entries, include_system=True
        )
        self.assertEqual(len(events), 1)
        sa = events[0]
        self.assertEqual(sa["type"], "subagent")
        self.assertEqual(sa["agent"], "worker")
        self.assertEqual(sa["task"], "Implement feature")
        self.assertIsNone(sa["output"])

    def test_subagent_result_without_call_emits_partial(self) -> None:
        entries = [
            {
                "type": "message",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "call_orphan",
                    "toolName": "subagent",
                    "content": [{"type": "text", "text": "Done"}],
                },
            },
        ]
        events, _meta, _flags, _diag = pi_messages.normalize_pi_entries(
            entries, include_system=True
        )
        self.assertEqual(len(events), 1)
        sa = events[0]
        self.assertEqual(sa["type"], "subagent")
        self.assertEqual(sa["agent"], "subagent")
        self.assertIsNone(sa["task"])
        self.assertEqual(sa["output"], "Done")

    def test_subagent_with_string_arguments(self) -> None:
        entries = [
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "toolCall",
                            "id": "call_str",
                            "name": "subagent",
                            "arguments": '{"agent": "delegate", "task": "Do stuff"}',
                        }
                    ],
                },
            },
            {
                "type": "message",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "call_str",
                    "toolName": "subagent",
                    "content": [{"type": "text", "text": "Stuff done"}],
                },
            },
        ]
        events, _meta, _flags, _diag = pi_messages.normalize_pi_entries(
            entries, include_system=True
        )
        self.assertEqual(len(events), 1)
        sa = events[0]
        self.assertEqual(sa["agent"], "delegate")
        self.assertEqual(sa["task"], "Do stuff")
        self.assertEqual(sa["output"], "Stuff done")

    def test_subagent_events_do_not_break_existing_normalization(self) -> None:
        entries = [
            _pi_message_entry("user", "hello"),
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "toolCall",
                            "id": "call_sa",
                            "name": "subagent",
                            "arguments": {"agent": "reviewer", "task": "Check"},
                        }
                    ],
                },
            },
            {"type": "tool.started", "tool_name": "bash"},
            {
                "type": "message",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "call_sa",
                    "toolName": "subagent",
                    "content": [{"type": "text", "text": "OK"}],
                },
            },
            _pi_message_entry("assistant", "all done"),
        ]
        events, meta, _flags, diag = pi_messages.normalize_pi_entries(entries)
        types = [(e.get("role") or e.get("type")) for e in events]
        self.assertEqual(types, ["user", "tool", "subagent", "assistant"])
        self.assertEqual(meta["tool"], 1)
        self.assertIn("bash", diag["tool_names"])

    def test_non_subagent_toolcall_emits_tool_event(self) -> None:
        entries = [
            _pi_message_entry("user", "list files"),
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "toolCall",
                            "id": "call_bash_1",
                            "name": "bash",
                            "arguments": {"command": "ls"},
                        }
                    ],
                },
            },
            {
                "type": "message",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "call_bash_1",
                    "toolName": "bash",
                    "content": [{"type": "text", "text": "file1.py\nfile2.py"}],
                },
            },
            _pi_message_entry("assistant", "Here are the files."),
        ]
        events, meta, _flags, diag = pi_messages.normalize_pi_entries(entries)
        types = [(e.get("role") or e.get("type")) for e in events]
        self.assertEqual(types, ["user", "tool", "tool_result", "assistant"])
        self.assertEqual(events[1]["name"], "bash")
        self.assertEqual(events[2]["text"], "file1.py\nfile2.py")
        self.assertGreaterEqual(meta["tool"], 1)
        self.assertIn("bash", diag["tool_names"])

    def test_non_subagent_toolcall_only_message_emits_tool_event(self) -> None:
        """Assistant message with only toolCall blocks (no text) still produces tool events."""
        entries = [
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "toolCall",
                            "id": "c1",
                            "name": "read",
                            "arguments": {"file": "a.py"},
                        },
                        {
                            "type": "toolCall",
                            "id": "c2",
                            "name": "grep",
                            "arguments": {"pattern": "foo"},
                        },
                    ],
                },
            },
        ]
        events, meta, _flags, diag = pi_messages.normalize_pi_entries(entries)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0], {"type": "tool", "name": "read", "ts": 0.0})
        self.assertEqual(events[1], {"type": "tool", "name": "grep", "ts": 0.1})
        self.assertEqual(meta["tool"], 2)
        self.assertEqual(diag["tool_names"], ["read", "grep"])
        self.assertEqual(diag["last_tool"], "grep")

    def test_mixed_text_and_toolcall_emits_both(self) -> None:
        """Assistant message with both text and toolCall blocks emits tool + text events."""
        entries = [
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me check that."},
                        {
                            "type": "toolCall",
                            "id": "c1",
                            "name": "bash",
                            "arguments": {"command": "pwd"},
                        },
                    ],
                },
            },
        ]
        events, meta, _flags, diag = pi_messages.normalize_pi_entries(entries)
        types = [(e.get("role") or e.get("type")) for e in events]
        self.assertEqual(types, ["assistant", "tool"])
        self.assertEqual(events[0]["text"], "Let me check that.")
        self.assertEqual(events[1]["name"], "bash")

    def test_non_subagent_tool_result_emits_visible_event(self) -> None:
        """Non-subagent toolResult messages should remain visible in the chat stream."""
        entries = [
            {
                "type": "message",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "call_1",
                    "toolName": "bash",
                    "content": [{"type": "text", "text": "output"}],
                },
            },
        ]
        events, meta, _flags, diag = pi_messages.normalize_pi_entries(entries)
        self.assertEqual(
            events,
            [
                {"type": "tool_result", "name": "bash", "text": "output", "ts": 0.0},
            ],
        )
        self.assertEqual(meta["tool"], 1)
        self.assertIn("bash", diag["tool_names"])

    def test_reasoning_blocks_emit_reasoning_events(self) -> None:
        entries = [
            {
                "type": "message",
                "timestamp": "2026-04-03T01:02:03.000Z",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "Investigating the repo layout.",
                            "thinkingSignature": '{"summary":[{"type":"summary_text","text":"Repo scan"}]}',
                        },
                        {"type": "text", "text": "Final answer."},
                    ],
                },
            },
        ]

        events, _meta, _flags, _diag = pi_messages.normalize_pi_entries(
            entries, include_system=True
        )

        self.assertEqual(events[0]["type"], "reasoning")
        self.assertEqual(events[0]["text"], "Investigating the repo layout.")
        self.assertEqual(events[0]["summary"], "Repo scan")
        self.assertEqual(events[1]["role"], "assistant")
        self.assertEqual(events[1]["text"], "Final answer.")

    def test_turn_failed_emits_visible_pi_event(self) -> None:
        entries = [
            {
                "type": "turn.failed",
                "timestamp": "2026-04-03T01:02:03.000Z",
                "payload": {
                    "source_event": "turn.failed",
                    "message": "model await failed",
                    "error": "upstream timeout",
                },
            },
        ]

        events, _meta, flags, _diag = pi_messages.normalize_pi_entries(entries)

        self.assertTrue(flags["turn_aborted"])
        self.assertEqual(events[0]["type"], "pi_event")
        self.assertTrue(events[0]["is_error"])
        self.assertEqual(events[0]["summary"], "turn.failed")
        self.assertEqual(events[0]["text"], "upstream timeout")

    def test_error_tool_result_without_output_still_emits_visible_event(self) -> None:
        entries = [
            {
                "type": "message",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "call_1",
                    "toolName": "bash",
                    "isError": True,
                },
            },
        ]

        events, _meta, _flags, _diag = pi_messages.normalize_pi_entries(entries)

        self.assertEqual(events[0]["type"], "tool_result")
        self.assertEqual(events[0]["name"], "bash")
        self.assertTrue(events[0]["is_error"])
        self.assertEqual(events[0]["text"], "bash failed")

    def test_manage_todo_tool_result_emits_todo_snapshot(self) -> None:
        entries = [
            {
                "type": "message",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "todo_1",
                    "toolName": "manage_todo_list",
                    "content": [{"type": "text", "text": "Todos updated."}],
                    "details": {
                        "operation": "write",
                        "todos": [
                            {
                                "id": 1,
                                "title": "Investigate Pi events",
                                "status": "completed",
                                "description": "done",
                            },
                            {
                                "id": 2,
                                "title": "Build event stream",
                                "status": "in-progress",
                                "description": "active",
                            },
                        ],
                    },
                },
            },
        ]

        events, meta, _flags, diag = pi_messages.normalize_pi_entries(entries)

        self.assertEqual(events[0]["type"], "todo_snapshot")
        self.assertEqual(events[0]["operation"], "write")
        self.assertEqual(
            events[0]["counts"],
            {"total": 2, "completed": 1, "in_progress": 1, "not_started": 0},
        )
        self.assertEqual(events[0]["progress_text"], "1/2 completed")
        self.assertEqual(meta["tool"], 1)
        self.assertIn("manage_todo_list", diag["tool_names"])

    def test_claude_todo_v2_task_assignment_custom_message_is_normalized(self) -> None:
        entries = [
            {
                "type": "custom_message",
                "customType": "claude-todo-v2-task-assignment",
                "content": "Task #1 assigned to @Codex",
                "display": True,
                "details": {
                    "taskId": "1",
                    "taskListId": "3c7c0443-c037-47f4-8326-86c13e21403c",
                    "subject": "Clarify Claude Todo V2 compatibility goal",
                    "description": "Ask the user a focused requirement question.",
                    "owner": "Codex",
                    "assignedBy": "team-lead",
                    "timestamp": "2026-04-09T15:01:23.735Z",
                },
                "id": "fe60f4ff",
                "parentId": "46834883",
                "timestamp": "2026-04-09T15:02:02.525Z",
            },
        ]

        events, _meta, _flags, _diag = pi_messages.normalize_pi_entries(entries)

        self.assertEqual(events[0]["type"], "custom_message")
        self.assertEqual(events[0]["custom_type"], "claude-todo-v2-task-assignment")
        self.assertEqual(events[0]["text"], "Task #1 assigned to @Codex")
        self.assertEqual(events[0]["display"], True)
        self.assertEqual(events[0]["task_id"], "1")
        self.assertEqual(
            events[0]["task_list_id"], "3c7c0443-c037-47f4-8326-86c13e21403c"
        )
        self.assertEqual(
            events[0]["subject"], "Clarify Claude Todo V2 compatibility goal"
        )
        self.assertEqual(
            events[0]["description"], "Ask the user a focused requirement question."
        )
        self.assertEqual(events[0]["owner"], "Codex")
        self.assertEqual(events[0]["assigned_by"], "team-lead")
        self.assertEqual(events[0]["details"]["taskId"], "1")
        self.assertAlmostEqual(events[0]["ts"], 1775746922.525)

    def test_unknown_custom_message_falls_back_to_generic_normalized_event(
        self,
    ) -> None:
        entries = [
            {
                "type": "custom_message",
                "customType": "claude-todo-v2-task-note",
                "content": "Task note added",
                "display": False,
                "details": {
                    "taskId": "3",
                    "note": "Need another sample before folding into snapshot.",
                },
                "timestamp": "2026-04-09T15:02:10.000Z",
            },
        ]

        events, _meta, _flags, _diag = pi_messages.normalize_pi_entries(entries)

        self.assertEqual(events[0]["type"], "custom_message")
        self.assertEqual(events[0]["custom_type"], "claude-todo-v2-task-note")
        self.assertEqual(events[0]["text"], "Task note added")
        self.assertEqual(events[0]["display"], False)
        self.assertEqual(events[0]["details"]["taskId"], "3")
        self.assertIn("ts", events[0])

    def test_claude_todo_v2_context_custom_message_is_normalized(self) -> None:
        entries = [
            {
                "type": "custom_message",
                "customType": "claude-todo-v2-context",
                "content": "Claude Todo V2 task tools are active for task list 0752b114-9039-45b0-a64e-b21df86819f5. No shared tasks exist yet.",
                "display": False,
                "id": "e3b53b0c",
                "parentId": "6e10532a",
                "timestamp": "2026-04-09T15:42:12.720Z",
            },
        ]

        events, _meta, _flags, _diag = pi_messages.normalize_pi_entries(entries)

        self.assertEqual(events[0]["type"], "custom_message")
        self.assertEqual(events[0]["custom_type"], "claude-todo-v2-context")
        self.assertEqual(events[0]["display"], False)
        self.assertEqual(
            events[0]["task_list_id"], "0752b114-9039-45b0-a64e-b21df86819f5"
        )
        self.assertEqual(events[0]["has_shared_tasks"], False)
        self.assertEqual(
            events[0]["text"],
            "Claude Todo V2 task tools are active for task list 0752b114-9039-45b0-a64e-b21df86819f5. No shared tasks exist yet.",
        )

    def test_model_and_thinking_level_changes_are_ignored_without_event_stream(
        self,
    ) -> None:
        entries = [
            {
                "type": "model_change",
                "timestamp": "2026-04-03T01:02:03.000Z",
                "provider": "macaron",
                "modelId": "gpt-5.4",
            },
            {
                "type": "thinking_level_change",
                "timestamp": "2026-04-03T01:02:04.000Z",
                "thinkingLevel": "high",
            },
        ]

        events, _meta, _flags, _diag = pi_messages.normalize_pi_entries(
            entries, include_system=True
        )

        self.assertEqual(events, [])

    def test_unknown_pi_entry_is_ignored_without_event_stream(self) -> None:
        entries = [
            {
                "type": "custom_event",
                "timestamp": "2026-04-03T01:02:05.000Z",
                "payload": {"phase": "queued", "count": 3},
            },
        ]

        events, _meta, _flags, _diag = pi_messages.normalize_pi_entries(
            entries, include_system=True
        )

        self.assertEqual(events, [])

    def test_last_tool_clears_after_new_user_turn_without_tool_activity(self) -> None:
        entries = [
            _pi_message_entry("user", "first prompt"),
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "toolCall",
                            "id": "c1",
                            "name": "bash",
                            "arguments": {"command": "pwd"},
                        },
                        {"type": "text", "text": "done"},
                    ],
                },
            },
            _pi_message_entry("user", "follow-up prompt"),
        ]

        _events, _meta, _flags, diag = pi_messages.normalize_pi_entries(entries)

        self.assertIsNone(diag["last_tool"])

    def test_pi_busy_state_without_rollout_log_comes_from_broker(self) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            session_path = Path(td) / "pi-session.jsonl"
            session_path.write_text(
                '{"type":"session","session_id":"pi-thread-001"}\n', encoding="utf-8"
            )
            sock = Path(td) / "pi.sock"
            sock.touch()
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=123.0,
                cwd=td,
                log_path=None,
                sock_path=sock,
                session_path=session_path,
                busy=True,
                pi_busy_activity_floor=float(session_path.stat().st_mtime),
            )
            mgr._sock_call = lambda *_args, **_kwargs: {
                "busy": True,
                "queue_len": 0,
                "token": None,
            }  # type: ignore[method-assign]

            rows = mgr.list_sessions()
            payload = mgr.get_messages_page(
                "pi-session", offset=0, init=True, limit=20, before=0
            )

        self.assertTrue(rows[0]["busy"])
        self.assertTrue(payload["busy"])
        self.assertEqual(payload["events"], [])
        self.assertEqual(payload["diag"], {"tool_names": [], "last_tool": None})

    def test_pi_busy_not_overridden_by_idle_session_file_when_no_new_events(
        self,
    ) -> None:
        """After resume + send, the session file still contains old content
        ending with an assistant message.  The idle check on the file would
        return True, but since there are no new events (Pi hasn't written yet),
        the broker's busy flag should be trusted."""
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            session_path = Path(td) / "pi-session.jsonl"
            # Old session content ending with assistant message (idle by file inspection)
            lines = [
                '{"type":"session","id":"pi-thread-001","cwd":"/tmp","timestamp":"2026-03-28T12:00:00Z"}',
                '{"type":"message","message":{"role":"user","content":[{"type":"text","text":"hello"}]}}',
                '{"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"hi there"}]}}',
            ]
            session_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            initial_offset = session_path.stat().st_size

            sock = Path(td) / "pi.sock"
            sock.touch()
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=123.0,
                cwd=td,
                log_path=None,
                sock_path=sock,
                session_path=session_path,
                busy=True,
                pi_busy_activity_floor=float(session_path.stat().st_mtime),
            )
            # Broker reports busy (prompt just sent)
            mgr._sock_call = lambda *_args, **_kwargs: {
                "busy": True,
                "queue_len": 0,
                "token": None,
            }  # type: ignore[method-assign]

            # Delta poll from end of file (no new content) — broker says busy
            payload = mgr.get_messages_page(
                "pi-session", offset=initial_offset, init=False, limit=20, before=0
            )

        # busy should be preserved because no new events from the session file
        self.assertTrue(
            payload["busy"],
            "busy should not be overridden when session file has no new events",
        )

    def test_pi_busy_clears_when_session_file_advanced_past_busy_floor(self) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            session_path = Path(td) / "pi-session.jsonl"
            lines = [
                '{"type":"session","id":"pi-thread-001","cwd":"/tmp","timestamp":"2026-03-28T12:00:00Z"}',
                '{"type":"message","message":{"role":"user","content":[{"type":"text","text":"hello"}]}}',
                '{"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"hi there"}]}}',
            ]
            session_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            old_mtime = 1700000100.0
            os.utime(session_path, (old_mtime, old_mtime))
            initial_offset = session_path.stat().st_size

            with session_path.open("a", encoding="utf-8") as f:
                f.write(
                    '{"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"done"}]}}\n'
                )
            new_mtime = old_mtime + 5.0
            os.utime(session_path, (new_mtime, new_mtime))
            settled_offset = session_path.stat().st_size

            sock = Path(td) / "pi.sock"
            sock.touch()
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-001",
                agent_backend="pi",
                backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=123.0,
                cwd=td,
                log_path=None,
                sock_path=sock,
                session_path=session_path,
                busy=True,
                pi_busy_activity_floor=old_mtime,
            )
            mgr._sock_call = lambda *_args, **_kwargs: {
                "busy": True,
                "queue_len": 0,
                "token": None,
            }  # type: ignore[method-assign]

            payload = mgr.get_messages_page(
                "pi-session", offset=settled_offset, init=False, limit=20, before=0
            )

        self.assertFalse(
            payload["busy"],
            "busy should clear once the session file advances past the busy floor and is idle",
        )

    def test_pi_session_auto_discovery_does_not_switch_back_to_older_file(self) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            old_session_path = root / "old-pi-session.jsonl"
            new_session_path = root / "new-pi-session.jsonl"
            old_session_path.write_text(
                "\n".join(
                    [
                        '{"type":"session","id":"pi-thread-old","cwd":"/tmp","timestamp":"2026-03-28T12:00:00Z"}',
                        '{"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"old reply"}]}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            new_session_path.write_text(
                "\n".join(
                    [
                        '{"type":"session","id":"pi-thread-new","cwd":"/tmp","timestamp":"2026-03-28T12:05:00Z"}',
                        '{"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"new reply"}]}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            old_mtime = 1000.0
            new_mtime = 1010.0
            os.utime(old_session_path, (old_mtime, old_mtime))
            os.utime(new_session_path, (new_mtime, new_mtime))

            sock = root / "pi.sock"
            sock.touch()
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-old",
                agent_backend="pi",
                backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=900.0,
                cwd=td,
                log_path=None,
                sock_path=sock,
                session_path=old_session_path,
                busy=False,
                pi_session_path_discovered=True,
            )
            mgr._sock_call = lambda *_args, **_kwargs: {
                "busy": False,
                "queue_len": 0,
                "token": None,
            }  # type: ignore[method-assign]

            def fake_discover_pi_session_for_cwd(
                _cwd: str, _start_ts: float, *, exclude: set[Path] | None = None
            ) -> Path | None:
                blocked = set(exclude or set())
                if old_session_path in blocked:
                    return new_session_path
                if new_session_path in blocked:
                    return old_session_path
                return None

            with (
                patch("codoxear.server.time.time", return_value=2000.0),
                patch(
                    "codoxear.server._find_session_log_for_session_id",
                    return_value=None,
                ),
                patch(
                    "codoxear.server._discover_pi_session_for_cwd",
                    side_effect=fake_discover_pi_session_for_cwd,
                ),
                patch("codoxear.server._patch_metadata_session_path") as patch_meta,
            ):
                first = mgr.get_messages_page(
                    "pi-session",
                    offset=old_session_path.stat().st_size,
                    init=False,
                    limit=20,
                    before=0,
                )
                self.assertEqual(
                    mgr._sessions["pi-session"].session_path, new_session_path
                )

                second = mgr.get_messages_page(
                    "pi-session", offset=first["offset"], init=False, limit=20, before=0
                )

        self.assertEqual(mgr._sessions["pi-session"].session_path, new_session_path)
        self.assertEqual(
            [call.args[1] for call in patch_meta.call_args_list], [new_session_path]
        )
        self.assertTrue(any(ev.get("text") == "new reply" for ev in first["events"]))
        self.assertEqual(second["events"], [])

    def test_get_messages_page_lazy_pi_binding_prefers_exact_thread_match(
        self,
    ) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            exact_session_path = root / "exact-pi-session.jsonl"
            wrong_session_path = root / "wrong-pi-session.jsonl"
            exact_session_path.write_text(
                "\n".join(
                    [
                        '{"type":"session","id":"pi-thread-exact","cwd":"/tmp"}',
                        '{"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"exact reply"}]}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            wrong_session_path.write_text(
                "\n".join(
                    [
                        '{"type":"session","id":"pi-thread-wrong","cwd":"/tmp"}',
                        '{"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"wrong reply"}]}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            sock = root / "pi.sock"
            sock.touch()
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-exact",
                agent_backend="pi",
                backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=900.0,
                cwd=td,
                log_path=None,
                sock_path=sock,
                session_path=None,
                busy=False,
            )
            mgr._sock_call = lambda *_args, **_kwargs: {
                "busy": False,
                "queue_len": 0,
                "token": None,
            }  # type: ignore[method-assign]

            with (
                patch(
                    "codoxear.server._find_session_log_for_session_id",
                    return_value=exact_session_path,
                ) as find_exact,
                patch(
                    "codoxear.server._discover_pi_session_for_cwd",
                    return_value=wrong_session_path,
                ) as discover,
                patch("codoxear.server._patch_metadata_session_path") as patch_meta,
            ):
                payload = mgr.get_messages_page(
                    "pi-session", offset=0, init=True, limit=20, before=0
                )

        self.assertEqual(mgr._sessions["pi-session"].session_path, exact_session_path)
        self.assertTrue(
            any(ev.get("text") == "exact reply" for ev in payload["events"])
        )
        self.assertFalse(
            any(ev.get("text") == "wrong reply" for ev in payload["events"])
        )
        find_exact.assert_called()
        discover.assert_not_called()
        patch_meta.assert_called_once()
        self.assertEqual(patch_meta.call_args.args[:2], (sock, exact_session_path))

    def test_pi_session_stale_recovery_prefers_exact_thread_match_over_same_cwd_file(
        self,
    ) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            stale_session_path = root / "stale-pi-session.jsonl"
            exact_session_path = root / "exact-pi-session.jsonl"
            wrong_session_path = root / "wrong-pi-session.jsonl"
            stale_session_path.write_text(
                '{"type":"session","id":"pi-thread-old","cwd":"/tmp","timestamp":"2026-03-28T12:00:00Z"}\n',
                encoding="utf-8",
            )
            exact_session_path.write_text(
                "\n".join(
                    [
                        '{"type":"session","id":"pi-thread-exact","cwd":"/tmp","timestamp":"2026-03-28T12:05:00Z"}',
                        '{"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"exact reply"}]}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            wrong_session_path.write_text(
                "\n".join(
                    [
                        '{"type":"session","id":"pi-thread-wrong","cwd":"/tmp","timestamp":"2026-03-28T12:06:00Z"}',
                        '{"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"wrong reply"}]}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            os.utime(stale_session_path, (1000.0, 1000.0))
            os.utime(exact_session_path, (1008.0, 1008.0))
            os.utime(wrong_session_path, (1010.0, 1010.0))

            sock = root / "pi.sock"
            sock.touch()
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-exact",
                agent_backend="pi",
                backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=900.0,
                cwd=td,
                log_path=None,
                sock_path=sock,
                session_path=stale_session_path,
                busy=False,
                pi_session_path_discovered=True,
            )
            mgr._sock_call = lambda *_args, **_kwargs: {
                "busy": False,
                "queue_len": 0,
                "token": None,
            }  # type: ignore[method-assign]

            with (
                patch("codoxear.server.time.time", return_value=2000.0),
                patch(
                    "codoxear.server._find_session_log_for_session_id",
                    return_value=exact_session_path,
                ) as find_exact,
                patch(
                    "codoxear.server._discover_pi_session_for_cwd",
                    return_value=wrong_session_path,
                ) as discover,
                patch("codoxear.server._patch_metadata_session_path") as patch_meta,
            ):
                payload = mgr.get_messages_page(
                    "pi-session",
                    offset=stale_session_path.stat().st_size,
                    init=False,
                    limit=20,
                    before=0,
                )

        self.assertEqual(mgr._sessions["pi-session"].session_path, exact_session_path)
        self.assertTrue(
            any(ev.get("text") == "exact reply" for ev in payload["events"])
        )
        self.assertFalse(
            any(ev.get("text") == "wrong reply" for ev in payload["events"])
        )
        find_exact.assert_called()
        discover.assert_not_called()
        self.assertEqual(
            [call.args[1] for call in patch_meta.call_args_list], [exact_session_path]
        )

    def test_pi_explicit_session_path_does_not_drift_to_newer_same_cwd_file(
        self,
    ) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            old_session_path = root / "old-pi-session.jsonl"
            new_session_path = root / "new-pi-session.jsonl"
            old_session_path.write_text(
                "\n".join(
                    [
                        '{"type":"session","id":"pi-thread-old","cwd":"/tmp","timestamp":"2026-03-28T12:00:00Z"}',
                        '{"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"old reply"}]}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            new_session_path.write_text(
                "\n".join(
                    [
                        '{"type":"session","id":"pi-thread-new","cwd":"/tmp","timestamp":"2026-03-28T12:05:00Z"}',
                        '{"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"new reply"}]}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            old_mtime = 1000.0
            new_mtime = 1010.0
            os.utime(old_session_path, (old_mtime, old_mtime))
            os.utime(new_session_path, (new_mtime, new_mtime))

            sock = root / "pi.sock"
            sock.touch()
            mgr._sessions["pi-session"] = Session(
                session_id="pi-session",
                thread_id="pi-thread-old",
                agent_backend="pi",
                backend="pi",
                broker_pid=3333,
                codex_pid=4444,
                owned=True,
                start_ts=900.0,
                cwd=td,
                log_path=None,
                sock_path=sock,
                session_path=old_session_path,
                busy=False,
            )
            mgr._sock_call = lambda *_args, **_kwargs: {
                "busy": False,
                "queue_len": 0,
                "token": None,
            }  # type: ignore[method-assign]

            with (
                patch("codoxear.server.time.time", return_value=2000.0),
                patch(
                    "codoxear.server._discover_pi_session_for_cwd",
                    return_value=new_session_path,
                ) as discover,
                patch("codoxear.server._patch_metadata_session_path") as patch_meta,
            ):
                payload = mgr.get_messages_page(
                    "pi-session",
                    offset=old_session_path.stat().st_size,
                    init=False,
                    limit=20,
                    before=0,
                )

        self.assertEqual(mgr._sessions["pi-session"].session_path, old_session_path)
        self.assertEqual(payload["events"], [])
        patch_meta.assert_not_called()
        discover.assert_not_called()


if __name__ == "__main__":
    unittest.main()
