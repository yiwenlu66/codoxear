from __future__ import annotations

import io
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from codoxear import pi_messages
from codoxear.server import Handler
from codoxear.server import Session


def _write_jsonl(path: Path, entries: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")


def _todo_result_entry(
    todos: object,
    *,
    call_id: str = "call_todo_1",
    is_error: bool = False,
) -> dict[str, object]:
    return {
        "type": "message",
        "message": {
            "role": "toolResult",
            "toolCallId": call_id,
            "toolName": "manage_todo_list",
            "isError": is_error,
            "details": {
                "operation": "write",
                "todos": todos,
            },
            "content": [
                {
                    "type": "text",
                    "text": "Todos have been modified successfully.",
                }
            ],
        },
    }


class TestPiTodoSnapshot(unittest.TestCase):
    def test_read_latest_pi_todo_snapshot_raises_for_missing_session_file(self) -> None:
        with TemporaryDirectory() as td:
            session_path = Path(td) / "missing-session.jsonl"

            with self.assertRaises(FileNotFoundError):
                pi_messages.read_latest_pi_todo_snapshot(session_path)

    def test_read_latest_pi_todo_snapshot_raises_for_unreadable_session_path(self) -> None:
        with TemporaryDirectory() as td:
            session_path = Path(td)

            with self.assertRaises(OSError):
                pi_messages.read_latest_pi_todo_snapshot(session_path)

    def test_read_latest_pi_todo_snapshot_returns_newest_valid_snapshot(self) -> None:
        with TemporaryDirectory() as td:
            session_path = Path(td) / "pi-session.jsonl"
            _write_jsonl(
                session_path,
                [
                    {"type": "session", "id": "pi-session-001"},
                    _todo_result_entry(
                        [
                            {
                                "id": 1,
                                "title": "Older task",
                                "description": "old snapshot",
                                "status": "completed",
                            }
                        ],
                        call_id="call_old",
                    ),
                    _todo_result_entry(
                        [
                            {
                                "id": 1,
                                "title": "Explore project context",
                                "description": "inspect files first",
                                "status": "completed",
                            },
                            {
                                "id": 2,
                                "title": "Ask clarifying questions",
                                "description": "confirm desired surface",
                                "status": "in-progress",
                            },
                            {
                                "id": 3,
                                "title": "Present design",
                                "description": "share recommended approach",
                                "status": "not-started",
                            },
                        ],
                        call_id="call_new",
                    ),
                ],
            )

            snapshot = pi_messages.read_latest_pi_todo_snapshot(session_path)

        assert snapshot is not None
        self.assertEqual(snapshot["progress_text"], "1/3 completed")
        self.assertEqual(
            snapshot["counts"],
            {
                "total": 3,
                "completed": 1,
                "in_progress": 1,
                "not_started": 1,
            },
        )
        self.assertEqual(snapshot["items"][0]["title"], "Explore project context")
        self.assertEqual(snapshot["items"][1]["status"], "in-progress")

    def test_read_latest_pi_todo_snapshot_skips_newer_malformed_result(self) -> None:
        with TemporaryDirectory() as td:
            session_path = Path(td) / "pi-session.jsonl"
            _write_jsonl(
                session_path,
                [
                    {"type": "session", "id": "pi-session-001"},
                    _todo_result_entry(
                        [
                            {
                                "id": 1,
                                "title": "Stable todo",
                                "description": "valid older entry",
                                "status": "completed",
                            }
                        ],
                        call_id="call_valid",
                    ),
                    _todo_result_entry({"broken": True}, call_id="call_bad"),
                ],
            )

            snapshot = pi_messages.read_latest_pi_todo_snapshot(session_path)

        assert snapshot is not None
        self.assertEqual(snapshot["progress_text"], "1/1 completed")
        self.assertEqual(snapshot["items"][0]["title"], "Stable todo")

    def test_read_latest_pi_todo_snapshot_falls_back_when_newer_todo_list_has_only_malformed_items(self) -> None:
        with TemporaryDirectory() as td:
            session_path = Path(td) / "pi-session.jsonl"
            _write_jsonl(
                session_path,
                [
                    {"type": "session", "id": "pi-session-001"},
                    _todo_result_entry(
                        [
                            {
                                "id": 1,
                                "title": "Recovered task",
                                "description": "older valid snapshot",
                                "status": "completed",
                            }
                        ],
                        call_id="call_valid",
                    ),
                    _todo_result_entry(
                        [
                            None,
                            "broken",
                            {"id": 2, "description": "missing title", "status": "completed"},
                            {"id": 3, "title": "   ", "status": "not-started"},
                        ],
                        call_id="call_malformed_list",
                    ),
                ],
            )

            snapshot = pi_messages.read_latest_pi_todo_snapshot(session_path)

        assert snapshot is not None
        self.assertEqual(snapshot["progress_text"], "1/1 completed")
        self.assertEqual(snapshot["items"][0]["title"], "Recovered task")

    def test_read_latest_pi_todo_snapshot_expands_scan_window_to_find_older_snapshot(self) -> None:
        with TemporaryDirectory() as td:
            session_path = Path(td) / "pi-session.jsonl"
            trailing_noise = "tail-noise-" * 12
            _write_jsonl(
                session_path,
                [
                    {"type": "session", "id": "pi-session-001"},
                    _todo_result_entry(
                        [
                            {
                                "id": 1,
                                "title": "Recovered task",
                                "description": "older snapshot still within scan cap",
                                "status": "completed",
                            }
                        ],
                        call_id="call_recovered",
                    ),
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": trailing_noise,
                                }
                            ],
                        },
                    },
                ],
            )

            with patch.object(pi_messages, "_PI_TODO_SCAN_START_BYTES", 128), patch.object(
                pi_messages,
                "_PI_TODO_SCAN_MAX_BYTES",
                1024,
            ):
                snapshot = pi_messages.read_latest_pi_todo_snapshot(session_path)

        assert snapshot is not None
        self.assertEqual(snapshot["items"][0]["title"], "Recovered task")

    def test_normalize_pi_todo_snapshot_strips_status_and_preserves_unknown_non_empty_values(self) -> None:
        snapshot = pi_messages._normalize_pi_todo_snapshot(
            [
                {"id": 1, "title": "Completed task", "status": " completed "},
                {"id": 2, "title": "Whitespace task", "status": "   \t  "},
                {"id": 3, "title": "Empty task", "status": ""},
                {"id": 4, "title": "Custom task", "status": " blocked "},
            ]
        )

        self.assertEqual(
            [item["status"] for item in snapshot["items"]],
            ["completed", "not-started", "not-started", "blocked"],
        )
        self.assertEqual(
            snapshot["counts"],
            {
                "total": 4,
                "completed": 1,
                "in_progress": 0,
                "not_started": 2,
            },
        )


class _HandlerHarness:
    def __init__(self, path: str) -> None:
        self.path = path
        self.headers = {"Content-Length": "0"}
        self.rfile = io.BytesIO(b"")
        self.wfile = io.BytesIO()
        self.status: int | None = None
        self.sent_headers: list[tuple[str, str]] = []

    def send_response(self, status: int) -> None:
        self.status = status

    def send_header(self, key: str, value: str) -> None:
        self.sent_headers.append((key, value))

    def end_headers(self) -> None:
        return


def _make_session(*, backend: str = "pi", session_path: Path | None = None) -> Session:
    return Session(
        session_id=f"{backend}-session",
        thread_id=f"{backend}-thread-001",
        backend=backend,
        broker_pid=3333,
        codex_pid=4444,
        owned=True,
        start_ts=123.0,
        cwd="/tmp/pi-cwd",
        log_path=None,
        sock_path=Path("/tmp/pi.sock"),
        session_path=session_path,
    )


class TestPiTodoDiagnostics(unittest.TestCase):
    def test_diagnostics_exposes_pi_session_file_path_and_file_activity_timestamp(self) -> None:
        handler = _HandlerHarness("/api/sessions/pi-session/diagnostics")
        with TemporaryDirectory() as td:
            session_path = Path(td) / "pi-session.jsonl"
            session_path.write_text('{"type":"session","id":"pi-session-001"}\n', encoding="utf-8")
            session_mtime = 1_700_000_100.0
            os.utime(session_path, (session_mtime, session_mtime))
            session = _make_session(session_path=session_path)
            session.last_chat_ts = 321.0

            with patch("codoxear.server._require_auth", return_value=True), \
                patch("codoxear.server.MANAGER") as manager, \
                patch("codoxear.server._current_git_branch", return_value=None), \
                patch("codoxear.server._pi_messages.read_latest_pi_todo_snapshot", return_value=None):
                manager.refresh_session_meta.return_value = None
                manager.get_session.return_value = session
                manager.get_state.return_value = {"busy": False, "queue_len": 0, "token": None}
                manager._queue_len.return_value = 0
                manager.sidebar_meta_get.return_value = {
                    "priority_offset": 0.0,
                    "snooze_until": None,
                    "dependency_session_id": None,
                }

                Handler.do_GET(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(payload["log_path"], None)
        self.assertEqual(payload["session_file_path"], str(session_path))
        self.assertEqual(payload["updated_ts"], session_mtime)

    def test_diagnostics_includes_todo_snapshot_for_pi_session(self) -> None:
        handler = _HandlerHarness("/api/sessions/pi-session/diagnostics")
        session = _make_session(session_path=Path("/tmp/pi-session.jsonl"))
        snapshot = {
            "items": [
                {
                    "id": 1,
                    "title": "Explore project context",
                    "description": "inspect files first",
                    "status": "completed",
                }
            ],
            "counts": {"total": 1, "completed": 1, "in_progress": 0, "not_started": 0},
            "progress_text": "1/1 completed",
        }

        with patch("codoxear.server._require_auth", return_value=True), \
            patch("codoxear.server.MANAGER") as manager, \
            patch("codoxear.server._current_git_branch", return_value=None), \
            patch("codoxear.server._pi_messages.read_latest_pi_todo_snapshot", return_value=snapshot):
            manager.refresh_session_meta.return_value = None
            manager.get_session.return_value = session
            manager.get_state.return_value = {"busy": False, "queue_len": 0, "token": None}
            manager._queue_len.return_value = 0
            manager.sidebar_meta_get.return_value = {
                "priority_offset": 0.0,
                "snooze_until": None,
                "dependency_session_id": None,
            }

            Handler.do_GET(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(handler.status, 200)
        self.assertEqual(payload["todo_snapshot"]["available"], True)
        self.assertEqual(payload["todo_snapshot"]["error"], False)
        self.assertEqual(payload["todo_snapshot"]["progress_text"], "1/1 completed")
        self.assertEqual(payload["todo_snapshot"]["items"][0]["title"], "Explore project context")

    def test_diagnostics_returns_empty_todo_snapshot_when_pi_session_has_no_todos(self) -> None:
        handler = _HandlerHarness("/api/sessions/pi-session/diagnostics")
        session = _make_session(session_path=Path("/tmp/pi-session.jsonl"))

        with patch("codoxear.server._require_auth", return_value=True), \
            patch("codoxear.server.MANAGER") as manager, \
            patch("codoxear.server._current_git_branch", return_value=None), \
            patch("codoxear.server._pi_messages.read_latest_pi_todo_snapshot", return_value=None):
            manager.refresh_session_meta.return_value = None
            manager.get_session.return_value = session
            manager.get_state.return_value = {"busy": False, "queue_len": 0, "token": None}
            manager._queue_len.return_value = 0
            manager.sidebar_meta_get.return_value = {
                "priority_offset": 0.0,
                "snooze_until": None,
                "dependency_session_id": None,
            }

            Handler.do_GET(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(
            payload["todo_snapshot"],
            {"available": False, "error": False, "items": []},
        )

    def test_diagnostics_marks_todo_unavailable_on_reader_error(self) -> None:
        handler = _HandlerHarness("/api/sessions/pi-session/diagnostics")
        session = _make_session(session_path=Path("/tmp/pi-session.jsonl"))

        with patch("codoxear.server._require_auth", return_value=True), \
            patch("codoxear.server.MANAGER") as manager, \
            patch("codoxear.server._current_git_branch", return_value=None), \
            patch("codoxear.server._pi_messages.read_latest_pi_todo_snapshot", side_effect=OSError("boom")):
            manager.refresh_session_meta.return_value = None
            manager.get_session.return_value = session
            manager.get_state.return_value = {"busy": False, "queue_len": 0, "token": None}
            manager._queue_len.return_value = 0
            manager.sidebar_meta_get.return_value = {
                "priority_offset": 0.0,
                "snooze_until": None,
                "dependency_session_id": None,
            }

            Handler.do_GET(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(
            payload["todo_snapshot"],
            {"available": False, "error": True, "items": []},
        )

    def test_diagnostics_returns_empty_todo_snapshot_when_snapshot_file_is_missing(self) -> None:
        handler = _HandlerHarness("/api/sessions/pi-session/diagnostics")
        session = _make_session(session_path=Path("/tmp/pi-session.jsonl"))

        with patch("codoxear.server._require_auth", return_value=True), \
            patch("codoxear.server.MANAGER") as manager, \
            patch("codoxear.server._current_git_branch", return_value=None), \
            patch("codoxear.server._pi_messages.read_latest_pi_todo_snapshot", side_effect=FileNotFoundError("missing")):
            manager.refresh_session_meta.return_value = None
            manager.get_session.return_value = session
            manager.get_state.return_value = {"busy": False, "queue_len": 0, "token": None}
            manager._queue_len.return_value = 0
            manager.sidebar_meta_get.return_value = {
                "priority_offset": 0.0,
                "snooze_until": None,
                "dependency_session_id": None,
            }

            Handler.do_GET(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(
            payload["todo_snapshot"],
            {"available": False, "error": False, "items": []},
        )

    def test_diagnostics_returns_empty_todo_snapshot_when_pi_session_path_is_none(self) -> None:
        handler = _HandlerHarness("/api/sessions/pi-session/diagnostics")
        session = _make_session(session_path=None)

        with patch("codoxear.server._require_auth", return_value=True), \
            patch("codoxear.server.MANAGER") as manager, \
            patch("codoxear.server._current_git_branch", return_value=None), \
            patch("codoxear.server._pi_messages.read_latest_pi_todo_snapshot") as read_snapshot:
            manager.refresh_session_meta.return_value = None
            manager.get_session.return_value = session
            manager.get_state.return_value = {"busy": False, "queue_len": 0, "token": None}
            manager._queue_len.return_value = 0
            manager.sidebar_meta_get.return_value = {
                "priority_offset": 0.0,
                "snooze_until": None,
                "dependency_session_id": None,
            }

            Handler.do_GET(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(
            payload["todo_snapshot"],
            {"available": False, "error": False, "items": []},
        )
        read_snapshot.assert_not_called()

    def test_diagnostics_keeps_non_pi_sessions_on_empty_todo_snapshot(self) -> None:
        handler = _HandlerHarness("/api/sessions/codex-session/diagnostics")
        session = _make_session(backend="codex", session_path=None)

        with patch("codoxear.server._require_auth", return_value=True), \
            patch("codoxear.server.MANAGER") as manager, \
            patch("codoxear.server._current_git_branch", return_value=None), \
            patch("codoxear.server._pi_messages.read_latest_pi_todo_snapshot") as read_snapshot:
            manager.refresh_session_meta.return_value = None
            manager.get_session.return_value = session
            manager.get_state.return_value = {"busy": False, "queue_len": 0, "token": None}
            manager._queue_len.return_value = 0
            manager.sidebar_meta_get.return_value = {
                "priority_offset": 0.0,
                "snooze_until": None,
                "dependency_session_id": None,
            }

            Handler.do_GET(handler)  # type: ignore[arg-type]

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(payload["backend"], "codex")
        self.assertEqual(
            payload["todo_snapshot"],
            {"available": False, "error": False, "items": []},
        )
        read_snapshot.assert_not_called()
