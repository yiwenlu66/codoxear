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
    path.write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8"
    )


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


def _task_create_result_entry(
    task_id: str, subject: str, *, task_list_id: str = "task-list-1"
) -> dict[str, object]:
    return {
        "type": "message",
        "message": {
            "role": "toolResult",
            "toolCallId": f"task_create_{task_id}",
            "toolName": "TaskCreate",
            "content": [{"type": "text", "text": f"Task #{task_id} created"}],
            "details": {
                "success": True,
                "taskListId": task_list_id,
                "task": {"id": task_id, "subject": subject},
            },
            "isError": False,
        },
    }


def _task_update_result_entry(
    task_id: str,
    *,
    task_list_id: str = "task-list-1",
    from_status: str,
    to_status: str,
    updated_fields: list[str] | None = None,
) -> dict[str, object]:
    return {
        "type": "message",
        "message": {
            "role": "toolResult",
            "toolCallId": f"task_update_{task_id}_{to_status}",
            "toolName": "TaskUpdate",
            "content": [{"type": "text", "text": f"Task #{task_id} updated"}],
            "details": {
                "success": True,
                "taskId": task_id,
                "taskListId": task_list_id,
                "updatedFields": updated_fields or ["status"],
                "statusChange": {"from": from_status, "to": to_status},
                "verificationNudgeNeeded": False,
            },
            "isError": False,
        },
    }


def _task_assignment_entry(
    task_id: str,
    subject: str,
    *,
    description: str,
    owner: str,
    assigned_by: str,
    task_list_id: str = "task-list-1",
    timestamp: str = "2026-04-09T15:02:02.525Z",
) -> dict[str, object]:
    return {
        "type": "custom_message",
        "customType": "claude-todo-v2-task-assignment",
        "content": f"Task #{task_id} assigned to @{owner}",
        "display": True,
        "details": {
            "taskId": task_id,
            "taskListId": task_list_id,
            "subject": subject,
            "description": description,
            "owner": owner,
            "assignedBy": assigned_by,
            "timestamp": timestamp,
        },
        "timestamp": timestamp,
    }


def _task_context_entry(*, task_list_id: str = "task-list-1") -> dict[str, object]:
    return {
        "type": "custom_message",
        "customType": "claude-todo-v2-context",
        "content": f"Claude Todo V2 task tools are active for task list {task_list_id}. No shared tasks exist yet.",
        "display": False,
        "timestamp": "2026-04-09T14:59:29.988Z",
    }


def _todo_v2_state_entry(
    *,
    panel_enabled: bool,
    timestamp: str = "2026-04-10T09:58:42.627Z",
    last_activation_key: str = "task-list-1:task-list-1",
) -> dict[str, object]:
    return {
        "type": "custom",
        "customType": "claude-todo-v2-state",
        "data": {
            "panelEnabled": panel_enabled,
            "lastActivationKey": last_activation_key,
        },
        "timestamp": timestamp,
    }


def _todo_write_result_entry(
    new_todos: object,
    *,
    task_list_id: str = "task-list-1",
    timestamp: str = "2026-04-10T09:58:52.177Z",
    call_id: str = "todo_write_1",
    is_error: bool = False,
) -> dict[str, object]:
    return {
        "type": "message",
        "message": {
            "role": "toolResult",
            "toolCallId": call_id,
            "toolName": "TodoWrite",
            "content": [{"type": "text", "text": "Todos updated."}],
            "details": {
                "taskListId": task_list_id,
                "newTodos": new_todos,
            },
            "isError": is_error,
            "timestamp": timestamp,
        },
    }


class TestPiTodoSnapshot(unittest.TestCase):
    def test_read_latest_pi_todo_snapshot_raises_for_missing_session_file(self) -> None:
        with TemporaryDirectory() as td:
            session_path = Path(td) / "missing-session.jsonl"

            with self.assertRaises(FileNotFoundError):
                pi_messages.read_latest_pi_todo_snapshot(session_path)

    def test_read_latest_pi_todo_snapshot_raises_for_unreadable_session_path(
        self,
    ) -> None:
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

    def test_read_latest_pi_todo_snapshot_falls_back_when_newer_todo_list_has_only_malformed_items(
        self,
    ) -> None:
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
                            {
                                "id": 2,
                                "description": "missing title",
                                "status": "completed",
                            },
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

    def test_read_latest_pi_todo_snapshot_expands_scan_window_to_find_older_snapshot(
        self,
    ) -> None:
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

            with (
                patch.object(pi_messages, "_PI_TODO_SCAN_START_BYTES", 128),
                patch.object(
                    pi_messages,
                    "_PI_TODO_SCAN_MAX_BYTES",
                    1024,
                ),
            ):
                snapshot = pi_messages.read_latest_pi_todo_snapshot(session_path)

        assert snapshot is not None
        self.assertEqual(snapshot["items"][0]["title"], "Recovered task")

    def test_read_latest_pi_todo_snapshot_aggregates_claude_todo_v2_task_state(
        self,
    ) -> None:
        with TemporaryDirectory() as td:
            session_path = Path(td) / "pi-session.jsonl"
            _write_jsonl(
                session_path,
                [
                    {"type": "session", "id": "pi-session-001"},
                    _task_context_entry(task_list_id="task-list-42"),
                    _task_create_result_entry(
                        "3",
                        "Explore current todo implementation",
                        task_list_id="task-list-42",
                    ),
                    _task_assignment_entry(
                        "3",
                        "Explore current todo implementation",
                        description="Inspect the existing web session todo feature.",
                        owner="Codex",
                        assigned_by="team-lead",
                        task_list_id="task-list-42",
                        timestamp="2026-04-09T15:01:41.841Z",
                    ),
                    _task_update_result_entry(
                        "3",
                        task_list_id="task-list-42",
                        from_status="pending",
                        to_status="in_progress",
                    ),
                    _task_create_result_entry(
                        "1", "Clarify compatibility goal", task_list_id="task-list-42"
                    ),
                    _task_update_result_entry(
                        "1",
                        task_list_id="task-list-42",
                        from_status="pending",
                        to_status="completed",
                    ),
                ],
            )

            snapshot = pi_messages.read_latest_pi_todo_snapshot(session_path)

        assert snapshot is not None
        self.assertEqual(snapshot["progress_text"], "1/2 completed")
        self.assertEqual(
            snapshot["counts"],
            {"total": 2, "completed": 1, "in_progress": 1, "not_started": 0},
        )
        self.assertEqual(
            snapshot["items"][0]["title"], "Explore current todo implementation"
        )
        self.assertEqual(snapshot["items"][0]["status"], "in-progress")
        self.assertEqual(
            snapshot["items"][0]["description"],
            "Inspect the existing web session todo feature.",
        )
        self.assertEqual(snapshot["items"][0]["owner"], "Codex")
        self.assertEqual(snapshot["items"][0]["assigned_by"], "team-lead")
        self.assertEqual(snapshot["items"][1]["status"], "completed")

    def test_read_latest_pi_todo_snapshot_prefers_manage_todo_list_over_claude_aggregate(
        self,
    ) -> None:
        with TemporaryDirectory() as td:
            session_path = Path(td) / "pi-session.jsonl"
            _write_jsonl(
                session_path,
                [
                    {"type": "session", "id": "pi-session-001"},
                    _task_create_result_entry(
                        "3",
                        "Explore current todo implementation",
                        task_list_id="task-list-42",
                    ),
                    _task_assignment_entry(
                        "3",
                        "Explore current todo implementation",
                        description="Inspect the existing web session todo feature.",
                        owner="Codex",
                        assigned_by="team-lead",
                        task_list_id="task-list-42",
                    ),
                    _todo_result_entry(
                        [
                            {
                                "id": 9,
                                "title": "Explicit todo snapshot",
                                "description": "authoritative tool output",
                                "status": "completed",
                            }
                        ],
                        call_id="call_authoritative",
                    ),
                ],
            )

            snapshot = pi_messages.read_latest_pi_todo_snapshot(session_path)

        assert snapshot is not None
        self.assertEqual(snapshot["items"][0]["title"], "Explicit todo snapshot")
        self.assertEqual(snapshot["progress_text"], "1/1 completed")

    def test_read_latest_pi_todo_snapshot_returns_none_for_context_without_tasks(
        self,
    ) -> None:
        with TemporaryDirectory() as td:
            session_path = Path(td) / "pi-session.jsonl"
            _write_jsonl(
                session_path,
                [
                    {"type": "session", "id": "pi-session-001"},
                    _task_context_entry(task_list_id="task-list-42"),
                ],
            )

            snapshot = pi_messages.read_latest_pi_todo_snapshot(session_path)

        self.assertIsNone(snapshot)

    def test_read_latest_pi_todo_snapshot_ignores_state_without_todos(self) -> None:
        with TemporaryDirectory() as td:
            session_path = Path(td) / "pi-session.jsonl"
            _write_jsonl(
                session_path,
                [
                    {"type": "session", "id": "pi-session-001"},
                    _todo_v2_state_entry(panel_enabled=True),
                ],
            )

            snapshot = pi_messages.read_latest_pi_todo_snapshot(session_path)

        self.assertIsNone(snapshot)

    def test_read_latest_pi_todo_snapshot_reads_claude_todo_v2_todowrite_snapshot(
        self,
    ) -> None:
        with TemporaryDirectory() as td:
            session_path = Path(td) / "pi-session.jsonl"
            _write_jsonl(
                session_path,
                [
                    {"type": "session", "id": "pi-session-001"},
                    _todo_v2_state_entry(panel_enabled=True),
                    _todo_write_result_entry(
                        [
                            {
                                "content": "Explore project context for todo state rendering",
                                "status": "completed",
                                "activeForm": "Exploring project context for todo state rendering",
                            },
                            {
                                "content": "Transition from design into implementation planning",
                                "status": "pending",
                                "activeForm": "Transitioning from design into implementation planning",
                            },
                        ]
                    ),
                ],
            )

            snapshot = pi_messages.read_latest_pi_todo_snapshot(session_path)

        assert snapshot is not None
        self.assertEqual(snapshot["progress_text"], "1/2 completed")
        self.assertEqual(
            snapshot["counts"],
            {"total": 2, "completed": 1, "in_progress": 0, "not_started": 1},
        )
        self.assertEqual(
            snapshot["items"][0]["title"],
            "Explore project context for todo state rendering",
        )
        self.assertEqual(
            snapshot["items"][0]["description"],
            "Exploring project context for todo state rendering",
        )
        self.assertEqual(snapshot["items"][0]["source"], "claude-todo-v2")
        self.assertEqual(snapshot["items"][1]["status"], "not-started")

    def test_read_latest_pi_todo_snapshot_prefers_latest_claude_todo_v2_todowrite_snapshot(
        self,
    ) -> None:
        with TemporaryDirectory() as td:
            session_path = Path(td) / "pi-session.jsonl"
            _write_jsonl(
                session_path,
                [
                    {"type": "session", "id": "pi-session-001"},
                    _todo_v2_state_entry(panel_enabled=True),
                    _todo_write_result_entry(
                        [
                            {
                                "content": "Older snapshot",
                                "status": "completed",
                                "activeForm": "Done",
                            }
                        ],
                        call_id="todo_write_old",
                        timestamp="2026-04-10T09:58:52.177Z",
                    ),
                    _todo_write_result_entry(
                        [
                            {
                                "content": "New snapshot",
                                "status": "in_progress",
                                "activeForm": "Working now",
                            }
                        ],
                        call_id="todo_write_new",
                        timestamp="2026-04-10T09:59:52.177Z",
                    ),
                ],
            )

            snapshot = pi_messages.read_latest_pi_todo_snapshot(session_path)

        assert snapshot is not None
        self.assertEqual(snapshot["items"][0]["title"], "New snapshot")
        self.assertEqual(snapshot["items"][0]["status"], "in-progress")

    def test_read_latest_pi_todo_snapshot_hides_disabled_claude_todo_v2_panel(
        self,
    ) -> None:
        with TemporaryDirectory() as td:
            session_path = Path(td) / "pi-session.jsonl"
            _write_jsonl(
                session_path,
                [
                    {"type": "session", "id": "pi-session-001"},
                    _todo_v2_state_entry(panel_enabled=False),
                    _todo_write_result_entry(
                        [
                            {
                                "content": "Should stay hidden",
                                "status": "completed",
                                "activeForm": "Done",
                            }
                        ]
                    ),
                ],
            )

            snapshot = pi_messages.read_latest_pi_todo_snapshot(session_path)

        self.assertIsNone(snapshot)

    def test_normalize_pi_todo_snapshot_strips_status_and_preserves_unknown_non_empty_values(
        self,
    ) -> None:
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
        agent_backend=backend,
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
    def test_diagnostics_exposes_pi_session_file_path_and_file_activity_timestamp(
        self,
    ) -> None:
        handler = _HandlerHarness("/api/sessions/pi-session/diagnostics")
        with TemporaryDirectory() as td:
            session_path = Path(td) / "pi-session.jsonl"
            session_path.write_text(
                '{"type":"session","id":"pi-session-001"}\n', encoding="utf-8"
            )
            session_mtime = 1_700_000_100.0
            os.utime(session_path, (session_mtime, session_mtime))
            session = _make_session(session_path=session_path)
            session.last_chat_ts = 321.0

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
                    "queue_len": 0,
                    "token": None,
                }
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
        self.assertEqual(payload["updated_ts"], 321.0)

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

        with (
            patch("codoxear.server._require_auth", return_value=True),
            patch("codoxear.server.MANAGER") as manager,
            patch("codoxear.server._current_git_branch", return_value=None),
            patch(
                "codoxear.server._pi_messages.read_latest_pi_todo_snapshot",
                return_value=snapshot,
            ),
        ):
            manager.refresh_session_meta.return_value = None
            manager.get_session.return_value = session
            manager.get_state.return_value = {
                "busy": False,
                "queue_len": 0,
                "token": None,
            }
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
        self.assertEqual(
            payload["todo_snapshot"]["items"][0]["title"], "Explore project context"
        )

    def test_diagnostics_returns_empty_todo_snapshot_when_pi_session_has_no_todos(
        self,
    ) -> None:
        handler = _HandlerHarness("/api/sessions/pi-session/diagnostics")
        session = _make_session(session_path=Path("/tmp/pi-session.jsonl"))

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
                "queue_len": 0,
                "token": None,
            }
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

        with (
            patch("codoxear.server._require_auth", return_value=True),
            patch("codoxear.server.MANAGER") as manager,
            patch("codoxear.server._current_git_branch", return_value=None),
            patch(
                "codoxear.server._pi_messages.read_latest_pi_todo_snapshot",
                side_effect=OSError("boom"),
            ),
        ):
            manager.refresh_session_meta.return_value = None
            manager.get_session.return_value = session
            manager.get_state.return_value = {
                "busy": False,
                "queue_len": 0,
                "token": None,
            }
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

    def test_diagnostics_returns_empty_todo_snapshot_when_snapshot_file_is_missing(
        self,
    ) -> None:
        handler = _HandlerHarness("/api/sessions/pi-session/diagnostics")
        session = _make_session(session_path=Path("/tmp/pi-session.jsonl"))

        with (
            patch("codoxear.server._require_auth", return_value=True),
            patch("codoxear.server.MANAGER") as manager,
            patch("codoxear.server._current_git_branch", return_value=None),
            patch(
                "codoxear.server._pi_messages.read_latest_pi_todo_snapshot",
                side_effect=FileNotFoundError("missing"),
            ),
        ):
            manager.refresh_session_meta.return_value = None
            manager.get_session.return_value = session
            manager.get_state.return_value = {
                "busy": False,
                "queue_len": 0,
                "token": None,
            }
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

    def test_diagnostics_returns_empty_todo_snapshot_when_pi_session_path_is_none(
        self,
    ) -> None:
        handler = _HandlerHarness("/api/sessions/pi-session/diagnostics")
        session = _make_session(session_path=None)

        with (
            patch("codoxear.server._require_auth", return_value=True),
            patch("codoxear.server.MANAGER") as manager,
            patch("codoxear.server._current_git_branch", return_value=None),
            patch(
                "codoxear.server._pi_messages.read_latest_pi_todo_snapshot"
            ) as read_snapshot,
        ):
            manager.refresh_session_meta.return_value = None
            manager.get_session.return_value = session
            manager.get_state.return_value = {
                "busy": False,
                "queue_len": 0,
                "token": None,
            }
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

        with (
            patch("codoxear.server._require_auth", return_value=True),
            patch("codoxear.server.MANAGER") as manager,
            patch("codoxear.server._current_git_branch", return_value=None),
            patch(
                "codoxear.server._pi_messages.read_latest_pi_todo_snapshot"
            ) as read_snapshot,
        ):
            manager.refresh_session_meta.return_value = None
            manager.get_session.return_value = session
            manager.get_state.return_value = {
                "busy": False,
                "queue_len": 0,
                "token": None,
            }
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
