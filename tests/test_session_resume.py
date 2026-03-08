import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import ANY
from unittest.mock import patch

from codoxear.server import SessionManager
from codoxear.server import _first_user_message_preview_from_log
from codoxear.server import _list_resume_candidates_for_cwd


def _write_jsonl(path: Path, objs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(obj) + "\n" for obj in objs), encoding="utf-8")


class TestSessionResumeCandidates(unittest.TestCase):
    def test_list_resume_candidates_filters_same_cwd_and_dedupes(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            same_new = root / "rollout-2026-03-08T01-00-00-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa.jsonl"
            same_old = root / "rollout-2026-03-07T01-00-00-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa.jsonl"
            other = root / "rollout-2026-03-08T02-00-00-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb.jsonl"
            child = root / "rollout-2026-03-08T03-00-00-cccccccc-cccc-cccc-cccc-cccccccccccc.jsonl"

            _write_jsonl(same_new, [{"type": "session_meta", "payload": {"id": "resume-a", "cwd": "/repo", "timestamp": "2026-03-08T01:00:00Z", "source": "cli"}}])
            _write_jsonl(same_old, [{"type": "session_meta", "payload": {"id": "resume-a", "cwd": "/repo", "timestamp": "2026-03-07T01:00:00Z", "source": "cli"}}])
            _write_jsonl(other, [{"type": "session_meta", "payload": {"id": "resume-b", "cwd": "/elsewhere", "timestamp": "2026-03-08T02:00:00Z", "source": "cli"}}])
            _write_jsonl(
                child,
                [
                    {
                        "type": "session_meta",
                        "payload": {
                            "id": "resume-child",
                            "cwd": "/repo",
                            "source": {"subagent": {"thread_spawn": {"parent_thread_id": "resume-a", "depth": 1}}},
                        },
                    }
                ],
            )

            with patch("codoxear.server._iter_session_logs", return_value=[same_new, child, other, same_old]):
                rows = _list_resume_candidates_for_cwd("/repo", limit=10)

        self.assertEqual([row["session_id"] for row in rows], ["resume-a"])
        self.assertEqual(rows[0]["cwd"], "/repo")
        self.assertEqual(rows[0]["log_path"], str(same_new))

    def test_first_user_message_preview_skips_harness_scaffolding(self) -> None:
        with TemporaryDirectory() as td:
            log_path = Path(td) / "rollout-2026-03-08T01-00-00-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa.jsonl"
            _write_jsonl(
                log_path,
                [
                    {"type": "session_meta", "payload": {"id": "resume-a", "cwd": "/repo", "source": "cli"}},
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": "# AGENTS.md instructions for /repo\n..."},
                                {"type": "input_text", "text": "<environment_context>\n  <cwd>/repo</cwd>\n</environment_context>"},
                            ],
                        },
                    },
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": "Is it possible to extract something like the conversation title or at least the first user message?",
                                }
                            ],
                        },
                    },
                ],
            )

            preview = _first_user_message_preview_from_log(log_path)

        self.assertEqual(
            preview,
            "Is it possible to extract something like the conversation title or at least the first user message?",
        )


class TestSpawnWebSessionResume(unittest.TestCase):
    def test_spawn_web_session_passes_resume_id_to_broker(self) -> None:
        manager = SessionManager.__new__(SessionManager)
        thread_calls: list[str] = []

        class _Proc:
            pid = 4321
            stderr = None

            def wait(self) -> int:
                return 0

        with TemporaryDirectory() as td, patch("codoxear.server._list_resume_candidates_for_cwd", return_value=[{"session_id": "resume-a"}]), patch(
            "codoxear.server._wait_or_raise", return_value=None
        ), patch("codoxear.server.subprocess.Popen", return_value=_Proc()) as popen_mock, patch.object(threading.Thread, "start", lambda self: thread_calls.append("start")):
            result = SessionManager.spawn_web_session(
                manager,
                cwd=td,
                args=["--search"],
                resume_session_id="resume-a",
            )

        argv = popen_mock.call_args.args[0]
        self.assertEqual(
            argv,
            [
                ANY,
                "-m",
                "codoxear.broker",
                "--cwd",
                td,
                "--",
                "resume",
                "resume-a",
                "--search",
            ],
        )
        self.assertEqual(result, {"broker_pid": 4321})
        self.assertEqual(thread_calls, ["start"])

    def test_spawn_web_session_rejects_resume_id_not_in_cwd(self) -> None:
        manager = SessionManager.__new__(SessionManager)
        with TemporaryDirectory() as td, patch("codoxear.server._list_resume_candidates_for_cwd", return_value=[]):
            with self.assertRaisesRegex(ValueError, "resume session not found for cwd"):
                SessionManager.spawn_web_session(manager, cwd=td, resume_session_id="missing")


if __name__ == "__main__":
    unittest.main()
