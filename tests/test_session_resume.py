import json
import os
import subprocess
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import ANY
from unittest.mock import patch

from codoxear.server import Session
from codoxear.server import SessionManager
from codoxear.server import _create_git_worktree
from codoxear.server import _describe_session_cwd
from codoxear.server import _default_worktree_path
from codoxear.server import _first_user_message_preview_from_log
from codoxear.server import _first_user_message_preview_from_pi_session
from codoxear.server import _list_resume_candidates_for_cwd


def _write_jsonl(path: Path, objs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(obj) + "\n" for obj in objs), encoding="utf-8")


class TestSessionResumeCandidates(unittest.TestCase):
    def test_describe_session_cwd_marks_missing_dir_for_creation(self) -> None:
        with TemporaryDirectory() as td:
            target = Path(td) / "missing" / "child"

            info = _describe_session_cwd(target.resolve())

        self.assertEqual(info["cwd"], str(target.resolve()))
        self.assertFalse(info["exists"])
        self.assertTrue(info["will_create"])
        self.assertFalse(info["git_repo"])

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

    def test_list_resume_candidates_supports_pi_native_sessions_for_same_cwd(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            pi_root = root / "pi-native"
            session_dir = pi_root / "--repo--"
            session_dir.mkdir(parents=True, exist_ok=True)
            newest = session_dir / "2026-03-28T12-00-00-000Z_aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa.jsonl"
            older = session_dir / "2026-03-27T12-00-00-000Z_bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb.jsonl"
            other_dir = pi_root / "--elsewhere--"
            other_dir.mkdir(parents=True, exist_ok=True)
            other = other_dir / "2026-03-29T12-00-00-000Z_cccccccc-cccc-cccc-cccc-cccccccccccc.jsonl"

            _write_jsonl(
                newest,
                [
                    {"type": "session", "id": "pi-session-new", "cwd": "/repo", "timestamp": "2026-03-28T12:00:00.000Z"},
                    {
                        "type": "message",
                        "message": {
                            "role": "user",
                            "content": [{"type": "text", "text": "Newest Pi session prompt"}],
                            "timestamp": 1,
                        },
                    },
                ],
            )
            _write_jsonl(
                older,
                [
                    {"type": "session", "id": "pi-session-old", "cwd": "/repo", "timestamp": "2026-03-27T12:00:00.000Z"},
                ],
            )
            _write_jsonl(
                other,
                [
                    {"type": "session", "id": "pi-session-other", "cwd": "/elsewhere", "timestamp": "2026-03-29T12:00:00.000Z"},
                ],
            )
            newer_mtime = 2000
            older_mtime = 1000
            other_mtime = 3000
            os.utime(newest, (newer_mtime, newer_mtime))
            os.utime(older, (older_mtime, older_mtime))
            os.utime(other, (other_mtime, other_mtime))

            with patch("codoxear.server.PI_NATIVE_SESSIONS_DIR", pi_root):
                rows = _list_resume_candidates_for_cwd("/repo", limit=10, backend="pi")

        self.assertEqual([row["session_id"] for row in rows], ["pi-session-new", "pi-session-old"])
        self.assertEqual(rows[0]["cwd"], "/repo")
        self.assertEqual(rows[0]["session_path"], str(newest))
        self.assertEqual(rows[0]["backend"], "pi")

    def test_list_resume_candidates_skips_pi_files_that_disappear_during_sort(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            pi_root = root / "pi-native"
            session_dir = pi_root / "--repo--"
            session_dir.mkdir(parents=True, exist_ok=True)
            stable = session_dir / "stable.jsonl"
            missing = session_dir / "missing.jsonl"

            _write_jsonl(stable, [{"type": "session", "id": "pi-stable", "cwd": "/repo", "timestamp": "2026-03-28T12:00:00.000Z"}])
            _write_jsonl(missing, [{"type": "session", "id": "pi-missing", "cwd": "/repo", "timestamp": "2026-03-28T11:00:00.000Z"}])

            real_stat = Path.stat

            def fake_stat(path_obj: Path, *args, **kwargs):
                if path_obj == missing:
                    raise FileNotFoundError(path_obj)
                return real_stat(path_obj, *args, **kwargs)

            with patch("codoxear.server.PI_NATIVE_SESSIONS_DIR", pi_root), patch("pathlib.Path.stat", autospec=True, side_effect=fake_stat):
                rows = _list_resume_candidates_for_cwd("/repo", limit=10, backend="pi")

        self.assertEqual([row["session_id"] for row in rows], ["pi-stable"])

    def test_list_resume_candidates_skips_unreadable_pi_files(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            pi_root = root / "pi-native"
            session_dir = pi_root / "--repo--"
            session_dir.mkdir(parents=True, exist_ok=True)
            stable = session_dir / "stable.jsonl"
            unreadable = session_dir / "unreadable.jsonl"

            _write_jsonl(stable, [{"type": "session", "id": "pi-stable", "cwd": "/repo", "timestamp": "2026-03-28T12:00:00.000Z"}])
            _write_jsonl(unreadable, [{"type": "session", "id": "pi-unreadable", "cwd": "/repo", "timestamp": "2026-03-28T11:00:00.000Z"}])

            real_open = Path.open

            def fake_open(path_obj: Path, *args, **kwargs):
                if path_obj == unreadable:
                    raise PermissionError(path_obj)
                return real_open(path_obj, *args, **kwargs)

            with patch("codoxear.server.PI_NATIVE_SESSIONS_DIR", pi_root), patch("pathlib.Path.open", autospec=True, side_effect=fake_open):
                rows = _list_resume_candidates_for_cwd("/repo", limit=10, backend="pi")

        self.assertEqual([row["session_id"] for row in rows], ["pi-stable"])

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

    def test_first_user_message_preview_from_pi_session_reads_persisted_payload_entries(self) -> None:
        with TemporaryDirectory() as td:
            session_path = Path(td) / "pi-session.jsonl"
            _write_jsonl(
                session_path,
                [
                    {"type": "session", "id": "pi-session", "cwd": "/repo", "timestamp": "2026-03-28T12:00:00.000Z"},
                    {
                        "type": "message",
                        "payload": {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {"type": "output_text", "text": "# AGENTS.md instructions for /repo\n..."},
                            ],
                        },
                    },
                    {
                        "type": "message",
                        "payload": {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {"type": "output_text", "text": "Summarize the current repository state."},
                            ],
                        },
                    },
                ],
            )

            preview = _first_user_message_preview_from_pi_session(session_path)

        self.assertEqual(preview, "Summarize the current repository state.")


class TestSpawnWebSessionResume(unittest.TestCase):
    def test_spawn_web_session_creates_missing_cwd(self) -> None:
        manager = SessionManager.__new__(SessionManager)
        thread_calls: list[str] = []

        class _Proc:
            pid = 2468
            stderr = None

            def wait(self) -> int:
                return 0

        with TemporaryDirectory() as td, patch("codoxear.server._wait_or_raise", return_value=None), patch(
            "codoxear.server.subprocess.Popen", return_value=_Proc()
        ) as popen_mock, patch.object(threading.Thread, "start", lambda self: thread_calls.append("start")):
            target = Path(td) / "new" / "session"
            result = SessionManager.spawn_web_session(manager, cwd=str(target))
            self.assertTrue(target.is_dir())

        argv = popen_mock.call_args.args[0]
        trust_override = f'projects={{ {json.dumps(str(target.resolve()))} = {{ trust_level = "trusted" }} }}'
        self.assertEqual(
            argv,
            [
                ANY,
                "-m",
                "codoxear.broker",
                "--cwd",
                str(target.resolve()),
                "--",
                "-c",
                trust_override,
                "--dangerously-bypass-approvals-and-sandbox",
            ],
        )
        self.assertEqual(result, {"broker_pid": 2468})
        self.assertEqual(thread_calls, ["start"])

    def test_spawn_web_session_surfaces_mkdir_failure(self) -> None:
        manager = SessionManager.__new__(SessionManager)
        with TemporaryDirectory() as td, patch("pathlib.Path.mkdir", side_effect=PermissionError(13, "Permission denied")):
            target = Path(td) / "blocked" / "session"
            with self.assertRaisesRegex(ValueError, r"cwd could not be created: .*Permission denied"):
                SessionManager.spawn_web_session(manager, cwd=str(target))

    def test_spawn_web_session_marks_spawn_cwd_trusted(self) -> None:
        manager = SessionManager.__new__(SessionManager)
        thread_calls: list[str] = []

        class _Proc:
            pid = 3210
            stderr = None

            def wait(self) -> int:
                return 0

        with TemporaryDirectory() as td, patch("codoxear.server._wait_or_raise", return_value=None), patch(
            "codoxear.server.subprocess.Popen", return_value=_Proc()
        ) as popen_mock, patch.object(threading.Thread, "start", lambda self: thread_calls.append("start")):
            result = SessionManager.spawn_web_session(manager, cwd=td, args=["--search"])

        argv = popen_mock.call_args.args[0]
        env = popen_mock.call_args.kwargs["env"]
        resolved_td = str(Path(td).resolve())
        trust_override = f'projects={{ {json.dumps(str(Path(td).resolve()))} = {{ trust_level = "trusted" }} }}'
        self.assertEqual(
            argv,
            [
                ANY,
                "-m",
                "codoxear.broker",
                "--cwd",
                resolved_td,
                "--",
                "-c",
                trust_override,
                "--dangerously-bypass-approvals-and-sandbox",
                "--search",
            ],
        )
        self.assertEqual(result, {"broker_pid": 3210})
        self.assertEqual(thread_calls, ["start"])

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
        env = popen_mock.call_args.kwargs["env"]
        resolved_td = str(Path(td).resolve())
        trust_override = f'projects={{ {json.dumps(str(Path(td).resolve()))} = {{ trust_level = "trusted" }} }}'
        self.assertEqual(
            argv,
            [
                ANY,
                "-m",
                "codoxear.broker",
                "--cwd",
                resolved_td,
                "--",
                "-c",
                trust_override,
                "--dangerously-bypass-approvals-and-sandbox",
                "resume",
                "resume-a",
                "--search",
            ],
        )
        self.assertEqual(env["CODEX_WEB_RESUME_SESSION_ID"], "resume-a")
        self.assertEqual(result, {"broker_pid": 4321})
        self.assertEqual(thread_calls, ["start"])

    def test_spawn_web_session_passes_pi_resume_session_file_to_pi_broker(self) -> None:
        manager = SessionManager.__new__(SessionManager)
        thread_calls: list[str] = []

        class _Proc:
            pid = 5321
            stderr = None

            def wait(self) -> int:
                return 0

        resume_file = Path("/tmp/native-pi-session.jsonl")
        with TemporaryDirectory() as td, patch(
            "codoxear.server._list_resume_candidates_for_cwd",
            return_value=[{"session_id": "pi-resume-a", "session_path": str(resume_file)}],
        ), patch("codoxear.server._wait_or_raise", return_value=None), patch(
            "codoxear.server.subprocess.Popen", return_value=_Proc()
        ) as popen_mock, patch.object(threading.Thread, "start", lambda self: thread_calls.append("start")):
            result = SessionManager.spawn_web_session(
                manager,
                cwd=td,
                backend="pi",
                resume_session_id="pi-resume-a",
            )

        argv = popen_mock.call_args.args[0]
        self.assertEqual(
            argv,
            [
                ANY,
                "-m",
                "codoxear.pi_broker",
                "--cwd",
                str(Path(td).resolve()),
                "--session-file",
                str(resume_file),
            ],
        )
        self.assertEqual(result, {"broker_pid": 5321, "backend": "pi"})
        self.assertEqual(thread_calls, ["start"])

    def test_spawn_web_session_passes_model_and_reasoning_to_broker(self) -> None:
        manager = SessionManager.__new__(SessionManager)
        thread_calls: list[str] = []

        class _Proc:
            pid = 6543
            stderr = None

            def wait(self) -> int:
                return 0

        with TemporaryDirectory() as td, patch("codoxear.server._wait_or_raise", return_value=None), patch(
            "codoxear.server.subprocess.Popen", return_value=_Proc()
        ) as popen_mock, patch.object(threading.Thread, "start", lambda self: thread_calls.append("start")):
            result = SessionManager.spawn_web_session(
                manager,
                cwd=td,
                model_provider="bytecat",
                preferred_auth_method="apikey",
                model="gpt-5.4",
                reasoning_effort="xhigh",
                service_tier="fast",
            )

        argv = popen_mock.call_args.args[0]
        env = popen_mock.call_args.kwargs["env"]
        resolved_td = str(Path(td).resolve())
        trust_override = f'projects={{ {json.dumps(str(Path(td).resolve()))} = {{ trust_level = "trusted" }} }}'
        self.assertEqual(
            argv,
            [
                ANY,
                "-m",
                "codoxear.broker",
                "--cwd",
                resolved_td,
                "--",
                "-c",
                trust_override,
                "--dangerously-bypass-approvals-and-sandbox",
                "--model",
                "gpt-5.4",
                "-c",
                'model_reasoning_effort="xhigh"',
                "-c",
                'model_provider="bytecat"',
                "-c",
                'preferred_auth_method="apikey"',
                "-c",
                'service_tier="fast"',
            ],
        )
        self.assertEqual(env["CODEX_WEB_MODEL_PROVIDER"], "bytecat")
        self.assertEqual(env["CODEX_WEB_PREFERRED_AUTH_METHOD"], "apikey")
        self.assertEqual(env["CODEX_WEB_MODEL"], "gpt-5.4")
        self.assertEqual(env["CODEX_WEB_REASONING_EFFORT"], "xhigh")
        self.assertEqual(env["CODEX_WEB_SERVICE_TIER"], "fast")
        self.assertEqual(result, {"broker_pid": 6543})
        self.assertEqual(thread_calls, ["start"])

    def test_spawn_web_session_can_start_in_tmux(self) -> None:
        manager = SessionManager.__new__(SessionManager)

        with TemporaryDirectory() as td, patch("codoxear.server.shutil.which", return_value="/usr/bin/tmux"), patch(
            "codoxear.server._wait_for_spawned_broker_meta", return_value={"broker_pid": 7777}
        ) as wait_mock, patch(
            "codoxear.server.subprocess.run",
            side_effect=[
                subprocess.CompletedProcess(["/usr/bin/tmux", "has-session", "-t", "codoxear"], 1, stdout="", stderr=""),
                subprocess.CompletedProcess(["/usr/bin/tmux", "new-session"], 0, stdout="%8\n", stderr=""),
            ],
        ) as run_mock:
            result = SessionManager.spawn_web_session(manager, cwd=td, model_provider="crs", preferred_auth_method="apikey", model="gpt-5.4", service_tier="fast", create_in_tmux=True)

        self.assertEqual(result, {"broker_pid": 7777, "tmux_session": "codoxear", "tmux_window": ANY})
        tmux_argv = run_mock.call_args_list[1].args[0]
        self.assertEqual(tmux_argv[:8], ["/usr/bin/tmux", "new-session", "-d", "-P", "-F", "#{pane_id}", "-s", "codoxear"])
        shell_cmd = tmux_argv[-1]
        self.assertIn("CODEX_WEB_TRANSPORT=tmux", shell_cmd)
        self.assertIn("CODEX_WEB_TMUX_SESSION=codoxear", shell_cmd)
        self.assertIn("CODEX_WEB_TMUX_WINDOW=", shell_cmd)
        self.assertIn("CODEX_WEB_MODEL_PROVIDER=crs", shell_cmd)
        self.assertIn("CODEX_WEB_PREFERRED_AUTH_METHOD=apikey", shell_cmd)
        self.assertIn("CODEX_WEB_MODEL=gpt-5.4", shell_cmd)
        self.assertIn("CODEX_WEB_SERVICE_TIER=fast", shell_cmd)
        self.assertIn("codoxear.broker", shell_cmd)
        wait_mock.assert_called_once()

    def test_spawn_web_session_rejects_tmux_when_unavailable(self) -> None:
        manager = SessionManager.__new__(SessionManager)
        with TemporaryDirectory() as td, patch("codoxear.server.shutil.which", return_value=None):
            with self.assertRaisesRegex(ValueError, "tmux is unavailable on this host"):
                SessionManager.spawn_web_session(manager, cwd=td, create_in_tmux=True)

    def test_spawn_web_session_rejects_resume_id_not_in_cwd(self) -> None:
        manager = SessionManager.__new__(SessionManager)
        with TemporaryDirectory() as td, patch("codoxear.server._list_resume_candidates_for_cwd", return_value=[]):
            with self.assertRaisesRegex(ValueError, "resume session not found for cwd"):
                SessionManager.spawn_web_session(manager, cwd=td, resume_session_id="missing")

    def test_spawn_web_session_accepts_agent_backend_for_pi_broker(self) -> None:
        manager = SessionManager.__new__(SessionManager)
        thread_calls: list[str] = []

        class _Proc:
            pid = 7654
            stderr = None

            def wait(self) -> int:
                return 0

        with TemporaryDirectory() as td, patch("codoxear.server._wait_or_raise", return_value=None), patch(
            "codoxear.server.subprocess.Popen", return_value=_Proc()
        ) as popen_mock, patch.object(threading.Thread, "start", lambda self: thread_calls.append("start")):
            result = SessionManager.spawn_web_session(
                manager,
                cwd=td,
                agent_backend="pi",
                model_provider="macaron",
                model="gpt-5.4",
                reasoning_effort="medium",
            )

        argv = popen_mock.call_args.args[0]
        env = popen_mock.call_args.kwargs["env"]
        self.assertEqual(
            argv,
            [
                ANY,
                "-m",
                "codoxear.pi_broker",
                "--cwd",
                str(Path(td).resolve()),
                "--session-file",
                ANY,
            ],
        )
        self.assertEqual(env["CODEX_WEB_OWNER"], "web")
        self.assertEqual(result, {"broker_pid": 7654, "backend": "pi"})
        self.assertEqual(thread_calls, ["start"])

    def test_spawn_web_session_resolves_pi_resume_from_agent_backend_alias(self) -> None:
        manager = SessionManager.__new__(SessionManager)
        thread_calls: list[str] = []

        class _Proc:
            pid = 7655
            stderr = None

            def wait(self) -> int:
                return 0

        resume_file = Path("/tmp/pi-resume.jsonl")
        with TemporaryDirectory() as td, patch("codoxear.server._list_resume_candidates_for_cwd", return_value=[{"session_id": "resume-a", "session_path": str(resume_file)}]), patch(
            "codoxear.server._wait_or_raise", return_value=None
        ), patch("codoxear.server.subprocess.Popen", return_value=_Proc()) as popen_mock, patch.object(threading.Thread, "start", lambda self: thread_calls.append("start")):
            result = SessionManager.spawn_web_session(
                manager,
                cwd=td,
                agent_backend="pi",
                resume_session_id="resume-a",
            )

        argv = popen_mock.call_args.args[0]
        env = popen_mock.call_args.kwargs["env"]
        self.assertEqual(
            argv,
            [
                ANY,
                "-m",
                "codoxear.pi_broker",
                "--cwd",
                str(Path(td).resolve()),
                "--session-file",
                str(resume_file),
            ],
        )
        self.assertEqual(env["CODEX_WEB_OWNER"], "web")
        self.assertEqual(result, {"broker_pid": 7655, "backend": "pi"})
        self.assertEqual(thread_calls, ["start"])

    def test_create_git_worktree_creates_new_checkout(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td) / "repo"
            root.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init"], cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (root / "README.md").write_text("x\n", encoding="utf-8")
            subprocess.run(["git", "add", "README.md"], cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(["git", "commit", "-m", "init"], cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            branch = "feature/test-worktree"
            worktree = _default_worktree_path(root, branch)
            result = _create_git_worktree(root, branch)
            self.assertEqual(result, worktree.resolve())
            self.assertTrue((worktree / ".git").exists())
            branch_name = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=worktree,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ).stdout.strip()
            self.assertEqual(branch_name, branch)

    def test_spawn_web_session_uses_created_worktree_as_cwd(self) -> None:
        manager = SessionManager.__new__(SessionManager)
        thread_calls: list[str] = []

        class _Proc:
            pid = 5432
            stderr = None

            def wait(self) -> int:
                return 0

        with TemporaryDirectory() as td, patch("codoxear.server._create_git_worktree", return_value=Path(td) / "repo-worktree"), patch(
            "codoxear.server._wait_or_raise", return_value=None
        ), patch("codoxear.server.subprocess.Popen", return_value=_Proc()) as popen_mock, patch.object(threading.Thread, "start", lambda self: thread_calls.append("start")):
            result = SessionManager.spawn_web_session(manager, cwd=td, worktree_branch="feature/test-worktree")

        argv = popen_mock.call_args.args[0]
        trust_override = f'projects={{ {json.dumps(str((Path(td) / "repo-worktree").resolve()))} = {{ trust_level = "trusted" }} }}'
        self.assertEqual(
            argv,
            [
                ANY,
                "-m",
                "codoxear.broker",
                "--cwd",
                str(Path(td) / "repo-worktree"),
                "--",
                "-c",
                trust_override,
                "--dangerously-bypass-approvals-and-sandbox",
            ],
        )
        self.assertEqual(result, {"broker_pid": 5432})
        self.assertEqual(thread_calls, ["start"])

    def test_spawn_web_session_rejects_worktree_when_resuming(self) -> None:
        manager = SessionManager.__new__(SessionManager)
        with TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "worktree_branch cannot be used when resuming a session"):
                SessionManager.spawn_web_session(manager, cwd=td, resume_session_id="resume-a", worktree_branch="feature/test-worktree")

    def test_spawn_web_session_rejects_worktree_outside_git(self) -> None:
        manager = SessionManager.__new__(SessionManager)
        with TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "cwd is not inside a git worktree"):
                SessionManager.spawn_web_session(manager, cwd=td, worktree_branch="feature/test-worktree")

    def test_resume_catchup_suppresses_delivery_until_resume_marker_clears(self) -> None:
        class _FakeVoicePush:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def observe_messages(self, *, session_id: str, session_display_name: str, messages: list[object]) -> None:
                self.calls.append(
                    {
                        "session_id": session_id,
                        "session_display_name": session_display_name,
                        "messages": messages,
                    }
                )

        with TemporaryDirectory() as td:
            root = Path(td)
            sock_path = root / "broker.sock"
            sock_path.touch()
            log_path = root / "rollout-2026-03-29T10-00-00-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa.jsonl"
            meta_path = sock_path.with_suffix(".json")

            replay_row = {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "phase": "final_answer",
                    "content": [{"type": "output_text", "text": "historical replay reply"}],
                },
                "ts": 1.0,
            }
            _write_jsonl(
                log_path,
                [
                    {
                        "type": "session_meta",
                        "payload": {
                            "id": "thread-1",
                            "cwd": str(root),
                            "source": "cli",
                        },
                    },
                    replay_row,
                ],
            )
            meta_path.write_text(
                json.dumps(
                    {
                        "session_id": "thread-1",
                        "owner": "web",
                        "broker_pid": 1,
                        "codex_pid": 2,
                        "cwd": str(root),
                        "start_ts": 100.0,
                        "log_path": str(log_path),
                        "sock_path": str(sock_path),
                        "resume_session_id": "resume-old",
                    }
                ),
                encoding="utf-8",
            )

            manager = SessionManager.__new__(SessionManager)
            manager._lock = threading.Lock()
            manager._sessions = {}
            manager._aliases = {}
            manager._queues = {}
            manager._voice_push = _FakeVoicePush()
            manager._discover_existing_if_stale = lambda *args, **kwargs: None  # type: ignore[method-assign]
            manager._prune_dead_sessions = lambda *args, **kwargs: None  # type: ignore[method-assign]

            session = Session(
                session_id="broker-1",
                thread_id="thread-1",
                broker_pid=1,
                codex_pid=2,
                agent_backend="codex",
                owned=True,
                start_ts=100.0,
                cwd=str(root),
                log_path=log_path,
                sock_path=sock_path,
            )
            manager._sessions[session.session_id] = session

            manager.refresh_session_meta(session.session_id)
            manager._observe_rollout_delta(session.session_id, objs=[replay_row], new_off=10)
            self.assertEqual(manager._voice_push.calls, [])
            self.assertEqual(session.delivery_log_off, 10)
            self.assertEqual(session.resume_session_id, "resume-old")

            fresh_row = {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "phase": "final_answer",
                    "content": [{"type": "output_text", "text": "fresh reply after resume"}],
                },
                "ts": 2.0,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(fresh_row) + "\n")
            meta_path.write_text(
                json.dumps(
                    {
                        "session_id": "thread-1",
                        "owner": "web",
                        "broker_pid": 1,
                        "codex_pid": 2,
                        "cwd": str(root),
                        "start_ts": 100.0,
                        "log_path": str(log_path),
                        "sock_path": str(sock_path),
                        "resume_session_id": None,
                    }
                ),
                encoding="utf-8",
            )

            manager.refresh_session_meta(session.session_id)
            manager._observe_rollout_delta(session.session_id, objs=[fresh_row], new_off=20)
            self.assertIsNone(session.resume_session_id)
            self.assertEqual(len(manager._voice_push.calls), 1)
            delivered = manager._voice_push.calls[0]
            self.assertEqual(delivered["session_id"], "broker-1")
            messages = delivered["messages"]
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0].text, "fresh reply after resume")


if __name__ == "__main__":
    unittest.main()
