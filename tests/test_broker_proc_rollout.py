from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from codoxear.broker import _find_recent_claude_project_log
from codoxear.broker import _find_recent_gemini_chat_log
from codoxear.util import proc_find_open_rollout_log, proc_open_rollout_logs


def _write_jsonl(path: Path, objs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(o) + "\n")


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


class TestBrokerProcRolloutDiscovery(unittest.TestCase):
    def test_proc_finds_rollout_log_open_in_descendant(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            proc_root = root / "proc"
            logs = root / "logs"
            logs.mkdir(parents=True, exist_ok=True)

            want = logs / "rollout-2026-02-04T00-00-02-11111111-1111-1111-1111-111111111111.jsonl"
            other = logs / "rollout-2026-02-04T00-00-03-22222222-2222-2222-2222-222222222222.jsonl"
            _write_jsonl(want, [{"type": "session_meta", "payload": {"id": "want", "cwd": "/x", "source": "cli"}}])
            _write_jsonl(other, [{"type": "session_meta", "payload": {"id": "other", "cwd": "/y", "source": "cli"}}])

            t0 = 1000.0
            os.utime(want, (t0 + 10, t0 + 10))
            os.utime(other, (t0 + 20, t0 + 20))

            # Fake process tree:
            # 100
            #  |-- 101 (opens other)
            #  `-- 102 (opens want)
            (proc_root / "100" / "task" / "100").mkdir(parents=True, exist_ok=True)
            (proc_root / "101" / "task" / "101").mkdir(parents=True, exist_ok=True)
            (proc_root / "102" / "task" / "102").mkdir(parents=True, exist_ok=True)
            (proc_root / "100" / "task" / "100" / "children").write_text("101 102\n", encoding="utf-8")
            (proc_root / "101" / "task" / "101" / "children").write_text("\n", encoding="utf-8")
            (proc_root / "102" / "task" / "102" / "children").write_text("\n", encoding="utf-8")

            for pid in ("100", "101", "102"):
                (proc_root / pid / "fd").mkdir(parents=True, exist_ok=True)

            os.symlink(str(other), proc_root / "101" / "fd" / "3")
            os.symlink(str(want), proc_root / "102" / "fd" / "4")

            opened = proc_open_rollout_logs(proc_root, 100)
            self.assertIn(want, opened)
            self.assertIn(other, opened)

            found = proc_find_open_rollout_log(proc_root=proc_root, root_pid=100, cwd="/x")
            self.assertEqual(found, want)

    def test_proc_disambiguates_same_cwd_by_root_pid(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            proc_root = root / "proc"
            logs = root / "logs"
            logs.mkdir(parents=True, exist_ok=True)

            a = logs / "rollout-2026-02-04T00-00-02-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa.jsonl"
            b = logs / "rollout-2026-02-04T00-00-03-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb.jsonl"
            _write_jsonl(a, [{"type": "session_meta", "payload": {"id": "a", "cwd": "/x", "source": "cli"}}])
            _write_jsonl(b, [{"type": "session_meta", "payload": {"id": "b", "cwd": "/x", "source": "cli"}}])
            os.utime(a, (1000.0, 1000.0))
            os.utime(b, (2000.0, 2000.0))

            # Fake two independent process trees.
            # 100 -> 101 opens a
            # 200 -> 201 opens b
            for pid in ("100", "101", "200", "201"):
                (proc_root / pid / "task" / pid).mkdir(parents=True, exist_ok=True)
                (proc_root / pid / "task" / pid / "children").write_text("\n", encoding="utf-8")
                (proc_root / pid / "fd").mkdir(parents=True, exist_ok=True)
            (proc_root / "100" / "task" / "100" / "children").write_text("101\n", encoding="utf-8")
            (proc_root / "200" / "task" / "200" / "children").write_text("201\n", encoding="utf-8")

            os.symlink(str(a), proc_root / "101" / "fd" / "3")
            os.symlink(str(b), proc_root / "201" / "fd" / "4")

            found_a = proc_find_open_rollout_log(proc_root=proc_root, root_pid=100, cwd="/x")
            found_b = proc_find_open_rollout_log(proc_root=proc_root, root_pid=200, cwd="/x")
            self.assertEqual(found_a, a)
            self.assertEqual(found_b, b)

    def test_proc_finds_claude_project_log_by_cwd(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            proc_root = root / "proc"
            claude_home = root / ".claude-home"
            projects = claude_home / "projects" / "workspace"
            projects.mkdir(parents=True, exist_ok=True)

            want = projects / "11111111-1111-1111-1111-111111111111.jsonl"
            _write_jsonl(
                want,
                [
                    {"type": "system", "cwd": "/claude/work", "subtype": "init"},
                    {"type": "user", "message": {"content": [{"type": "text", "text": "hello"}]}},
                ],
            )

            (proc_root / "100" / "task" / "100").mkdir(parents=True, exist_ok=True)
            (proc_root / "101" / "task" / "101").mkdir(parents=True, exist_ok=True)
            (proc_root / "100" / "task" / "100" / "children").write_text("101\n", encoding="utf-8")
            (proc_root / "101" / "task" / "101" / "children").write_text("\n", encoding="utf-8")
            for pid in ("100", "101"):
                (proc_root / pid / "fd").mkdir(parents=True, exist_ok=True)
            os.symlink(str(want), proc_root / "101" / "fd" / "3")

            with patch.dict(os.environ, {"CLAUDE_HOME": str(claude_home)}, clear=False):
                opened = proc_open_rollout_logs(proc_root, 100)
                self.assertIn(want, opened)
                found = proc_find_open_rollout_log(proc_root=proc_root, root_pid=100, cwd="/claude/work")
                self.assertEqual(found, want)

    def test_proc_finds_gemini_chat_log_by_cwd(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            proc_root = root / "proc"
            gem_home = root / ".gemini-home"
            project = gem_home / "tmp" / "workspace"
            chats = project / "chats"
            chats.mkdir(parents=True, exist_ok=True)
            (project / ".project_root").write_text("/gemini/work\n", encoding="utf-8")

            want = chats / "session-2026-03-02T00-00-abcd1234.json"
            _write_json(
                want,
                {
                    "sessionId": "aaaaaaaa-1111-2222-3333-bbbbbbbbbbbb",
                    "messages": [{"type": "user", "content": [{"text": "hello"}], "timestamp": "2026-03-02T00:00:00.000Z"}],
                },
            )

            (proc_root / "100" / "task" / "100").mkdir(parents=True, exist_ok=True)
            (proc_root / "101" / "task" / "101").mkdir(parents=True, exist_ok=True)
            (proc_root / "100" / "task" / "100" / "children").write_text("101\n", encoding="utf-8")
            (proc_root / "101" / "task" / "101" / "children").write_text("\n", encoding="utf-8")
            for pid in ("100", "101"):
                (proc_root / pid / "fd").mkdir(parents=True, exist_ok=True)
            os.symlink(str(want), proc_root / "101" / "fd" / "3")

            with patch.dict(os.environ, {"GEMINI_HOME": str(gem_home)}, clear=False):
                opened = proc_open_rollout_logs(proc_root, 100)
                self.assertIn(want, opened)
                found = proc_find_open_rollout_log(proc_root=proc_root, root_pid=100, cwd="/gemini/work")
                self.assertEqual(found, want)

    def test_broker_fallback_finds_recent_claude_log(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            projects = root / "projects" / "workspace"
            projects.mkdir(parents=True, exist_ok=True)

            stale = projects / "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa.jsonl"
            wrong = projects / "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb.jsonl"
            want = projects / "cccccccc-cccc-cccc-cccc-cccccccccccc.jsonl"
            _write_jsonl(stale, [{"type": "system", "cwd": "/want", "subtype": "init"}])
            _write_jsonl(wrong, [{"type": "system", "cwd": "/other", "subtype": "init"}])
            _write_jsonl(want, [{"type": "system", "cwd": "/want", "subtype": "init"}])

            os.utime(stale, (1000.0, 1000.0))
            os.utime(wrong, (1015.0, 1015.0))
            os.utime(want, (1020.0, 1020.0))

            found = _find_recent_claude_project_log(sessions_dir=root / "projects", cwd="/want", after_ts=1010.0)
            self.assertEqual(found, want)

    def test_broker_fallback_ignores_prestart_logs(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            projects = root / "projects" / "workspace"
            projects.mkdir(parents=True, exist_ok=True)

            old = projects / "dddddddd-dddd-dddd-dddd-dddddddddddd.jsonl"
            _write_jsonl(old, [{"type": "system", "cwd": "/want", "subtype": "init"}])
            os.utime(old, (1000.0, 1000.0))

            found = _find_recent_claude_project_log(sessions_dir=root / "projects", cwd="/want", after_ts=2000.0)
            self.assertIsNone(found)

    def test_broker_fallback_finds_recent_gemini_log(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            tmp = root / "tmp"
            stale_proj = tmp / "stale"
            wrong_proj = tmp / "wrong"
            want_proj = tmp / "want"
            for proj in (stale_proj, wrong_proj, want_proj):
                (proj / "chats").mkdir(parents=True, exist_ok=True)
            (stale_proj / ".project_root").write_text("/want\n", encoding="utf-8")
            (wrong_proj / ".project_root").write_text("/other\n", encoding="utf-8")
            (want_proj / ".project_root").write_text("/want\n", encoding="utf-8")

            stale = stale_proj / "chats" / "session-2026-03-02T00-00-aaaa1111.json"
            wrong = wrong_proj / "chats" / "session-2026-03-02T00-00-bbbb2222.json"
            want = want_proj / "chats" / "session-2026-03-02T00-00-cccc3333.json"
            _write_json(stale, {"sessionId": "aaaaaaaa-1111-2222-3333-bbbbbbbbbbbb", "messages": []})
            _write_json(wrong, {"sessionId": "aaaaaaaa-1111-2222-3333-cccccccccccc", "messages": []})
            _write_json(want, {"sessionId": "aaaaaaaa-1111-2222-3333-dddddddddddd", "messages": []})

            os.utime(stale, (1000.0, 1000.0))
            os.utime(wrong, (1015.0, 1015.0))
            os.utime(want, (1020.0, 1020.0))

            found = _find_recent_gemini_chat_log(sessions_dir=tmp, cwd="/want", after_ts=1010.0)
            self.assertEqual(found, want)


if __name__ == "__main__":
    unittest.main()
