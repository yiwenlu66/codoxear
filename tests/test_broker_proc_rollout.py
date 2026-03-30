import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from codoxear.util import proc_find_open_rollout_log, proc_open_rollout_logs


def _write_jsonl(path: Path, objs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(o) + "\n")


def _ensure_proc_pid(proc_root: Path, pid: str) -> None:
    (proc_root / pid / "task" / pid).mkdir(parents=True, exist_ok=True)
    (proc_root / pid / "task" / pid / "children").write_text("\n", encoding="utf-8")
    (proc_root / pid / "fd").mkdir(parents=True, exist_ok=True)
    (proc_root / pid / "fdinfo").mkdir(parents=True, exist_ok=True)


def _link_fd(proc_root: Path, pid: str, fd: str, target: Path, *, flags_octal: str) -> None:
    os.symlink(str(target), proc_root / pid / "fd" / fd)
    (proc_root / pid / "fdinfo" / fd).write_text(f"flags:\t{flags_octal}\n", encoding="utf-8")


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
            _ensure_proc_pid(proc_root, "100")
            _ensure_proc_pid(proc_root, "101")
            _ensure_proc_pid(proc_root, "102")
            (proc_root / "100" / "task" / "100" / "children").write_text("101 102\n", encoding="utf-8")
            _link_fd(proc_root, "101", "3", other, flags_octal="0100001")
            _link_fd(proc_root, "102", "4", want, flags_octal="0100001")

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
                _ensure_proc_pid(proc_root, pid)
            (proc_root / "100" / "task" / "100" / "children").write_text("101\n", encoding="utf-8")
            (proc_root / "200" / "task" / "200" / "children").write_text("201\n", encoding="utf-8")

            _link_fd(proc_root, "101", "3", a, flags_octal="0100001")
            _link_fd(proc_root, "201", "4", b, flags_octal="0100001")

            found_a = proc_find_open_rollout_log(proc_root=proc_root, root_pid=100, cwd="/x")
            found_b = proc_find_open_rollout_log(proc_root=proc_root, root_pid=200, cwd="/x")
            self.assertEqual(found_a, a)
            self.assertEqual(found_b, b)

    def test_proc_prefers_writable_log_over_newer_read_only_log(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            proc_root = root / "proc"
            logs = root / "logs"
            logs.mkdir(parents=True, exist_ok=True)

            old = logs / "rollout-2026-02-04T00-00-02-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa.jsonl"
            want = logs / "rollout-2026-02-04T00-00-03-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb.jsonl"
            _write_jsonl(old, [{"type": "session_meta", "payload": {"id": "old", "cwd": "/x", "source": "cli"}}])
            _write_jsonl(want, [{"type": "session_meta", "payload": {"id": "want", "cwd": "/x", "source": "cli"}}])
            os.utime(old, (2000.0, 2000.0))
            os.utime(want, (1000.0, 1000.0))

            _ensure_proc_pid(proc_root, "100")
            _ensure_proc_pid(proc_root, "101")
            (proc_root / "100" / "task" / "100" / "children").write_text("101\n", encoding="utf-8")

            _link_fd(proc_root, "101", "3", old, flags_octal="0100000")
            _link_fd(proc_root, "101", "4", want, flags_octal="0100001")

            found = proc_find_open_rollout_log(proc_root=proc_root, root_pid=100, cwd="/x")
            self.assertEqual(found, want)

    def test_proc_returns_none_when_multiple_writable_same_cwd_candidates_exist(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            proc_root = root / "proc"
            logs = root / "logs"
            logs.mkdir(parents=True, exist_ok=True)

            a = logs / "rollout-2026-02-04T00-00-02-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa.jsonl"
            b = logs / "rollout-2026-02-04T00-00-03-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb.jsonl"
            _write_jsonl(a, [{"type": "session_meta", "payload": {"id": "a", "cwd": "/x", "source": "cli"}}])
            _write_jsonl(b, [{"type": "session_meta", "payload": {"id": "b", "cwd": "/x", "source": "cli"}}])

            _ensure_proc_pid(proc_root, "100")
            _ensure_proc_pid(proc_root, "101")
            (proc_root / "100" / "task" / "100" / "children").write_text("101\n", encoding="utf-8")

            _link_fd(proc_root, "101", "3", a, flags_octal="0100001")
            _link_fd(proc_root, "101", "4", b, flags_octal="0100001")

            found = proc_find_open_rollout_log(proc_root=proc_root, root_pid=100, cwd="/x")
            self.assertIsNone(found)

    def test_proc_ignores_explicitly_ignored_paths(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            proc_root = root / "proc"
            logs = root / "logs"
            logs.mkdir(parents=True, exist_ok=True)

            old = logs / "rollout-2026-02-04T00-00-02-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa.jsonl"
            want = logs / "rollout-2026-02-04T00-00-03-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb.jsonl"
            _write_jsonl(old, [{"type": "session_meta", "payload": {"id": "old", "cwd": "/x", "source": "cli"}}])
            _write_jsonl(want, [{"type": "session_meta", "payload": {"id": "want", "cwd": "/x", "source": "cli"}}])

            _ensure_proc_pid(proc_root, "100")
            _ensure_proc_pid(proc_root, "101")
            (proc_root / "100" / "task" / "100" / "children").write_text("101\n", encoding="utf-8")

            _link_fd(proc_root, "101", "3", old, flags_octal="0100001")
            _link_fd(proc_root, "101", "4", want, flags_octal="0100001")

            found = proc_find_open_rollout_log(
                proc_root=proc_root,
                root_pid=100,
                cwd="/x",
                ignored_paths={old},
            )
            self.assertEqual(found, want)

    def test_proc_finds_pi_session_log_open_in_descendant(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            proc_root = root / "proc"
            pi_home = root / ".pi"
            sessions = pi_home / "agent" / "sessions" / "--pi-work--"
            want = sessions / "2026-02-04T00-00-02-11111111-1111-1111-1111-111111111111.jsonl"
            other = sessions / "2026-02-04T00-00-03-22222222-2222-2222-2222-222222222222.jsonl"
            _write_jsonl(want, [{"type": "session", "id": "want", "cwd": "/x", "timestamp": "2026-02-04T00:00:00Z"}])
            _write_jsonl(other, [{"type": "session", "id": "other", "cwd": "/y", "timestamp": "2026-02-04T00:00:00Z"}])

            _ensure_proc_pid(proc_root, "100")
            _ensure_proc_pid(proc_root, "101")
            _ensure_proc_pid(proc_root, "102")
            (proc_root / "100" / "task" / "100" / "children").write_text("101 102\n", encoding="utf-8")
            _link_fd(proc_root, "101", "3", other, flags_octal="0100001")
            _link_fd(proc_root, "102", "4", want, flags_octal="0100001")

            with patch.dict(os.environ, {"PI_HOME": str(pi_home)}, clear=False):
                opened = proc_open_rollout_logs(proc_root, 100, agent_backend="pi")
                self.assertIn(want, opened)
                self.assertIn(other, opened)
                found = proc_find_open_rollout_log(proc_root=proc_root, root_pid=100, agent_backend="pi", cwd="/x")
            self.assertEqual(found, want)


if __name__ == "__main__":
    unittest.main()
