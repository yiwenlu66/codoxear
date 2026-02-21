import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from codoxear.util import proc_find_open_rollout_log, proc_open_rollout_logs


def _write_jsonl(path: Path, objs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(o) + "\n")


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


if __name__ == "__main__":
    unittest.main()
