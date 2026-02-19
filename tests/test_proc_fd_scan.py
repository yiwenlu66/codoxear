import os
import tempfile
import unittest
from pathlib import Path

from codoxear.broker import (
    _fd_is_writable,
    _iter_writable_rollout_paths,
    _proc_descendants,
    _proc_descendants_owned,
    _proc_pid_uid,
    _rollout_path_from_fd_link,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class TestProcFdScan(unittest.TestCase):
    def test_proc_descendants_from_children_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            proc_root = Path(td)
            _write(proc_root / "100" / "task" / "100" / "children", "101 102\n")
            _write(proc_root / "101" / "task" / "101" / "children", "103\n")
            _write(proc_root / "102" / "task" / "102" / "children", "\n")
            _write(proc_root / "103" / "task" / "103" / "children", "\n")

            self.assertEqual(_proc_descendants(proc_root, 100), {100, 101, 102, 103})

    def test_proc_descendants_raises_children_permission_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            proc_root = Path(td)
            p = proc_root / "100" / "task" / "100" / "children"
            _write(p, "101\n")
            try:
                os.chmod(p, 0o000)
                with self.assertRaises(PermissionError):
                    _proc_descendants(proc_root, 100)
            finally:
                os.chmod(p, 0o600)

    def test_proc_pid_uid_missing_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            proc_root = Path(td)
            self.assertIsNone(_proc_pid_uid(proc_root, 999))

    def test_proc_descendants_owned_skips_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            proc_root = Path(td)
            # PID directory missing.
            self.assertEqual(_proc_descendants_owned(proc_root, 100, uid=os.getuid()), set())

    def test_rollout_path_from_fd_link(self) -> None:
        self.assertEqual(
            _rollout_path_from_fd_link("/x/rollout-abc.jsonl"),
            Path("/x/rollout-abc.jsonl"),
        )
        self.assertEqual(
            _rollout_path_from_fd_link("/x/rollout-abc.jsonl (deleted)"),
            Path("/x/rollout-abc.jsonl"),
        )
        self.assertIsNone(_rollout_path_from_fd_link("/x/nope.jsonl"))
        self.assertIsNone(_rollout_path_from_fd_link("rollout-abc.jsonl"))

    def test_fd_is_writable_from_fdinfo_flags(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            proc_root = Path(td)
            _write(proc_root / "200" / "fdinfo" / "3", "flags:\t0100000\n")
            _write(proc_root / "200" / "fdinfo" / "4", "flags:\t0100001\n")
            _write(proc_root / "200" / "fdinfo" / "5", "flags:\t0100002\n")

            self.assertFalse(_fd_is_writable(proc_root, 200, 3))
            self.assertTrue(_fd_is_writable(proc_root, 200, 4))
            self.assertTrue(_fd_is_writable(proc_root, 200, 5))

    def test_iter_writable_rollout_paths_filters_sessions_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            proc_root = root / "proc"
            sessions_dir = root / "sessions"

            rollout = sessions_dir / "2026" / "01" / "01" / "rollout-abc.jsonl"
            rollout.parent.mkdir(parents=True, exist_ok=True)
            rollout.write_text("", encoding="utf-8")

            outside = root / "other" / "rollout-zzz.jsonl"
            outside.parent.mkdir(parents=True, exist_ok=True)
            outside.write_text("", encoding="utf-8")

            fd_dir = proc_root / "300" / "fd"
            fd_dir.mkdir(parents=True, exist_ok=True)
            (proc_root / "300" / "fdinfo").mkdir(parents=True, exist_ok=True)

            os.symlink(str(rollout), str(fd_dir / "3"))
            _write(proc_root / "300" / "fdinfo" / "3", "flags:\t0100001\n")

            os.symlink(str(outside), str(fd_dir / "4"))
            _write(proc_root / "300" / "fdinfo" / "4", "flags:\t0100001\n")

            os.symlink(str(rollout), str(fd_dir / "5"))
            _write(proc_root / "300" / "fdinfo" / "5", "flags:\t0100000\n")

            got = _iter_writable_rollout_paths(proc_root=proc_root, pid=300, sessions_dir=sessions_dir)
            self.assertEqual(got, [rollout.resolve()])

    def test_iter_writable_rollout_paths_raises_fd_permission_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            proc_root = root / "proc"
            sessions_dir = root / "sessions"
            sessions_dir.mkdir(parents=True, exist_ok=True)

            fd_dir = proc_root / "400" / "fd"
            fd_dir.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(fd_dir, 0o000)
                with self.assertRaises(PermissionError):
                    _iter_writable_rollout_paths(proc_root=proc_root, pid=400, sessions_dir=sessions_dir)
            finally:
                os.chmod(fd_dir, 0o700)
