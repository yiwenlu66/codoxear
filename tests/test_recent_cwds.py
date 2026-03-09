import json
import os
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from codoxear.server import RECENT_CWD_MAX
from codoxear.server import SessionManager


def _write_jsonl(path: Path, objs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(obj) + "\n" for obj in objs), encoding="utf-8")


def _make_manager() -> SessionManager:
    mgr = SessionManager.__new__(SessionManager)
    mgr._lock = threading.Lock()
    mgr._recent_cwds = {}
    return mgr


class TestRecentCwds(unittest.TestCase):
    def test_backfill_recent_cwds_uses_rollout_log_history(self) -> None:
        mgr = _make_manager()
        with TemporaryDirectory() as td:
            root = Path(td)
            newest = root / "rollout-2026-03-08T03-00-00-cccccccc-cccc-cccc-cccc-cccccccccccc.jsonl"
            dup = root / "rollout-2026-03-08T02-00-00-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb.jsonl"
            older = root / "rollout-2026-03-08T01-00-00-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa.jsonl"
            _write_jsonl(newest, [{"type": "session_meta", "payload": {"id": "c", "cwd": "/repo/newest", "source": "cli"}}])
            _write_jsonl(dup, [{"type": "session_meta", "payload": {"id": "b", "cwd": "/repo/shared", "source": "cli"}}])
            _write_jsonl(older, [{"type": "session_meta", "payload": {"id": "a", "cwd": "/repo/shared", "source": "cli"}}])
            os.utime(newest, (300, 300))
            os.utime(dup, (200, 200))
            os.utime(older, (100, 100))
            with patch("codoxear.server._iter_session_logs", return_value=[newest, dup, older]):
                with patch.object(SessionManager, "_save_recent_cwds", lambda self: None):
                    SessionManager._backfill_recent_cwds_from_logs(mgr)

        self.assertEqual(mgr.recent_cwds(limit=RECENT_CWD_MAX), ["/repo/newest", "/repo/shared"])


if __name__ == "__main__":
    unittest.main()
