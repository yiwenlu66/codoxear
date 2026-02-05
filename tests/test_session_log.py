import json
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from codoxear.util import (
    classify_session_log,
    find_session_log_for_session_id,
    is_subagent_session_meta,
    read_session_meta_payload,
    subagent_parent_thread_id,
)


def _write_jsonl(path: Path, objs: list[dict]) -> None:
    path.write_text("".join(json.dumps(o) + "\n" for o in objs), encoding="utf-8")


class TestSessionLogClassification(unittest.TestCase):
    def test_classify_main(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout-2026-02-04T00-00-00-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa.jsonl"
            _write_jsonl(p, [{"type": "session_meta", "payload": {"id": "main", "source": "cli"}}])
            self.assertEqual(classify_session_log(p), "main")

    def test_classify_subagent_and_parent(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout-2026-02-04T00-00-00-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb.jsonl"
            payload = {
                "id": "child",
                "source": {"subagent": {"thread_spawn": {"parent_thread_id": "parent", "depth": 1}}},
            }
            _write_jsonl(p, [{"type": "session_meta", "payload": payload}])
            self.assertTrue(is_subagent_session_meta(payload))
            self.assertEqual(subagent_parent_thread_id(payload), "parent")
            self.assertEqual(classify_session_log(p), "subagent")

    def test_read_session_meta_payload_waits(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout-2026-02-04T00-00-00-cccccccc-cccc-cccc-cccc-cccccccccccc.jsonl"
            p.write_text("", encoding="utf-8")

            def writer() -> None:
                time.sleep(0.15)
                _write_jsonl(p, [{"type": "session_meta", "payload": {"id": "late", "source": "cli"}}])

            t = threading.Thread(target=writer, daemon=True)
            t.start()
            payload = read_session_meta_payload(p, timeout_s=1.0, poll_s=0.02)
            self.assertIsInstance(payload, dict)
            self.assertEqual(payload.get("id"), "late")

    def test_find_session_log_for_session_id(self) -> None:
        with TemporaryDirectory() as td:
            sessions = Path(td)
            a = sessions / "2026" / "02" / "04"
            a.mkdir(parents=True, exist_ok=True)
            p = a / "rollout-2026-02-04T00-00-00-dddddddd-dddd-dddd-dddd-dddddddddddd.jsonl"
            _write_jsonl(p, [{"type": "session_meta", "payload": {"id": "dd", "source": "cli"}}])
            found = find_session_log_for_session_id(sessions, "dd")
            self.assertEqual(found, p)


if __name__ == "__main__":
    unittest.main()

