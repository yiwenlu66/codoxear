import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from codoxear.util import find_session_log_for_session_id
from codoxear.util import iter_session_logs
from codoxear.util import read_session_meta_payload


SESSION_ID = "11111111-2222-3333-4444-555555555555"


def cc_row(session_id=SESSION_ID, cwd="/repo"):
    return {
        "type": "user",
        "sessionId": session_id,
        "timestamp": "2026-06-11T00:00:00.000Z",
        "cwd": cwd,
        "message": {"role": "user", "content": "hello"},
    }


class TestCcSessionLog(unittest.TestCase):
    def test_read_cc_session_meta_payload(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / ".claude" / "projects" / "-repo" / f"{SESSION_ID}.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(cc_row()) + "\n", encoding="utf-8")
            payload = read_session_meta_payload(path, agent_backend="cc")
        self.assertEqual(payload, {"id": SESSION_ID, "sessionId": SESSION_ID, "cwd": "/repo", "timestamp": "2026-06-11T00:00:00.000Z"})

    def test_iter_cc_logs_excludes_subagents_and_history(self) -> None:
        with TemporaryDirectory() as td:
            sessions = Path(td) / ".claude" / "projects"
            main = sessions / "-repo" / f"{SESSION_ID}.jsonl"
            sub = sessions / "-repo" / SESSION_ID / "subagents" / "agent-abc.jsonl"
            history = sessions / "history.jsonl"
            main.parent.mkdir(parents=True)
            sub.parent.mkdir(parents=True)
            main.write_text(json.dumps(cc_row()) + "\n", encoding="utf-8")
            sub.write_text(json.dumps(cc_row("22222222-2222-3333-4444-555555555555")) + "\n", encoding="utf-8")
            history.write_text("{}\n", encoding="utf-8")
            self.assertEqual(iter_session_logs(sessions, agent_backend="cc"), [main])
            self.assertEqual(find_session_log_for_session_id(sessions, SESSION_ID, agent_backend="cc"), main)


if __name__ == "__main__":
    unittest.main()
