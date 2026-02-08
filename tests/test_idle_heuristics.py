import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from codoxear.server import _compute_idle_from_log


def _write_jsonl(path: Path, objs: list[dict]) -> None:
    path.write_text("".join(json.dumps(o) + "\n" for o in objs), encoding="utf-8")


class TestIdleHeuristics(unittest.TestCase):
    def test_fresh_session_is_idle(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout.jsonl"
            _write_jsonl(p, [{"type": "session_meta", "payload": {"id": "s"}}])
            self.assertIs(_compute_idle_from_log(p, max_scan_bytes=64 * 1024), True)

    def test_open_turn_without_assistant_candidate_is_busy(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout.jsonl"
            _write_jsonl(
                p,
                [
                    {"type": "session_meta", "payload": {"id": "s"}},
                    {"type": "event_msg", "payload": {"type": "user_message", "message": "hi"}},
                    {"type": "event_msg", "payload": {"type": "agent_reasoning"}},
                ],
            )
            self.assertIs(_compute_idle_from_log(p, max_scan_bytes=64 * 1024), False)

    def test_assistant_candidate_after_user_is_idle(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout.jsonl"
            _write_jsonl(
                p,
                [
                    {"type": "session_meta", "payload": {"id": "s"}},
                    {"type": "event_msg", "payload": {"type": "user_message", "message": "hi"}},
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "done"}],
                        },
                    },
                ],
            )
            self.assertIs(_compute_idle_from_log(p, max_scan_bytes=64 * 1024), True)

    def test_tool_after_assistant_reopens_busy(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout.jsonl"
            _write_jsonl(
                p,
                [
                    {"type": "session_meta", "payload": {"id": "s"}},
                    {"type": "event_msg", "payload": {"type": "user_message", "message": "hi"}},
                    {"type": "event_msg", "payload": {"type": "agent_message", "message": "starting"}},
                    {"type": "response_item", "payload": {"type": "function_call", "call_id": "c1"}},
                ],
            )
            self.assertIs(_compute_idle_from_log(p, max_scan_bytes=64 * 1024), False)

    def test_web_search_after_assistant_reopens_busy(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout.jsonl"
            _write_jsonl(
                p,
                [
                    {"type": "session_meta", "payload": {"id": "s"}},
                    {"type": "event_msg", "payload": {"type": "user_message", "message": "hi"}},
                    {"type": "event_msg", "payload": {"type": "agent_message", "message": "starting"}},
                    {"type": "response_item", "payload": {"type": "web_search_call", "status": "completed"}},
                ],
            )
            self.assertIs(_compute_idle_from_log(p, max_scan_bytes=64 * 1024), False)

    def test_local_shell_after_assistant_reopens_busy(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout.jsonl"
            _write_jsonl(
                p,
                [
                    {"type": "session_meta", "payload": {"id": "s"}},
                    {"type": "event_msg", "payload": {"type": "user_message", "message": "hi"}},
                    {"type": "event_msg", "payload": {"type": "agent_message", "message": "starting"}},
                    {"type": "response_item", "payload": {"type": "local_shell_call", "status": "completed"}},
                ],
            )
            self.assertIs(_compute_idle_from_log(p, max_scan_bytes=64 * 1024), False)

    def test_turn_aborted_is_idle(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout.jsonl"
            _write_jsonl(
                p,
                [
                    {"type": "session_meta", "payload": {"id": "s"}},
                    {"type": "event_msg", "payload": {"type": "user_message", "message": "hi"}},
                    {"type": "event_msg", "payload": {"type": "turn_aborted"}},
                ],
            )
            self.assertIs(_compute_idle_from_log(p, max_scan_bytes=64 * 1024), True)


if __name__ == "__main__":
    unittest.main()
