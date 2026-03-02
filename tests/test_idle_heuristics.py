from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

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

    def test_task_complete_is_idle(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout.jsonl"
            _write_jsonl(
                p,
                [
                    {"type": "session_meta", "payload": {"id": "s"}},
                    {"type": "event_msg", "payload": {"type": "user_message", "message": "hi"}},
                    {"type": "response_item", "payload": {"type": "function_call", "call_id": "c1"}},
                    {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "c1"}},
                    {"type": "event_msg", "payload": {"type": "task_complete"}},
                ],
            )
            self.assertIs(_compute_idle_from_log(p, max_scan_bytes=64 * 1024), True)

    def test_claude_turn_duration_is_idle(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "claude.jsonl"
            _write_jsonl(
                p,
                [
                    {"type": "user", "message": {"content": [{"type": "text", "text": "hi"}]}},
                    {"type": "assistant", "message": {"content": [{"type": "text", "text": "done"}]}},
                    {"type": "system", "subtype": "turn_duration"},
                ],
            )
            self.assertIs(_compute_idle_from_log(p, max_scan_bytes=64 * 1024), True)

    def test_claude_tool_use_after_text_is_busy(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "claude.jsonl"
            _write_jsonl(
                p,
                [
                    {"type": "user", "message": {"content": [{"type": "text", "text": "hi"}]}},
                    {"type": "assistant", "message": {"content": [{"type": "text", "text": "starting"}]}},
                    {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "t1", "name": "Read"}]}},
                ],
            )
            self.assertIs(_compute_idle_from_log(p, max_scan_bytes=64 * 1024), False)

    def test_claude_api_error_is_idle(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "claude.jsonl"
            _write_jsonl(
                p,
                [
                    {"type": "user", "message": {"content": [{"type": "text", "text": "hi"}]}},
                    {"type": "system", "subtype": "api_error"},
                ],
            )
            self.assertIs(_compute_idle_from_log(p, max_scan_bytes=64 * 1024), True)

    def test_gemini_chat_json_with_assistant_reply_is_idle(self) -> None:
        with TemporaryDirectory() as td:
            gem_home = Path(td) / ".gemini"
            chats = gem_home / "tmp" / "proj" / "chats"
            chats.mkdir(parents=True, exist_ok=True)
            p = chats / "session-2026-03-02T00-00-abcd1234.json"
            p.write_text(
                json.dumps(
                    {
                        "sessionId": "aaaaaaaa-1111-2222-3333-bbbbbbbbbbbb",
                        "messages": [
                            {"type": "user", "timestamp": "2026-03-02T00:00:00.000Z", "content": [{"text": "hi"}]},
                            {"type": "gemini", "timestamp": "2026-03-02T00:00:01.000Z", "content": "done"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"GEMINI_HOME": str(gem_home)}, clear=False):
                self.assertIs(_compute_idle_from_log(p, max_scan_bytes=64 * 1024), True)

    def test_gemini_chat_json_thinking_only_stays_busy(self) -> None:
        with TemporaryDirectory() as td:
            gem_home = Path(td) / ".gemini"
            chats = gem_home / "tmp" / "proj" / "chats"
            chats.mkdir(parents=True, exist_ok=True)
            p = chats / "session-2026-03-02T00-00-abcd1234.json"
            p.write_text(
                json.dumps(
                    {
                        "sessionId": "aaaaaaaa-1111-2222-3333-bbbbbbbbbbbb",
                        "messages": [
                            {"type": "user", "timestamp": "2026-03-02T00:00:00.000Z", "content": [{"text": "hi"}]},
                            {"type": "gemini", "timestamp": "2026-03-02T00:00:01.000Z", "thoughts": [{"text": "thinking"}]},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"GEMINI_HOME": str(gem_home)}, clear=False):
                self.assertIs(_compute_idle_from_log(p, max_scan_bytes=64 * 1024), False)


if __name__ == "__main__":
    unittest.main()
